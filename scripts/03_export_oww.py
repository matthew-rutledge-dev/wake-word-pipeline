#!/usr/bin/env python3
"""
Export trained OWW model to ONNX + TFLite.

ONNX-to-TFLite conversion is a graph transformation that does not benefit
from GPU. CUDA_VISIBLE_DEVICES is set to -1 to prevent TensorFlow from
allocating GPU memory while other pipeline scripts may be training.

Usage: python 03_export_oww.py <word_id>
"""
import _compat  # noqa: F401 — must be first
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def load_config(word_id: str) -> dict:
    cfg_path = REPO_DIR / "words" / word_id / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def convert_onnx_to_tflite(onnx_path: str, tflite_path: str):
    """Convert OWW DNN ONNX model to TFLite format (CPU only).

    Architecture: Reshape -> Gemm+LayerNorm+ReLU (xN blocks) -> Gemm+Sigmoid
    """
    try:
        from openwakeword.train import convert_onnx_to_tflite as oww_convert
        oww_convert(onnx_path, tflite_path)
        return
    except (ImportError, AttributeError):
        pass

    import onnx
    from onnx import numpy_helper
    import tensorflow as tf

    log.info("Converting ONNX to TFLite (manual fallback)")
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    # Collect Gemm and LayerNorm nodes in order
    gemm_nodes = [n for n in graph.node if n.op_type == "Gemm"]
    ln_nodes = [n for n in graph.node if n.op_type == "LayerNormalization"]

    # Build Keras functional model matching the ONNX graph
    inp = tf.keras.Input(shape=(16, 96))
    x = tf.keras.layers.Flatten()(inp)

    for gi, gemm in enumerate(gemm_nodes):
        w = weights[gemm.input[1]]
        units = w.shape[0]
        has_bias = len(gemm.input) > 2 and gemm.input[2] in weights
        is_last = (gi == len(gemm_nodes) - 1)

        dense = tf.keras.layers.Dense(
            units, use_bias=has_bias,
            activation="sigmoid" if is_last else None,
            name=f"dense_{gi}",
        )
        x = dense(x)

        # Set Dense weights (Gemm convention: weight is transposed)
        w_list = [w.T]
        if has_bias:
            w_list.append(weights[gemm.input[2]])
        dense.set_weights(w_list)

        if not is_last and gi < len(ln_nodes):
            ln = ln_nodes[gi]
            ln_layer = tf.keras.layers.LayerNormalization(name=f"ln_{gi}")
            x = ln_layer(x)
            ln_layer.set_weights([weights[ln.input[1]], weights[ln.input[2]]])
            x = tf.keras.layers.ReLU()(x)

    model = tf.keras.Model(inputs=inp, outputs=x)

    # Verify ONNX vs TF match
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    input_name = graph.input[0].name
    test_in = np.random.randn(1, 16, 96).astype(np.float32)
    onnx_out = sess.run(None, {input_name: test_in})[0]
    tf_out = model.predict(test_in, verbose=0)
    diff = float(np.max(np.abs(onnx_out - tf_out)))
    log.info("ONNX vs TF max diff: %.6f", diff)
    if diff > 0.01:
        raise ValueError(f"Model mismatch too large: {diff}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


def validate_tflite(tflite_path: str):
    """Validate TFLite model has correct shape."""
    try:
        import ai_edge_litert.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=tflite_path)
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=tflite_path)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_shape = input_details[0]["shape"]
    out_shape = output_details[0]["shape"]

    log.info("TFLite input shape:  %s dtype=%s", in_shape, input_details[0]["dtype"])
    log.info("TFLite output shape: %s dtype=%s", out_shape, output_details[0]["dtype"])

    # Smoke test with random data
    test_input = np.random.randn(*in_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    log.info("Smoke test output: %s (range [%.4f, %.4f])", output.shape, output.min(), output.max())

    if output.shape[-1] != 1:
        log.error("Expected output dim 1 (binary), got %d", output.shape[-1])
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Export OWW model to ONNX + TFLite")
    parser.add_argument("word_id", help="Wake word identifier")
    args = parser.parse_args()

    cfg = load_config(args.word_id)
    word_id = cfg["word_id"]
    artifact_dir = REPO_DIR / "artifacts" / word_id / "oww"

    onnx_path = artifact_dir / f"{word_id}.onnx"
    tflite_path = artifact_dir / f"{word_id}.tflite"

    if not onnx_path.exists():
        log.error("ONNX model not found: %s — run 02_train_oww.py first", onnx_path)
        sys.exit(1)

    # Convert ONNX → TFLite
    log.info("Converting %s → %s", onnx_path, tflite_path)
    convert_onnx_to_tflite(str(onnx_path), str(tflite_path))

    fsize = tflite_path.stat().st_size
    log.info("TFLite model: %s (%.1f KB)", tflite_path.name, fsize / 1024)

    # Validate
    if validate_tflite(str(tflite_path)):
        log.info("Export complete — model validated successfully")
    else:
        log.error("Model validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
