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
    """Convert ONNX model to TFLite format (CPU only)."""
    try:
        from openwakeword.train import convert_onnx_to_tflite as oww_convert
        oww_convert(onnx_path, tflite_path)
        return
    except ImportError:
        pass

    # Manual conversion fallback
    import onnx
    from onnx import numpy_helper
    import tensorflow as tf

    log.info("Converting ONNX → TFLite manually")
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    input_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]

    # Build equivalent TF model from ONNX weights
    weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    layers = []
    layers.append(tf.keras.layers.InputLayer(input_shape=input_shape[1:]))
    layers.append(tf.keras.layers.Flatten())

    for node in graph.node:
        if node.op_type == "MatMul":
            w_name = node.input[1]
            w = weights[w_name]
            units = w.shape[1]
            layers.append(tf.keras.layers.Dense(units, use_bias=False))
        elif node.op_type == "Add":
            pass  # bias handled by Dense
        elif node.op_type == "Relu":
            layers.append(tf.keras.layers.ReLU())
        elif node.op_type == "Sigmoid":
            layers.append(tf.keras.layers.Activation("sigmoid"))

    model = tf.keras.Sequential(layers)
    model.build(input_shape=[1] + input_shape[1:])

    # Load weights
    for i, layer in enumerate(model.layers):
        if hasattr(layer, "kernel"):
            for node in graph.node:
                if node.op_type == "MatMul":
                    w = weights[node.input[1]]
                    b_name = None
                    for n2 in graph.node:
                        if n2.op_type == "Add" and node.output[0] in n2.input:
                            b_name = [x for x in n2.input if x != node.output[0]][0]
                            break
                    if b_name and b_name in weights:
                        layer.set_weights([w, weights[b_name]])
                    else:
                        layer.set_weights([w])
                    break

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
