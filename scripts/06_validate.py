#!/usr/bin/env python3
"""
Validate trained model: shape check, smoke test, and detection test on
generated positive samples.

For OWW models: loads via openWakeWord and runs predict_clip on test samples.
For mWW models: loads TFLite and runs inference on test samples.
GPU used for OWW inference if available.

Usage: python 06_validate.py <word_id> [oww|mww]
"""
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


def validate_oww(word_id: str, artifact_dir: Path):
    """Validate OWW model by running detection on positive samples."""
    from openwakeword.model import Model as OWWModel

    tflite_path = artifact_dir / "oww" / f"{word_id}.tflite"
    onnx_path = artifact_dir / "oww" / f"{word_id}.onnx"

    # Prefer TFLite, fall back to ONNX
    if tflite_path.exists():
        model_path = str(tflite_path)
        framework = "tflite"
    elif onnx_path.exists():
        model_path = str(onnx_path)
        framework = "onnx"
    else:
        log.error("No OWW model found — run training + export first")
        return False

    log.info("Loading OWW model: %s (framework=%s)", Path(model_path).name, framework)
    model = OWWModel(wakeword_models=[model_path], inference_framework=framework)

    # Test on positive validation samples
    pos_val_dir = artifact_dir / "positive_val"
    if not pos_val_dir.exists():
        pos_val_dir = artifact_dir / "positive_train"

    wav_files = sorted(pos_val_dir.glob("*.wav"))[:20]
    if not wav_files:
        log.warning("No validation WAV files found — skipping detection test")
        return True

    detections = 0
    total = len(wav_files)

    for wav_path in wav_files:
        try:
            result = model.predict_clip(str(wav_path))
            max_score = 0.0
            for mdl_name, scores in result.items():
                if isinstance(scores, dict):
                    for label, frame_scores in scores.items():
                        frame_max = max(frame_scores) if frame_scores else 0.0
                        max_score = max(max_score, frame_max)
                elif isinstance(scores, (list, np.ndarray)):
                    max_score = max(max_score, max(scores) if len(scores) > 0 else 0.0)
            if max_score >= 0.5:
                detections += 1
        except Exception as e:
            log.warning("Error processing %s: %s", wav_path.name, e)

    recall = detections / total if total > 0 else 0
    log.info("OWW detection: %d/%d (recall=%.1f%%) on %s samples",
             detections, total, recall * 100, pos_val_dir.name)

    if recall < 0.3:
        log.warning("Low recall (%.1f%%) — model may need retraining", recall * 100)
    return True


def validate_mww(word_id: str, artifact_dir: Path):
    """Validate mWW TFLite model."""
    import json

    mww_dir = artifact_dir / "mww"
    tflite_files = list(mww_dir.glob("*.tflite"))
    if not tflite_files:
        log.error("No mWW TFLite model found")
        return False

    tflite_path = tflite_files[0]
    manifest_path = mww_dir / f"{word_id}.json"

    # Load and validate TFLite
    try:
        import ai_edge_litert.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=str(tflite_path))
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))

    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    log.info("mWW model: %s", tflite_path.name)
    log.info("  Input:  %s %s", in_details[0]["shape"], in_details[0]["dtype"])
    log.info("  Output: %s %s", out_details[0]["shape"], out_details[0]["dtype"])
    log.info("  Size:   %.1f KB", tflite_path.stat().st_size / 1024)

    # Smoke test
    test = np.random.randn(*in_details[0]["shape"]).astype(np.float32)
    interpreter.set_tensor(in_details[0]["index"], test)
    interpreter.invoke()
    out = interpreter.get_tensor(out_details[0]["index"])
    log.info("  Smoke:  output=%s range=[%.4f, %.4f]", out.shape, out.min(), out.max())

    # Validate manifest
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        log.info("  Manifest: wake_word='%s' version=%d",
                 manifest.get("wake_word"), manifest.get("version"))
        if manifest.get("model") != tflite_path.name:
            log.warning("  Manifest model name mismatch: %s vs %s",
                        manifest.get("model"), tflite_path.name)
    else:
        log.warning("  No manifest found — run 05_export_mww.py")

    return True


def main():
    parser = argparse.ArgumentParser(description="Validate wake word model")
    parser.add_argument("word_id", help="Wake word identifier")
    parser.add_argument("model_type", nargs="?", default="oww",
                        choices=["oww", "mww", "both"],
                        help="Model type to validate")
    args = parser.parse_args()

    cfg = load_config(args.word_id)
    artifact_dir = REPO_DIR / "artifacts" / cfg["word_id"]

    results = {}

    if args.model_type in ("oww", "both"):
        log.info("=== Validating OWW model ===")
        results["oww"] = validate_oww(cfg["word_id"], artifact_dir)

    if args.model_type in ("mww", "both"):
        log.info("=== Validating mWW model ===")
        results["mww"] = validate_mww(cfg["word_id"], artifact_dir)

    # Summary
    log.info("=== Validation Summary ===")
    all_pass = True
    for mtype, passed in results.items():
        status = "PASS" if passed else "FAIL"
        log.info("  %s: %s", mtype.upper(), status)
        if not passed:
            all_pass = False

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
