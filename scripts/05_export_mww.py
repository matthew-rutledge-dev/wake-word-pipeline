#!/usr/bin/env python3
"""
Export trained microWakeWord model to streaming TFLite + JSON manifest.

Produces the model + manifest needed for ESPHome custom firmware builds.
The JSON manifest follows microWakeWord v2 schema.

Usage: python 05_export_mww.py <word_id>
"""
import argparse
import json
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


def create_manifest(cfg: dict, tflite_path: Path, arena_size: int = 30000) -> dict:
    """Create microWakeWord v2 JSON manifest for ESPHome."""
    return {
        "type": "micro",
        "wake_word": cfg["display_name"],
        "author": "cdccentral",
        "website": "https://github.com/cdccentral-sourcecontrol/wake-word-pipeline",
        "trained_languages": cfg["trained_languages"],
        "model": tflite_path.name,
        "version": 2,
        "micro": {
            "probability_cutoff": 0.97,
            "sliding_window_size": 5,
            "feature_step_size": 10,
            "tensor_arena_size": arena_size,
            "minimum_esphome_version": "2026.3.0",
        },
    }


def validate_mww_tflite(tflite_path: str) -> dict:
    """Validate mWW TFLite model and return metadata."""
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
    fsize = Path(tflite_path).stat().st_size

    log.info("mWW TFLite: input=%s output=%s size=%.1f KB", in_shape, out_shape, fsize / 1024)

    # Smoke test
    test_input = np.random.randn(*in_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    log.info("Smoke test: output=%s range=[%.4f, %.4f]", output.shape, output.min(), output.max())

    # ESP32-S3 has ~512KB SRAM; model should be well under that
    if fsize > 500_000:
        log.warning("Model is %.1f KB — may be too large for ESP32-S3", fsize / 1024)

    return {
        "input_shape": in_shape.tolist(),
        "output_shape": out_shape.tolist(),
        "size_bytes": fsize,
    }


def main():
    parser = argparse.ArgumentParser(description="Export mWW model + manifest")
    parser.add_argument("word_id", help="Wake word identifier")
    args = parser.parse_args()

    cfg = load_config(args.word_id)
    word_id = cfg["word_id"]
    mww_dir = REPO_DIR / "artifacts" / word_id / "mww"

    # Find the trained TFLite model
    tflite_files = list(mww_dir.glob("*.tflite"))
    if not tflite_files:
        log.error("No TFLite model found in %s — run 04_train_mww.py first", mww_dir)
        sys.exit(1)

    tflite_path = tflite_files[0]

    # Validate
    meta = validate_mww_tflite(str(tflite_path))

    # Create manifest
    manifest = create_manifest(cfg, tflite_path, arena_size=30000)
    manifest_path = mww_dir / f"{word_id}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Manifest: %s", manifest_path)
    log.info("Model:    %s (%.1f KB)", tflite_path.name, meta["size_bytes"] / 1024)
    log.info("mWW export complete for '%s'", cfg["display_name"])


if __name__ == "__main__":
    main()
