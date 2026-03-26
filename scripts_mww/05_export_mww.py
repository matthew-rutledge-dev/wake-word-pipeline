#!/usr/bin/env python3
"""Export trained mWW model to TFLite + v2 JSON manifest.

Usage: python3 05_export_mww.py <word_id>
Requires mww-venv activated.
"""
import argparse
import json
import os
import shutil
import sys
import yaml
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Export mWW model")
    parser.add_argument("word_id", help="Word identifier (e.g. hey_ara)")
    args = parser.parse_args()

    word_id = args.word_id
    base = "/opt/ai/wakeword-train/wake-word-pipeline"
    config_path = os.path.join(base, "words", word_id, "config.yaml")
    mww_dir = os.path.join(base, "artifacts", word_id, "mww")
    train_dir = os.path.join(mww_dir, "trained_models", word_id)
    tflite_src = os.path.join(
        train_dir,
        "tflite_stream_state_internal_quant",
        "stream_state_internal_quant.tflite",
    )
    tflite_dst = os.path.join(mww_dir, f"{word_id}.tflite")
    json_dst = os.path.join(mww_dir, f"{word_id}.json")

    if not os.path.isfile(config_path):
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(tflite_src):
        print(f"ERROR: Trained TFLite not found: {tflite_src}", file=sys.stderr)
        print("Run 04_train_mww.py first.", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    display_name = config.get("display_name", word_id.replace("_", " ").title())
    languages = config.get("trained_languages", ["en"])

    # Copy TFLite model
    shutil.copy2(tflite_src, tflite_dst)
    tflite_size = os.path.getsize(tflite_dst)
    print(f"Copied TFLite: {tflite_dst} ({tflite_size:,} bytes, {tflite_size / 1024:.1f} KB)")

    # Create v2 JSON manifest
    manifest = {
        "type": "micro",
        "wake_word": display_name,
        "author": "cdccentral",
        "website": "https://github.com/cdccentral-sourcecontrol/wake-word-pipeline",
        "trained_languages": languages,
        "model": f"{word_id}.tflite",
        "version": 2,
        "micro": {
            "probability_cutoff": 0.97,
            "sliding_window_size": 5,
            "feature_step_size": 10,
            "tensor_arena_size": 30000,
            "minimum_esphome_version": "2024.7.0",
        },
    }

    with open(json_dst, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {json_dst}")

    # Validate TFLite model (smoke test)
    print("Validating TFLite model...")
    try:
        import ai_edge_litert as tflite

        interpreter = tflite.Interpreter(model_path=tflite_dst)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"  Inputs:  {len(input_details)}")
        print(f"  Outputs: {len(output_details)}")
        print(f"  Input  shape={input_details[0]['shape']} dtype={input_details[0]['dtype']}")
        print(f"  Output shape={output_details[0]['shape']} dtype={output_details[0]['dtype']}")

        # Run with zeros
        for detail in input_details:
            interpreter.set_tensor(
                detail["index"],
                np.zeros(detail["shape"], dtype=detail["dtype"]),
            )
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        print(f"  Smoke test output: {output.flatten()}")
        print("  Validation PASSED.")
    except Exception as e:
        print(f"  WARNING: Validation failed: {e}", file=sys.stderr)

    print(f"\nExport complete for '{word_id}'.")
    print(f"  TFLite:   {tflite_dst} ({tflite_size / 1024:.1f} KB)")
    print(f"  Manifest: {json_dst}")


if __name__ == "__main__":
    main()
