#!/usr/bin/env python3
"""
Stage artifacts for submission to fwartner/home-assistant-wakewords-collection.

Copies validated models into staging/ with the directory structure expected by
the fwartner collection repo. NEVER auto-opens PRs — human gate enforced.

Usage: python 07_stage_submission.py <word_id> [--oww|--mww|--both]
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

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


def stage_oww(cfg: dict):
    """Stage OWW model for fwartner submission."""
    word_id = cfg["word_id"]
    display_name = cfg["display_name"]
    artifact_dir = REPO_DIR / "artifacts" / word_id / "oww"

    # fwartner expects: custom_wake_words/<word_id>/<word_id>.tflite
    stage_dir = REPO_DIR / "staging" / word_id / "oww"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Copy TFLite model
    tflite_src = artifact_dir / f"{word_id}.tflite"
    if not tflite_src.exists():
        # Try to find any tflite
        tflite_files = list(artifact_dir.glob("*.tflite"))
        if tflite_files:
            tflite_src = tflite_files[0]
        else:
            log.error("No OWW TFLite model found in %s", artifact_dir)
            return False

    tflite_dst = stage_dir / f"{word_id}.tflite"
    shutil.copy2(tflite_src, tflite_dst)
    log.info("Staged OWW TFLite: %s (%.1f KB)", tflite_dst, tflite_dst.stat().st_size / 1024)

    # Copy ONNX model if present
    onnx_src = artifact_dir / f"{word_id}.onnx"
    if onnx_src.exists():
        shutil.copy2(onnx_src, stage_dir / f"{word_id}.onnx")

    # Create README for the submission
    readme = stage_dir / "README.md"
    readme.write_text(
        f"# {display_name}\n\n"
        f"Wake word: **{display_name}**\n\n"
        f"- Type: openWakeWord (Wyoming)\n"
        f"- Languages: {', '.join(cfg['trained_languages'])}\n"
        f"- Author: cdccentral\n"
        f"- Training: Piper TTS (neural, multi-speaker) + data augmentation\n"
        f"- Model size: {tflite_dst.stat().st_size / 1024:.1f} KB\n"
    )

    return True


def stage_mww(cfg: dict):
    """Stage mWW model and manifest."""
    word_id = cfg["word_id"]
    artifact_dir = REPO_DIR / "artifacts" / word_id / "mww"
    stage_dir = REPO_DIR / "staging" / word_id / "mww"
    stage_dir.mkdir(parents=True, exist_ok=True)

    tflite_files = list(artifact_dir.glob("*.tflite"))
    if not tflite_files:
        log.error("No mWW TFLite model found in %s", artifact_dir)
        return False

    for f in tflite_files:
        shutil.copy2(f, stage_dir / f.name)
        log.info("Staged mWW TFLite: %s (%.1f KB)", f.name, f.stat().st_size / 1024)

    manifest_src = artifact_dir / f"{word_id}.json"
    if manifest_src.exists():
        shutil.copy2(manifest_src, stage_dir / f"{word_id}.json")
        log.info("Staged manifest: %s", manifest_src.name)

    return True


def main():
    parser = argparse.ArgumentParser(description="Stage models for submission")
    parser.add_argument("word_id", help="Wake word identifier")
    parser.add_argument("--target", choices=["oww", "mww", "both"], default="both",
                        help="Which model type to stage")
    args = parser.parse_args()

    cfg = load_config(args.word_id)
    word_id = cfg["word_id"]

    log.info("Staging '%s' for submission", cfg["display_name"])

    results = {}
    if args.target in ("oww", "both"):
        results["oww"] = stage_oww(cfg)
    if args.target in ("mww", "both"):
        results["mww"] = stage_mww(cfg)

    # Summary
    log.info("")
    log.info("=== Staging Summary ===")
    for mtype, ok in results.items():
        log.info("  %s: %s", mtype.upper(), "STAGED" if ok else "FAILED")

    staging_dir = REPO_DIR / "staging" / word_id
    log.info("")
    log.info("Staged files: %s", staging_dir)
    log.info("")
    log.info("HUMAN GATE: Review the staging directory, then submit manually:")
    log.info("  1. Fork https://github.com/fwartner/home-assistant-wakewords-collection")
    log.info("  2. Copy staging/%s/ into the fork's custom_wake_words/ directory", word_id)
    log.info("  3. Open a PR with a description of the model and training data")
    log.info("  4. DO NOT auto-submit from this pipeline")


if __name__ == "__main__":
    main()
