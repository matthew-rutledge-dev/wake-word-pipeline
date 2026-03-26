#!/usr/bin/env python3
"""Basic sanity check for an exported mWW model.

Usage: python3 06_validate_mww.py <word_id>
Requires mww-venv activated.
"""
import argparse
import glob
import os
import random
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Validate mWW model")
    parser.add_argument("word_id", help="Word identifier (e.g. hey_ara)")
    args = parser.parse_args()

    word_id = args.word_id
    base = "/opt/ai/wakeword-train/wake-word-pipeline"
    mww_dir = os.path.join(base, "artifacts", word_id, "mww")
    tflite_path = os.path.join(mww_dir, f"{word_id}.tflite")
    positive_dir = os.path.join(base, "artifacts", word_id, "positive_train")

    if not os.path.isfile(tflite_path):
        print(f"ERROR: TFLite not found: {tflite_path}", file=sys.stderr)
        print("Run 05_export_mww.py first.", file=sys.stderr)
        sys.exit(1)

    from microwakeword.inference import Model
    from scipy.io import wavfile

    print(f"Loading model: {tflite_path}")
    model_size = os.path.getsize(tflite_path)
    print(f"  Size: {model_size:,} bytes ({model_size / 1024:.1f} KB)")
    model = Model(tflite_path)

    # --- Test on positive samples ---
    wavs = sorted(glob.glob(os.path.join(positive_dir, "*.wav")))
    if not wavs:
        print(f"WARNING: No WAV files found in {positive_dir}", file=sys.stderr)
    else:
        test_wavs = random.sample(wavs, min(10, len(wavs)))
        detections = 0
        print(f"\nTesting {len(test_wavs)} positive samples...")
        for wav_path in test_wavs:
            sr, audio = wavfile.read(wav_path)
            if audio.dtype != np.int16:
                audio = (audio * 32767).astype(np.int16)
            predictions = model.predict_clip(audio, step_ms=10)
            max_prob = max(predictions) if predictions else 0.0
            detected = max_prob >= 0.5
            if detected:
                detections += 1
            tag = "DETECTED" if detected else "missed"
            print(f"  {os.path.basename(wav_path)}: max_prob={max_prob:.4f} [{tag}]")

        rate = 100 * detections / len(test_wavs)
        print(f"\nPositive detection rate: {detections}/{len(test_wavs)} ({rate:.0f}%)")

    # --- Test on silence ---
    print("\nTesting on silence (2 s of zeros)...")
    silence = np.zeros(32000, dtype=np.int16)
    silence_preds = model.predict_clip(silence, step_ms=10)
    max_silence = max(silence_preds) if silence_preds else 0.0
    fp = max_silence >= 0.5
    tag = "FALSE POSITIVE" if fp else "OK"
    print(f"  Silence max_prob={max_silence:.4f} [{tag}]")

    print(f"\nValidation complete for '{word_id}'.")


if __name__ == "__main__":
    main()
