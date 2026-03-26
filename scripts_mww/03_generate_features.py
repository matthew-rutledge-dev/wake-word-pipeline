#!/usr/bin/env python3
"""Generate augmented spectrogram features for mWW training.

Usage: python3 03_generate_features.py <word_id>
Requires mww-venv activated.
"""
import argparse
import os
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description="Generate mWW spectrogram features")
    parser.add_argument("word_id", help="Word identifier (e.g. hey_ara)")
    args = parser.parse_args()

    word_id = args.word_id
    base = "/opt/ai/wakeword-train/wake-word-pipeline"
    config_path = os.path.join(base, "words", word_id, "config.yaml")
    positive_dir = os.path.join(base, "artifacts", word_id, "positive_train")
    out_dir = os.path.join(base, "artifacts", word_id, "mww", "features")

    if not os.path.isfile(config_path):
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(positive_dir):
        print(f"ERROR: Positive WAVs not found: {positive_dir}", file=sys.stderr)
        sys.exit(1)

    # Check if features already exist
    if os.path.isdir(out_dir):
        training_dir = os.path.join(out_dir, "training")
        has_mmap = False
        if os.path.isdir(training_dir):
            has_mmap = any(
                d.endswith("_mmap")
                for d in os.listdir(training_dir)
                if os.path.isdir(os.path.join(training_dir, d))
            )
        if has_mmap:
            print(f"Features already exist at {out_dir}, skipping.")
            return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Generating features for '{word_id}'...")
    print(f"  Positive WAVs: {positive_dir}")
    print(f"  Output dir:    {out_dir}")

    from microwakeword.audio.clips import Clips
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    # Load positive clips with train/validation/test split
    clips = Clips(
        input_directory=positive_dir,
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    print(f"  Loaded {len(clips.clips)} clips")

    # Audio augmentation (basic — no background noise or RIR for positive features)
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "Gain": 1.0,
        },
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    os.makedirs(out_dir, exist_ok=True)

    # Generate features for each split
    splits_config = [
        {"name": "training",   "split": "train",      "repeat": 2, "slide_frames": 10},
        {"name": "validation", "split": "validation",  "repeat": 1, "slide_frames": 10},
        {"name": "testing",    "split": "test",        "repeat": 1, "slide_frames": 1},
    ]

    for sc in splits_config:
        split_out = os.path.join(out_dir, sc["name"])
        mmap_out = os.path.join(split_out, "wakeword_mmap")

        if os.path.isdir(mmap_out) and os.listdir(mmap_out):
            print(f"  [{sc['name']}] Already exists, skipping.")
            continue

        os.makedirs(split_out, exist_ok=True)
        print(f"  [{sc['name']}] Generating spectrograms "
              f"(repeat={sc['repeat']}, slide={sc['slide_frames']})...")

        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=sc["slide_frames"],
            step_ms=10,
        )

        RaggedMmap.from_generator(
            out_dir=mmap_out,
            sample_generator=spectrograms.spectrogram_generator(
                split=sc["split"], repeat=sc["repeat"]
            ),
            batch_size=100,
            verbose=True,
        )
        print(f"  [{sc['name']}] Done.")

    print(f"Feature generation complete for '{word_id}'.")


if __name__ == "__main__":
    main()
