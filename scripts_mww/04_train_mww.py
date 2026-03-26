#!/usr/bin/env python3
"""Train a microWakeWord model.

Usage: python3 04_train_mww.py <word_id> [--training-steps N] [--restore-checkpoint]
Requires mww-venv activated.
"""
import argparse
import os
import subprocess
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description="Train mWW model")
    parser.add_argument("word_id", help="Word identifier (e.g. hey_ara)")
    parser.add_argument("--training-steps", type=int, default=10000,
                        help="Number of training steps (default: 10000)")
    parser.add_argument("--restore-checkpoint", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    word_id = args.word_id
    base = "/opt/ai/wakeword-train/wake-word-pipeline"
    neg_dir = "/opt/ai/wakeword-train/mww-negatives"
    config_path = os.path.join(base, "words", word_id, "config.yaml")
    mww_dir = os.path.join(base, "artifacts", word_id, "mww")
    features_dir = os.path.join(mww_dir, "features")
    train_dir = os.path.join(mww_dir, "trained_models", word_id)
    yaml_path = os.path.join(mww_dir, "training_parameters.yaml")

    if not os.path.isfile(config_path):
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(features_dir):
        print(f"ERROR: Features not found at {features_dir}", file=sys.stderr)
        print("Run 03_generate_features.py first.", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Enable TF GPU memory growth
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    # Build training_parameters.yaml
    training_config = {
        "window_step_ms": 10,
        "train_dir": train_dir,
        "features": [
            {
                "features_dir": features_dir,
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": os.path.join(neg_dir, "speech"),
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": os.path.join(neg_dir, "dinner_party"),
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": os.path.join(neg_dir, "no_speech"),
                "sampling_weight": 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": os.path.join(neg_dir, "dinner_party_eval"),
                "sampling_weight": 0.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "split",
                "type": "mmap",
            },
        ],
        "training_steps": [args.training_steps],
        "positive_class_weight": [1],
        "negative_class_weight": [20],
        "learning_rates": [0.001],
        "batch_size": 128,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 500,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }

    os.makedirs(mww_dir, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(training_config, f, default_flow_style=False)
    print(f"Wrote training config: {yaml_path}")

    # Build training command
    restore_flag = "1" if args.restore_checkpoint else "0"
    cmd = [
        sys.executable, "-m", "microwakeword.model_train_eval",
        "--training_config", yaml_path,
        "--train", "1",
        "--restore_checkpoint", restore_flag,
        "--test_tf_nonstreaming", "0",
        "--test_tflite_nonstreaming", "0",
        "--test_tflite_nonstreaming_quantized", "0",
        "--test_tflite_streaming", "0",
        "--test_tflite_streaming_quantized", "1",
        "--use_weights", "best_weights",
        "mixednet",
        "--pointwise_filters", "64,64,64,64",
        "--repeat_in_block", "1,1,1,1",
        "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
        "--residual_connection", "0,0,0,0",
        "--first_conv_filters", "32",
        "--first_conv_kernel_size", "5",
        "--stride", "3",
    ]

    print(f"Starting training for '{word_id}' ({args.training_steps} steps)...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=os.path.dirname(yaml_path))
    if result.returncode != 0:
        print(f"ERROR: Training failed (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"Training complete for '{word_id}'.")


if __name__ == "__main__":
    main()
