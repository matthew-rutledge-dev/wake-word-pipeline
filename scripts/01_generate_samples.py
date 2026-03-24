#!/usr/bin/env python3
"""
Generate positive and negative WAV samples for a wake word.

PRIMARY:  Piper TTS (neural, multi-speaker, GPU accelerated)
FALLBACK: espeak-ng (formant synthesis, CPU only)

Usage: python 01_generate_samples.py <word_id> [--engine piper|espeak]
"""
import _compat  # noqa: F401 — must be first
import argparse
import logging
import os
import sys
import subprocess
import uuid
from pathlib import Path

import yaml
import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
PIPER_DIR = Path(os.environ.get("PIPER_SAMPLE_GENERATOR_PATH",
                                 str(REPO_DIR / "piper-sample-generator")))


def detect_device():
    """Return 'cuda' if GPU available, else 'cpu'."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log.info("GPU detected: %s (%.1f GB)", name, mem)
        return "cuda"
    log.warning("No CUDA GPU detected — falling back to CPU")
    return "cpu"


def load_config(word_id: str) -> dict:
    cfg_path = REPO_DIR / "words" / word_id / "config.yaml"
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def ensure_piper_setup(cfg: dict):
    """Clone piper-sample-generator and download voice model if missing."""
    if not (PIPER_DIR / "piper_sample_generator" / "__main__.py").exists():
        log.info("Cloning piper-sample-generator...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/rhasspy/piper-sample-generator",
             str(PIPER_DIR)],
            check=True,
        )

    model_name = cfg["piper"]["model"]
    model_path = PIPER_DIR / "models" / model_name
    if not model_path.exists():
        model_url = cfg["piper"]["model_url"]
        log.info("Downloading Piper voice model: %s", model_name)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["wget", "-q", "-O", str(model_path), model_url],
            check=True,
        )
    return model_path


TARGET_SR = 16000


def resample_to_16k(output_dir: Path):
    """Resample all WAV files in output_dir to 16kHz mono if not already."""
    import torchaudio
    import soundfile as sf

    wavs = list(output_dir.glob("*.wav"))
    if not wavs:
        return
    # Check first file
    info = torchaudio.info(str(wavs[0]))
    if info.sample_rate == TARGET_SR:
        return

    src_sr = info.sample_rate
    log.info("Resampling %d files from %d Hz → %d Hz", len(wavs), src_sr, TARGET_SR)
    resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=TARGET_SR)

    for wav_path in wavs:
        data, sr = torchaudio.load(str(wav_path))
        if sr != TARGET_SR:
            data = resampler(data)
        # Ensure mono
        if data.shape[0] > 1:
            data = data.mean(dim=0, keepdim=True)
        sf.write(str(wav_path), data.squeeze().numpy(), TARGET_SR, subtype="PCM_16")


def generate_piper_samples(
    phrases: list[str],
    output_dir: Path,
    max_samples: int,
    cfg: dict,
    device: str,
):
    """Generate WAV samples using Piper TTS (GPU accelerated)."""
    from piper_sample_generator.__main__ import generate_samples

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("*.wav")))
    if existing >= max_samples * 0.95:
        log.info("Skipping generation for %s — %d samples already exist", output_dir, existing)
        return

    needed = max_samples - existing
    piper_cfg = cfg["piper"]

    model_path = PIPER_DIR / "models" / piper_cfg["model"]
    if not model_path.exists():
        log.error("Piper voice model not found: %s", model_path)
        sys.exit(1)

    log.info("Generating %d Piper TTS samples → %s (device=%s)", needed, output_dir, device)
    generate_samples(
        text=phrases,
        model=str(model_path),
        max_samples=needed,
        batch_size=piper_cfg["batch_size"],
        noise_scales=piper_cfg["noise_scales"],
        noise_scale_ws=piper_cfg["noise_scale_ws"],
        length_scales=piper_cfg["length_scales"],
        output_dir=str(output_dir),
        file_names=[uuid.uuid4().hex + ".wav" for _ in range(needed)],
    )
    torch.cuda.empty_cache()

    # Resample to 16kHz if needed (OWW expects 16000 Hz)
    resample_to_16k(output_dir)

    generated = len(list(output_dir.glob("*.wav")))
    log.info("Total samples in %s: %d", output_dir.name, generated)


def generate_espeak_samples(
    phrase: str,
    output_dir: Path,
    max_samples: int,
    cfg: dict,
):
    """Fallback: generate WAV samples using espeak-ng + sox augmentation."""
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("*.wav")))
    if existing >= max_samples * 0.95:
        log.info("Skipping espeak generation — %d samples already exist", existing)
        return

    voices = cfg["espeak"]["voices"]
    speeds = range(120, 200, 10)
    pitches = range(30, 70, 5)
    count = 0

    log.info("Generating up to %d espeak-ng samples → %s", max_samples - existing, output_dir)
    for voice in voices:
        for speed in speeds:
            for pitch in pitches:
                if count + existing >= max_samples:
                    break
                out_path = output_dir / f"{uuid.uuid4().hex}.wav"
                try:
                    subprocess.run(
                        ["espeak-ng", "-v", voice, "-s", str(speed),
                         "-p", str(pitch), "-w", str(out_path), phrase],
                        check=True, capture_output=True,
                    )
                    count += 1
                except subprocess.CalledProcessError:
                    continue

    log.info("Generated %d espeak-ng samples", count)


def generate_adversarial_negatives(
    target_phrase: str,
    output_dir: Path,
    max_samples: int,
    custom_negatives: list[str],
    cfg: dict,
    device: str,
    use_piper: bool,
):
    """Generate phonetically similar negative samples."""
    try:
        from openwakeword.data import generate_adversarial_texts
    except ImportError:
        log.warning("openwakeword not installed — using custom negatives only")
        adversarial_texts = custom_negatives
    else:
        adversarial_texts = list(custom_negatives)
        adversarial_texts.extend(
            generate_adversarial_texts(
                input_text=target_phrase,
                N=max_samples,
                include_partial_phrase=1.0,
                include_input_words=0.2,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("*.wav")))
    if existing >= max_samples * 0.95:
        log.info("Skipping negative generation — %d already exist", existing)
        return

    if use_piper:
        generate_piper_samples(adversarial_texts, output_dir, max_samples, cfg, device)
    else:
        for phrase in adversarial_texts[:max_samples]:
            generate_espeak_samples(phrase, output_dir, max_samples, cfg)


def main():
    parser = argparse.ArgumentParser(description="Generate wake word training samples")
    parser.add_argument("word_id", help="Wake word identifier (e.g. hey_ara)")
    parser.add_argument("--engine", choices=["piper", "espeak", "auto"], default="auto",
                        help="TTS engine: piper (GPU), espeak (CPU), auto (prefer piper)")
    args = parser.parse_args()

    cfg = load_config(args.word_id)
    device = detect_device()
    target_phrase = cfg["display_name"].lower()
    samples = cfg["samples"]
    custom_negs = cfg.get("custom_negative_phrases", [])

    # Determine engine
    use_piper = args.engine != "espeak"
    if args.engine == "auto":
        try:
            ensure_piper_setup(cfg)
            use_piper = True
            log.info("Using Piper TTS (neural, %s)", device)
        except Exception as e:
            log.warning("Piper setup failed (%s) — falling back to espeak-ng", e)
            use_piper = False
    elif args.engine == "piper":
        ensure_piper_setup(cfg)

    # Output directories
    out_base = REPO_DIR / "artifacts" / cfg["word_id"]

    # --- Positive training samples ---
    log.info("=== Positive training samples ===")
    pos_train_dir = out_base / "positive_train"
    if use_piper:
        generate_piper_samples([target_phrase], pos_train_dir, samples["positive_train"], cfg, device)
    else:
        generate_espeak_samples(target_phrase, pos_train_dir, samples["positive_train"], cfg)

    # --- Positive validation samples ---
    log.info("=== Positive validation samples ===")
    pos_val_dir = out_base / "positive_val"
    if use_piper:
        generate_piper_samples([target_phrase], pos_val_dir, samples["positive_val"], cfg, device)
    else:
        generate_espeak_samples(target_phrase, pos_val_dir, samples["positive_val"], cfg)

    # --- Adversarial negative training samples ---
    log.info("=== Adversarial negative training samples ===")
    neg_train_dir = out_base / "negative_train"
    generate_adversarial_negatives(
        target_phrase, neg_train_dir, samples["negative_train"],
        custom_negs, cfg, device, use_piper,
    )

    # --- Adversarial negative validation samples ---
    log.info("=== Adversarial negative validation samples ===")
    neg_val_dir = out_base / "negative_val"
    generate_adversarial_negatives(
        target_phrase, neg_val_dir, samples["negative_val"],
        custom_negs, cfg, device, use_piper,
    )

    # Summary
    dirs = [pos_train_dir, pos_val_dir, neg_train_dir, neg_val_dir]
    for d in dirs:
        n = len(list(d.glob("*.wav"))) if d.exists() else 0
        log.info("  %s: %d files", d.name, n)

    log.info("Sample generation complete for '%s'", cfg["display_name"])


if __name__ == "__main__":
    main()
