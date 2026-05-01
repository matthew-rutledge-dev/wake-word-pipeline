#!/usr/bin/env python3
"""
Test wake word models for false positive rates against ambient audio.

Loads ONNX wake word models and runs them against WAV files (ambient noise,
environmental sounds, speech, music). Reports false activations per hour.

Optimizations:
  - Pre-loads all audio into memory once (avoids repeated disk I/O)
  - Multiprocessing across models (one model per worker)
  - Feeds whole files to OWW predict() (internal chunking)

Usage:
    python3 test_false_positives.py
    python3 test_false_positives.py --models hey_ara hey_bender
    python3 test_false_positives.py --audio-dir /path/to/recordings
    python3 test_false_positives.py --threshold 0.3 --workers 8
"""
import argparse
import logging
import os
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_DIR / "artifacts"
NEGATIVE_AUDIO_DIR = REPO_DIR / "negative_audio"
DEFAULT_THRESHOLD = 0.5
CHUNK_SIZE = 1280  # OWW processes 80ms frames (1280 samples @ 16kHz)


def load_audio(path):
    """Load a WAV file as 16kHz mono int16 numpy array."""
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception:
        return None

    audio = np.frombuffer(raw, dtype=np.int16)
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

    if framerate != 16000:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-ar", "16000", "-ac", "1",
                 "-sample_fmt", "s16", tmp_path],
                capture_output=True, check=True,
            )
            return load_audio(Path(tmp_path))
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return audio


def preload_audio(audio_dir):
    """Load all WAV files into memory. Returns list of (filename, np.array)."""
    files = sorted(Path(audio_dir).rglob("*.wav"))
    loaded = []
    failed = 0
    for f in files:
        audio = load_audio(f)
        if audio is not None and len(audio) > 0:
            loaded.append((f.name, audio))
        else:
            failed += 1
    return loaded, failed


def _test_model_worker(args):
    """Worker function for multiprocessing. Runs one model against all audio."""
    model_path, audio_data, threshold = args
    from openwakeword.model import Model

    model_name = Path(model_path).stem
    oww = Model(wakeword_models=[model_path])

    total_duration_s = 0.0
    total_false_positives = 0
    activations = []

    for fname, audio in audio_data:
        duration_s = len(audio) / 16000.0
        total_duration_s += duration_s
        oww.reset()

        # Feed whole file — OWW handles internal chunking
        for i in range(0, len(audio) - CHUNK_SIZE + 1, CHUNK_SIZE):
            chunk = audio[i : i + CHUNK_SIZE]
            result = oww.predict(chunk)
            score = result.get(model_name, 0.0)
            if score >= threshold:
                time_s = i / 16000.0
                total_false_positives += 1
                activations.append((fname, time_s, score))

    total_hours = total_duration_s / 3600.0
    fp_per_hour = total_false_positives / total_hours if total_hours > 0 else 0.0

    return {
        "model": model_name,
        "threshold": threshold,
        "total_audio_hours": total_hours,
        "total_files": len(audio_data),
        "false_positives": total_false_positives,
        "fp_per_hour": fp_per_hour,
        "activations": activations[:50],  # cap to avoid huge data transfer
    }


def find_models(model_names=None):
    if model_names:
        models = []
        for name in model_names:
            onnx = ARTIFACTS_DIR / name / "oww" / f"{name}.onnx"
            if onnx.exists():
                models.append(str(onnx))
            else:
                log.warning("Model not found: %s", onnx)
        return models
    return sorted(str(p) for p in ARTIFACTS_DIR.glob("*/oww/*.onnx"))


def main():
    parser = argparse.ArgumentParser(description="Test wake word models for false positives")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--audio-dir", default=str(NEGATIVE_AUDIO_DIR))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", default=None, help="Save results to CSV")
    parser.add_argument("--workers", "-w", type=int, default=0,
                        help="Parallel workers (0=auto, 1=sequential)")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        log.error("Audio directory not found: %s", audio_dir)
        log.error("Run download_background_audio.py first.")
        sys.exit(1)

    models = find_models(args.models)
    if not models:
        log.error("No models found. Check artifacts/ directory.")
        sys.exit(1)

    # Pre-load all audio into memory
    log.info("Pre-loading audio from %s...", audio_dir)
    t_load = time.time()
    audio_data, failed = preload_audio(audio_dir)
    load_time = time.time() - t_load
    total_hours = sum(len(a) for _, a in audio_data) / 16000.0 / 3600.0
    log.info("Loaded %d files (%.2f hours, %d failed) in %.1fs",
             len(audio_data), total_hours, failed, load_time)

    if not audio_data:
        log.error("No audio loaded.")
        sys.exit(1)

    max_workers = args.workers if args.workers > 0 else min(len(models), os.cpu_count() or 4)
    # Cap to avoid OOM — each OWW model uses ~200MB with preprocessor
    max_workers = min(max_workers, 12)

    log.info("Testing %d models (threshold=%.2f, workers=%d)",
             len(models), args.threshold, max_workers)

    t_start = time.time()
    results = []

    if max_workers == 1:
        # Sequential mode
        for i, model_path in enumerate(models, 1):
            log.info("[%d/%d] Testing %s...", i, len(models), Path(model_path).stem)
            t0 = time.time()
            r = _test_model_worker((model_path, audio_data, args.threshold))
            elapsed = time.time() - t0
            results.append(r)
            status = "PASS" if r["fp_per_hour"] < 1.0 else "WARN" if r["fp_per_hour"] < 5.0 else "FAIL"
            log.info("  %s: %.1f FP/hr (%d FP in %.2fh) [%s] %.1fs",
                     r["model"], r["fp_per_hour"], r["false_positives"],
                     r["total_audio_hours"], status, elapsed)
    else:
        # Parallel mode
        work_items = [(m, audio_data, args.threshold) for m in models]
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_test_model_worker, w): w[0] for w in work_items}
            done = 0
            for future in as_completed(futures):
                done += 1
                model_path = futures[future]
                try:
                    r = future.result()
                    results.append(r)
                    status = "PASS" if r["fp_per_hour"] < 1.0 else "WARN" if r["fp_per_hour"] < 5.0 else "FAIL"
                    log.info("[%d/%d] %s: %.1f FP/hr (%d FP) [%s]",
                             done, len(models), r["model"],
                             r["fp_per_hour"], r["false_positives"], status)
                except Exception as e:
                    log.error("[%d/%d] %s FAILED: %s", done, len(models),
                              Path(model_path).stem, e)

    total_time = time.time() - t_start

    # Report
    print()
    print("=" * 85)
    print("FALSE POSITIVE REPORT")
    print("=" * 85)
    print(f"{'Model':<24} {'Files':>5} {'Hours':>6} {'FP':>5} {'FP/hr':>7} {'Status':>8}")
    print("-" * 85)
    for r in sorted(results, key=lambda x: -x["fp_per_hour"]):
        status = "PASS" if r["fp_per_hour"] < 1.0 else "WARN" if r["fp_per_hour"] < 5.0 else "FAIL"
        print(f"{r['model']:<24} {r['total_files']:>5} {r['total_audio_hours']:>6.2f} "
              f"{r['false_positives']:>5} {r['fp_per_hour']:>7.2f} {status:>8}")

    total_fp = sum(r["false_positives"] for r in results)
    avg_fp_hr = sum(r["fp_per_hour"] for r in results) / len(results) if results else 0
    pass_count = sum(1 for r in results if r["fp_per_hour"] < 1.0)
    warn_count = sum(1 for r in results if 1.0 <= r["fp_per_hour"] < 5.0)
    fail_count = sum(1 for r in results if r["fp_per_hour"] >= 5.0)

    print("-" * 85)
    print(f"{'TOTAL':<24} {'':>5} {'':>6} {total_fp:>5} {avg_fp_hr:>7.2f}")
    print(f"\nPASS (<1 FP/hr): {pass_count}  |  WARN (1-5): {warn_count}  |  FAIL (>5): {fail_count}")
    print(f"Threshold: {args.threshold}  |  Models: {len(results)}  |  Workers: {max_workers}")
    print(f"Total time: {total_time:.1f}s  |  Audio: {len(audio_data)} files ({total_hours:.2f}h)")
    print("=" * 85)

    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "threshold", "audio_hours", "files", "false_positives", "fp_per_hour"])
            for r in results:
                w.writerow([r["model"], r["threshold"], f"{r['total_audio_hours']:.4f}",
                           r["total_files"], r["false_positives"], f"{r['fp_per_hour']:.4f}"])
        log.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
