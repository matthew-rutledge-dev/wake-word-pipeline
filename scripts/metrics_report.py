#!/usr/bin/env python3
"""
Generate a training metrics report from per-word CSV files.
Reads all CSVs in /opt/ai/wakeword-train/metrics/ and produces
a summary table + recommendations.

Usage: python3 metrics_report.py [--csv-dir /path/to/metrics]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

METRICS_DIR = Path("/opt/ai/wakeword-train/metrics")
ARTIFACTS_DIR = Path("/opt/ai/wakeword-train/wake-word-pipeline/artifacts")


def parse_csv(path):
    """Read a metrics CSV and return list of row dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in row:
                try:
                    row[k] = float(row[k])
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def summarize_word(word_id, phase, rows):
    """Return a summary dict for one word+phase."""
    if not rows:
        return None
    duration = rows[-1].get("elapsed_s", 0)
    cpu_vals = [r.get("cpu_usr_pct", 0) for r in rows]
    ram_vals = [r.get("ram_used_mb", 0) for r in rows]
    rss_vals = [r.get("process_rss_mb", 0) for r in rows]
    gpu_vals = [r.get("gpu_util_pct", 0) for r in rows]
    vram_vals = [r.get("gpu_mem_used_mb", 0) for r in rows]
    temp_vals = [r.get("gpu_temp_c", 0) for r in rows]
    power_vals = [r.get("gpu_power_w", 0) for r in rows]
    swap_vals = [r.get("swap_used_mb", 0) for r in rows]
    return {
        "word_id": word_id,
        "phase": phase,
        "duration_s": duration,
        "cpu_avg": sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0,
        "cpu_peak": max(cpu_vals) if cpu_vals else 0,
        "ram_peak_mb": max(ram_vals) if ram_vals else 0,
        "rss_peak_mb": max(rss_vals) if rss_vals else 0,
        "swap_peak_mb": max(swap_vals) if swap_vals else 0,
        "gpu_avg": sum(gpu_vals) / len(gpu_vals) if gpu_vals else 0,
        "gpu_peak": max(gpu_vals) if gpu_vals else 0,
        "vram_peak_mb": max(vram_vals) if vram_vals else 0,
        "temp_max_c": max(temp_vals) if temp_vals else 0,
        "power_max_w": max(power_vals) if power_vals else 0,
        "ram_total_mb": rows[0].get("ram_total_mb", 0),
        "gpu_mem_total_mb": rows[0].get("gpu_mem_total_mb", 0),
        "samples": len(rows),
    }


def check_onnx(word_id):
    """Check if ONNX model exists and return its size."""
    onnx_path = ARTIFACTS_DIR / word_id / "oww" / f"{word_id}.onnx"
    if onnx_path.exists():
        return onnx_path.stat().st_size
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default=str(METRICS_DIR))
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    csvs = sorted(csv_dir.glob("*_01_gen.csv")) + sorted(csv_dir.glob("*_02_train.csv"))

    if not csvs:
        print("No per-word metrics CSVs found. Metrics are collected automatically")
        print("when the pipeline runs with _metrics integration.")
        sys.exit(0)

    summaries = []
    for csv_path in csvs:
        rows = parse_csv(csv_path)
        if not rows:
            continue
        # Parse word_id and phase from filename: hey_ara_01_gen.csv
        name = csv_path.stem
        if "_01_gen" in name:
            word_id = name.replace("_01_gen", "")
            phase = "01_gen"
        elif "_02_train" in name:
            word_id = name.replace("_02_train", "")
            phase = "02_train"
        else:
            continue
        s = summarize_word(word_id, phase, rows)
        if s:
            summaries.append(s)

    if not summaries:
        print("No metrics data to report.")
        sys.exit(0)

    # Print report
    print()
    print("=" * 90)
    print("WAKE WORD PIPELINE — METRICS REPORT")
    print("=" * 90)

    # Per-phase tables
    for phase_name, phase_code in [("SAMPLE GENERATION (01_gen)", "01_gen"),
                                     ("DNN TRAINING (02_train)", "02_train")]:
        phase_data = [s for s in summaries if s["phase"] == phase_code]
        if not phase_data:
            continue
        print(f"\n--- {phase_name} ---")
        print(f"{'Word':<20} {'Time':>7} {'CPU%':>6} {'RAM MB':>8} {'RSS MB':>8} "
              f"{'GPU%':>6} {'VRAM MB':>9} {'Temp C':>7} {'Power W':>8} {'ONNX':>8}")
        print("-" * 90)
        for s in phase_data:
            onnx_size = check_onnx(s["word_id"])
            onnx_str = f"{onnx_size:,}" if onnx_size else "-"
            mins = s["duration_s"] / 60
            print(f"{s['word_id']:<20} {mins:>6.1f}m {s['cpu_avg']:>5.1f} "
                  f"{s['ram_peak_mb']:>8.0f} {s['rss_peak_mb']:>8.0f} "
                  f"{s['gpu_avg']:>5.1f} {s['vram_peak_mb']:>9.0f} "
                  f"{s['temp_max_c']:>7.0f} {s['power_max_w']:>8.1f} {onnx_str:>8}")

    # Aggregate stats
    gen_data = [s for s in summaries if s["phase"] == "01_gen"]
    train_data = [s for s in summaries if s["phase"] == "02_train"]

    print(f"\n--- AGGREGATE ---")
    if gen_data:
        total_gen = sum(s["duration_s"] for s in gen_data)
        avg_gen = total_gen / len(gen_data)
        print(f"  Sample gen:  {len(gen_data)} words, total {total_gen/60:.1f}m, avg {avg_gen/60:.1f}m/word")
        print(f"    CPU avg:   {sum(s['cpu_avg'] for s in gen_data)/len(gen_data):.1f}%")
        print(f"    GPU avg:   {sum(s['gpu_avg'] for s in gen_data)/len(gen_data):.1f}%")
        print(f"    RAM peak:  {max(s['ram_peak_mb'] for s in gen_data):.0f} MB")
        print(f"    VRAM peak: {max(s['vram_peak_mb'] for s in gen_data):.0f} MB")

    if train_data:
        total_train = sum(s["duration_s"] for s in train_data)
        avg_train = total_train / len(train_data)
        print(f"  DNN train:   {len(train_data)} words, total {total_train/60:.1f}m, avg {avg_train/60:.1f}m/word")
        print(f"    CPU avg:   {sum(s['cpu_avg'] for s in train_data)/len(train_data):.1f}%")
        print(f"    GPU avg:   {sum(s['gpu_avg'] for s in train_data)/len(train_data):.1f}%")
        print(f"    RAM peak:  {max(s['ram_peak_mb'] for s in train_data):.0f} MB")
        print(f"    VRAM peak: {max(s['vram_peak_mb'] for s in train_data):.0f} MB")

    # Tuning recommendations
    print(f"\n--- TUNING RECOMMENDATIONS ---")
    all_gpu = [s["gpu_avg"] for s in summaries]
    all_cpu = [s["cpu_avg"] for s in summaries]
    all_vram = [s["vram_peak_mb"] for s in summaries]
    all_ram = [s["ram_peak_mb"] for s in summaries]

    if gen_data:
        gen_gpu_avg = sum(s["gpu_avg"] for s in gen_data) / len(gen_data)
        if gen_gpu_avg < 10:
            print("  * Sample gen is CPU-bound (GPU <10% avg). Piper TTS runs on CPU.")
            print("    Consider: more CPU cores or faster clock speed for gen phase.")
    if train_data:
        train_gpu_avg = sum(s["gpu_avg"] for s in train_data) / len(train_data)
        if train_gpu_avg < 30:
            print("  * DNN training GPU underutilized (<30% avg). Model is small (~5K params).")
            print("    Already mitigated by running 3 concurrent training jobs.")
        max_vram = max(s["vram_peak_mb"] for s in train_data)
        gpu_total = train_data[0]["gpu_mem_total_mb"]
        if gpu_total > 0 and max_vram < gpu_total * 0.3:
            parallel = int(gpu_total // (max_vram if max_vram > 0 else 300))
            print(f"  * VRAM headroom: peak {max_vram:.0f}/{gpu_total:.0f} MB.")
            print(f"    Could run up to {min(parallel, 8)} concurrent training jobs (currently 3).")
        max_ram = max(s["ram_peak_mb"] for s in train_data)
        ram_total = train_data[0]["ram_total_mb"]
        if ram_total > 0:
            ram_free = ram_total - max_ram
            print(f"  * RAM headroom: peak {max_ram:.0f}/{ram_total:.0f} MB ({ram_free:.0f} MB free).")

    swap_peak = max(s["swap_peak_mb"] for s in summaries)
    if swap_peak > 1000:
        print(f"  * Swap usage high: {swap_peak:.0f} MB. Consider clearing stale swap or adding RAM.")

    total_onnx = sum(1 for s in summaries if s["phase"] == "02_train" and check_onnx(s["word_id"]))
    total_words = len(set(s["word_id"] for s in summaries if s["phase"] == "02_train"))
    if total_words > 0:
        print(f"  * Models: {total_onnx}/{total_words} ONNX files produced.")

    print()
    print("=" * 90)
    print(f"Report generated from {len(csvs)} CSV files in {csv_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()