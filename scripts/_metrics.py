"""
Pipeline metrics collector — auto-captures CPU, RAM, GPU, VRAM in a background thread.
Import and call start()/stop() from any pipeline script.

Usage:
    from _metrics import MetricsCollector
    mc = MetricsCollector(word_id="hey_ara", phase="01_gen")
    mc.start()
    # ... do work ...
    mc.stop()   # writes CSV + prints summary
"""
import csv
import os
import re
import subprocess
import threading
import time
from pathlib import Path

METRICS_DIR = Path("/opt/ai/wakeword-train/metrics")


def _gpu_stats():
    """Query nvidia-smi for GPU utilization, mem, temp, power."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_util_pct": float(parts[0]),
            "gpu_mem_util_pct": float(parts[1]),
            "gpu_mem_used_mb": float(parts[2]),
            "gpu_mem_total_mb": float(parts[3]),
            "gpu_temp_c": float(parts[4]),
            "gpu_power_w": float(parts[5]),
        }
    except Exception:
        return {"gpu_util_pct": 0, "gpu_mem_util_pct": 0,
                "gpu_mem_used_mb": 0, "gpu_mem_total_mb": 0,
                "gpu_temp_c": 0, "gpu_power_w": 0}


def _cpu_ram_stats():
    """Read /proc/stat and /proc/meminfo for CPU and RAM."""
    stats = {}
    # CPU — instantaneous from /proc/stat (user, nice, system, idle)
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        user, nice, system, idle, iowait = (int(parts[i]) for i in range(1, 6))
        total = user + nice + system + idle + iowait
        stats["cpu_usr_pct"] = round(100 * user / total, 1) if total else 0
        stats["cpu_sys_pct"] = round(100 * system / total, 1) if total else 0
        stats["cpu_idle_pct"] = round(100 * idle / total, 1) if total else 0
    except Exception:
        stats["cpu_usr_pct"] = stats["cpu_sys_pct"] = stats["cpu_idle_pct"] = 0

    # RAM
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split()
                mem[parts[0].rstrip(":")] = int(parts[1])
        total_mb = mem.get("MemTotal", 0) // 1024
        avail_mb = mem.get("MemAvailable", 0) // 1024
        used_mb = total_mb - avail_mb
        swap_total = mem.get("SwapTotal", 0) // 1024
        swap_free = mem.get("SwapFree", 0) // 1024
        stats["ram_used_mb"] = used_mb
        stats["ram_total_mb"] = total_mb
        stats["ram_pct"] = round(100 * used_mb / total_mb, 1) if total_mb else 0
        stats["swap_used_mb"] = swap_total - swap_free
    except Exception:
        stats["ram_used_mb"] = stats["ram_total_mb"] = 0
        stats["ram_pct"] = stats["swap_used_mb"] = 0

    # Process RSS
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    stats["process_rss_mb"] = int(line.split()[1]) // 1024
                    break
    except Exception:
        stats["process_rss_mb"] = 0

    return stats


CSV_FIELDS = [
    "timestamp", "word_id", "phase", "elapsed_s",
    "cpu_usr_pct", "cpu_sys_pct", "cpu_idle_pct",
    "ram_used_mb", "ram_total_mb", "ram_pct", "swap_used_mb",
    "gpu_util_pct", "gpu_mem_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb",
    "gpu_temp_c", "gpu_power_w", "process_rss_mb",
]


class MetricsCollector:
    """Background thread that samples system metrics every `interval` seconds."""

    def __init__(self, word_id: str, phase: str, interval: int = 10):
        self.word_id = word_id
        self.phase = phase
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._rows = []
        self._start_time = None
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        self._csv_path = METRICS_DIR / f"{word_id}_{phase}.csv"

    def start(self):
        self._start_time = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            row = self._sample()
            self._rows.append(row)
            self._stop.wait(self.interval)

    def _sample(self):
        now = time.time()
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "word_id": self.word_id,
            "phase": self.phase,
            "elapsed_s": round(now - self._start_time, 1),
        }
        row.update(_cpu_ram_stats())
        row.update(_gpu_stats())
        return row

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        # Final sample
        self._rows.append(self._sample())
        self._write_csv()
        self._print_summary()

    def _write_csv(self):
        with open(self._csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(self._rows)

    def _print_summary(self):
        if not self._rows:
            return
        duration = self._rows[-1]["elapsed_s"]
        cpu_avg = sum(r["cpu_usr_pct"] for r in self._rows) / len(self._rows)
        ram_peak = max(r["ram_used_mb"] for r in self._rows)
        rss_peak = max(r["process_rss_mb"] for r in self._rows)
        gpu_avg = sum(r["gpu_util_pct"] for r in self._rows) / len(self._rows)
        gpu_mem_peak = max(r["gpu_mem_used_mb"] for r in self._rows)
        gpu_temp_max = max(r["gpu_temp_c"] for r in self._rows)
        gpu_power_max = max(r["gpu_power_w"] for r in self._rows)
        swap_peak = max(r["swap_used_mb"] for r in self._rows)

        print(f"\n{'='*60}")
        print(f"METRICS SUMMARY: {self.word_id} / {self.phase}")
        print(f"{'='*60}")
        print(f"  Duration:        {duration:.0f}s ({duration/60:.1f} min)")
        print(f"  CPU avg:         {cpu_avg:.1f}%")
        print(f"  RAM peak:        {ram_peak} MB / {self._rows[0]['ram_total_mb']} MB")
        print(f"  Process RSS peak:{rss_peak} MB")
        print(f"  Swap peak:       {swap_peak} MB")
        print(f"  GPU util avg:    {gpu_avg:.1f}%")
        print(f"  VRAM peak:       {gpu_mem_peak} MB / {self._rows[0]['gpu_mem_total_mb']} MB")
        print(f"  GPU temp max:    {gpu_temp_max} C")
        print(f"  GPU power max:   {gpu_power_max} W")
        print(f"  CSV:             {self._csv_path}")
        print(f"{'='*60}\n")