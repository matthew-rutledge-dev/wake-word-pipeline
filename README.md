# wake-word-pipeline

End-to-end training pipeline for custom wake word models targeting both
**openWakeWord** (Wyoming/OWW add-on) and **microWakeWord** (ESPHome HA Voice PE firmware).

**GPU-first** design — every step prefers CUDA GPU acceleration, with CPU fallback.
**Piper TTS primary** — neural multi-speaker synthesis (not espeak-ng) for training data.

## Quick Start

```bash
# Train a single wake word (both OWW + mWW)
./pipeline.sh hey_ara

# Train OWW only
./pipeline.sh hey_bender --oww-only

# Train mWW only
./pipeline.sh hey_ara --mww-only
```

## Batch Training

Train all 24 configured wake words autonomously:

```bash
# From the server (servergen1.cdclocal)
cd /opt/ai/wakeword-train
source venv/bin/activate

# Start batch training (runs in background)
nohup bash batch_train.sh > batch_train.log 2>&1 &

# Monitor progress
tail -f batch_train.log

# Check which ONNX models are done
find wake-word-pipeline/artifacts -name '*.onnx' -printf '%f  %s bytes  %T+\n'
```

`batch_train.sh` runs in two phases:
1. **Phase 1** — Sequential sample generation (Piper TTS, ~7-8 min/word)
2. **Phase 2** — Parallel DNN training in groups of 3 (~3 min/word)

A metrics report is generated automatically at the end.

## Pipeline Steps

| Step | Script | GPU Used | Description |
|------|--------|----------|-------------|
| — | _compat.py | N/A | Compatibility shims (auto-imported by 01/02) |
| — | _metrics.py | N/A | Background resource metrics collector |
| 1 | 01_generate_samples.py | Piper TTS inference | Generate positive + adversarial negative WAV samples |
| 2 | 02_train_oww.py | Embedding + DNN training | Augment clips, compute OWW features, train model |
| 3 | 03_export_oww.py | CPU only (TFLite) | Convert ONNX to TFLite, validate shapes |
| 4 | 04_train_mww.py | TF training | Train microWakeWord streaming model |
| 5 | 05_export_mww.py | N/A | Export mWW TFLite + JSON manifest |
| 6 | 06_validate.py | OWW inference | Shape check + detection recall on validation clips |
| 7 | 07_stage_submission.py | N/A | Stage for fwartner collection (human gate) |
| — | metrics_report.py | N/A | Aggregate metrics report generator |

## Structure

```
words/<word_id>/config.yaml       # Per-word configuration (TTS, training params)
scripts/                          # Training and export scripts (01-07)
  _compat.py                      # Monkey-patches for dependency version mismatches
  _metrics.py                     # Pipeline-integrated resource metrics collector
  metrics_report.py               # Aggregate metrics report generator
docker/                           # GPU-enabled Dockerfiles (NVIDIA CUDA base)
artifacts/                        # Trained model outputs (gitignored)
  <word_id>/oww/<word_id>.onnx    # Exported OWW model
staging/                          # Prepared submissions (human-reviewed before PR)
batch_train.sh                    # Autonomous batch trainer for all words
metrics_monitor.sh                # Standalone system resource monitor (CSV)
```

## Metrics & Reporting

Resource usage is collected automatically during pipeline execution.

### How It Works

`_metrics.py` provides a `MetricsCollector` class that runs a background daemon thread during both sample generation (`01_gen`) and training (`02_train`). It records:

- **CPU**: user/system/idle percentages (from `/proc/stat`)
- **RAM**: used/total/swap (from `/proc/meminfo`)
- **GPU**: utilization, VRAM, temperature, power (from `nvidia-smi`)
- **Process RSS**: resident memory of the training process

Data is written to per-word CSV files in the `metrics/` directory at 10-second intervals.

### Viewing Metrics

```bash
# Generate aggregate report after training
cd /opt/ai/wakeword-train/wake-word-pipeline/scripts
python3 metrics_report.py

# Watch live system metrics (standalone monitor)
bash /opt/ai/wakeword-train/metrics_monitor.sh

# Interactive GPU + CPU monitor
nvtop
```

### Metrics Output

Per-word CSVs: `metrics/<word_id>_01_gen.csv`, `metrics/<word_id>_02_train.csv`

The report includes per-phase tables (duration, CPU avg, RAM peak, GPU avg, VRAM peak, temperature, power) and tuning recommendations (parallelism suggestions, swap warnings, bottleneck identification).

## Interrupt / Resume Behavior

The pipeline supports safe interrupt (Ctrl+C / kill) and resumes from where it left off:

| Component | Resume Behavior |
|-----------|----------------|
| `batch_train.sh` | Skips words with ≥1000 WAVs (gen) or existing ONNX (train) |
| `01_generate_samples.py` | Per-split resume — skips splits with ≥95% WAVs, generates shortfall only |
| `02_train_oww.py` features | Skips existing `.npy` feature files per split |
| `02_train_oww.py` assets | Skips already-downloaded embedding model / validation data |
| DNN training | Restarts from step 0 (~3 min, no checkpointing needed) |
| Metrics | New CSV started per run; old CSVs preserved |

## TTS Architecture

| Engine | Type | GPU | Quality | Use Case |
|--------|------|-----|---------|----------|
| **Piper TTS** | Neural VITS | CUDA accelerated | Natural, multi-speaker | Primary for all production training |
| espeak-ng | Formant synthesis | CPU only | Robotic | Fallback when Piper unavailable |

Piper uses en_US-libritts_r-medium.pt (~900 speakers) via
[piper-sample-generator](https://github.com/rhasspy/piper-sample-generator).
Samples are augmented with room impulse responses + background noise for realism.

## Wake Word Queue (24 words)

| # | ID | Display | OWW | mWW |
|---|-----|---------|-----|-----|
| 1 | hey_ara | Hey Ara | Done | Pending |
| 2 | ok_ara | Ok Ara | Batch | Pending |
| 3 | hey_bender | Hey Bender | Batch | Pending |
| 4 | ok_bender | Ok Bender | Batch | Pending |
| 5 | hey_cortana | Hey Cortana | Batch | Pending |
| 6 | ok_cortana | Ok Cortana | Batch | Pending |
| 7 | hey_spongebob | Hey SpongeBob | Batch | Pending |
| 8 | ok_spongebob | Ok SpongeBob | Batch | Pending |
| 9 | hey_anya | Hey Anya | Batch | Pending |
| 10 | ok_anya | Ok Anya | Batch | Pending |
| 11 | hey_naruto | Hey Naruto | Batch | Pending |
| 12 | ok_naruto | Ok Naruto | Batch | Pending |
| 13 | hey_veldora | Hey Veldora | Batch | Pending |
| 14 | ok_veldora | Ok Veldora | Batch | Pending |
| 15 | hey_rimuru | Hey Rimuru | Batch | Pending |
| 16 | ok_rimuru | Ok Rimuru | Batch | Pending |
| 17 | hey_santa | Hey Santa | Batch | Pending |
| 18 | ok_santa | Ok Santa | Batch | Pending |
| 19 | hey_chief | Hey Chief | Batch | Pending |
| 20 | ok_chief | Ok Chief | Batch | Pending |
| 21 | hey_my_goddess | Hey My Goddess | Batch | Pending |
| 22 | ok_my_goddess | Ok My Goddess | Batch | Pending |
| 23 | hey_my_knight | Hey My Knight | Batch | Pending |
| 24 | ok_my_knight | Ok My Knight | Batch | Pending |

**Status**: `Done` = ONNX exported, `Batch` = queued in batch_train.sh

## Training Environment

- **Server**: servergen1.cdclocal (Ubuntu 25.10)
- **GPU**: NVIDIA RTX 5060 Ti 16GB (CUDA 13.0, Driver 580.126.09)
- **Docker**: nvidia-container-toolkit v1.19.0-1, Docker 29.1.3 + Compose v5.1.1
- **PyTorch**: CUDA 12.x wheels (GPU training + Piper inference)
- **TensorFlow**: GPU support for mWW training + OWW embedding computation

## Docker

```bash
# Build GPU-enabled images
docker build -t oww-train -f docker/Dockerfile.oww .
docker build -t mww-train -f docker/Dockerfile.mww .

# Run with GPU
docker run --gpus all -v /opt/ai/wakeword-train:/workspace oww-train scripts/01_generate_samples.py hey_ara

# Run CPU fallback
docker run -v /opt/ai/wakeword-train:/workspace oww-train scripts/01_generate_samples.py hey_ara --engine espeak
```

## Naming Convention

- **File/ID**: lowercase_underscore (e.g., hey_ara)
- **Display**: Title Case in manifests (e.g., "Hey Ara")

## Deployment to Home Assistant

After training completes, deploy ONNX models to the openWakeWord add-on:

```bash
# Copy models to HA (from servergen1)
scp /opt/ai/wakeword-train/wake-word-pipeline/artifacts/*/oww/*.onnx \
    root@homeassistant.local:/share/openwakeword/

# Then in HA:
# 1. Settings > Add-ons > openWakeWord > Configuration
# 2. Add custom model paths under "Custom model directory"
# 3. Restart the add-on
# 4. Settings > Voice assistants > select new wake word
```

## Known Issues & Fixes

The `_compat.py` module patches several dependency mismatches:

| Issue | Fix |
|-------|-----|
| `torchaudio.list_audio_backends` removed in 2.11 | Monkey-patch returns `["soundfile"]` |
| `scipy.special.sph_harm` removed in 1.14 | Stub returning `0.0` |
| String labels from OWW batch generator | Cast to `float32` in `_tensor_next` |
| Feature tensors are `float16`, model expects `float32` | `.float()` on all tensor returns |
| FP validation data OOM (17 GB .npy) | Chunked `IterableDataset` with 8192-row reads |
| Infinite `mmap_batch_generator` for validation | Finite `TensorDataset` + `DataLoader` for `X_val` |

## Submission

OWW models staged for [fwartner/home-assistant-wakewords-collection](https://github.com/fwartner/home-assistant-wakewords-collection).
**Human gate enforced** — pipeline stages artifacts but never auto-opens PRs.

## License

MIT
