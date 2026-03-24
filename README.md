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

## Pipeline Steps

| Step | Script | GPU Used | Description |
|------|--------|----------|-------------|
| 1 | 01_generate_samples.py | Piper TTS inference | Generate positive + adversarial negative WAV samples |
| 2 | 02_train_oww.py | Embedding + DNN training | Augment clips, compute OWW features, train model |
| 3 | 03_export_oww.py | CPU only (TFLite) | Convert ONNX to TFLite, validate shapes |
| 4 | 04_train_mww.py | TF training | Train microWakeWord streaming model |
| 5 | 05_export_mww.py | N/A | Export mWW TFLite + JSON manifest |
| 6 | 06_validate.py | OWW inference | Shape check + detection recall on validation clips |
| 7 | 07_stage_submission.py | N/A | Stage for fwartner collection (human gate) |

## Structure

```
words/<word_id>/config.yaml   # Per-word configuration (TTS, training params)
scripts/                      # Training and export scripts (01-07)
docker/                       # GPU-enabled Dockerfiles (NVIDIA CUDA base)
artifacts/                    # Trained model outputs (gitignored)
staging/                      # Prepared submissions (human-reviewed before PR)
```

## TTS Architecture

| Engine | Type | GPU | Quality | Use Case |
|--------|------|-----|---------|----------|
| **Piper TTS** | Neural VITS | CUDA accelerated | Natural, multi-speaker | Primary for all production training |
| espeak-ng | Formant synthesis | CPU only | Robotic | Fallback when Piper unavailable |

Piper uses en_US-libritts_r-medium.pt (~900 speakers) via
[piper-sample-generator](https://github.com/rhasspy/piper-sample-generator).
Samples are augmented with room impulse responses + background noise for realism.

## Wake Word Queue

| # | ID | Display | OWW | mWW |
|---|-----|---------|-----|-----|
| 1 | hey_ara | Hey Ara | Retrain (Piper) | Pending |
| 2 | hey_bender | Hey Bender | Pending | Pending |
| 3 | hey_spongebob | Hey SpongeBob | Pending | Pending |
| 4 | hey_anya | Hey Anya | Pending | Pending |
| 5 | hey_naruto | Hey Naruto | Pending | Pending |
| 6 | hey_veldora | Hey Veldora | Pending | Pending |
| 7 | hey_rimuru | Hey Rimuru | Pending | Pending |

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

## Submission

OWW models staged for [fwartner/home-assistant-wakewords-collection](https://github.com/fwartner/home-assistant-wakewords-collection).
**Human gate enforced** — pipeline stages artifacts but never auto-opens PRs.

## License

MIT
