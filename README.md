# wake-word-pipeline

End-to-end training pipeline for custom wake word models targeting both
**openWakeWord** (Wyoming/OWW add-on) and **microWakeWord** (ESPHome HA Voice PE firmware).

## Quick Start

`ash
# Train a single wake word (both OWW + mWW)
./pipeline.sh hey_ara

# Train OWW only
./pipeline.sh hey_bender --oww-only

# Train mWW only
./pipeline.sh hey_ara --mww-only
`

## Structure

`
words/<word_id>/config.yaml   # Per-word configuration
scripts/                      # Training and export scripts
docker/                       # Dockerfiles for training environments
artifacts/                    # Trained model outputs (gitignored, large files)
staging/                      # Prepared submissions for fwartner collection
`

## Wake Word Queue

| Priority | ID | Display | OWW | mWW |
|----------|----|---------|-----|-----|
| 1 | hey_ara | Hey Ara | Trained | Pending |
| 2 | hey_bender | Hey Bender | Pending | Pending |
| 3 | hey_spongebob | Hey SpongeBob | Pending | Pending |
| 4 | hey_anya | Hey Anya | Pending | Pending |
| 5 | hey_naruto | Hey Naruto | Pending | Pending |
| 6 | hey_veldora | Hey Veldora | Pending | Pending |
| 7 | hey_rimuru | Hey Rimuru | Pending | Pending |

## Training Environment

- **Server**: servergen1.cdclocal (Ubuntu 25.10, RTX 5060 Ti 16GB, CUDA 13.0)
- **OWW Python env**: Python 3.13.7, openwakeword 0.4.0, TensorFlow 2.21.0
- **mWW Python env**: TBD (microWakeWord framework from OHF-Voice)
- **Docker**: nvidia-container-toolkit v1.19.0-1, Docker 29.1.3 + Compose v5.1.1

## Naming Convention

- **File/ID**: `lowercase_underscore` (e.g., `hey_ara`, `hey_bender`)
- **Display string**: Title Case in JSON manifests (e.g., `"Hey Ara"`)

## Submission

OWW models are staged for submission to
[fwartner/home-assistant-wakewords-collection](https://github.com/fwartner/home-assistant-wakewords-collection).
A human gate is enforced: the pipeline stages artifacts and prints PR instructions but never auto-opens PRs.

## License

MIT
