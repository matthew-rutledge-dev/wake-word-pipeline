#!/usr/bin/env bash
set -euo pipefail

# Wake Word Training Pipeline
# Usage: ./pipeline.sh <word_id> [--oww-only|--mww-only]
#
# GPU-first: all steps prefer CUDA GPU when available, fall back to CPU.
# Piper-first: sample generation uses Piper TTS (neural), espeak-ng as fallback.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORD_ID="${1:?Usage: $0 <word_id> [--oww-only|--mww-only]}"
MODE="${2:-both}"
CONFIG="${SCRIPT_DIR}/words/${WORD_ID}/config.yaml"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
    echo "Create words/${WORD_ID}/config.yaml first."
    exit 1
fi

DISPLAY_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['display_name'])")

echo "=========================================="
echo " Wake Word Pipeline: '${DISPLAY_NAME}' (${WORD_ID})"
echo " Mode: ${MODE}"
echo "=========================================="

# Detect GPU
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo " GPU:  ${GPU_NAME}"
else
    echo " GPU:  None detected (CPU fallback)"
fi
echo ""

mkdir -p "${SCRIPT_DIR}/artifacts/${WORD_ID}/oww"
mkdir -p "${SCRIPT_DIR}/artifacts/${WORD_ID}/mww"

if [ "${MODE}" != "--mww-only" ]; then
    echo "--- Step 1: Generate samples (Piper TTS, GPU) ---"
    python3 "${SCRIPT_DIR}/scripts/01_generate_samples.py" "${WORD_ID}"

    echo ""
    echo "--- Step 2: Train openWakeWord model (GPU) ---"
    python3 "${SCRIPT_DIR}/scripts/02_train_oww.py" "${WORD_ID}"

    echo ""
    echo "--- Step 3: Export OWW TFLite + ONNX (CPU) ---"
    CUDA_VISIBLE_DEVICES=-1 python3 "${SCRIPT_DIR}/scripts/03_export_oww.py" "${WORD_ID}"

    echo ""
    echo "--- Step 6: Validate OWW model ---"
    python3 "${SCRIPT_DIR}/scripts/06_validate.py" "${WORD_ID}" oww
fi

if [ "${MODE}" != "--oww-only" ]; then
    echo ""
    echo "--- Step 4: Train microWakeWord model (GPU) ---"
    python3 "${SCRIPT_DIR}/scripts/04_train_mww.py" "${WORD_ID}"

    echo ""
    echo "--- Step 5: Export mWW TFLite + manifest ---"
    python3 "${SCRIPT_DIR}/scripts/05_export_mww.py" "${WORD_ID}"

    echo ""
    echo "--- Step 6: Validate mWW model ---"
    python3 "${SCRIPT_DIR}/scripts/06_validate.py" "${WORD_ID}" mww
fi

echo ""
echo "--- Step 7: Stage for submission ---"
python3 "${SCRIPT_DIR}/scripts/07_stage_submission.py" "${WORD_ID}"

echo ""
echo "=========================================="
echo " Pipeline complete: ${DISPLAY_NAME}"
echo " Artifacts: artifacts/${WORD_ID}/"
echo " Staging:   staging/${WORD_ID}/"
echo "=========================================="
