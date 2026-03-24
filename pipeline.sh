#!/usr/bin/env bash
set -euo pipefail

# Wake Word Training Pipeline
# Usage: ./pipeline.sh <word_id> [--oww-only|--mww-only]

WORD_ID=""
MODE=""
SCRIPT_DIR="C:\Users\devru"
CONFIG="/words//config.yaml"

if [ ! -f "" ]; then
    echo "ERROR: Config not found: "
    echo "Create words//config.yaml first."
    exit 1
fi

DISPLAY_NAME=
echo "=== Training '' () ==="
echo "Mode: "
echo ""

mkdir -p "/artifacts//oww"
mkdir -p "/artifacts//mww"

if [ "" != "--mww-only" ]; then
    echo "--- Step 1: Generate positive samples ---"
    python3 "/scripts/01_generate_samples.py" ""

    echo "--- Step 2: Train openWakeWord model ---"
    python3 "/scripts/02_train_oww.py" ""

    echo "--- Step 3: Export OWW TFLite + ONNX ---"
    CUDA_VISIBLE_DEVICES=-1 python3 "/scripts/03_export_oww.py" ""

    echo "--- Step 6: Validate OWW model ---"
    python3 "/scripts/06_validate.py" "" oww
fi

if [ "" != "--oww-only" ]; then
    echo "--- Step 4: Train microWakeWord model ---"
    python3 "/scripts/04_train_mww.py" ""

    echo "--- Step 5: Export mWW TFLite + JSON ---"
    python3 "/scripts/05_export_mww.py" ""

    echo "--- Step 6: Validate mWW model ---"
    python3 "/scripts/06_validate.py" "" mww
fi

echo "--- Step 7: Stage for submission ---"
python3 "/scripts/07_stage_submission.py" ""

echo ""
echo "=== Pipeline complete for  ==="
echo "Artifacts: artifacts//"
echo "Staging:   staging//"
