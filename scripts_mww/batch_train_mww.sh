#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="/opt/ai/wakeword-train/mww-venv"
LOG_DIR="/opt/ai/wakeword-train/logs"

# Default words if none supplied
DEFAULT_WORDS="hey_ara ok_ara hey_bender ok_bender"
if [ $# -gt 0 ]; then
    WORDS="$*"
else
    WORDS="$DEFAULT_WORDS"
fi

source "$VENV/bin/activate"
mkdir -p "$LOG_DIR"

PASSED=""
FAILED=""
PASS_COUNT=0
FAIL_COUNT=0

for word in $WORDS; do
    echo "========================================"
    echo "[$(date '+%H:%M:%S')] Processing: $word"
    echo "========================================"
    LOG="$LOG_DIR/${word}_mww.log"

    RC=0
    (
        set -euo pipefail
        echo "[$(date)] Starting mWW pipeline for $word"

        echo "[$(date)] Step 03: Generate features..."
        python3 "$SCRIPT_DIR/03_generate_features.py" "$word"

        echo "[$(date)] Step 04: Train model..."
        python3 "$SCRIPT_DIR/04_train_mww.py" "$word"

        echo "[$(date)] Step 05: Export model..."
        python3 "$SCRIPT_DIR/05_export_mww.py" "$word"

        echo "[$(date)] Step 06: Validate model..."
        python3 "$SCRIPT_DIR/06_validate_mww.py" "$word"

        echo "[$(date)] Pipeline complete for $word"
    ) > "$LOG" 2>&1 || RC=$?

    if [ $RC -eq 0 ]; then
        echo "  $word: PASSED"
        PASSED="$PASSED $word"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  $word: FAILED (exit $RC, see $LOG)"
        FAILED="$FAILED $word"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Passed ($PASS_COUNT):${PASSED:- none}"
echo "Failed ($FAIL_COUNT):${FAILED:- none}"
echo "Logs: $LOG_DIR/*_mww.log"
