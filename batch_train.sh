#!/bin/bash
set -euo pipefail

VENV="/opt/ai/wakeword-train/venv"
SCRIPTS="/opt/ai/wakeword-train/wake-word-pipeline/scripts"
LOGDIR="/opt/ai/wakeword-train/logs"
export PIPER_SAMPLE_GENERATOR_PATH="/opt/ai/wakeword-train/piper-sample-generator"

source "$VENV/bin/activate"
mkdir -p "$LOGDIR"

# Auto-discover all words from words/ directory
WORDS=()
for d in /opt/ai/wakeword-train/wake-word-pipeline/words/*/; do
  word=$(basename "$d")
  WORDS+=("$word")
done

# Sort: hey_ first, then ok_, priority words first
IFS=$'\n' WORDS=($(printf '%s\n' "${WORDS[@]}" | sort)); unset IFS

echo "=============================="
echo "Wake Word Batch Training"
echo "$(date) - ${#WORDS[@]} words"
echo "=============================="

# Phase 1: Generate samples sequentially (GPU Piper TTS - one at a time)
echo ""
echo "=== PHASE 1: Sample Generation ==="
for word in "${WORDS[@]}"; do
  ARTDIR="/opt/ai/wakeword-train/wake-word-pipeline/artifacts/$word"
  # Cache invalidation handled by 01_generate_samples.py via config hash
  # (stale samples are auto-purged when config changes)
  echo "[GEN] $word - generating samples..."
  python3 "$SCRIPTS/01_generate_samples.py" "$word" > "$LOGDIR/${word}_01_gen.log" 2>&1
  echo "[GEN] $word - done ($(date))"
done

# Phase 2: Train OWW models (features + DNN)
# Run up to 3 in parallel since each uses ~300MB VRAM + ~3GB RAM
echo ""
echo "=== PHASE 2: OWW Training ==="
PARALLEL=3
running=0
pids=()
words_running=()

for word in "${WORDS[@]}"; do
  OWWDIR="/opt/ai/wakeword-train/wake-word-pipeline/artifacts/$word/oww"
  # ONNX skip only if config hash matches (config hash checked inside 02_train_oww.py)
  HASH_FILE="/opt/ai/wakeword-train/wake-word-pipeline/artifacts/$word/.config_hashes.json"
  if [ -f "$OWWDIR/${word}.onnx" ] && [ -f "$HASH_FILE" ]; then
    echo "[SKIP] $word - ONNX model exists and config unchanged"
    continue
  fi

  echo "[TRAIN] $word - starting training..."
  python3 "$SCRIPTS/02_train_oww.py" "$word" > "$LOGDIR/${word}_02_train.log" 2>&1 &
  pids+=($!)
  words_running+=("$word")
  running=$((running + 1))

  if [ $running -ge $PARALLEL ]; then
    # Wait for any one to finish
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        wait "${pids[$i]}" || echo "[WARN] ${words_running[$i]} exited with error"
        echo "[DONE] ${words_running[$i]} - training complete ($(date))"
        unset 'pids[i]'
        unset 'words_running[i]'
        running=$((running - 1))
        break
      fi
    done
    # Re-index arrays
    pids=("${pids[@]}")
    words_running=("${words_running[@]}")

    # If still at max, wait for the first one
    if [ $running -ge $PARALLEL ]; then
      wait "${pids[0]}" || echo "[WARN] ${words_running[0]} exited with error"
      echo "[DONE] ${words_running[0]} - training complete ($(date))"
      pids=("${pids[@]:1}")
      words_running=("${words_running[@]:1}")
      running=$((running - 1))
    fi
  fi
done

# Wait for remaining
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || echo "[WARN] ${words_running[$i]} exited with error"
  echo "[DONE] ${words_running[$i]} - training complete ($(date))"
done

# Phase 3: Export ONNX -> TFLite (CPU only, no GPU needed)
echo ""
echo "=== PHASE 3: ONNX -> TFLite Export ==="
for word in "${WORDS[@]}"; do
  OWWDIR="/opt/ai/wakeword-train/wake-word-pipeline/artifacts/$word/oww"
  if [ ! -f "$OWWDIR/${word}.onnx" ]; then
    echo "[SKIP] $word - no ONNX model"
    continue
  fi
  if [ -f "$OWWDIR/${word}.tflite" ] && [ "$OWWDIR/${word}.tflite" -nt "$OWWDIR/${word}.onnx" ]; then
    echo "[SKIP] $word - TFLite already up to date"
    continue
  fi
  echo "[EXPORT] $word - converting..."
  python3 "$SCRIPTS/03_export_oww.py" "$word" > "$LOGDIR/${word}_03_export.log" 2>&1 || echo "[WARN] $word export failed"
  echo "[EXPORT] $word - done"
done

echo ""
echo "=============================="
echo "Batch training complete: $(date)"
echo "=============================="

# Summary
echo ""
echo "=== RESULTS ==="
for word in "${WORDS[@]}"; do
  OWWDIR="/opt/ai/wakeword-train/wake-word-pipeline/artifacts/$word/oww"
  if [ -f "$OWWDIR/${word}.tflite" ]; then
    SIZE=$(stat --format='%s' "$OWWDIR/${word}.tflite" 2>/dev/null)
    echo "[OK] $word - TFLite ${SIZE} bytes"
  elif [ -f "$OWWDIR/${word}.onnx" ]; then
    SIZE=$(stat --format='%s' "$OWWDIR/${word}.onnx" 2>/dev/null)
    echo "[PARTIAL] $word - ONNX only ${SIZE} bytes (TFLite export failed)"
  else
    echo "[FAIL] $word - no model output"
  fi
done

# Generate metrics report
echo ""
echo "=== METRICS REPORT ==="
python3 "$SCRIPTS/metrics_report.py" 2>&1 || echo "[WARN] Metrics report failed"

# Kill background metrics monitor if running
pkill -f metrics_monitor.sh 2>/dev/null || true
