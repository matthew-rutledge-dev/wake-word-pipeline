#!/bin/bash
set -euo pipefail

VENV="/opt/ai/wakeword-train/venv"
SCRIPTS="/opt/ai/wakeword-train/wake-word-pipeline/scripts"
LOGDIR="/opt/ai/wakeword-train/logs"
export PIPER_SAMPLE_GENERATOR_PATH="/opt/ai/wakeword-train/piper-sample-generator"

source "$VENV/bin/activate"
mkdir -p "$LOGDIR"

# Training order: cortana after bender, then remaining
WORDS=(
  hey_ara ok_ara
  hey_bender ok_bender
  hey_cortana ok_cortana
  hey_spongebob ok_spongebob
  hey_anya ok_anya
  hey_naruto ok_naruto
  hey_veldora ok_veldora
  hey_rimuru ok_rimuru
  hey_chief ok_chief
  hey_my_knight ok_my_knight
  hey_my_goddess ok_my_goddess
  hey_santa ok_santa
)

echo "=============================="
echo "Wake Word Batch Training"
echo "$(date) - ${#WORDS[@]} words"
echo "=============================="

# Phase 1: Generate samples sequentially (GPU Piper TTS - one at a time)
echo ""
echo "=== PHASE 1: Sample Generation ==="
for word in "${WORDS[@]}"; do
  ARTDIR="/opt/ai/wakeword-train/wake-word-pipeline/artifacts/$word"
  if [ -d "$ARTDIR/positive_train" ] && [ "$(find "$ARTDIR/positive_train" -name '*.wav' 2>/dev/null | wc -l)" -ge 1000 ]; then
    echo "[SKIP] $word - samples already exist"
    continue
  fi
  echo "[GEN] $word - generating samples..."
  python3 "$SCRIPTS/01_generate_samples.py" "$word" > "$LOGDIR/${word}_01_gen.log" 2>&1
  echo "[GEN] $word - done ($(date))"
done

# Phase 2: Train OWW models (features + DNN + export)
# Run up to 3 in parallel since each uses ~300MB VRAM + ~3GB RAM
echo ""
echo "=== PHASE 2: OWW Training ==="
PARALLEL=3
running=0
pids=()
words_running=()

for word in "${WORDS[@]}"; do
  OWWDIR="/opt/ai/wakeword-train/wake-word-pipeline/artifacts/$word/oww"
  if [ -f "$OWWDIR/${word}.onnx" ]; then
    echo "[SKIP] $word - ONNX model already exists"
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

echo ""
echo "=============================="
echo "Batch training complete: $(date)"
echo "=============================="

# Summary
echo ""
echo "=== RESULTS ==="
for word in "${WORDS[@]}"; do
  OWWDIR="/opt/ai/wakeword-train/wake-word-pipeline/artifacts/$word/oww"
  if [ -f "$OWWDIR/${word}.onnx" ]; then
    SIZE=$(stat --format='%s' "$OWWDIR/${word}.onnx" 2>/dev/null)
    echo "[OK] $word - ${SIZE} bytes"
  else
    echo "[FAIL] $word - no ONNX output"
  fi
done

# Generate metrics report
echo ""
echo "=== METRICS REPORT ==="
python3 "$SCRIPTS/metrics_report.py" 2>&1 || echo "[WARN] Metrics report failed"

# Kill background metrics monitor if running
pkill -f metrics_monitor.sh 2>/dev/null || true