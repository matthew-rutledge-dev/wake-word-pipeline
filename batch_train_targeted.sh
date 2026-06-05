#!/bin/bash
set -euo pipefail

# Targeted retrain: FAIL models + boosted WARN models + new my_ words
# 2026-06-02

VENV="/opt/ai/wakeword-train/venv"
SCRIPTS="/opt/ai/wakeword-train/wake-word-pipeline/scripts"
REPO="/opt/ai/wakeword-train/wake-word-pipeline"
LOGDIR="/opt/ai/wakeword-train/logs"
export PIPER_SAMPLE_GENERATOR_PATH="/opt/ai/wakeword-train/piper-sample-generator"

source "$VENV/bin/activate"
cd "$REPO"
mkdir -p "$LOGDIR"

# FAIL models (retrain with boosted configs)
FAIL_WORDS=(hey_count ok_ara ok_frieren)

# WARN models (boosted configs)
WARN_WORDS=(hey_anya hey_ara hey_frieren hey_knight hey_witch ok_chief ok_knight)

# New words (first-time training)
NEW_WORDS=(my_goddess my_knight my_god my_man my_lord)

# Combine all
ALL_WORDS=("${FAIL_WORDS[@]}" "${WARN_WORDS[@]}" "${NEW_WORDS[@]}")

echo "=============================="
echo "Targeted Training Batch"
echo "$(date) — ${#ALL_WORDS[@]} words"
echo "FAIL (retrain): ${FAIL_WORDS[*]}"
echo "WARN (retrain): ${WARN_WORDS[*]}"
echo "NEW  (train):   ${NEW_WORDS[*]}"
echo "=============================="

# Phase 0: Invalidate caches for retrain words (NOT new words — they have no cache)
echo ""
echo "=== PHASE 0: Invalidating caches for retrain targets ==="
for word in "${FAIL_WORDS[@]}" "${WARN_WORDS[@]}"; do
  ARTDIR="artifacts/$word"
  rm -f "$ARTDIR/$word/positive_features_train.npy" 2>/dev/null
  rm -f "$ARTDIR/$word/positive_features_val.npy" 2>/dev/null
  rm -f "$ARTDIR/$word/negative_features_train.npy" 2>/dev/null
  rm -f "$ARTDIR/$word/negative_features_val.npy" 2>/dev/null
  rm -f "$ARTDIR/oww/${word}.onnx" 2>/dev/null
  rm -f "$ARTDIR/oww/${word}.tflite" 2>/dev/null
  rm -f "$ARTDIR/.config_hashes.json" 2>/dev/null
done
echo "Invalidated caches for ${#FAIL_WORDS[@]} FAIL + ${#WARN_WORDS[@]} WARN words"

# Phase 1: Generate samples (new words need full gen, retrain words may skip)
echo ""
echo "=== PHASE 1: Sample Generation ==="
for word in "${ALL_WORDS[@]}"; do
  echo "[GEN] $word — $(date '+%H:%M:%S')"
  python3 "$SCRIPTS/01_generate_samples.py" "$word" > "$LOGDIR/${word}_01_targeted.log" 2>&1 || echo "[WARN] $word generation failed"
done

# Phase 2: Train OWW models (3 parallel)
echo ""
echo "=== PHASE 2: OWW Training (3 parallel) ==="
PARALLEL=3
running=0
pids=()
words_running=()

for word in "${ALL_WORDS[@]}"; do
  echo "[TRAIN] $word — starting $(date '+%H:%M:%S')"
  python3 "$SCRIPTS/02_train_oww.py" "$word" > "$LOGDIR/${word}_02_targeted.log" 2>&1 &
  pids+=($!)
  words_running+=("$word")
  running=$((running + 1))

  if [ $running -ge $PARALLEL ]; then
    wait "${pids[0]}" && echo "[DONE] ${words_running[0]} — $(date '+%H:%M:%S')" || echo "[FAIL] ${words_running[0]} exited with error"
    pids=("${pids[@]:1}")
    words_running=("${words_running[@]:1}")
    running=$((running - 1))
  fi
done

# Wait for remaining
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" && echo "[DONE] ${words_running[$i]} — $(date '+%H:%M:%S')" || echo "[FAIL] ${words_running[$i]} exited with error"
done

# Phase 3: Export ONNX → TFLite
echo ""
echo "=== PHASE 3: ONNX → TFLite Export ==="
for word in "${ALL_WORDS[@]}"; do
  OWWDIR="artifacts/$word/oww"
  if [ ! -f "$OWWDIR/${word}.onnx" ]; then
    echo "[SKIP] $word — no ONNX model"
    continue
  fi
  echo "[EXPORT] $word"
  CUDA_VISIBLE_DEVICES=-1 python3 "$SCRIPTS/03_export_oww.py" "$word" > "$LOGDIR/${word}_03_targeted.log" 2>&1 || echo "[WARN] $word export failed"
done

echo ""
echo "=============================="
echo "Targeted training complete: $(date)"
ONNX_COUNT=$(find artifacts/ -name '*.onnx' -path '*/oww/*' | wc -l)
TFLITE_COUNT=$(find artifacts/ -name '*.tflite' -path '*/oww/*' | wc -l)
echo "Total ONNX:   $ONNX_COUNT"
echo "Total TFLite: $TFLITE_COUNT"
echo "=============================="
