#!/bin/bash
set -euo pipefail

VENV="/opt/ai/wakeword-train/venv"
SCRIPTS="/opt/ai/wakeword-train/wake-word-pipeline/scripts"
REPO="/opt/ai/wakeword-train/wake-word-pipeline"
LOGDIR="/opt/ai/wakeword-train/logs"
export PIPER_SAMPLE_GENERATOR_PATH="/opt/ai/wakeword-train/piper-sample-generator"

source "$VENV/bin/activate"
cd "$REPO"
mkdir -p "$LOGDIR"

# Auto-discover all words
WORDS=()
for d in words/*/; do
  WORDS+=("$(basename "$d")")
done
IFS=$'\n' WORDS=($(printf '%s\n' "${WORDS[@]}" | sort)); unset IFS

echo "=============================="
echo "Cloud TTS Retrain Batch v2"
echo "$(date) - ${#WORDS[@]} words"
echo "=============================="

# Phase 0: Invalidate ONLY positive feature caches + ONNX + TFLite
# Keep negative features if they exist (they haven't changed)
echo ""
echo "=== PHASE 0: Invalidating positive caches ==="
for word in "${WORDS[@]}"; do
  ARTDIR="artifacts/$word"
  FEATDIR="$ARTDIR/$word"
  rm -f "$FEATDIR/positive_features_train.npy" 2>/dev/null
  rm -f "$FEATDIR/positive_features_val.npy" 2>/dev/null
  rm -f "$ARTDIR/oww/${word}.onnx" 2>/dev/null
  rm -f "$ARTDIR/oww/${word}.tflite" 2>/dev/null
  rm -f "$ARTDIR/.config_hashes.json" 2>/dev/null
done
echo "Invalidated: positive features, ONNX, TFLite for ${#WORDS[@]} words"

# Phase 1: Generate negative samples (positive will be skipped — already > 95%)
# Must be sequential — uses GPU Piper TTS
echo ""
echo "=== PHASE 1: Generate negative samples ==="
for word in "${WORDS[@]}"; do
  NEGDIR="artifacts/$word/negative_train"
  FEATDIR="artifacts/$word/$word"
  # Skip if negative features already exist
  if [ -f "$FEATDIR/negative_features_train.npy" ] && [ -f "$FEATDIR/negative_features_val.npy" ]; then
    echo "[SKIP-NEG] $word - negative features already exist"
    continue
  fi
  echo "[GEN] $word - generating samples (negatives)... $(date '+%H:%M:%S')"
  python3 "$SCRIPTS/01_generate_samples.py" "$word" > "$LOGDIR/${word}_01_regen.log" 2>&1 || echo "[WARN] $word sample generation failed"
  echo "[GEN] $word - done $(date '+%H:%M:%S')"
done

# Phase 2: Train OWW models (features + DNN) — 3 in parallel
echo ""
echo "=== PHASE 2: OWW Training (3 parallel) ==="
PARALLEL=3
running=0
pids=()
words_running=()

for word in "${WORDS[@]}"; do
  echo "[TRAIN] $word - starting $(date '+%H:%M:%S')"
  python3 "$SCRIPTS/02_train_oww.py" "$word" > "$LOGDIR/${word}_02_retrain_cloud.log" 2>&1 &
  pids+=($!)
  words_running+=("$word")
  running=$((running + 1))

  if [ $running -ge $PARALLEL ]; then
    wait "${pids[0]}" && echo "[DONE] ${words_running[0]} - $(date '+%H:%M:%S')" || echo "[WARN] ${words_running[0]} exited with error"
    pids=("${pids[@]:1}")
    words_running=("${words_running[@]:1}")
    running=$((running - 1))
  fi
done

# Wait for remaining
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" && echo "[DONE] ${words_running[$i]} - $(date '+%H:%M:%S')" || echo "[WARN] ${words_running[$i]} exited with error"
done

# Phase 3: Export ONNX -> TFLite (CPU only)
echo ""
echo "=== PHASE 3: ONNX -> TFLite Export ==="
for word in "${WORDS[@]}"; do
  OWWDIR="artifacts/$word/oww"
  if [ ! -f "$OWWDIR/${word}.onnx" ]; then
    echo "[SKIP] $word - no ONNX model (training may have failed)"
    continue
  fi
  echo "[EXPORT] $word"
  CUDA_VISIBLE_DEVICES=-1 python3 "$SCRIPTS/03_export_oww.py" "$word" > "$LOGDIR/${word}_03_export_cloud.log" 2>&1 || echo "[WARN] $word export failed"
done

echo ""
echo "=============================="
echo "Retrain complete: $(date)"
echo "ONNX:   $(find artifacts/ -name '*.onnx' -path '*/oww/*' | wc -l)"
echo "TFLite: $(find artifacts/ -name '*.tflite' -path '*/oww/*' | wc -l)"
echo "=============================="
