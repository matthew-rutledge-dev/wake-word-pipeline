#!/usr/bin/env bash
set -euo pipefail

# Pre-generated spectrogram features for microWakeWord training.
# License: CC-BY-NC-4.0 — personal non-commercial use only.
# Source: https://huggingface.co/datasets/kahrendt/microwakeword

NEG_DIR="/opt/ai/wakeword-train/mww-negatives"
BASE_URL="https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main"

mkdir -p "$NEG_DIR"

declare -A SIZES
SIZES[dinner_party]="444 MB"
SIZES[dinner_party_eval]="82 MB"
SIZES[speech]="3.18 GB"
SIZES[no_speech]="2 GB"

FILES=(dinner_party dinner_party_eval speech no_speech)

for name in "${FILES[@]}"; do
    zip_file="$NEG_DIR/${name}.zip"

    # Check if already extracted (zip creates {name}/ directory inside NEG_DIR)
    if [ -d "$NEG_DIR/$name" ] && [ "$(find "$NEG_DIR/$name" -maxdepth 3 -name '*_mmap' -type d 2>/dev/null | head -1)" ]; then
        echo "[$name] Already extracted with mmap data, skipping."
        continue
    fi

    echo "[$name] Downloading (~${SIZES[$name]})..."
    wget -c "$BASE_URL/${name}.zip" -O "$zip_file"

    echo "[$name] Extracting to $NEG_DIR..."
    unzip -o -q "$zip_file" -d "$NEG_DIR"

    echo "[$name] Done."
done

echo ""
echo "All negative datasets ready at $NEG_DIR"
ls -la "$NEG_DIR"/
