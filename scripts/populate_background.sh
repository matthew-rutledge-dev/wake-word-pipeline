#!/usr/bin/env bash
# populate_background.sh — Symlink negative_audio WAVs into each word's background_data/ dir.
#
# The OWW training script (02_train_oww.py) reads background audio from
# artifacts/<word>/background_data/. This script populates those dirs with
# symlinks to the shared negative_audio/ directory (downloaded by
# download_background_audio.py).
#
# Usage:
#   ./populate_background.sh              # all words
#   ./populate_background.sh hey_ara hey_bender  # specific words

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NEGATIVE_AUDIO_DIR="${REPO_DIR}/negative_audio"
ARTIFACTS_DIR="${REPO_DIR}/artifacts"

if [[ ! -d "${NEGATIVE_AUDIO_DIR}" ]]; then
    echo "ERROR: negative_audio/ not found at ${NEGATIVE_AUDIO_DIR}"
    echo "Run download_background_audio.py first."
    exit 1
fi

wav_count=$(find "${NEGATIVE_AUDIO_DIR}" -name "*.wav" | wc -l)
if [[ "${wav_count}" -eq 0 ]]; then
    echo "ERROR: No WAV files in ${NEGATIVE_AUDIO_DIR}"
    exit 1
fi

echo "Source: ${NEGATIVE_AUDIO_DIR} (${wav_count} WAV files)"

# Determine words to populate
if [[ $# -gt 0 ]]; then
    words=("$@")
else
    words=()
    for d in "${ARTIFACTS_DIR}"/*/; do
        [[ -d "$d" ]] || continue
        word=$(basename "$d")
        [[ "$word" == ".shared" ]] && continue
        words+=("$word")
    done
fi

if [[ ${#words[@]} -eq 0 ]]; then
    echo "ERROR: No word directories found in ${ARTIFACTS_DIR}/"
    exit 1
fi

echo "Populating background_data/ for ${#words[@]} words..."

linked=0
skipped=0
for word in "${words[@]}"; do
    bg_dir="${ARTIFACTS_DIR}/${word}/background_data"
    mkdir -p "${bg_dir}"

    # Remove stale symlinks
    find "${bg_dir}" -maxdepth 1 -type l ! -exec test -e {} \; -delete 2>/dev/null || true

    existing=$(find "${bg_dir}" -maxdepth 1 -name "*.wav" | wc -l)
    if [[ "${existing}" -ge "${wav_count}" ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    # Create symlinks for all WAVs
    count=0
    while IFS= read -r wav; do
        fname=$(basename "$wav")
        cat=$(basename "$(dirname "$wav")")
        link_name="${bg_dir}/${cat}__${fname}"
        if [[ ! -e "${link_name}" ]]; then
            ln -sf "${wav}" "${link_name}"
            count=$((count + 1))
        fi
    done < <(find "${NEGATIVE_AUDIO_DIR}" -name "*.wav" -type f)

    linked=$((linked + 1))
    echo "  ${word}: ${count} symlinks created"
done

echo ""
echo "Done. Populated: ${linked}, Already up-to-date: ${skipped}"
echo "Total words: ${#words[@]}"
