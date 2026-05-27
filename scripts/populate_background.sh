#!/usr/bin/env bash
# populate_background.sh — Symlink negative_audio WAVs into each word's background_data/ dir.
#
# The OWW training script (02_train_oww.py) reads background audio from
# artifacts/<word>/background_data/. This script populates those dirs with
# symlinks to the shared negative_audio/ directory (room recordings) and
# optionally a random subset of Common Voice WAVs from negative_audio_cv/.
#
# Usage:
#   ./populate_background.sh                        # all words, default 5000 CV files
#   ./populate_background.sh --max-cv 10000         # more CV diversity
#   ./populate_background.sh hey_ara hey_bender     # specific words
#   ./populate_background.sh --max-cv 0 hey_ara     # room recordings only

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NEGATIVE_AUDIO_DIR="${REPO_DIR}/negative_audio"
NEGATIVE_AUDIO_CV_DIR="${REPO_DIR}/negative_audio_cv"
ARTIFACTS_DIR="${REPO_DIR}/artifacts"

MAX_CV=5000  # default: 5000 random CV files per word

# Parse --max-cv flag
words_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-cv)
            MAX_CV="$2"
            shift 2
            ;;
        *)
            words_args+=("$1")
            shift
            ;;
    esac
done
set -- "${words_args[@]+"${words_args[@]}"}"

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

# Count Common Voice WAVs (optional — may be a symlink to /var/lib/vms/common_voice/wav/)
cv_count=0
if [[ -d "${NEGATIVE_AUDIO_CV_DIR}" && "${MAX_CV}" -gt 0 ]]; then
    cv_count=$(find -L "${NEGATIVE_AUDIO_CV_DIR}" -name "*.wav" -type f | wc -l)
fi

echo "Source: ${NEGATIVE_AUDIO_DIR} (${wav_count} room WAV files)"
if [[ "${cv_count}" -gt 0 ]]; then
    cv_use=$((cv_count < MAX_CV ? cv_count : MAX_CV))
    echo "Source: ${NEGATIVE_AUDIO_CV_DIR} (${cv_count} total, using ${cv_use} per word)"
else
    cv_use=0
fi
total_wav=$((wav_count + cv_use))
echo "Total negative WAV files per word: ${total_wav}"

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

# Pre-build the random CV subset list once (same subset for all words for reproducibility)
cv_tmplist=""
if [[ "${cv_use}" -gt 0 ]]; then
    cv_tmplist=$(mktemp)
    find -L "${NEGATIVE_AUDIO_CV_DIR}" -name "*.wav" -type f | shuf -n "${cv_use}" > "${cv_tmplist}"
    echo "  Selected ${cv_use} random CV files for background data"
fi

linked=0
skipped=0
for word in "${words[@]}"; do
    bg_dir="${ARTIFACTS_DIR}/${word}/background_data"
    mkdir -p "${bg_dir}"

    # Remove stale symlinks
    find "${bg_dir}" -maxdepth 1 -type l ! -exec test -e {} \; -delete 2>/dev/null || true

    existing=$(find "${bg_dir}" -maxdepth 1 -name "*.wav" | wc -l)
    if [[ "${existing}" -ge "${total_wav}" ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    # Create symlinks for room recordings
    count=0
    while IFS= read -r wav; do
        fname=$(basename "$wav")
        link_name="${bg_dir}/negative_audio__${fname}"
        if [[ ! -e "${link_name}" ]]; then
            ln -sf "${wav}" "${link_name}"
            count=$((count + 1))
        fi
    done < <(find "${NEGATIVE_AUDIO_DIR}" -name "*.wav" -type f)

    # Create symlinks for Common Voice subset (if available)
    cv_linked=0
    if [[ -n "${cv_tmplist}" && -s "${cv_tmplist}" ]]; then
        while IFS= read -r wav; do
            fname=$(basename "$wav")
            link_name="${bg_dir}/cv__${fname}"
            if [[ ! -e "${link_name}" ]]; then
                ln -sf "${wav}" "${link_name}"
                cv_linked=$((cv_linked + 1))
            fi
        done < "${cv_tmplist}"
    fi

    linked=$((linked + 1))
    echo "  ${word}: ${count} room + ${cv_linked} CV symlinks created"
done

# Cleanup
[[ -n "${cv_tmplist}" ]] && rm -f "${cv_tmplist}"

echo ""
echo "Done. Populated: ${linked}, Already up-to-date: ${skipped}"
echo "Total words: ${#words[@]}"
