#!/usr/bin/env bash
# Downloads ACE-Step 1.5 turbo (2B) and converts it to the diffusers format that
# music_gen.generation expects. Needs ~12 GB free during conversion, ~6 GB after.
#
# Usage: bash scripts/download_acestep.sh
set -euo pipefail

REPO="ACE-Step/Ace-Step1.5"
REVISION="19671f406d603126926c1b7e2adc169acbcade22"  # pinned for reproducibility
RAW="data/models/acestep-raw"
OUT="data/models/acestep-v15-turbo-diffusers"

if [ -f "$OUT/model_index.json" ]; then
  echo "Already converted: $OUT"
  exit 0
fi

# Separate calls: one call with several --include patterns silently skipped a folder.
for part in "acestep-v15-turbo/*" "vae/*" "Qwen3-Embedding-0.6B/*"; do
  uv run hf download "$REPO" --revision "$REVISION" --include "$part" --local-dir "$RAW"
done

uv run python scripts/third_party/convert_ace_step_to_diffusers.py \
  --checkpoint_dir "$RAW" --dit_config acestep-v15-turbo --output_dir "$OUT" --dtype bf16

rm -rf "$RAW"  # the converted copy is all we need
echo "Done: $OUT"
