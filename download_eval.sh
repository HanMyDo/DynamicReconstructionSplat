#!/bin/bash
# Usage: ./download_eval.sh <local_output_dir>
# Example: ./download_eval.sh 260503_output_5epochs

LOCAL=/home/hanmy/Dokumente/MA_WS2526/Masterthesis/Repos/DynamicReconstructionSplat/${1:?Usage: $0 <output_dir_name>}
REMOTE=hanmydo@131.159.11.60:/mnt/home/hanmydo/DynamicReconstructionSplat

mkdir -p "$LOCAL"

for model in vggt_baseline vggt4d_baseline finetuned_singleseq finetuned_multiseq; do
  echo "=== $model ==="
  mkdir -p "$LOCAL/$model/images" "$LOCAL/$model/dyn_mask"

  # PLY + metrics + videos
  rsync -av \
    --exclude='images/' \
    --exclude='dyn_mask/' \
    "$REMOTE/output_crossseq_${model}/" \
    "$LOCAL/$model/"

  # Middle 10 batches of images
  rsync -av \
    --filter='+ b022[0-9]_*.png' \
    --filter='- *' \
    "$REMOTE/output_crossseq_${model}/images/" \
    "$LOCAL/$model/images/" 2>/dev/null || true

  # Middle 10 batches of dyn_mask
  rsync -av \
    --filter='+ b022[0-9]_*.png' \
    --filter='- *' \
    "$REMOTE/output_crossseq_${model}/dyn_mask/" \
    "$LOCAL/$model/dyn_mask/" 2>/dev/null || true
done
