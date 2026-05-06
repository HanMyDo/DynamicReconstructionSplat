#!/bin/bash
# Usage: ./download_eval.sh <local_output_dir>
# Example: ./download_eval.sh 260507_output_crossseq

LOCAL=/home/hanmy/Dokumente/MA_WS2526/Masterthesis/Repos/DynamicReconstructionSplat/${1:?Usage: $0 <output_dir_name>}
REMOTE=hanmydo@131.159.11.60:/mnt/home/hanmydo/DynamicReconstructionSplat

mkdir -p "$LOCAL"

download_model() {
  local label="$1"
  local remote_dir="$2"
  local local_dir="$3"

  echo "=== $label ==="
  mkdir -p "$local_dir/images" "$local_dir/dyn_mask"

  # PLY + metrics + videos (exclude large image/mask dirs)
  rsync -av \
    --exclude='images/' \
    --exclude='dyn_mask/' \
    "$REMOTE/$remote_dir/" \
    "$local_dir/"

  # Middle 10 batches of images
  rsync -av \
    --filter='+ b022[0-9]_*.png' \
    --filter='- *' \
    "$REMOTE/$remote_dir/images/" \
    "$local_dir/images/" 2>/dev/null || true

  # Middle 10 batches of dyn_mask
  rsync -av \
    --filter='+ b022[0-9]_*.png' \
    --filter='- *' \
    "$REMOTE/$remote_dir/dyn_mask/" \
    "$local_dir/dyn_mask/" 2>/dev/null || true
}

download_model "vggt_baseline"        "output_crossseq_vggt_baseline"        "$LOCAL/vggt_baseline"
download_model "vggt4d_tokensuppress" "output_crossseq_vggt4d_tokensuppress" "$LOCAL/vggt4d_tokensuppress"
download_model "vggt4d_pretrained"    "output_crossseq_vggt4d_pretrained"    "$LOCAL/vggt4d_pretrained"
