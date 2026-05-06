#!/bin/bash
# Usage:
#   ./download_eval.sh crossseq <local_output_dir>     — downloads crossseq results (balloon2)
#   ./download_eval.sh multiseq <local_output_dir>     — downloads multiseq results (crowd etc.)
#
# Example:
#   ./download_eval.sh crossseq 260506_output_tokensuppress
#   ./download_eval.sh multiseq 260506_output_multiseq

MODE=${1:?Usage: $0 <crossseq|multiseq> <output_dir_name>}
LOCAL=/home/hanmy/Dokumente/MA_WS2526/Masterthesis/Repos/DynamicReconstructionSplat/${2:?Usage: $0 <crossseq|multiseq> <output_dir_name>}
REMOTE=hanmydo@131.159.11.60:/mnt/home/hanmydo/DynamicReconstructionSplat

mkdir -p "$LOCAL"

download_model() {
  local label="$1"   # e.g. vggt_baseline
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

if [ "$MODE" = "crossseq" ]; then
  download_model "vggt_baseline"        "output_crossseq_vggt_baseline"        "$LOCAL/vggt_baseline"
  download_model "vggt4d_tokensuppress" "output_crossseq_vggt4d_tokensuppress" "$LOCAL/vggt4d_tokensuppress"

elif [ "$MODE" = "multiseq" ]; then
  SEQ="rgbd_bonn_crowd"
  download_model "${SEQ}_vggt_baseline"        "output_eval_multiseq/${SEQ}_vggt_baseline"        "$LOCAL/${SEQ}_vggt_baseline"
  download_model "${SEQ}_vggt4d_tokensuppress" "output_eval_multiseq/${SEQ}_vggt4d_tokensuppress" "$LOCAL/${SEQ}_vggt4d_tokensuppress"

else
  echo "Unknown mode: $MODE. Use 'crossseq' or 'multiseq'."
  exit 1
fi
