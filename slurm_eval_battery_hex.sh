#!/bin/bash
#SBATCH --job-name=eval_battery
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --output=slurm_logs/eval_battery_%j.out
#SBATCH --error=slurm_logs/eval_battery_%j.err
# =============================================================================
# The WHOLE re-validation battery in ONE job, on hex-4gpu (conda, no enroot).
#
# Nine evaluations chained in one allocation so a single sbatch finishes the lot
# unattended. The dataset and masks are already on this machine, so unlike the
# old cluster there is no zip extraction and no mask relay.
#
#   removing_obstructing_box : control, flow, rigid-4, ft-control, ft-flow
#   placing_obstructing_box  : control, flow
#   synchronous2             : control, flow
#
# The control is repeated per sequence ON PURPOSE: a new mask changes both the
# dyn/static split and which Gaussians the flow displaces, so a control computed
# on different masks is not a valid baseline.
#
# ft runs are SKIPPED automatically when the anchor checkpoint is absent (it
# lives on the old cluster and has to be relayed) -- the seven frozen runs still
# complete, and the two ft ones can be added later.
#
# USAGE: sbatch slurm_eval_battery_hex.sh [MASK_DIR] [DATE_TAG]
# =============================================================================
set -uo pipefail

MASK_DIR=${1:-$HOME/data/mask_out/output_dyn_masks_precomputed_cs64_r518_st3_fs1_m6}
DATE_TAG=${2:-m6}
# Minimum free VRAM. An nf6 eval needs ~10 GB (these ran on 24 GB 3090s), far less
# than the mask precompute -- so do not inherit that script's 20 GB bar, or a card
# with a partial occupant gets refused when it would have been perfectly usable.
MIN_FREE=${3:-12000}
NF=6

REPO="${HOME}/DynamicReconstructionSplat"
DATA_ROOT="${HOME}/data/bonn/rgbd_bonn_dataset"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
cd ${REPO}; mkdir -p slurm_logs

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate dynrec
export PATH=/usr/local/cuda-12.9/bin:${PATH}
export CUDA_HOME=/usr/local/cuda-12.9
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Refuse to start on a card another process is occupying (slurm does not see them
# all on this node) -- otherwise every config OOMs and the job burns hours to
# produce nothing.
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -1 | tr -d ' ')
PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=${PICK}
echo "GPU: local index ${PICK}, ${FREE} MiB free"
[ "${FREE}" -lt "${MIN_FREE}" ] && { echo "ERROR: only ${FREE} MiB free (need ${MIN_FREE}); requeue when a card frees."; nvidia-smi; exit 1; }

# The anchor checkpoint has to be relayed from the old cluster; without it the two
# fine-tuned configs are skipped rather than failing the whole battery.
CKPT=$(ls -t ${REPO}/ckpts/checkpoint_best_ep5_*.pt ${REPO}/output_*anc1p0*/checkpoint_best_ep5_*.pt 2>/dev/null | head -1)
echo "anchor checkpoint: [${CKPT:-NOT PRESENT -- ft runs skipped}]"

BASE="--eval_loo --frame_stride 8 --max_image_batches 0 --dyn_mask_dir ${MASK_DIR}"
FLOW="--track_dynamic --dyn_motion_knn 8 --dyn_motion_strict --dyn_motion_pred_bandwidth 1.5"
RIGID="--track_dynamic --dyn_motion_groups 4"

RUNS=(
"rgbd_bonn_removing_obstructing_box|frozen||_loo_s8_pcm_nf6"
"rgbd_bonn_removing_obstructing_box|frozen|${FLOW}|_loo_s8_flow8sb1p5_pcm_nf6"
"rgbd_bonn_removing_obstructing_box|frozen|${RIGID}|_loo_s8_trk4_pcm_nf6"
"rgbd_bonn_removing_obstructing_box|ft||_loo_s8_pcm_nf6"
"rgbd_bonn_removing_obstructing_box|ft|${FLOW}|_loo_s8_flow8sb1p5_pcm_nf6"
"rgbd_bonn_placing_obstructing_box|frozen||_loo_s8_pcm_nf6"
"rgbd_bonn_placing_obstructing_box|frozen|${FLOW}|_loo_s8_flow8sb1p5_pcm_nf6"
"rgbd_bonn_synchronous2|frozen||_loo_s8_pcm_nf6"
"rgbd_bonn_synchronous2|frozen|${FLOW}|_loo_s8_flow8sb1p5_pcm_nf6"
)

# A missing mask does NOT fail the eval -- it silently falls back to live detection
# and reports a number computed under the wrong protocol. Check all of them up front.
for SEQ in rgbd_bonn_removing_obstructing_box rgbd_bonn_placing_obstructing_box rgbd_bonn_synchronous2; do
  N=$(ls "${MASK_DIR}/${SEQ}/masks"/*.png 2>/dev/null | wc -l)
  echo "masks ${SEQ}: ${N}"
  [ "${N}" -eq 0 ] && { echo "ERROR: no masks for ${SEQ} under ${MASK_DIR}"; exit 1; }
  [ -d "${DATA_ROOT}/${SEQ}/rgb" ] || { echo "ERROR: no rgb/ for ${SEQ}"; exit 1; }
done

echo "=============================================="; date
i=0
for R in "${RUNS[@]}"; do
  i=$((i+1))
  SEQ=$(echo "$R" | cut -d'|' -f1); MODE=$(echo "$R" | cut -d'|' -f2)
  XFLAGS=$(echo "$R" | cut -d'|' -f3); FTAG=$(echo "$R" | cut -d'|' -f4)
  if [ "${MODE}" = "ft" ] && [ -z "${CKPT}" ]; then
    echo "[${i}/${#RUNS[@]}] SKIP ${MODE} ${SEQ} (no anchor checkpoint)"; continue
  fi
  CKPT_FLAG=""; [ "${MODE}" = "ft" ] && CKPT_FLAG="--checkpoint ${CKPT}"
  SEQ_TAG=$(echo ${SEQ} | sed 's/rgbd_bonn_//')
  OUT="output_eval_${MODE}${FTAG}_${SEQ_TAG}_${DATE_TAG}"
  echo "===== [${i}/${#RUNS[@]}] ${MODE} ${SEQ_TAG} ${XFLAGS:-no-motion} -> ${OUT}  $(date +%H:%M)"
  # one bad config must not cost the other eight
  python eval_gaussian_head.py --data_dir "${DATA_ROOT}" --dataset_name "${SEQ}" \
    --intrinsics bonn --num_frames ${NF} --split all \
    --vggt4d_weights_path "${VGGT4D_CKPT}" --output_dir "${OUT}" \
    ${CKPT_FLAG} ${BASE} ${XFLAGS} || echo "FAILED: ${OUT}"
done

echo "=============================================="
echo "Battery done at $(date):"
ls -d output_eval_*_${DATE_TAG} 2>/dev/null
