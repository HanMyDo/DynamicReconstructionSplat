#!/bin/bash
#SBATCH --job-name=figures
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/figures_%j.out
#SBATCH --error=slurm_logs/figures_%j.err
# =============================================================================
# Renders + 4D PLYs for the thesis figure, on hex-4gpu.
#
# The metrics battery runs with --max_image_batches 0 (no images) and leaves only
# a PLY of the LAST window, which is rarely where the moving object is. This runs
# the two configs that make the visible comparison -- VANILLA frozen vs the FULL
# SYSTEM (fine-tuned head + scene flow) -- over a chosen window, writing images
# and per-timestamp PLYs.
#
# The +2.78 dB full-system gap is what reads visually; flow-vs-no-flow (+0.33 dB
# on ~3% of pixels) does not, which is why this compares system to system.
#
# USAGE: sbatch slurm_figures_hex.sh [BATCH] [NWIN] [SEQ]
#   sbatch slurm_figures_hex.sh 500 40 rgbd_bonn_removing_obstructing_box
# NWIN>1 with --image_views 0 gives one frame per window = a video of that span.
# =============================================================================
set -uo pipefail
BATCH=${1:-500}; NWIN=${2:-40}; SEQ=${3:-rgbd_bonn_removing_obstructing_box}
DATE_TAG=${4:-fig}

REPO="${HOME}/DynamicReconstructionSplat"; cd ${REPO}; mkdir -p slurm_logs
DATA_ROOT="${HOME}/data/bonn/rgbd_bonn_dataset"
MASK_DIR="${HOME}/data/mask_out/output_dyn_masks_precomputed_cs64_r518_st3_fs1_m6"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
CKPT=$(ls -t ${REPO}/ckpts/checkpoint_best_ep5_*.pt 2>/dev/null | head -1)

source /opt/miniforge3/etc/profile.d/conda.sh; conda activate dynrec
export PATH=/usr/local/cuda-12.9/bin:${PATH} CUDA_HOME=/usr/local/cuda-12.9
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -1 | tr -d ' ')
PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=${PICK}
echo "GPU ${PICK}: ${FREE} MiB free | ckpt [${CKPT:-NONE}] | batch ${BATCH} x ${NWIN} windows"
[ "${FREE}" -lt 12000 ] && { echo "ERROR: only ${FREE} MiB free"; exit 1; }

# --images_only skips every window outside the range BEFORE the forward pass, so
# this is minutes rather than a full pass. Its metrics cover only that range and
# are NOT comparable to the battery's -- they exist to make pictures.
FIG="--images_only --image_batch_start ${BATCH} --max_image_batches ${NWIN} \
--image_views 0 --ply_batch ${BATCH} --ply_per_frame --ply_dyn_source 0"
BASE="--eval_loo --frame_stride 8 --dyn_mask_dir ${MASK_DIR}"
FLOW="--track_dynamic --dyn_motion_knn 8 --dyn_motion_strict --dyn_motion_pred_bandwidth 1.5"
TAG=$(echo ${SEQ} | sed 's/rgbd_bonn_//')

echo "===== vanilla baseline (frozen, vanilla VGGT, no motion)"
python eval_gaussian_head.py --data_dir "${DATA_ROOT}" --dataset_name "${SEQ}" \
  --intrinsics bonn --num_frames 6 --split all --vggt4d_weights_path "${VGGT4D_CKPT}" \
  --output_dir "output_fig_vanilla_${TAG}_${DATE_TAG}" --no_vggt4d ${BASE} ${FIG} \
  || echo "FAILED: vanilla"

if [ -n "${CKPT}" ]; then
  echo "===== full system (fine-tuned head + scene flow)"
  python eval_gaussian_head.py --data_dir "${DATA_ROOT}" --dataset_name "${SEQ}" \
    --intrinsics bonn --num_frames 6 --split all --vggt4d_weights_path "${VGGT4D_CKPT}" \
    --checkpoint "${CKPT}" --output_dir "output_fig_ours_${TAG}_${DATE_TAG}" \
    ${BASE} ${FLOW} ${FIG} || echo "FAILED: ours"
fi
echo "done $(date)"; ls -d output_fig_*_${DATE_TAG}
