#!/bin/bash
#SBATCH --job-name=train_hex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/train_hex_%j.out
#SBATCH --error=slurm_logs/train_hex_%j.err
# =============================================================================
# Fine-tuning on hex-4gpu (conda, no enroot).
#
# WHY RETRAIN. Two reasons, neither of them "more compute":
#  1. The anchor checkpoint was trained on the BROKEN masks (--frame_stride 0, the
#     60x temporal-window bug: 11.9% of pixels labelled dynamic where only ~3%
#     move). Those masks drove the loss downweight and the compositing gate for
#     the whole run.
#  2. It was trained at nf6 / stride 1 / LOO / no compositing, but the deliverable
#     renders at a wider window without LOO and WITH compositing. Without LOO all
#     frames' static Gaussians composite into every view, while the head learned
#     opacities for 5 contributors -- that mismatch over-accumulates.
#
# Everything else is the ANCHOR RECIPE, which is the one configuration in this
# project's history that trains stably for 5 epochs and renders a correct PLY:
# scale_reg 0, sh_reg 0 (the trust-region anchor replaces both), lpips 0.05,
# depth_head unfrozen, depth_consis 1.0, anchor 1.0, dyn downweight 0, lr 2e-5.
# Do not change those without a reason -- earlier recipes diverged (f_dc runaway)
# or collapsed Gaussian scales 26x.
#
# USAGE: sbatch slurm_train_hex.sh [NUM_FRAMES] [FRAME_STRIDE] [LOO_PROB] [PFD] [EPOCHS]
#   sbatch slurm_train_hex.sh 16 4 0.5 1 5
#     LOO_PROB 0.5 = half the steps leave-one-out, half not, so the head learns
#     both regimes: LOO is what the metrics measure, non-LOO is what the demo renders.
#     PFD 1 = per-frame dynamic compositing during training, matching the demo.
# =============================================================================
set -uo pipefail

NUM_FRAMES=${1:-16}
FRAME_STRIDE=${2:-4}
LOO_PROB=${3:-0.5}
PFD=${4:-1}
EPOCHS=${5:-5}

REPO="${HOME}/DynamicReconstructionSplat"; cd ${REPO}; mkdir -p slurm_logs
DATA_ROOT="${HOME}/data/bonn/rgbd_bonn_dataset"
MASKS="${HOME}/data/mask_out/output_dyn_masks_precomputed_cs64_r518_st3_fs1_m6"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
# Family-disjoint from the eval split. With balloon promoted to EVAL, its whole
# family (balloon2, balloon_tracking*) must leave training too -- the split rule is
# per scene family, not per sequence, because variants of one scene share geometry
# and appearance. person_tracking* is excluded on quality grounds: the camera
# follows the person, so they are static in-image while the background sweeps, and
# the attention detector flags the background instead.
# Override with DATASETS=... to change the training set.
DATASETS=${DATASETS:-"rgbd_bonn_crowd,rgbd_bonn_crowd2,rgbd_bonn_crowd3,\
rgbd_bonn_kidnapping_box,rgbd_bonn_kidnapping_box2,\
rgbd_bonn_moving_nonobstructing_box,rgbd_bonn_moving_nonobstructing_box2,\
rgbd_bonn_moving_obstructing_box,rgbd_bonn_moving_obstructing_box2,\
rgbd_bonn_placing_nonobstructing_box,rgbd_bonn_placing_nonobstructing_box2,\
rgbd_bonn_placing_nonobstructing_box3,\
rgbd_bonn_removing_nonobstructing_box,rgbd_bonn_removing_nonobstructing_box2"}
DATASETS=$(echo "${DATASETS}" | tr -d ' \\\n')
OUT="${REPO}/output_train_hex_nf${NUM_FRAMES}_s${FRAME_STRIDE}_loo${LOO_PROB}_pfd${PFD}_$(date +%Y%m%d)"

source /opt/miniforge3/etc/profile.d/conda.sh; conda activate dynrec
export PATH=/usr/local/cuda-12.9/bin:${PATH} CUDA_HOME=/usr/local/cuda-12.9
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# wandb has no API key on this cluster and its init raises rather than degrading,
# which killed the run 6 s in. Offline keeps the full local log (syncable later
# with `wandb sync`) and needs no account. Set WANDB_MODE=online after `wandb login`
# if you want live curves.
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${REPO}/wandb

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -1 | tr -d ' ')
PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=${PICK}
echo "GPU ${PICK}: ${FREE} MiB free"
[ "${FREE}" -lt 30000 ] && { echo "ERROR: only ${FREE} MiB free; training needs headroom."; exit 1; }

# Masks must exist for EVERY training sequence: a missing one does not fail, it
# silently falls back to live detection and trains against the wrong pixels.
for S in $(echo ${DATASETS} | tr ',' ' '); do
  N=$(ls "${MASKS}/${S}/masks"/*.png 2>/dev/null | wc -l)
  echo "masks ${S}: ${N}"
  [ "${N}" -eq 0 ] && { echo "ERROR: no masks for ${S} -- precompute it first"; exit 1; }
  [ -d "${DATA_ROOT}/${S}/rgb" ] || { echo "ERROR: no rgb/ for ${S}"; exit 1; }
done

PFD_FLAG=""; [ "${PFD}" = "1" ] && PFD_FLAG="--per_frame_dynamic"
echo "=============================================="
echo "nf${NUM_FRAMES} stride${FRAME_STRIDE} loo_prob${LOO_PROB} pfd${PFD} epochs${EPOCHS}"
echo "train sequences ($(echo ${DATASETS} | tr ',' '\n' | wc -l)): ${DATASETS}"
echo "-> ${OUT}   $(date)"
echo "=============================================="

python train_temporal_gaussian_head.py \
  --data_dir "${DATA_ROOT}" --dataset_names "${DATASETS}" --output_dir "${OUT}" \
  --num_epochs ${EPOCHS} --val_every_epochs 1 --save_every_n_steps 200 --batch_size 1 \
  --learning_rate 2e-5 --warmup_ratio 0.15 --gradient_clip 0.5 \
  --num_frames ${NUM_FRAMES} --frame_stride ${FRAME_STRIDE} \
  --dyn_mask_dir "${MASKS}" --dynamic_loss_downweight 0.0 \
  --train_loo --train_loo_prob ${LOO_PROB} ${PFD_FLAG} \
  --scale_reg_weight 0.0 --sh_reg_weight 0.0 --temporal_weight 0.0 \
  --lpips_weight 0.05 --unfreeze_depth_head --depth_consis_weight 1.0 --anchor_weight 1.0 \
  --no_gt_poses --intrinsics bonn --vggt4d_weights_path "${VGGT4D_CKPT}"

echo "Done $(date). Checkpoints -> ${OUT}"
