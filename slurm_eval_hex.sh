#!/bin/bash
#SBATCH --job-name=eval_hex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/eval_hex_%j.out
#SBATCH --error=slurm_logs/eval_hex_%j.err
# =============================================================================
# ONE evaluation config on hex-4gpu (conda, no enroot). The battery script runs a
# fixed nine-config list; this is for a single run you choose, e.g. a new tracker.
#
# USAGE: sbatch slurm_eval_hex.sh [CKPT|baseline] ["FLAGS"] [SEQ] [NUM_FRAMES]
#   sbatch slurm_eval_hex.sh baseline \
#     "--eval_loo --frame_stride 8 --max_image_batches 0 --dyn_mask_dir $M \
#      --track_dynamic --dyn_motion_knn 8 --dyn_motion_tracker raft" \
#     rgbd_bonn_removing_obstructing_box 6
#
# The output dir is built from the flags, matching the old cluster's convention, so
# results from both machines sit in one comparison table. EVAL_DATE=... overrides
# the date suffix (use it to group a batch of related runs).
# =============================================================================
set -uo pipefail

CKPT_ARG=${1:-baseline}
EXTRA_FLAGS=${2:-}
EVAL_SEQ=${3:-rgbd_bonn_removing_obstructing_box}
NUM_FRAMES=${4:-6}

# A stray < or > would become a shell redirect and kill python before it prints
# anything -- the job then "completes" in seconds with no metrics.json.
case "${EXTRA_FLAGS}" in *"<"*|*">"*)
  echo "ERROR: EXTRA_FLAGS contains < or > (shell redirect): ${EXTRA_FLAGS}"; exit 1 ;;
esac
case "${EXTRA_FLAGS}" in *dyn_motion_knn*) case "${EXTRA_FLAGS}" in *track_dynamic*) ;; *)
  echo "ERROR: --dyn_motion_knn without --track_dynamic (it would be ignored)."; exit 1 ;; esac ;;
esac

REPO="${HOME}/DynamicReconstructionSplat"; cd ${REPO}; mkdir -p slurm_logs
DATA_ROOT="${HOME}/data/bonn/rgbd_bonn_dataset"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"

if [ "${CKPT_ARG}" = "baseline" ]; then CKPT_FLAG=""; MODE_TAG="frozen"
else
  [ -f "${CKPT_ARG}" ] || { echo "ERROR: checkpoint not found: ${CKPT_ARG}"; exit 1; }
  CKPT_FLAG="--checkpoint ${CKPT_ARG}"; MODE_TAG="ft"
fi

# Same tag convention as the old cluster, so both machines' outputs compare directly.
T=""
case "${EXTRA_FLAGS}" in *no_vggt4d*) T="${T}_vggt" ;; esac
case "${EXTRA_FLAGS}" in *eval_loo*)  T="${T}_loo" ;; esac
case "${EXTRA_FLAGS}" in *frame_stride*)
  T="${T}_s$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--frame_stride[= ]*\([0-9][0-9]*\).*/\1/p')" ;;
esac
case "${EXTRA_FLAGS}" in *track_dynamic*)
  KNN=$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--dyn_motion_knn[= ]*\([0-9][0-9]*\).*/\1/p')
  if [ -n "${KNN}" ] && [ "${KNN}" != "0" ]; then
    T="${T}_flow${KNN}"
    case "${EXTRA_FLAGS}" in *"dyn_motion_tracker raft"*|*"dyn_motion_tracker=raft"*) T="${T}raft" ;; esac
    case "${EXTRA_FLAGS}" in *dyn_motion_strict*)
      T="${T}s"
      BW=$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--dyn_motion_pred_bandwidth[= ]*\([0-9.]*\).*/\1/p')
      [ -n "${BW}" ] && [ "${BW}" != "0" ] && [ "${BW}" != "0.0" ] && T="${T}b$(echo ${BW} | tr '.' 'p')" ;;
    esac
    case "${EXTRA_FLAGS}" in *dyn_motion_chain*) T="${T}c" ;; esac
    case "${EXTRA_FLAGS}" in *dyn_motion_clean_tokens*) T="${T}ct" ;; esac
  else
    T="${T}_trk$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--dyn_motion_groups[= ]*\([0-9][0-9]*\).*/\1/p')"
  fi ;;
esac
case "${EXTRA_FLAGS}" in *gain_correct*) T="${T}_gc" ;; esac
case "${EXTRA_FLAGS}" in *dyn_mask_dir*) T="${T}_pcm" ;; esac
T="${T}_nf${NUM_FRAMES}"
SEQ_TAG=$(echo ${EVAL_SEQ} | sed 's/rgbd_bonn_//')
OUT_DIR="output_eval_${MODE_TAG}${T}_${SEQ_TAG}_${EVAL_DATE:-$(date +%Y%m%d)}"

source /opt/miniforge3/etc/profile.d/conda.sh; conda activate dynrec
export PATH=/usr/local/cuda-12.9/bin:${PATH} CUDA_HOME=/usr/local/cuda-12.9
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Refuse a card another process already occupies (slurm does not see them all here).
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -1 | tr -d ' ')
PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=${PICK}
echo "GPU ${PICK}: ${FREE} MiB free"
[ "${FREE}" -lt 12000 ] && { echo "ERROR: only ${FREE} MiB free; requeue later."; nvidia-smi; exit 1; }

# A missing mask does not fail eval -- it silently falls back to live detection and
# reports a number computed under the wrong protocol. Check before spending hours.
MASK_DIR=$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--dyn_mask_dir[= ]*\([^ ]*\).*/\1/p')
if [ -n "${MASK_DIR}" ]; then
  N=$(ls "${MASK_DIR}/${EVAL_SEQ}/masks"/*.png 2>/dev/null | wc -l)
  [ "${N}" -eq 0 ] && { echo "ERROR: no masks for ${EVAL_SEQ} under ${MASK_DIR}"; exit 1; }
  echo "masks: ${N}"
fi
[ -d "${DATA_ROOT}/${EVAL_SEQ}/rgb" ] || { echo "ERROR: no rgb/ for ${EVAL_SEQ}"; exit 1; }

echo "=============================================="
echo "${MODE_TAG} | ${EVAL_SEQ} | nf${NUM_FRAMES} | ${EXTRA_FLAGS}"
echo "-> ${OUT_DIR}   $(date)"
echo "=============================================="

python eval_gaussian_head.py --data_dir "${DATA_ROOT}" --dataset_name "${EVAL_SEQ}" \
  --intrinsics bonn --num_frames ${NUM_FRAMES} --split all \
  --vggt4d_weights_path "${VGGT4D_CKPT}" --output_dir "${OUT_DIR}" \
  ${CKPT_FLAG} ${EXTRA_FLAGS}

echo "Result -> ${OUT_DIR}/metrics.json   $(date)"
