#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/probe_%j.out
#SBATCH --error=slurm_logs/probe_%j.err
# =============================================================================
# CHEAP LOOK before spending an hour per config. A full-sequence eval is ~1h;
# --images_only skips every window outside the range BEFORE the forward pass, so
# a 30-window probe is minutes. Runs the no-motion control, scene flow, and
# clamped scene flow over the SAME window range in ONE job, so the three are
# directly comparable and cost one GPU allocation instead of three.
#
# The [DynFlow] diagnostic prints on the first window, so `moved %` and the
# displacement `max` are known long before the job finishes.
#
# USAGE: sbatch slurm_probe_hex.sh CKPT MASKDIR [SEQ] [NUM_FRAMES] [STRIDE] [START] [NWIN]
#   sbatch slurm_probe_hex.sh "$CK" "$M2" rgbd_bonn_balloon 6 8 0 30
#
# Metrics here cover ONLY the probed range and are not comparable to a full run.
# =============================================================================
set -uo pipefail
CKPT=${1:?usage: sbatch slurm_probe_hex.sh CKPT MASKDIR [SEQ] [NF] [STRIDE] [START] [NWIN]}
MASKS=${2:?missing MASKDIR}
SEQ=${3:-rgbd_bonn_balloon}; NF=${4:-6}; STRIDE=${5:-8}; START=${6:-0}; NWIN=${7:-30}
TAG=${EVAL_DATE:-probe}

REPO="${HOME}/DynamicReconstructionSplat"; cd ${REPO}; mkdir -p slurm_logs
DATA_ROOT="${HOME}/data/bonn/rgbd_bonn_dataset"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
[ -f "${CKPT}" ] || { echo "ERROR: no checkpoint ${CKPT}"; exit 1; }
N=$(ls "${MASKS}/${SEQ}/masks"/*.png 2>/dev/null | wc -l)
[ "${N}" -eq 0 ] && { echo "ERROR: no masks for ${SEQ} under ${MASKS}"; exit 1; }
echo "masks: ${N}"

source /opt/miniforge3/etc/profile.d/conda.sh; conda activate dynrec
export PATH=/usr/local/cuda-12.9/bin:${PATH} CUDA_HOME=/usr/local/cuda-12.9
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -1 | tr -d ' ')
PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=${PICK}
echo "GPU ${PICK}: ${FREE} MiB free | ${SEQ} nf${NF} s${STRIDE} windows ${START}..$((START+NWIN))"
[ "${FREE}" -lt 12000 ] && { echo "ERROR: only ${FREE} MiB free"; exit 1; }

WIN="--images_only --image_batch_start ${START} --max_image_batches ${NWIN} --image_views 0"
BASE="--frame_stride ${STRIDE} --dyn_mask_dir ${MASKS}"
FLOW="--track_dynamic --dyn_motion_knn 8 --dyn_motion_strict --dyn_motion_pred_bandwidth 1.5 --dyn_motion_tracker raft"
SEQ_TAG=$(echo ${SEQ} | sed 's/rgbd_bonn_//')

run () {   # name, extra flags
  local name=$1; shift
  local out="output_probe_${name}_${SEQ_TAG}_${TAG}"
  echo "===================================================================="
  echo "PROBE ${name} -> ${out}   $(date +%H:%M:%S)"
  echo "===================================================================="
  python eval_gaussian_head.py --data_dir "${DATA_ROOT}" --dataset_name "${SEQ}" \
    --intrinsics bonn --num_frames ${NF} --split all \
    --vggt4d_weights_path "${VGGT4D_CKPT}" --checkpoint "${CKPT}" \
    --output_dir "${out}" ${BASE} ${WIN} "$@" || echo "FAILED: ${name}"
}

run ctl
run flow  ${FLOW}
run clamp ${FLOW} --dyn_motion_max_disp_mult 3.0

echo "done $(date)"
echo "--- displacement diagnostics (first window of each) ---"
grep -m 9 -E "DynFlow" slurm_logs/probe_${SLURM_JOB_ID}.out || true
ls -d output_probe_*_${SEQ_TAG}_${TAG}
