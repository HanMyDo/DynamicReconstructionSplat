#!/bin/bash
#SBATCH --job-name=precomp_hex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/precomp_hex_%j.out
#SBATCH --error=slurm_logs/precomp_hex_%j.err
# =============================================================================
# Dynamic-mask precompute for the hex-4gpu cluster (conda, no enroot container).
#
# WHY THIS EXISTS SEPARATELY. slurm_precompute_masks_20260731.sh targets the old
# cluster: enroot + squashfs, --partition=24g, --qos=students_normal, fixed node
# list, and memory tied to GPU count. Here it is plain conda, one partition
# (gpu), and MaxMemPerNode=UNLIMITED with 377 GB of host RAM.
#
# WHY THAT MATTERS. Host RAM was the binding constraint on the old cluster: the
# detector copies Q/K to CPU and they scale with FRAMES PER PASS, which capped
# chunk_size at 16 -- and at chunk 16 only 26% of frames get the full
# [-6,-4,-2,2,4,6] comparison window even with --pass_margin. With ~12x the RAM
# the chunk can finally be large enough that a margin is cheap, which is the
# whole reason for moving here.
#
# USAGE: sbatch slurm_precompute_masks_hex.sh SEQUENCE [CHUNK] [RES] [STAGES] [STRIDE] [MARGIN]
#   sbatch slurm_precompute_masks_hex.sh rgbd_bonn_removing_obstructing_box 128 518 3 1 6
# Find the largest CHUNK that fits by trying 64 -> 128 -> 256 (see README notes).
# =============================================================================
set -euo pipefail

SEQUENCE=${1:?"give a sequence, e.g. rgbd_bonn_removing_obstructing_box"}
CHUNK_SIZE=${2:-64}
DET_RES=${3:-518}
STAGES=${4:-3}
STRIDE=${5:-1}      # 1 = consecutive frames = the paper's ~0.2 s window. Do not use 0.
MARGIN=${6:-6}      # overlap passes so every emitted frame keeps its full window
NORM=${7:-per_frame}  # per_frame = original (rescales EVERY frame to [0,1], so a frame with
                      # nothing moving still contributes its brightest patches to a globally
                      # thresholded mask); global = one min/max over the pass, letting quiet
                      # frames be rejected outright.

REPO="${HOME}/DynamicReconstructionSplat"
DATA_ROOT="${HOME}/data/bonn/rgbd_bonn_dataset"
CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
# Outputs go to /data (14 TB) not home (100 GB quota); symlinked back for convenience.
OUT_ROOT="${HOME}/data/mask_out"
OUT_DIR="${OUT_ROOT}/output_dyn_masks_precomputed_cs${CHUNK_SIZE}_r${DET_RES}_st${STAGES}_fs${STRIDE}"
[ "${MARGIN}" != "0" ] && OUT_DIR="${OUT_DIR}_m${MARGIN}"
[ "${NORM}" != "per_frame" ] && OUT_DIR="${OUT_DIR}_${NORM}"   # different masks -> own dir

mkdir -p "${REPO}/slurm_logs" "${OUT_ROOT}"
cd "${REPO}"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate dynrec
export PATH=/usr/local/cuda-12.9/bin:${PATH}
export CUDA_HOME=/usr/local/cuda-12.9
# Q/K copies are large and numpy/BLAS threads multiply the footprint; 64 cores
# would otherwise spawn 64 threads per op for no speedup on this workload.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # large activations fragment badly

# Refuse to start on a card someone else is already occupying. Slurm does not always
# account for non-slurm processes, so an allocated GPU can arrive with most of its
# memory gone -- that is an instant OOM on the first pass, and the traceback blames
# chunk_size for what is really a busy card. Better to exit in seconds with the real
# reason and requeue when the GPU is free.
# If MORE THAN ONE GPU was allocated (sbatch --gres=gpu:2), use the emptiest of them.
# Default is one GPU; this only engages when the submitter deliberately asked for more,
# e.g. to work around a card held by a process slurm cannot see. Indices are job-local,
# so this never selects a device outside our own allocation.
PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
       | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=${PICK}
echo "GPU free memory: ${FREE} MiB (using local index ${PICK})"
if [ "${FREE}" -lt 20000 ]; then
  echo "ERROR: allocated GPU has only ${FREE} MiB free (need >= ~20 GB)."
  echo "       Another process is resident on it. Requeue when it frees up."
  nvidia-smi
  exit 1
fi

echo "=============================================="
echo "Masks: ${SEQUENCE}  chunk ${CHUNK_SIZE}  res ${DET_RES}  stages ${STAGES}  stride ${STRIDE}  margin ${MARGIN}  norm ${NORM}"
echo "node: $(hostname)   gpu: ${CUDA_VISIBLE_DEVICES:-unset}   $(date)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo "=============================================="

[ -d "${DATA_ROOT}/${SEQUENCE}/rgb" ] || { echo "ERROR: no rgb/ under ${DATA_ROOT}/${SEQUENCE}"; exit 1; }
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint missing: ${CKPT}"; exit 1; }

python precompute_dyn_masks.py \
    --data_dir "${DATA_ROOT}" \
    --dataset_name "${SEQUENCE}" \
    --output_dir "${OUT_DIR}" \
    --vggt4d_weights_path "${CKPT}" \
    --chunk_size "${CHUNK_SIZE}" \
    --det_resolution "${DET_RES}" \
    --stages "${STAGES}" \
    --frame_stride "${STRIDE}" \
    --pass_margin "${MARGIN}" \
    --mask_normalize "${NORM}" \
    --save_overlays

echo "=============================================="
echo "Masks -> ${OUT_DIR}/${SEQUENCE}/{masks,overlays}"
echo "LOOK at the overlays: is the moving object covered and the background clean?"
echo "Job finished at: $(date)"
