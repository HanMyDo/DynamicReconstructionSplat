#!/bin/sh
#SBATCH --job-name=precomp_masks
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/precomp_masks_20260731_%j.out
#SBATCH --error=slurm_logs/precomp_masks_20260731_%j.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=bonn,heidelberg,muenchen,stuttgart,koblenz
#SBATCH --time=03:00:00
# NOTE: this cluster forbids --mem (host RAM is fixed per GPU). cs96 host-OOM'd at
# 518. To fit more temporal context within fixed RAM, lower --det_resolution ($3)
# instead of raising memory: fewer tokens -> less host+GPU memory, mask upsamples
# on load anyway.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

# =============================================================================
# Precompute VGGT4D dynamic masks over LONG temporal windows — 2026-07-31
# -----------------------------------------------------------------------------
# Detection-only (NO Gaussians, NO rendering), 518 aspect-preserved preprocessing
# (matches original VGGT4D), CHUNKED so it fits 24g regardless of sequence length.
# FULL original VGGT4D pipeline by default ($4=3): Stage 1 (attention coarse mask)
# -> Stage 2 (re-run backbone with mask -> refined poses) -> Stage 3 (geometric
# refine, open3d), feeding Stage 3 the ORIGINAL combo of Stage-1 depth/intrinsic +
# Stage-2 poses. $4=1 stops at the coarse mask. Fast + non-resumable -> short wall.
#
# USAGE:  sbatch slurm_precompute_masks_20260731.sh <SEQUENCE> [CHUNK_SIZE] [DET_RES] [STAGES] [STRIDE]
#   $1 SEQUENCE   e.g. rgbd_bonn_moving_nonobstructing_box (a clean independent-
#                 motion, NON-training seq — avoid person_tracking / crowd)
#   $2 CHUNK_SIZE frames per detection pass (default 32; bigger = more memory. OOM -> lower)
#   $3 DET_RES    detection long-edge (default 518; lower to fit more frames)
#   $4 STAGES     3 = full original pipeline (default); 1 = coarse mask only
#   $5 STRIDE     1 = consecutive (default). >1 = each pass takes every STRIDE-th frame,
#                 spanning chunk_size*STRIDE frames -> gives the detector real object
#                 motion at fixed memory (the fix for weak masks on 24g). Try 8, then 4/16.
#
# AFTER: pull output_dyn_masks_precomputed_*/<SEQ>/overlays/ locally and LOOK —
#        is the moving object covered AND are static false-positives (e.g. the
#        chair) removed by Stage 3?
# =============================================================================

SEQUENCE=${1:?"give a sequence, e.g. rgbd_bonn_moving_nonobstructing_box"}
CHUNK_SIZE=${2:-32}
DET_RES=${3:-518}   # detection long-edge; lower (e.g. 378) to fit more frames within fixed RAM
STAGES=${4:-3}      # 3 = full original VGGT4D (Stage 1->2->3); 1 = coarse only. 3 needs smaller chunks (more memory).
STRIDE=${5:-1}      # 0 = AUTO full-sequence span (VALIDATED: use with res 518, reproduced original mask quality). 1 = consecutive. >1 = every STRIDE-th frame (spans chunk_size*STRIDE).
MARGIN=${6:-0}     # overlap passes by N frames each side, emit only the interior, so every
                   # frame gets its FULL +-2/4/6 detector window (at chunk 16, margin 0
                   # leaves only 26% of frames with a complete window; margin 6 -> 100%).
                   # Costs runtime, not memory: ~chunk/(chunk-2*margin)x = 4x at 16/6.

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH
mkdir -p slurm_logs

REPO="/mnt/home/hanmydo/DynamicReconstructionSplat"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
OUT_DIR="output_dyn_masks_precomputed_cs${CHUNK_SIZE}_r${DET_RES}_st${STAGES}_fs${STRIDE}"
[ "${MARGIN}" != "0" ] && OUT_DIR="${OUT_DIR}_m${MARGIN}"   # margin changes the masks -> own dir   # chunk+res+stages+stride in name so runs don't overwrite; matches gitignore output_*/

echo "=============================================="
echo "Precompute dynamic masks — ${SEQUENCE} (chunk_size ${CHUNK_SIZE})"
echo "node: $(hostname)   time: $(date)"
echo "=============================================="
echo ""

BONN_DATA=/tmp/bonn_data_precomp_${SLURM_JOB_ID}
mkdir -p ${BONN_DATA}
echo "Extracting ${SEQUENCE} ..."
python3 -c "
import zipfile
prefix = 'rgbd_bonn_dataset/${SEQUENCE}/'
with zipfile.ZipFile('/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip', 'r') as zf:
    members = [m for m in zf.namelist() if m.startswith(prefix)]
    print(f'Extracting {len(members)} files...')
    zf.extractall('${BONN_DATA}/', members)
print('Extraction done.')
"
echo ""

if [ ! -f "$VGGT4D_CKPT" ]; then
  echo "Downloading VGGT4D weights..."
  mkdir -p "$(dirname "$VGGT4D_CKPT")"
  wget -c "https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt" -O "$VGGT4D_CKPT" || { echo "download failed"; exit 1; }
fi

CONTAINER=precomp_masks_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash -c "
  cd ${REPO}
  export CUDA_VISIBLE_DEVICES=0
  python --version
  nvidia-smi
  echo ''
  pip install open3d --quiet   # not used by Stage 1, but avoids import warnings

  python precompute_dyn_masks.py \
    --data_dir ${BONN_DATA}/rgbd_bonn_dataset \
    --dataset_name ${SEQUENCE} \
    --output_dir ${OUT_DIR} \
    --vggt4d_weights_path ${VGGT4D_CKPT} \
    --chunk_size ${CHUNK_SIZE} \
    --det_resolution ${DET_RES} \
    --stages ${STAGES} \
    --frame_stride ${STRIDE} \
    --pass_margin ${MARGIN} \
    --save_overlays
"

enroot remove -f ${CONTAINER}
rm -rf ${BONN_DATA}

echo ""
echo "=============================================="
echo "Masks -> ${OUT_DIR}/${SEQUENCE}/{masks,overlays}/"
echo "LOOK at the overlays: does the moving object light up?"
echo "=============================================="
echo "Job finished at: $(date)"
