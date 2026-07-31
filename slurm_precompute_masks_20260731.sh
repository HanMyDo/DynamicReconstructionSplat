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
#SBATCH --mem=96G
# ^ HOST RAM. The Stage-1 mask stage moves attention maps + features to CPU for
# KMeans/Otsu, scaling with chunk_size — cs96 OOM-killed (host, not GPU) at the
# default. Bump host RAM; if the GPU itself OOMs (a CUDA error, not a SLURM
# oom_kill), lower chunk_size or add a lower detection resolution.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

# =============================================================================
# Precompute VGGT4D dynamic masks over LONG temporal windows — 2026-07-31
# -----------------------------------------------------------------------------
# Detection-only (Pass 1: backbone attention -> mask; NO Gaussians, NO rendering),
# 518 aspect-preserved preprocessing (matches original VGGT4D), CHUNKED so it fits
# 24g regardless of sequence length. STAGE-1 ONLY (no Stage 2/3 — see the plan and
# precompute_dyn_masks.py docstring). Fast + non-resumable -> short wall, normal QOS.
#
# USAGE:  sbatch slurm_precompute_masks_20260731.sh <SEQUENCE> [CHUNK_SIZE]
#   $1 SEQUENCE   e.g. rgbd_bonn_moving_nonobstructing_box (a clean independent-
#                 motion, NON-training seq — avoid person_tracking / crowd)
#   $2 CHUNK_SIZE frames per detection window (default 32 ~= 1s; bigger = more
#                 motion but more memory. If it OOMs, lower it.)
#
# AFTER: pull output_dyn_masks_precomputed/<SEQ>/overlays/ locally and LOOK —
#        does the moving object light up? That decides Stage-1-only vs adding
#        Stage 2/3.
# =============================================================================

SEQUENCE=${1:?"give a sequence, e.g. rgbd_bonn_moving_nonobstructing_box"}
CHUNK_SIZE=${2:-32}

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH
mkdir -p slurm_logs

REPO="/mnt/home/hanmydo/DynamicReconstructionSplat"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
OUT_DIR="output_dyn_masks_precomputed_cs${CHUNK_SIZE}"   # chunk size in the name so runs don't overwrite; matches gitignore output_*/

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
