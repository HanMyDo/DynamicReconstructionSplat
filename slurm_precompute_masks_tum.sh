#!/bin/sh
#SBATCH --job-name=precomp_tum
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/precomp_tum_%j.out
#SBATCH --error=slurm_logs/precomp_tum_%j.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=03:00:00

# =============================================================================
# TUM VARIANT of slurm_precompute_masks_20260731.sh — the Bonn script is UNTOUCHED.
#
# Two differences from the Bonn version, both about where the data lives:
#   1. Source zip: the public read-only /mnt/datasets/tum-rgbd/<seq>.zip (already on
#      the cluster -- do NOT wget it; the head node has a file-size limit that kills
#      the download at ~192MB, and the docs say downloads belong on the data partition).
#   2. Layout: TUM zips contain <seq>/rgb/... directly, with no parent dataset folder,
#      whereas Bonn nests everything under rgbd_bonn_dataset/<seq>/. So --data_dir is
#      the extraction root itself.
#
# Everything else -- the detector, the 3-stage pipeline, the validated 518 + full-span
# recipe -- is identical, and the masks land in the SAME output dir the eval already
# points at, so no eval flag changes are needed.
#
# WHY TUM fr3/walking_xyz: two people walking in the foreground against a static desk
# scene, i.e. the clear foreground/background separation Bonn's obstructing-box
# sequences lack. It is also a CROSS-DATASET check -- the model was trained only on
# Bonn -- which is worth more than a nicer picture.
#
# USAGE:  sbatch slurm_precompute_masks_tum.sh [SEQUENCE] [CHUNK] [RES] [STAGES] [STRIDE]
#   defaults: rgbd_dataset_freiburg3_walking_xyz 16 518 3 0   <- the VALIDATED recipe
#             (16/518/3/0 reproduced the original VGGT4D mask quality on Bonn)
#
# AFTER: LOOK at the overlays before trusting them. TUM is a different room and camera,
#        so the detector may behave differently:
#   scp -r hanmydo@head:.../output_dyn_masks_precomputed_cs16_r518_st3_fs0/<SEQ>/overlays ~
# =============================================================================

SEQUENCE=${1:-rgbd_dataset_freiburg3_walking_xyz}
CHUNK_SIZE=${2:-16}
DET_RES=${3:-518}
STAGES=${4:-3}
STRIDE=${5:-0}

TUM_ZIP="/mnt/datasets/tum-rgbd/${SEQUENCE}.zip"
REPO="/mnt/home/hanmydo/DynamicReconstructionSplat"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
OUT_DIR="output_dyn_masks_precomputed_cs${CHUNK_SIZE}_r${DET_RES}_st${STAGES}_fs${STRIDE}"

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH slurm_logs

echo "=============================================="
echo "TUM precompute — ${SEQUENCE}"
echo "  zip     : ${TUM_ZIP}"
echo "  out dir : ${OUT_DIR}   (same dir the eval already reads)"
echo "  recipe  : chunk ${CHUNK_SIZE}, res ${DET_RES}, stages ${STAGES}, stride ${STRIDE}"
echo "  node: $(hostname)   time: $(date)"
echo "=============================================="

if [ ! -f "${TUM_ZIP}" ]; then
  echo "ERROR: not found: ${TUM_ZIP}"
  echo "available:"; ls /mnt/datasets/tum-rgbd/ | head -30
  exit 1
fi

TUM_DATA=/tmp/tum_data_precomp_${SLURM_JOB_ID}
mkdir -p ${TUM_DATA}
echo "Extracting ${SEQUENCE} ..."
python3 -c "
import zipfile
with zipfile.ZipFile('${TUM_ZIP}', 'r') as zf:
    members = [m for m in zf.namelist() if m.startswith('${SEQUENCE}/')]
    print(f'Extracting {len(members)} files...')
    zf.extractall('${TUM_DATA}/', members)
print('Extraction done.')
"
# TUM has no parent dataset folder -> the extraction root IS the data_dir
ls ${TUM_DATA}/${SEQUENCE} | head
echo ""

if [ ! -f "$VGGT4D_CKPT" ]; then
  echo "Downloading VGGT4D weights..."
  mkdir -p "$(dirname "$VGGT4D_CKPT")"
  wget -c "https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt" -O "$VGGT4D_CKPT" || { echo "download failed"; exit 1; }
fi

CONTAINER=precomp_tum_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash -c "
  cd ${REPO}
  export CUDA_VISIBLE_DEVICES=0
  nvidia-smi
  pip install open3d --quiet

  python precompute_dyn_masks.py \
    --data_dir ${TUM_DATA} \
    --dataset_name ${SEQUENCE} \
    --output_dir ${OUT_DIR} \
    --vggt4d_weights_path ${VGGT4D_CKPT} \
    --chunk_size ${CHUNK_SIZE} \
    --det_resolution ${DET_RES} \
    --stages ${STAGES} \
    --frame_stride ${STRIDE} \
    --save_overlays
"
STATUS=$?

enroot remove -f ${CONTAINER}
rm -rf ${TUM_DATA}

echo ""
echo "=============================================="
echo "exit=${STATUS}   masks -> ${OUT_DIR}/${SEQUENCE}/"
echo "NEXT: look at ${OUT_DIR}/${SEQUENCE}/overlays/ — are the WALKING PEOPLE fully"
echo "covered, and are static objects (desk, monitors) NOT falsely marked?"
echo "Job finished at: $(date)"
echo "=============================================="
exit ${STATUS}
