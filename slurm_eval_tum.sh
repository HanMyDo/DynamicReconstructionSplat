#!/bin/sh
#SBATCH --job-name=eval_tum
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/eval_tum_%j.out
#SBATCH --error=slurm_logs/eval_tum_%j.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:59:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

# =============================================================================
# TUM VARIANT of slurm_eval_compositing_20260711.sh — the Bonn script is UNTOUCHED.
#
# Differences from the Bonn version, all about data location and camera:
#   1. Source zip: public read-only /mnt/datasets/tum-rgbd/<seq>.zip
#   2. Layout: TUM zips hold <seq>/rgb/... with no parent dataset folder, so
#      --data_dir is the extraction root (Bonn nests under rgbd_bonn_dataset/).
#   3. --intrinsics tum_fr3 instead of bonn (already a supported preset).
# The model, protocol and metric code are identical.
#
# CROSS-DATASET NOTE: the checkpoint was trained ONLY on Bonn, so numbers here are
# zero-shot transfer. Report them as such -- they are not comparable to the Bonn
# table, and they are interesting precisely because they are out of domain.
#
# USAGE:  sbatch slurm_eval_tum.sh [CKPT|baseline] ["FLAGS"] [SEQUENCE] [NUM_FRAMES]
#   e.g.  sbatch slurm_eval_tum.sh baseline "--no_vggt4d --eval_loo --frame_stride 8 --gain_correct --dyn_mask_dir output_dyn_masks_precomputed_cs16_r518_st3_fs0"
# Set EVAL_DATE=<tag> to control the output dir suffix (default: today).
# =============================================================================

CKPT_ARG=${1:-baseline}
EXTRA_FLAGS=${2:-}
EVAL_SEQ=${3:-rgbd_dataset_freiburg3_walking_xyz}
NUM_FRAMES=${4:-6}

TUM_ZIP="/mnt/datasets/tum-rgbd/${EVAL_SEQ}.zip"
REPO="/mnt/home/hanmydo/DynamicReconstructionSplat"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"

if [ "${CKPT_ARG}" = "baseline" ]; then
  CKPT_FLAG=""; MODE_TAG="frozen"
else
  if [ ! -f "${CKPT_ARG}" ]; then echo "ERROR: checkpoint not found: ${CKPT_ARG}"; exit 1; fi
  CKPT_FLAG="--checkpoint ${CKPT_ARG}"; MODE_TAG="ft"
fi

# same tagging convention as the Bonn script, so nothing can silently overwrite
FLAG_TAG=""
case "${EXTRA_FLAGS}" in *no_vggt4d*)   FLAG_TAG="${FLAG_TAG}_vggt" ;; esac
case "${EXTRA_FLAGS}" in *eval_loo*)    FLAG_TAG="${FLAG_TAG}_loo" ;; esac
case "${EXTRA_FLAGS}" in *frame_stride*)
  STRIDE_VAL=$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--frame_stride[= ]*\([0-9][0-9]*\).*/\1/p')
  FLAG_TAG="${FLAG_TAG}_s${STRIDE_VAL}" ;;
esac
case "${EXTRA_FLAGS}" in *gain_correct*) FLAG_TAG="${FLAG_TAG}_gc" ;; esac
case "${EXTRA_FLAGS}" in *dyn_mask_dir*) FLAG_TAG="${FLAG_TAG}_pcm" ;; esac
[ "${NUM_FRAMES}" != "12" ] && FLAG_TAG="${FLAG_TAG}_nf${NUM_FRAMES}"
[ -z "${FLAG_TAG}" ] && FLAG_TAG="_plain"

SEQ_TAG=$(echo ${EVAL_SEQ} | sed 's/rgbd_dataset_//')
DATE_TAG=${EVAL_DATE:-$(date +%Y%m%d)}
OUT_DIR="output_eval_tum_${MODE_TAG}${FLAG_TAG}_${SEQ_TAG}_${DATE_TAG}"

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH slurm_logs

echo "=============================================="
echo "TUM eval — ${EVAL_SEQ}   (ZERO-SHOT: trained on Bonn only)"
echo "  mode       : ${MODE_TAG}   (${CKPT_ARG})"
echo "  extra flags: ${EXTRA_FLAGS}"
echo "  output dir : ${OUT_DIR}"
echo "  node: $(hostname)   time: $(date)"
echo "=============================================="

if [ ! -f "${TUM_ZIP}" ]; then
  echo "ERROR: not found: ${TUM_ZIP}"; ls /mnt/datasets/tum-rgbd/ | head -30; exit 1
fi

TUM_DATA=/tmp/tum_data_eval_${SLURM_JOB_ID}
mkdir -p ${TUM_DATA}
python3 -c "
import zipfile
with zipfile.ZipFile('${TUM_ZIP}', 'r') as zf:
    members = [m for m in zf.namelist() if m.startswith('${EVAL_SEQ}/')]
    print(f'Extracting {len(members)} files...')
    zf.extractall('${TUM_DATA}/', members)
print('Extraction done.')
"

CONTAINER=eval_tum_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash -c "
  cd ${REPO}
  export CUDA_VISIBLE_DEVICES=0
  nvidia-smi
  pip install open3d --quiet

  python eval_gaussian_head.py \
    --data_dir ${TUM_DATA} \
    --dataset_name ${EVAL_SEQ} \
    --intrinsics tum_fr3 \
    --num_frames ${NUM_FRAMES} \
    --split all \
    --vggt4d_weights_path ${VGGT4D_CKPT} \
    --output_dir ${OUT_DIR} \
    ${CKPT_FLAG} ${EXTRA_FLAGS}
"
STATUS=$?

enroot remove -f ${CONTAINER}
rm -rf ${TUM_DATA}

echo ""
echo "=============================================="
echo "exit=${STATUS}   metrics -> ${OUT_DIR}/metrics.json"
echo "Zero-shot transfer (Bonn-trained). Compare ONLY against the matching TUM"
echo "baseline run -- never against the Bonn table."
echo "Job finished at: $(date)"
echo "=============================================="
exit ${STATUS}
