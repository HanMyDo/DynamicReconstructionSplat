#!/bin/sh
#SBATCH --job-name=eval_comp
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/eval_comp_20260711_%j.out
#SBATCH --error=slurm_logs/eval_comp_20260711_%j.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:59:00
#SBATCH --nodelist=bonn,heidelberg,muenchen,stuttgart,koblenz
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

# =============================================================================
# Held-out crowd eval — per-frame dynamic compositing — 2026-07-11
# -----------------------------------------------------------------------------
# THE POINT: per-frame compositing is a RENDERING change, not a training change.
# So it should improve dynamic PSNR with the FROZEN head and ZERO GPU-hours of
# training. That makes this the cheapest possible test of the whole hypothesis.
#
# STORED REFERENCE (frozen VGGT4D, no compositing, same protocol):
#   overall 20.96 | dynamic 20.58 | static ~21.17
#   (and the fine-tuned 0.90 ep1 head scored dynamic 18.72 — WORSE than frozen)
# If `frozen + --per_frame_dynamic` pushes DYNAMIC above 20.58, the ghosting
# hypothesis is confirmed before we train anything.
#
# Protocol is byte-identical to the stored baseline (--split all, --num_frames 12,
# --image_batch_start 400, max_image_batches default 50) so the delta is real.
# NOTE: eval is NOT resumable (metrics.json is written only at the end) -> full
# 23:59 wall + non-preemptible students_normal.
#
# USAGE:  sbatch slurm_eval_compositing_20260711.sh [CKPT|baseline] ["FLAGS"]
#
#   # 1. THE FIRST RUN — frozen head + compositing (no training!):
#   sbatch slurm_eval_compositing_20260711.sh baseline "--per_frame_dynamic"
#
#   # 2. The honest control — leave-one-out (view j rebuilt from the OTHER frames).
#   #    A big drop here is EXPECTED and reportable: this architecture cannot model
#   #    motion, so dynamics can't be recovered from neighbouring frames.
#   sbatch slurm_eval_compositing_20260711.sh baseline "--per_frame_dynamic --eval_loo"
#
#   # 3. Fine-tuned head + compositing (after a --per_frame_dynamic training run):
#   sbatch slurm_eval_compositing_20260711.sh /path/to/checkpoint_best.pt "--per_frame_dynamic"
#
#   # 4. Sanity: reproduce the stored baseline exactly (expect ~20.96 / 20.58):
#   sbatch slurm_eval_compositing_20260711.sh baseline ""
# =============================================================================

CKPT_ARG=${1:-baseline}
EXTRA_FLAGS=${2:-}

if [ "${CKPT_ARG}" = "baseline" ]; then
  CKPT_FLAG=""
  MODE_TAG="frozen"
else
  if [ ! -f "${CKPT_ARG}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT_ARG}"
    exit 1
  fi
  CKPT_FLAG="--checkpoint ${CKPT_ARG}"
  MODE_TAG="ft"
fi

FLAG_TAG=""
# backbone tag: vanilla VGGT (--no_vggt4d) vs VGGT4D (default), so the anysplat+vggt
# baseline doesn't overwrite the VGGT4D-frozen output dir
case "${EXTRA_FLAGS}" in *no_vggt4d*) FLAG_TAG="${FLAG_TAG}_vggt" ;; esac
case "${EXTRA_FLAGS}" in *per_frame_dynamic*) FLAG_TAG="${FLAG_TAG}_pfd" ;; esac
case "${EXTRA_FLAGS}" in *eval_loo*)          FLAG_TAG="${FLAG_TAG}_loo" ;; esac
# stride goes in the tag so parallel runs at different strides don't share an output dir
case "${EXTRA_FLAGS}" in *frame_stride*)
  STRIDE_VAL=$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--frame_stride[= ]*\([0-9][0-9]*\).*/\1/p')
  FLAG_TAG="${FLAG_TAG}_s${STRIDE_VAL}" ;;
esac
# precomputed-mask runs get their own tag so they don't share an output dir with the
# live-detection run (the A/B: same rendering, dyn/static split from good vs live masks)
case "${EXTRA_FLAGS}" in *dyn_mask_dir*) FLAG_TAG="${FLAG_TAG}_pcm" ;; esac
[ -z "${FLAG_TAG}" ] && FLAG_TAG="_plain"

# $3 = eval sequence (default held-out crowd). Use an easier single-moving-object
# sequence (e.g. rgbd_bonn_person_tracking, rgbd_bonn_balloon) to test whether the
# dynamic MASK works at all in our integration, independent of crowd's difficulty.
EVAL_SEQ=${3:-rgbd_bonn_crowd}
SEQ_TAG=$(echo ${EVAL_SEQ} | sed 's/rgbd_bonn_//')
REPO="/mnt/home/hanmydo/DynamicReconstructionSplat"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
OUT_DIR="output_eval_${MODE_TAG}${FLAG_TAG}_${SEQ_TAG}_20260711"

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH
mkdir -p slurm_logs

echo "=============================================="
echo "Held-out eval on ${EVAL_SEQ}"
echo "  mode       : ${MODE_TAG}   (${CKPT_ARG})"
echo "  extra flags: ${EXTRA_FLAGS}"
echo "  output dir : ${OUT_DIR}"
echo "  node: $(hostname)   time: $(date)"
echo "=============================================="
echo ""

BONN_DATA=/tmp/bonn_data_evalcomp_${SLURM_JOB_ID}
mkdir -p ${BONN_DATA}
echo "Extracting ${EVAL_SEQ} ..."
python3 -c "
import zipfile
prefix = 'rgbd_bonn_dataset/${EVAL_SEQ}/'
with zipfile.ZipFile('/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip', 'r') as zf:
    members = [m for m in zf.namelist() if m.startswith(prefix)]
    print(f'Extracting {len(members)} files...')
    zf.extractall('${BONN_DATA}/', members)
print('Extraction done.')
"
echo ""

CONTAINER=eval_comp_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash -c "
  cd ${REPO}
  export CUDA_VISIBLE_DEVICES=0
  python --version
  nvidia-smi
  echo ''
  pip install open3d --quiet
  echo ''

  python eval_gaussian_head.py \
    --data_dir ${BONN_DATA}/rgbd_bonn_dataset \
    --dataset_name ${EVAL_SEQ} \
    --intrinsics bonn \
    --num_frames 12 \
    --split all \
    --image_batch_start 400 \
    --vggt4d_weights_path ${VGGT4D_CKPT} \
    --output_dir ${OUT_DIR} \
    ${CKPT_FLAG} ${EXTRA_FLAGS}
"

enroot remove -f ${CONTAINER}
rm -rf ${BONN_DATA}

echo ""
echo "=============================================="
echo "Result -> ${OUT_DIR}/metrics.json"
echo "Compare against the stored FROZEN baseline (same protocol, no compositing):"
echo "  overall 20.96 | dynamic 20.58 | static ~21.17"
echo "The number that matters is psnr_dynamic."
echo "=============================================="
echo "Job finished at: $(date)"
