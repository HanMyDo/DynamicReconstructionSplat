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
#
#   # 5. TRACK-CORRESPONDENCE SCENE FLOW (branch dyn_handling) vs its own control.
#   #    MASKS is a real path -- a "<MASKS>" placeholder is a shell redirect and the
#   #    job dies in 10s (job 20991). Both runs must be the SAME except the motion:
#   M=output_dyn_masks_precomputed_cs16_r518_st3_fs0
#   B="--eval_loo --frame_stride 8 --dyn_mask_dir $M"
#   sbatch slurm_eval_compositing_20260711.sh baseline "$B" rgbd_bonn_removing_obstructing_box 6
#   sbatch slurm_eval_compositing_20260711.sh baseline "$B --track_dynamic --dyn_motion_knn 8" \
#          rgbd_bonn_removing_obstructing_box 6
#   # -> compare psnr_dynamic; headroom is the +3.26 dB static-dynamic gap at this point.
# =============================================================================

CKPT_ARG=${1:-baseline}
EXTRA_FLAGS=${2:-}

# --- FAIL FAST on flag mistakes that otherwise cost a whole run -----------------
# EXTRA_FLAGS is expanded UNQUOTED into the container's bash -c, so a stray < or >
# (e.g. a copy-pasted "<MASKS>" placeholder) becomes a shell REDIRECT: the python
# command dies before printing anything and the job "completes" in ~10s with no
# metrics.json. Happened on job 20991.
case "${EXTRA_FLAGS}" in *"<"*|*">"*)
  echo "ERROR: EXTRA_FLAGS contains < or >, which the shell would treat as a redirect."
  echo "       Replace the placeholder with a real path: ${EXTRA_FLAGS}"
  exit 1 ;;
esac
# (the --dyn_mask_dir check needs EVAL_SEQ and lives just below it)
MASK_DIR=$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--dyn_mask_dir[= ]*\([^ ]*\).*/\1/p')
# --dyn_motion_knn only takes effect together with --track_dynamic; silently ignoring
# it would produce a 'scene flow does not help' null for a purely trivial reason.
case "${EXTRA_FLAGS}" in
  *dyn_motion_knn*) case "${EXTRA_FLAGS}" in *track_dynamic*) ;; *)
    echo "ERROR: --dyn_motion_knn given without --track_dynamic (the flag would be ignored)."
    exit 1 ;; esac ;;
esac
# -------------------------------------------------------------------------------
# $4 NUM_FRAMES: views per window. 12 = the protocol EVERY stored baseline used -> keep 12
# for anything you want to compare against them. Use a different value ONLY to match a
# checkpoint's TRAINING window (e.g. 6): under leave-one-out each pixel is rebuilt from
# V-1 sources, so a head trained with 5 contributors but evaluated with 11 over-
# accumulates opacity/brightness. Non-12 values get their own output dir.
# Must be set BEFORE FLAG_TAG is built below.
NUM_FRAMES=${4:-12}

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
case "${EXTRA_FLAGS}" in *track_dynamic*)
  # Mechanism + its cardinal parameter belong in the tag: scene-flow (KNN), K=1 and
  # K=4 groups are DIFFERENT mechanisms and must not share (or overwrite) an output
  # dir — and none may clobber the earlier single-centroid _trk result.
  KNN=$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--dyn_motion_knn[= ]*\([0-9][0-9]*\).*/\1/p')
  if [ -n "${KNN}" ] && [ "${KNN}" != "0" ]; then
    FLAG_TAG="${FLAG_TAG}_flow${KNN}"
    case "${EXTRA_FLAGS}" in *dyn_motion_strict*) FLAG_TAG="${FLAG_TAG}s" ;; esac
    case "${EXTRA_FLAGS}" in *dyn_motion_query_first_only*) FLAG_TAG="${FLAG_TAG}q0" ;; esac
  else
    GRP=$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--dyn_motion_groups[= ]*\([0-9][0-9]*\).*/\1/p')
    FLAG_TAG="${FLAG_TAG}_trk${GRP:-1}"
  fi ;;
esac
# hybrid fusion: voxel size AND scale multiplier must both be in the tag — they are
# different operating points and must never share (or clobber) an output dir.
case "${EXTRA_FLAGS}" in *hybrid_voxelize*)
  VOX=$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--voxel_size[= ]*\([0-9.]*\).*/\1/p')
  SM=$(echo "${EXTRA_FLAGS}" | sed -n 's/.*--scale_mult[= ]*\([0-9.]*\).*/\1/p')
  FLAG_TAG="${FLAG_TAG}_hyb$(echo ${VOX:-0.001} | tr '.' 'p')"
  [ -n "${SM}" ] && FLAG_TAG="${FLAG_TAG}_sm$(echo ${SM} | tr '.' 'p')" ;;
esac
case "${EXTRA_FLAGS}" in *gain_correct*) FLAG_TAG="${FLAG_TAG}_gc" ;; esac
case "${EXTRA_FLAGS}" in *dyn_mask_dir*) FLAG_TAG="${FLAG_TAG}_pcm" ;; esac
[ "${NUM_FRAMES}" != "12" ] && FLAG_TAG="${FLAG_TAG}_nf${NUM_FRAMES}"
[ -z "${FLAG_TAG}" ] && FLAG_TAG="_plain"

# $3 = eval sequence. VALID EVAL SET (2026-08-05 split, family-disjoint from training,
# chosen for SLOW CAMERA so held-out frames differ because things MOVED rather than
# because the viewpoint jumped — Bonn is monocular, so time and viewpoint are coupled):
#   rgbd_bonn_synchronous2             (0.032 m/s, 6.64 deg/s)  <- default, best decoupled
#   rgbd_bonn_removing_obstructing_box (0.086 m/s, 11.88 deg/s)
#   rgbd_bonn_placing_obstructing_box  (0.098 m/s, 12.82 deg/s)
# DO NOT eval on: crowd/crowd2/crowd3, balloon*, synchronous, moving_nonobstructing_box
# — all are training sequences or the same scene family (leakage).
EVAL_SEQ=${3:-rgbd_bonn_synchronous2}
SEQ_TAG=$(echo ${EVAL_SEQ} | sed 's/rgbd_bonn_//')

# Missing masks for THIS sequence are worse than a crash: load_precomputed_masks()
# returns None when no mask file is found, and eval SILENTLY falls back to the weak
# live detection — reporting a plausible-looking number computed under the wrong
# protocol. The base dir existing is not enough; masks are precomputed PER SEQUENCE,
# so check the layout _resolve_mask_path actually reads.
if [ -n "${MASK_DIR}" ]; then
  if [ -d "${MASK_DIR}/${EVAL_SEQ}/masks" ]; then
    N_MASKS=$(ls "${MASK_DIR}/${EVAL_SEQ}/masks"/*.png 2>/dev/null | wc -l)
  else
    N_MASKS=$(ls "${MASK_DIR}"/*.png 2>/dev/null | wc -l)   # FLAT layout (points at .../masks)
  fi
  if [ "${N_MASKS}" -eq 0 ]; then
    echo "ERROR: no precomputed masks for ${EVAL_SEQ} under --dyn_mask_dir ${MASK_DIR}"
    echo "       looked for ${MASK_DIR}/${EVAL_SEQ}/masks/*.png and ${MASK_DIR}/*.png"
    echo "       (eval would NOT fail — it falls back to live detection and reports a"
    echo "        number computed with the wrong masks; precompute this sequence first)"
    exit 1
  fi
  echo "  masks      : ${N_MASKS} precomputed for ${EVAL_SEQ}"
fi
REPO="/mnt/home/hanmydo/DynamicReconstructionSplat"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
# Date suffix = TODAY by default, so a rerun on a different day can never overwrite an
# earlier result. (The flag tag encodes flags but NOT which checkpoint was used, so
# without this an `ft` rerun silently clobbers a previous `ft` number — it already
# happened once.) Override with EVAL_DATE=20260711 to reproduce an old path exactly.
DATE_TAG=${EVAL_DATE:-$(date +%Y%m%d)}
OUT_DIR="output_eval_${MODE_TAG}${FLAG_TAG}_${SEQ_TAG}_${DATE_TAG}"

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
    --num_frames ${NUM_FRAMES} \
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
