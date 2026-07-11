#!/bin/sh
#SBATCH --job-name=eval_dw090
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/eval_dw090_crowd_20260710_%j.out
#SBATCH --error=slurm_logs/eval_dw090_crowd_20260710_%j.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:59:00
#SBATCH --nodelist=bonn,heidelberg,muenchen,stuttgart,koblenz
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

# NOTE: eval_gaussian_head.py evaluates the FULL sequence (`--split all` = every
# crowd window) and writes metrics.json only at the very end — it is NOT
# resumable. So: (1) full wall time (23:59, not 4h — job 13831 timed out at 4h),
# and (2) students_normal, NON-preemptible — a non-resumable job must never sit on
# an opportunistic queue where a preemption throws away the whole pass. (Your
# training jobs crashed, so the normal slot is free; relaunch them after eval.)
# max_image_batches only limits IMAGE/VIDEO dumping, NOT the metrics, so the
# comparison to the stored 20.96 dB baseline is unaffected.

# ============================================================================
# Held-out eval of the 0.90 (10% grounding) run's EPOCH-1 checkpoint — 2026-07-10
# ----------------------------------------------------------------------------
# WHY: in-domain val (22.92 dB) is NOT comparable to the 20.95/20.96 dB frozen
# baseline — different data (baseline = held-out `crowd`, val = temporal tail of
# the TRAINING seqs). Only a held-out-crowd eval measures the real fine-tuning
# effect. See memory: v4 scored 22.64 in-domain and ZERO held-out gain.
#
# NEW QUESTION THIS ANSWERS: v4's held-out eval only ever used its EPOCH-5
# (post-drift) checkpoint, which showed no gain. The EPOCH-1 (pre-drift)
# checkpoint was never held-out-evaluated. Does stopping before the drift
# generalize where epoch 5 didn't?
#
# PROTOCOL — must match the stored baseline EXACTLY or the delta is meaningless:
#   --split all, --num_frames 12, --image_batch_start 400,
#   max_image_batches = default (50; deliberately NOT passed, mirroring
#   slurm_eval_v4_crowd.sh). Predicted poses are hardcoded in eval_gaussian_head.py.
#   Stored matched baseline: 260507_output_debugged/vggt4d_pretrained/metrics.json
#   = 20.96 dB on crowd. (PSNR - 20.96) IS the real held-out fine-tuning effect.
#
# CHECKPOINT: pins the DATED checkpoint_best_ep1_*.pt (never overwritten). Do NOT
# use checkpoint_best.pt — the 0.90 job is STILL TRAINING and will move that
# pointer to a later epoch if one beats 22.92, silently changing what we eval.
#
# Runs on students_opportunistic so it never disturbs the training jobs.
#
# LAUNCH:  sbatch slurm_eval_dw090_crowd_20260710.sh
#          sbatch slurm_eval_dw090_crowd_20260710.sh /path/to/other_checkpoint.pt   # optional override
# ============================================================================

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

mkdir -p slurm_logs

EVAL_SEQ="rgbd_bonn_crowd"
REPO="/mnt/home/hanmydo/DynamicReconstructionSplat"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
TRAIN_DIR="${REPO}/output_train_testbed_dw0p90_lr5e-5_20260706"
OUT_DIR="output_eval_dw090_ep1_crowd_20260710"

# Resolve the epoch-1 checkpoint (or use an explicit override as $1).
if [ -n "$1" ]; then
  CKPT="$1"
else
  CKPT=$(ls -1 ${TRAIN_DIR}/checkpoint_best_ep1_*.pt 2>/dev/null | head -1)
fi
if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
  echo "ERROR: could not find an epoch-1 checkpoint."
  echo "Looked for: ${TRAIN_DIR}/checkpoint_best_ep1_*.pt"
  echo "Contents of ${TRAIN_DIR}:"
  ls -la "${TRAIN_DIR}" 2>/dev/null || echo "  (dir missing)"
  echo "If keep_best_n pruned it, pass a checkpoint explicitly as \$1 —"
  echo "but do NOT silently substitute checkpoint_best.pt: it may be a later epoch."
  exit 1
fi

echo "=============================================="
echo "Held-out eval on ${EVAL_SEQ}"
echo "  checkpoint: ${CKPT}"
echo "  ($(du -h "${CKPT}" | cut -f1))"
echo "  node: $(hostname)   time: $(date)"
echo "=============================================="
echo ""

echo "Extracting ${EVAL_SEQ} ..."
BONN_DATA=/tmp/bonn_data_eval_${SLURM_JOB_ID}
mkdir -p ${BONN_DATA}
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

CONTAINER=eval_dw090_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash -c "
  cd ${REPO}
  export CUDA_VISIBLE_DEVICES=0
  python --version
  nvidia-smi
  echo ''

  echo 'Installing open3d (Stage 3 dynamic mask refinement)...'
  pip install open3d --quiet
  echo ''

  echo '=============================================='
  echo 'VGGT4D + 0.90 fine-tuned head (epoch 1) on held-out crowd...'
  echo '=============================================='
  # max_image_batches intentionally omitted -> default 50, matching
  # slurm_eval_v4_crowd.sh and the stored baseline protocol.
  python eval_gaussian_head.py \
    --data_dir ${BONN_DATA}/rgbd_bonn_dataset \
    --dataset_name ${EVAL_SEQ} \
    --intrinsics bonn \
    --num_frames 12 \
    --split all \
    --checkpoint ${CKPT} \
    --vggt4d_weights_path ${VGGT4D_CKPT} \
    --image_batch_start 400 \
    --output_dir ${OUT_DIR}
"

enroot remove -f ${CONTAINER}
rm -rf ${BONN_DATA}

echo ""
echo "=============================================="
echo "Held-out (${EVAL_SEQ}) result — same protocol as the stored baseline:"
echo "  VGGT4D pretrained (baseline, stored):  20.96 dB"
echo "  VGGT4D + 0.90 head, EPOCH 1:           ${OUT_DIR}/metrics.json"
echo ""
echo "  (PSNR - 20.96) = the REAL held-out fine-tuning effect."
echo "  Also read psnr_dynamic vs psnr_static: v4 lost -2.9 dB on dynamic regions."
echo "=============================================="
echo "Job finished at: $(date)"
