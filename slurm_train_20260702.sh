#!/bin/sh
#SBATCH --job-name=train_testbed
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/train_testbed_20260702_%j.out
#SBATCH --error=slurm_logs/train_testbed_20260702_%j.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=heidelberg,muenchen,koblenz
#SBATCH --time=23:59:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

# =============================================================================
# Testbed training run — 2026-07-02 (PARAMETERIZED: grounding × peak LR; VGGT-Ω recipe)
# -----------------------------------------------------------------------------
# HISTORY — two prior attempts DIVERGED via f_dc (SH color) runaway, watchdog
# CRITICAL at f_dc_absmax > 25:
#   v1 (job 13070, sh_reg=0):    diverged ~step 1278 (in the ramp), f_dc 6 -> 27.
#   v2 (job 13351, sh_reg=0.01): diverged EARLIER ~step 831 — the STATIC phase,
#                                BEFORE the ramp — f_dc -> 28.9. sh_reg didn't help.
# DIAGNOSIS: not the ramp phase-in, not the pose bug (static phase healthy ~20 dB).
# It is the FULL MASKING itself: curriculum_static_downweight=1.0 gives the
# Gaussians over moving people ZERO photometric gradient, so their f_dc is
# unconstrained and a few drift -> f_dc_absmax (MAX) runs away while sh_reg (MEAN
# of f_dc^2) stays moderate (~6). sh_reg is a MEAN penalty so it can't catch the
# MAX outliers. v4 never had this: it kept dynamic at 10% weight always, so all
# Gaussians stayed grounded and f_dc only crept to ~3.
# FIX (v3): --curriculum_static_downweight 0.95 — keep dynamic at 5% weight even
# in the "static" phase to GROUND those Gaussians, while staying static-first.
# Keep sh_reg 0.01. Single change vs v2. Fresh output dir.
#
# GOAL: a REAL training run (no batch caps) on the small 4-sequence testbed, to
# confirm training is HEALTHY over multiple epochs with the static-first
# curriculum on. This is NOT the final experiment — the "real real" training
# (whole dataset, >=10 epochs) comes later. Here we do 5 epochs, which mirror
# the FIRST 5 epochs of that future longer run, to see the curriculum play out
# and nothing drift/diverge before paying full compute.
#
# Recipe (= v4/v5 baseline + curriculum): LR 5e-5, warmup 0.15, grad_clip 0.5,
# num_frames 12, temporal_weight 0.25, sh_reg 0.01, predicted poses. Curriculum
# over these 5 epochs (curriculum_static_epochs=2, curriculum_ramp_epochs=3):
#   epochs 1-2  static  (dynamic heavily downweighted at 0.95, NOT fully masked)
#   epochs 3-5  ramp    (downweight 0.95 -> 0.9, landing on 0.9 at epoch 5)
#   (the joint phase at 0.9 would begin epoch 6 in the future longer run)
# Validation every epoch -> dense curve.
#
# WHAT TO WATCH ON W&B (run train_testbed_curriculum_20260702_<jobid>):
#   * train/dynamic_downweight  -> should step 1.0 (ep1-2) -> ramp -> 0.9 (ep6-8),
#                                  i.e. the curriculum schedule is visibly working
#   * val/psnr_db, val/psnr_dynamic_db, val/psnr_static_db  -> per-epoch curve
#   * train/temporal            -> should stay flat/declining, not reverse
#   * train/f_dc_absmax (<~3.5), train/scale_max (not collapsing to ~0) -> health
#
# RESUME AFTER A 24h KILL (what you asked to verify):
#   * Output dir is FIXED (output_train_testbed_curriculum_20260702), so
#     checkpoint_latest.pt persists across submissions. On (re)launch the block
#     below sets --resume checkpoint_latest.pt when it exists, and
#     load_checkpoint restores head weights + optimizer(Adam) + LR scheduler +
#     epoch + global_step. Saves are atomic (os.replace) so a kill mid-write
#     cannot corrupt it. => a fresh `sbatch` after a kill CONTINUES, never
#     restarts from scratch.
#   * Worst-case lost work on a kill is bounded by --save_every_n_steps (100
#     opt steps here, ~1.3h) — every completed epoch is also saved at its
#     boundary, so no finished 16h epoch is ever lost.
#   * You have TWO layers: (a) automatic self-requeue via the USR1 trap ~120s
#     before --time (same job id, no new submission, no QOS-limit hit), and
#     (b) manual `sbatch` again — both resume identically. If (a) fires you may
#     not need to resubmit at all; (b) is the fallback if the auto-requeue is
#     ever preempted/blocked.
#
# PARAMETERIZED — keeps the VGGT-Ω / v4 recipe FIXED (LR peak, warmup_ratio 0.15,
# cosine full cycle, grad_clip 0.5, temporal 0.25, nf 12, predicted poses) and
# varies ONLY grounding + peak LR:
#   $1 STATIC_DOWNWEIGHT  (default 0.90 = 10% grounding, the v4-stable level)
#   $2 LEARNING_RATE      (default 5e-5 = v4/Ω; 3e-5 still "small peak LR" per Ω,
#                          gentles the compressed 5-epoch warmup)
# Each (dw, lr) gets its own output dir + wandb run. RUN BOTH IN PARALLEL:
#   sbatch --qos=students_normal        slurm_train_20260702.sh 0.90 5e-5   # reliable, fully v4-faithful
#   sbatch --qos=students_opportunistic slurm_train_20260702.sh 0.95 3e-5   # curriculum experiment (gentler LR)
# RESUME: re-run the SAME command (auto-detects checkpoint_latest.pt in its dir).
# WATCH : tail -f slurm_logs/train_testbed_20260702_<jobid>.out  (+ train/f_dc_absmax on W&B)
# =============================================================================

STATIC_DW=${1:-0.90}
LR=${2:-5e-5}
TAG="dw$(echo ${STATIC_DW} | tr '.' 'p')_lr${LR}"
OUTPUT_DIR_NAME=output_train_testbed_${TAG}_20260706
RUN_NAME=train_testbed_${TAG}_20260706_${SLURM_JOB_ID}

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

mkdir -p slurm_logs

echo "=============================================="
echo "Testbed health-check training (static-first curriculum)"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

# 4 training sequences — held-out for eval: rgbd_bonn_crowd
TRAIN_SEQUENCES="rgbd_bonn_crowd3 rgbd_bonn_crowd2 rgbd_bonn_balloon rgbd_bonn_synchronous"

# Per-JOB extraction dir. Two parallel runs of this launcher can co-schedule on the
# same node; a shared /tmp path makes them race on extraction. (SLURM_JOB_ID is
# preserved across requeues, so this stays stable for a given run.)
BONN_DATA=/tmp/bonn_data_train_${SLURM_JOB_ID}
echo "Extracting training sequences to ${BONN_DATA}/ ..."
mkdir -p ${BONN_DATA}
python3 -c "
import zipfile
sequences = '${TRAIN_SEQUENCES}'.split()
with zipfile.ZipFile('/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip', 'r') as zf:
    all_members = zf.namelist()
    for seq in sequences:
        prefix = f'rgbd_bonn_dataset/{seq}/'
        members = [m for m in all_members if m.startswith(prefix)]
        print(f'  {seq}: {len(members)} files')
        zf.extractall('${BONN_DATA}/', members)
print('Extraction done.')
"
echo ""

VGGT4D_CKPT="/mnt/home/hanmydo/DynamicReconstructionSplat/ckpts/vggt4d_model_tracker_fixed_e20.pt"
if [ ! -f "$VGGT4D_CKPT" ]; then
  echo "Downloading VGGT4D pretrained weights..."
  mkdir -p "$(dirname "$VGGT4D_CKPT")"
  wget -c "https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt" \
    -O "$VGGT4D_CKPT"
  if [ $? -ne 0 ] || [ ! -s "$VGGT4D_CKPT" ]; then
    echo "ERROR: Failed to download VGGT4D weights. Aborting."
    exit 1
  fi
  echo "Download complete: $(du -sh "$VGGT4D_CKPT" | cut -f1)"
else
  echo "VGGT4D weights already present: $(du -sh "$VGGT4D_CKPT" | cut -f1)"
fi
echo ""

# Source wandb API key from cluster home if present; otherwise run offline.
if [ -f ~/.wandb_key ]; then
  WANDB_SETUP="export WANDB_API_KEY='$(cat ~/.wandb_key)'"
  echo "wandb: API key found, will log online."
else
  WANDB_SETUP="export WANDB_MODE=offline"
  echo "wandb: no ~/.wandb_key found, will log in offline mode."
fi
echo ""

# Per-JOB container name. CRITICAL: with a shared name, two parallel runs on the
# same node destroy each other's container via `enroot remove -f` (this killed
# jobs 13688+13689 simultaneously on heidelberg, exit 120).
CONTAINER=train_testbed_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

# --- Self-continuation across the wall-clock limit ---------------------------
# SLURM sends USR1 to this batch shell ~120s before --time (see #SBATCH
# --signal=B:USR1@120). We requeue THIS job (same job id) so: no new sbatch ->
# per-QOS submit limits untouched; no duplicate run; the requeued job re-runs
# from the top and auto-resumes from checkpoint_latest.pt (intact via atomic
# saves). Requeue lives ONLY in this handler, so normal completion and real
# crashes fall through and do NOT loop. Training runs in the background with
# `wait` so the trap fires promptly.
TRAINING_DONE=0
on_walltime() {
  [ "${TRAINING_DONE}" = "1" ] && exit 0
  echo "[SELF-RESUBMIT] USR1 received ~120s before wall-clock limit — requeueing job ${SLURM_JOB_ID} to resume from checkpoint."
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap on_walltime USR1

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  export CUDA_VISIBLE_DEVICES=0
  $WANDB_SETUP
  echo 'Current directory:' \$(pwd)
  python --version
  nvidia-smi
  echo ''

  echo 'Installing open3d (mask refinement) and wandb (telemetry)...'
  pip install open3d wandb --quiet
  echo ''

  # Auto-resume: re-running this sbatch (or a self-requeue) continues from
  # checkpoint_latest.pt in the output dir; otherwise starts fresh.
  OUTPUT_DIR=${OUTPUT_DIR_NAME}
  LATEST_CKPT=/mnt/home/hanmydo/DynamicReconstructionSplat/\${OUTPUT_DIR}/checkpoint_latest.pt
  if [ -f \"\${LATEST_CKPT}\" ]; then
    echo \"Resuming from \${LATEST_CKPT}\"
    RESUME_FLAG=\"--resume \${LATEST_CKPT}\"
  else
    echo \"No checkpoint_latest.pt found — starting fresh.\"
    RESUME_FLAG=\"\"
  fi

  python train_temporal_gaussian_head.py \
    --data_dir ${BONN_DATA}/rgbd_bonn_dataset \
    --dataset_names rgbd_bonn_crowd3,rgbd_bonn_crowd2,rgbd_bonn_balloon,rgbd_bonn_synchronous \
    --output_dir \${OUTPUT_DIR} \
    --num_epochs 5 \
    --val_every_epochs 1 \
    --save_every_n_steps 100 \
    --batch_size 1 \
    --learning_rate ${LR} \
    --warmup_ratio 0.15 \
    --gradient_clip 0.5 \
    --num_frames 12 \
    --temporal_weight 0.25 \
    --sh_reg_weight 0.01 \
    --dynamic_loss_downweight 0.9 \
    --static_first \
    --curriculum_static_epochs 2 \
    --curriculum_ramp_epochs 3 \
    --curriculum_static_downweight ${STATIC_DW} \
    --no_gt_poses \
    --intrinsics bonn \
    --vggt4d_weights_path /mnt/home/hanmydo/DynamicReconstructionSplat/ckpts/vggt4d_model_tracker_fixed_e20.pt \
    --wandb_project dynrecsplat \
    --wandb_run_name ${RUN_NAME} \
    \${RESUME_FLAG}
" &
TRAIN_PID=$!
wait ${TRAIN_PID}
TRAIN_RC=$?
TRAINING_DONE=1   # reaching here means training exited on its own (done or crashed), not wall-clock-killed

enroot remove -f ${CONTAINER}

echo ""
echo "Job finished at: $(date) (training exit code: ${TRAIN_RC})"
exit ${TRAIN_RC}
