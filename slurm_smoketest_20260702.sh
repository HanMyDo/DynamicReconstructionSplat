#!/bin/sh
#SBATCH --job-name=smoketest
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/smoketest_%j.out
#SBATCH --error=slurm_logs/smoketest_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=heidelberg,muenchen,koblenz
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

# =============================================================================
# Pipeline smoke test — 2026-07-02
# -----------------------------------------------------------------------------
# Purpose: confirm the CURRENT pipeline runs end-to-end after the recent changes
# (static-first curriculum, checkpoint household, eval-consistent validate()).
# This is NOT a result run — it's a "does it crash?" check that exercises every
# code path once, fast:
#   * encoder forward + Gaussian head
#   * static-first curriculum (--static_first): epoch 1 sits in the STATIC phase,
#     so the dyn_mask weighting branch in compute_rendering_loss runs with
#     downweight=1.0 (dynamic pixels fully masked)
#   * temporal loss, scale/sh reg
#   * validate() with per-frame PSNR/SSIM (matches eval_gaussian_head.py)
#   * periodic atomic checkpoint save (save_every_n_steps=50)
#   * best/final/dated checkpoint bookkeeping
#
# Scope kept tiny: ONE sequence (crowd3), num_frames 8, ONE epoch. ~1-1.5h.
# Auto-resumes from checkpoint_latest.pt if re-submitted, so re-running this
# sbatch also smoke-tests the RESUME path.
#
# LAUNCH:  sbatch slurm_smoketest_20260702.sh
# WATCH :  tail -f slurm_logs/smoketest_<jobid>.out
# PASS   :  runs to "Training complete!", prints a val PSNR (any value — a low
#           1-epoch number is fine), and writes checkpoint_best.pt +
#           checkpoint_latest.pt + a dated checkpoint_final_*.pt in the output dir.
# =============================================================================

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

mkdir -p slurm_logs

echo "=============================================="
echo "Pipeline smoke test (static-first curriculum path)"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

# Single sequence keeps the run short. crowd3 is dynamic, so the curriculum's
# static-phase masking actually has something to mask.
SMOKE_SEQUENCE="rgbd_bonn_crowd3"

echo "Extracting ${SMOKE_SEQUENCE} to /tmp/bonn_data_smoke/ ..."
mkdir -p /tmp/bonn_data_smoke
python3 -c "
import zipfile
seq = '${SMOKE_SEQUENCE}'
with zipfile.ZipFile('/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip', 'r') as zf:
    prefix = f'rgbd_bonn_dataset/{seq}/'
    members = [m for m in zf.namelist() if m.startswith(prefix)]
    print(f'  {seq}: {len(members)} files')
    zf.extractall('/tmp/bonn_data_smoke/', members)
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

enroot remove -f smoketest 2>/dev/null || true
enroot create --name smoketest ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp smoketest bash -c "
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

  # Auto-resume: re-submitting this sbatch continues from checkpoint_latest.pt,
  # which also smoke-tests the resume path.
  OUTPUT_DIR=output_smoketest_20260702
  LATEST_CKPT=/mnt/home/hanmydo/DynamicReconstructionSplat/\${OUTPUT_DIR}/checkpoint_latest.pt
  if [ -f \"\${LATEST_CKPT}\" ]; then
    echo \"Resuming from \${LATEST_CKPT}\"
    RESUME_FLAG=\"--resume \${LATEST_CKPT}\"
  else
    echo \"No checkpoint_latest.pt found — starting fresh.\"
    RESUME_FLAG=\"\"
  fi

  python train_temporal_gaussian_head.py \
    --data_dir /tmp/bonn_data_smoke/rgbd_bonn_dataset \
    --dataset_name ${SMOKE_SEQUENCE} \
    --output_dir \${OUTPUT_DIR} \
    --num_epochs 1 \
    --val_every_epochs 1 \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.15 \
    --gradient_clip 0.5 \
    --num_frames 8 \
    --temporal_weight 0.25 \
    --sh_reg_weight 0.0 \
    --dynamic_loss_downweight 0.9 \
    --static_first \
    --curriculum_static_epochs 2 \
    --curriculum_ramp_epochs 3 \
    --curriculum_static_downweight 1.0 \
    --no_gt_poses \
    --intrinsics bonn \
    --save_every_n_steps 50 \
    --log_every_n_steps 10 \
    --vggt4d_weights_path /mnt/home/hanmydo/DynamicReconstructionSplat/ckpts/vggt4d_model_tracker_fixed_e20.pt \
    --wandb_project dynrecsplat \
    --wandb_run_name smoketest_20260702_${SLURM_JOB_ID} \
    \${RESUME_FLAG}
"

enroot remove -f smoketest

echo ""
echo "Job finished at: $(date)"
