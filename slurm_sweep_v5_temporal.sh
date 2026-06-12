#!/bin/sh
#SBATCH --job-name=sweep_v5_tw
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --output=slurm_logs/sweep_v5_tw_%A_%a.out
#SBATCH --error=slurm_logs/sweep_v5_tw_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=heidelberg,muenchen,koblenz
#SBATCH --time=23:59:00
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

# =============================================================================
# v5 temporal_weight sweep (× early-stop characterization)
# -----------------------------------------------------------------------------
# Why: v4 (temporal_weight=0.25) plateaued on val PSNR by epoch 1 (22.64) and
# then DRIFTED — train/temporal reversed (0.004 -> 0.018), f_dc_absmax crept
# 2.0 -> 3.13, scale_max grew 0.005 -> 0.013. Diagnosis: once the easy global
# gains are exhausted, the cheapest way to lower train MSE is per-frame
# overfitting that trades away temporal consistency, and tw=0.25 is too weak to
# hold the line. This sweep tests whether a stronger temporal weight keeps
# consistency without destroying PSNR.
#
# Design: ONE variable changes (temporal_weight). Everything else is identical
# to v4. Each array task is an independent, resumable run with its own output
# dir. Horizon is short (8 epochs) because the model saturates at epoch 1 and
# drift is visible by epoch ~5-6 — no need for 20. Validation runs EVERY epoch
# so we get a dense val curve and can read off the peak/drift point per config.
#
# Array index -> temporal_weight:
#   0 -> 0.25  (control: reproduce v4 drift under the 8-epoch schedule)
#   1 -> 0.5
#   2 -> 1.0
# Runs concurrently across the 3 named nodes (one task per node) if free.
#
# LAUNCH (dual-QOS, recommended): students_normal and students_opportunistic
# have SEPARATE per-user job limits (1 and 2), so run the control on the stable
# (non-preemptible) normal QOS and the treatments on opportunistic:
#   sbatch --qos=students_normal        --array=0   slurm_sweep_v5_temporal.sh
#   sbatch --qos=students_opportunistic --array=1-2 slurm_sweep_v5_temporal.sh
# This protects the baseline curve from preemption while still running all 3.
# (The #SBATCH --qos below is the default if you submit without overriding.)
# =============================================================================

TEMPORAL_WEIGHTS="0.25 0.5 1.0"
TW_TAGS="0p25 0p5 1p0"
TW=$(echo $TEMPORAL_WEIGHTS | cut -d' ' -f$((SLURM_ARRAY_TASK_ID + 1)))
TAG=$(echo $TW_TAGS | cut -d' ' -f$((SLURM_ARRAY_TASK_ID + 1)))

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

mkdir -p slurm_logs

echo "=============================================="
echo "v5 Temporal-Weight Sweep — task ${SLURM_ARRAY_TASK_ID}"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo "temporal_weight = ${TW}  (tag ${TAG})"
echo ""

# 4 training sequences — held-out for eval: rgbd_bonn_crowd (same split as v4)
TRAIN_SEQUENCES="rgbd_bonn_crowd3 rgbd_bonn_crowd2 rgbd_bonn_balloon rgbd_bonn_synchronous"

# Per-task extraction dir so concurrent array tasks on the same node never race.
BONN_DATA=/tmp/bonn_data_${SLURM_ARRAY_TASK_ID}
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

# Per-task container name so concurrent tasks on the same node don't collide.
CONTAINER=sweep_v5_tw_${SLURM_ARRAY_TASK_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

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

  # Per-config output dir -> independent auto-resume (one checkpoint_latest.pt
  # per temporal_weight). A wallclock-killed task resumes its own config on the
  # next sbatch without touching the others.
  OUTPUT_DIR=output_finetune_omega_recipe_v5_tw${TAG}
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
    --num_epochs 8 \
    --val_every_epochs 1 \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.15 \
    --gradient_clip 0.5 \
    --num_frames 12 \
    --temporal_weight ${TW} \
    --sh_reg_weight 0.0 \
    --no_gt_poses \
    --intrinsics bonn \
    --vggt4d_weights_path /mnt/home/hanmydo/DynamicReconstructionSplat/ckpts/vggt4d_model_tracker_fixed_e20.pt \
    --wandb_project dynrecsplat \
    --wandb_run_name omega_recipe_v5_tw${TAG}_${SLURM_ARRAY_JOB_ID} \
    \${RESUME_FLAG}
"

enroot remove -f ${CONTAINER}

echo ""
echo "Job finished at: $(date)"
