#!/bin/sh
#SBATCH --job-name=curr_probe
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --output=slurm_logs/probe_20260706_%j.out
#SBATCH --error=slurm_logs/probe_20260706_%j.err
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
# Fast curriculum-STABILITY probe — 2026-07-06 (parameterized)
# -----------------------------------------------------------------------------
# Purpose: iterate FAST on "does this curriculum config diverge?" without waiting
# days. Both prior divergences happened by ~epoch 2-3 (v2 died at step 831 in the
# STATIC phase; v1 at step 1278 in the ramp), so we only run 3 epochs — enough to
# clear BOTH known failure points (full static phase + first ramp epoch). Val is
# capped (we read train/f_dc_absmax, not val PSNR). You can scancel the moment
# f_dc_absmax clearly trends up past ~5-8; a run that holds ~2-3 through epoch 3
# is stable.
#
# The lever that matters is GROUNDING: static_downweight < 1.0 keeps a little
# photometric gradient on the moving-object Gaussians so their SH color (f_dc)
# can't run away. static_downweight=1.0 (full masking) is what diverged.
#
# USAGE:  sbatch slurm_probe_20260706.sh [STATIC_DOWNWEIGHT] [RAMP_EPOCHS]
#   $1 STATIC_DOWNWEIGHT : dynamic downweight during the static phase (default 0.95)
#   $2 RAMP_EPOCHS       : epochs to ramp static_dw -> 0.9 (default 3; larger = gentler)
# Each (dw, ramp) gets its OWN output dir + wandb run, so variants never collide.
#
# RUN 3 IN PARALLEL (1 normal + 2 opportunistic = the QOS max):
#   sbatch --qos=students_normal        slurm_probe_20260706.sh 0.95 3
#   sbatch --qos=students_opportunistic slurm_probe_20260706.sh 0.90 3
#   sbatch --qos=students_opportunistic slurm_probe_20260706.sh 0.95 5
# Three verdicts in one overnight window instead of three sequential multi-day waits.
#
# WATCH:  tail -f slurm_logs/probe_20260706_<jobid>.out   (and train/f_dc_absmax on W&B)
# =============================================================================

STATIC_DW=${1:-0.95}
RAMP_EP=${2:-3}
TAG="sdw$(echo ${STATIC_DW} | tr '.' 'p')_ramp${RAMP_EP}"
OUTPUT_DIR=output_probe_${TAG}_20260706
RUN_NAME=probe_${TAG}_20260706_${SLURM_JOB_ID}

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

mkdir -p slurm_logs

echo "=============================================="
echo "Curriculum stability probe"
echo "  static_downweight = ${STATIC_DW}   ramp_epochs = ${RAMP_EP}"
echo "  output_dir        = ${OUTPUT_DIR}"
echo "  node = $(hostname)   time = $(date)"
echo "=============================================="
echo ""

TRAIN_SEQUENCES="rgbd_bonn_crowd3 rgbd_bonn_crowd2 rgbd_bonn_balloon rgbd_bonn_synchronous"

# Per-job extraction dir so parallel probes on the same node never race.
BONN_DATA=/tmp/bonn_data_probe_${SLURM_JOB_ID}
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
else
  echo "VGGT4D weights already present."
fi
echo ""

if [ -f ~/.wandb_key ]; then
  WANDB_SETUP="export WANDB_API_KEY='$(cat ~/.wandb_key)'"
else
  WANDB_SETUP="export WANDB_MODE=offline"
fi

CONTAINER=curr_probe_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

# Self-continuation across the wallclock (a STABLE 3-epoch probe can exceed 24h;
# a diverging one dies at ~epoch 2 well inside the window). Same pattern as the
# main launcher: USR1 ~120s before --time -> requeue same job -> auto-resume.
TRAINING_DONE=0
on_walltime() {
  [ "${TRAINING_DONE}" = "1" ] && exit 0
  echo "[SELF-RESUBMIT] USR1 — requeueing job ${SLURM_JOB_ID} to resume from checkpoint."
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap on_walltime USR1

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  export CUDA_VISIBLE_DEVICES=0
  $WANDB_SETUP
  pip install open3d wandb --quiet

  LATEST_CKPT=/mnt/home/hanmydo/DynamicReconstructionSplat/${OUTPUT_DIR}/checkpoint_latest.pt
  if [ -f \"\${LATEST_CKPT}\" ]; then
    echo \"Resuming from \${LATEST_CKPT}\"
    RESUME_FLAG=\"--resume \${LATEST_CKPT}\"
  else
    echo \"Starting fresh.\"
    RESUME_FLAG=\"\"
  fi

  python train_temporal_gaussian_head.py \
    --data_dir ${BONN_DATA}/rgbd_bonn_dataset \
    --dataset_names rgbd_bonn_crowd3,rgbd_bonn_crowd2,rgbd_bonn_balloon,rgbd_bonn_synchronous \
    --output_dir ${OUTPUT_DIR} \
    --num_epochs 3 \
    --val_every_epochs 1 \
    --max_val_batches 20 \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.15 \
    --gradient_clip 0.5 \
    --num_frames 12 \
    --temporal_weight 0.25 \
    --sh_reg_weight 0.01 \
    --dynamic_loss_downweight 0.9 \
    --static_first \
    --curriculum_static_epochs 2 \
    --curriculum_ramp_epochs ${RAMP_EP} \
    --curriculum_static_downweight ${STATIC_DW} \
    --no_gt_poses \
    --intrinsics bonn \
    --save_every_n_steps 100 \
    --log_every_n_steps 5 \
    --vggt4d_weights_path /mnt/home/hanmydo/DynamicReconstructionSplat/ckpts/vggt4d_model_tracker_fixed_e20.pt \
    --wandb_project dynrecsplat \
    --wandb_run_name ${RUN_NAME} \
    \${RESUME_FLAG}
" &
TRAIN_PID=$!
wait ${TRAIN_PID}
TRAIN_RC=$?
TRAINING_DONE=1

enroot remove -f ${CONTAINER}
echo ""
echo "Job finished at: $(date) (exit ${TRAIN_RC})"
exit ${TRAIN_RC}
