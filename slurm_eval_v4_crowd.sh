#!/bin/sh
#SBATCH --job-name=eval_v4_crowd
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/eval_v4_crowd_%j.out
#SBATCH --error=slurm_logs/eval_v4_crowd_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --nodelist=bonn,heidelberg,muenchen,stuttgart,koblenz
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

# ============================================================================
# Apples-to-apples held-out-sequence eval for the v4 fine-tuned head.
#
# WHY THIS EXISTS:
#   The reported "20.95 -> 22.64" was NOT a clean before/after. 20.95 = frozen
#   head on `crowd` (held-out sequence, all frames); 22.64 = v4 fine-tuned head
#   on the temporal-tail of the *training* sequences (in-domain). Different model
#   AND different data -> the gap is uninterpretable.
#
#   This job renders the SAME held-out sequence (crowd, --split all, num_frames 12)
#   with the v4 fine-tuned head, using the exact protocol of the stored baseline:
#     VGGT4D backbone + v4 fine-tuned head  -> the run we care about
#   The matched VGGT4D-pretrained baseline on crowd is already on disk
#   (260507_output_debugged/vggt4d_pretrained/metrics.json = 20.96 dB) and its
#   eval path was unchanged by post-May-7 commits, so we don't recompute it.
#   The (v4 PSNR - 20.96) delta IS the real fine-tuning effect on a held-out sequence.
#
#   slurm_eval_finetuned.sh does the same thing but points at an OLD run
#   (output_finetune_lr6). This one points at v4.
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
V4_DIR="${REPO}/output_finetune_omega_recipe_v4"
# Exact epoch-1 high-water-mark checkpoint (manually renamed so later-epoch drift
# couldn't overwrite checkpoint_best.pt). No fallback to checkpoint_latest.pt on
# purpose: that's the drifted epoch-6 model and would be mislabeled as "v4 best".
V4_CKPT="${V4_DIR}/checkpoint_best_epoch1_22.64dB.pt"

echo "=============================================="
echo "v4 held-out eval on: ${EVAL_SEQ}"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

echo "Extracting ${EVAL_SEQ} to /tmp/bonn_data/ ..."
mkdir -p /tmp/bonn_data
python3 -c "
import zipfile
prefix = 'rgbd_bonn_dataset/${EVAL_SEQ}/'
with zipfile.ZipFile('/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip', 'r') as zf:
    members = [m for m in zf.namelist() if m.startswith(prefix)]
    print(f'Extracting {len(members)} files...')
    zf.extractall('/tmp/bonn_data/', members)
print('Extraction done.')
"
echo ""

enroot remove -f eval_v4_crowd 2>/dev/null || true
enroot create --name eval_v4_crowd ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp eval_v4_crowd bash -c "
  cd ${REPO}
  export CUDA_VISIBLE_DEVICES=0
  echo 'Current directory:' \$(pwd)
  python --version
  nvidia-smi
  echo ''

  echo 'Installing open3d for Stage 3 dynamic mask refinement...'
  pip install open3d --quiet
  echo ''

  # --- Use the exact epoch-1 best checkpoint; abort if missing (no silent fallback) ---
  if [ -f \"${V4_CKPT}\" ]; then
    CHECKPOINT=\"${V4_CKPT}\"
    echo \"Using v4 epoch-1 checkpoint: \${CHECKPOINT}\"
  else
    echo 'ERROR: ${V4_CKPT} not found — aborting (refusing to fall back to the drifted latest).'
    exit 1
  fi
  echo ''

  echo '=============================================='
  echo 'VGGT4D + v4 fine-tuned head on held-out crowd...'
  echo '=============================================='
  # Single pass only. The matched VGGT4D-pretrained baseline on crowd is already
  # stored (260507_output_debugged/vggt4d_pretrained/metrics.json = 20.96 dB,
  # same protocol) and its eval path was unchanged by post-May-7 commits, so we
  # compare against that rather than recomputing it.
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${EVAL_SEQ} \
    --intrinsics bonn \
    --num_frames 12 \
    --split all \
    --checkpoint \${CHECKPOINT} \
    --vggt4d_weights_path ${VGGT4D_CKPT} \
    --image_batch_start 400 \
    --output_dir output_eval_v4_crowd_finetuned
"

enroot remove -f eval_v4_crowd

echo ""
echo "=============================================="
echo "Held-out (${EVAL_SEQ}) result — same protocol as the stored baseline:"
echo "  VGGT4D pretrained (baseline, stored): 20.96 dB"
echo "  VGGT4D + v4 fine-tuned:               output_eval_v4_crowd_finetuned/metrics.json"
echo ""
echo "  (v4 PSNR - 20.96) = real fine-tuning effect on a held-out sequence."
echo "=============================================="
echo "Job finished at: $(date)"
