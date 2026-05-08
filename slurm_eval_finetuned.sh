#!/bin/sh
#SBATCH --job-name=eval_finetuned
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/eval_finetuned_%j.out
#SBATCH --error=slurm_logs/eval_finetuned_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=bonn,heidelberg,muenchen,stuttgart,koblenz
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

mkdir -p slurm_logs

# Same held-out sequence as crossseq eval
EVAL_SEQ="rgbd_bonn_crowd"
BEST_CKPT="/mnt/home/hanmydo/DynamicReconstructionSplat/output_finetune_vggt4d/checkpoint_best.pt"
LATEST_CKPT="/mnt/home/hanmydo/DynamicReconstructionSplat/output_finetune_vggt4d/checkpoint_latest.pt"
VGGT4D_CKPT="/mnt/home/hanmydo/DynamicReconstructionSplat/ckpts/vggt4d_model_tracker_fixed_e20.pt"

if [ -f "$BEST_CKPT" ]; then
  CHECKPOINT="$BEST_CKPT"
  echo "Using best checkpoint: ${CHECKPOINT}"
elif [ -f "$LATEST_CKPT" ]; then
  CHECKPOINT="$LATEST_CKPT"
  echo "checkpoint_best.pt not found, using latest: ${CHECKPOINT}"
else
  echo "ERROR: No checkpoint found in output_finetune_vggt4d/. Has training started?"
  exit 1
fi

echo "=============================================="
echo "Fine-tuned Eval on: ${EVAL_SEQ}"
echo "Checkpoint: ${CHECKPOINT}"
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

enroot remove -f eval_finetuned 2>/dev/null || true
enroot create --name eval_finetuned ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp eval_finetuned bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  export CUDA_VISIBLE_DEVICES=0
  echo 'Current directory:' \$(pwd)
  python --version
  nvidia-smi
  echo ''

  echo 'Installing open3d for Stage 3 dynamic mask refinement...'
  pip install open3d --quiet
  echo ''

  echo '=============================================='
  echo 'VGGT4D + fine-tuned temporal attention...'
  echo '=============================================='
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${EVAL_SEQ} \
    --intrinsics bonn \
    --num_frames 12 \
    --split all \
    --checkpoint ${CHECKPOINT} \
    --use_temporal_attention \
    --vggt4d_weights_path ${VGGT4D_CKPT} \
    --image_batch_start 400 \
    --output_dir output_eval_finetuned_vggt4d
"

enroot remove -f eval_finetuned

echo ""
echo "=============================================="
echo "Fine-tuned results on ${EVAL_SEQ}:"
echo "  Fine-tuned: output_eval_finetuned_vggt4d/metrics.json"
echo ""
echo "Compare against crossseq baselines:"
echo "  VGGT baseline:    output_crossseq_vggt_baseline/metrics.json"
echo "  VGGT4D pretrained: output_crossseq_vggt4d_pretrained/metrics.json"
echo "=============================================="
echo "Job finished at: $(date)"
