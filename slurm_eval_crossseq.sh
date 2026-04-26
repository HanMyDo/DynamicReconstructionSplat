#!/bin/sh
#SBATCH --job-name=eval_crossseq
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/eval_crossseq_%j.out
#SBATCH --error=slurm_logs/eval_crossseq_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=bonn,heidelberg,muenchen,stuttgart
#SBATCH --time=08:00:00

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

mkdir -p slurm_logs

echo "=============================================="
echo "Cross-Sequence Eval on held-out: rgbd_bonn_balloon2"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

# Held-out sequence — never seen during any training
HELD_OUT="rgbd_bonn_balloon2"

echo "Extracting ${HELD_OUT} to /tmp/bonn_data/ ..."
mkdir -p /tmp/bonn_data
python3 -c "
import zipfile
prefix = 'rgbd_bonn_dataset/${HELD_OUT}/'
with zipfile.ZipFile('/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip', 'r') as zf:
    members = [m for m in zf.namelist() if m.startswith(prefix)]
    print(f'Extracting {len(members)} files...')
    zf.extractall('/tmp/bonn_data/', members)
print('Extraction done.')
"
echo ""

enroot remove -f eval_crossseq 2>/dev/null || true
enroot create --name eval_crossseq ~/anysplat.sqsh

# All 4 runs in a single container start to avoid repeated startup/weight-loading overhead
enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp eval_crossseq bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  export CUDA_VISIBLE_DEVICES=0
  echo 'Current directory:' \$(pwd)
  python --version
  nvidia-smi
  echo ''

  echo '=============================================='
  echo '[1/4] VGGT original baseline...'
  echo '=============================================='
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${HELD_OUT} \
    --intrinsics bonn \
    --num_frames 24 \
    --split all \
    --no_vggt4d \
    --output_dir output_crossseq_vggt_baseline

  echo '=============================================='
  echo '[2/4] VGGT4D baseline (no fine-tuning)...'
  echo '=============================================='
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${HELD_OUT} \
    --intrinsics bonn \
    --num_frames 24 \
    --split all \
    --output_dir output_crossseq_vggt4d_baseline

  echo '=============================================='
  echo '[3/4] Single-sequence fine-tuned (crowd3 only)...'
  echo '=============================================='
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${HELD_OUT} \
    --intrinsics bonn \
    --num_frames 24 \
    --split all \
    --checkpoint output_finetune_initial/checkpoint_best.pt \
    --use_temporal_attention \
    --output_dir output_crossseq_finetuned_singleseq

  echo '=============================================='
  echo '[4/4] Multi-sequence fine-tuned (4 seqs)...'
  echo '=============================================='
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${HELD_OUT} \
    --intrinsics bonn \
    --num_frames 24 \
    --split all \
    --checkpoint output_finetune_multiseq/checkpoint_best.pt \
    --use_temporal_attention \
    --output_dir output_crossseq_finetuned_multiseq
"

enroot remove -f eval_crossseq

echo ""
echo "=============================================="
echo "Cross-sequence results on ${HELD_OUT}:"
echo "  [1] VGGT original:          output_crossseq_vggt_baseline/metrics.json"
echo "  [2] VGGT4D baseline:        output_crossseq_vggt4d_baseline/metrics.json"
echo "  [3] Fine-tuned (1 seq):     output_crossseq_finetuned_singleseq/metrics.json"
echo "  [4] Fine-tuned (4 seqs):    output_crossseq_finetuned_multiseq/metrics.json"
echo "=============================================="
echo "Job finished at: $(date)"
