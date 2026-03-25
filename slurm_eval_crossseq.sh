#!/bin/sh
#SBATCH --job-name=eval_crossseq
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/eval_crossseq_%j.out
#SBATCH --error=slurm_logs/eval_crossseq_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=essen,koblenz
#SBATCH --time=03:00:00

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

mkdir -p slurm_logs

echo "=============================================="
echo "Cross-Sequence Eval on held-out: rgbd_bonn_crowd2"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

# Held-out sequence — never seen during any training
HELD_OUT="rgbd_bonn_kidnapping_box"

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

# --- Run 1: VGGT original (no VGGT4D) ---
echo "=============================================="
echo "[1/4] VGGT original baseline..."
echo "=============================================="
enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp eval_crossseq bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${HELD_OUT} \
    --intrinsics bonn \
    --num_frames 4 \
    --split val \
    --no_vggt4d \
    --output_dir output_crossseq_vggt_baseline
"

# --- Run 2: VGGT4D baseline (no fine-tuning) ---
echo "=============================================="
echo "[2/4] VGGT4D baseline (no fine-tuning)..."
echo "=============================================="
enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp eval_crossseq bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${HELD_OUT} \
    --intrinsics bonn \
    --num_frames 4 \
    --split val \
    --output_dir output_crossseq_vggt4d_baseline
"

# --- Run 3: Single-sequence fine-tuned (crowd3 only) ---
echo "=============================================="
echo "[3/4] Single-sequence fine-tuned (crowd3 only)..."
echo "=============================================="
enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp eval_crossseq bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${HELD_OUT} \
    --intrinsics bonn \
    --num_frames 4 \
    --split val \
    --checkpoint output_finetune_initial/checkpoint_best.pt \
    --use_temporal_attention \
    --output_dir output_crossseq_finetuned_singleseq
"

# --- Run 4: Multi-sequence fine-tuned ---
echo "=============================================="
echo "[4/4] Multi-sequence fine-tuned (5 sequences)..."
echo "=============================================="
enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp eval_crossseq bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${HELD_OUT} \
    --intrinsics bonn \
    --num_frames 4 \
    --split val \
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
echo "  [4] Fine-tuned (5 seqs):    output_crossseq_finetuned_multiseq/metrics.json"
echo "=============================================="
echo "Job finished at: $(date)"
