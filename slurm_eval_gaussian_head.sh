#!/bin/sh
#SBATCH --job-name=eval_gaussian_head
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/eval_gaussian_head_%j.out
#SBATCH --error=slurm_logs/eval_gaussian_head_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=essen,koblenz
#SBATCH --time=02:00:00

# Override scratch paths
export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

mkdir -p slurm_logs

echo "=============================================="
echo "Evaluating Gaussian Head"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

# Extract dataset sequence to local /tmp
DATASET_SEQUENCE="rgbd_bonn_crowd3"
echo "Extracting ${DATASET_SEQUENCE} from zip to /tmp/bonn_data/ ..."
mkdir -p /tmp/bonn_data
python3 -c "
import zipfile
seq = 'rgbd_bonn_dataset/${DATASET_SEQUENCE}/'
with zipfile.ZipFile('/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip', 'r') as zf:
    members = [m for m in zf.namelist() if m.startswith(seq)]
    print(f'Extracting {len(members)} files...')
    zf.extractall('/tmp/bonn_data/', members)
print('Extraction done.')
"
echo ""

# Remove old container if exists
enroot remove -f eval_gaussian_head 2>/dev/null || true

# Create container from sqsh file
enroot create --name eval_gaussian_head ~/anysplat.sqsh

# --- Run 1: Original AnySplat (VGGT backbone, no VGGT4D) ---
echo "=============================================="
echo "Running VGGT (original AnySplat) evaluation..."
echo "=============================================="
enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp eval_gaussian_head bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name rgbd_bonn_crowd3 \
    --intrinsics bonn \
    --num_frames 4 \
    --split val \
    --no_vggt4d \
    --output_dir output_eval_vggt_baseline
"

# --- Run 2: VGGT4D baseline (no fine-tuning) ---
echo "=============================================="
echo "Running VGGT4D BASELINE evaluation..."
echo "=============================================="
enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp eval_gaussian_head bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name rgbd_bonn_crowd3 \
    --intrinsics bonn \
    --num_frames 4 \
    --split val \
    --output_dir output_eval_vggt4d_baseline
"

# --- Run 3: VGGT4D fine-tuned ---
echo "=============================================="
echo "Running VGGT4D FINE-TUNED evaluation..."
echo "=============================================="
enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp eval_gaussian_head bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name rgbd_bonn_crowd3 \
    --intrinsics bonn \
    --num_frames 4 \
    --split val \
    --checkpoint output_finetune_initial/checkpoint_best.pt \
    --use_temporal_attention \
    --output_dir output_eval_vggt4d_finetuned
"

# Cleanup container
enroot remove -f eval_gaussian_head

echo ""
echo "=============================================="
echo "All evaluations done. Results:"
echo "  VGGT (original):     output_eval_vggt_baseline/metrics.json"
echo "  VGGT4D baseline:     output_eval_vggt4d_baseline/metrics.json"
echo "  VGGT4D fine-tuned:   output_eval_vggt4d_finetuned/metrics.json"
echo "=============================================="
echo "Job finished at: $(date)"
