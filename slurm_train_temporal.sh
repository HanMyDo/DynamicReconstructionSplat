#!/bin/sh
#SBATCH --job-name=train_temporal_gs
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/train_temporal_%j.out
#SBATCH --error=slurm_logs/train_temporal_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=essen
#SBATCH --time=04:00:00

# Override scratch paths
export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

# Create log directory on host
mkdir -p slurm_logs

echo "=============================================="
echo "Training Temporal Gaussian Head"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

# Remove old container if exists
enroot remove -f train_temporal_gs 2>/dev/null || true

# Create container from sqsh file
enroot create --name train_temporal_gs ~/anysplat.sqsh

# Start container
enroot start --root --rw --mount /mnt:/mnt train_temporal_gs bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  echo 'Current directory:' \$(pwd)
  echo 'Python version:' \$(python --version)
  nvidia-smi
  echo ''

  python train_temporal_gaussian_head.py \
    --data_dir examples/vrnerf \
    --dataset_name rgbd_bonn_synchronous2 \
    --output_dir output_finetune_trial \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --num_frames 4 \
    --temporal_weight 0.1 \
    --intrinsics bonn
"

# Cleanup
enroot remove -f train_temporal_gs

echo ""
echo "Job finished at: $(date)"
