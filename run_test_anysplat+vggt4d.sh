#!/bin/sh
#SBATCH --job-name=anysplat+vggt4d
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=anysplat+vggt4d-%j.out
#SBATCH --error=anysplat+vggt4d-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

# Override scratch paths
export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

echo "=============================================="
echo "Testing AnySplat + VGGT4D Integration"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

# Remove old container if exists
enroot remove -f anysplat_vggt4d 2>/dev/null || true

# Create container from sqsh file
enroot create --name anysplat_vggt4d ~/anysplat.sqsh

# Start container with GPU access
enroot start --root --rw \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --mount /mnt:/mnt \
  anysplat_vggt4d bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  echo 'Current directory:' \$(pwd)
  echo 'Python version:' \$(python --version)
  nvidia-smi || echo 'nvidia-smi not available'
  echo ''
  python 'test_anysplat+vggt4d.py'
"

# Cleanup
enroot remove -f anysplat_vggt4d

echo ""
echo "Job finished at: $(date)"
