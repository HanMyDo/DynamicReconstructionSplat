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

# Create container and run test
enroot create --name anysplat_vggt4d ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt anysplat_vggt4d bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  echo 'Current directory:' \$(pwd)
  echo 'Python version:' \$(python --version)
  echo ''
  python 'test_anysplat+vggt4d.py'
"

# Cleanup
enroot remove anysplat_vggt4d

echo ""
echo "Job finished at: $(date)"
