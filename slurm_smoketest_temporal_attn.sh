#!/bin/sh
#SBATCH --job-name=smoketest_temporal_attn
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/smoketest_temporal_attn_%j.out
#SBATCH --error=slurm_logs/smoketest_temporal_attn_%j.err
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
echo "Smoke Test - Temporal Attention Enabled"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

# Extract one dataset sequence to fast local /tmp
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
enroot remove -f smoketest_temporal_attn 2>/dev/null || true

# Create container from sqsh file
enroot create --name smoketest_temporal_attn ~/anysplat.sqsh

# Start container
enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp smoketest_temporal_attn bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  echo 'Current directory:' \$(pwd)
  python --version
  nvidia-smi
  echo ''

  python train_temporal_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name rgbd_bonn_crowd3 \
    --output_dir output_smoketest_temporal_attn \
    --num_epochs 1 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --num_frames 4 \
    --temporal_weight 0.1 \
    --intrinsics bonn
"

# Cleanup container
enroot remove -f smoketest_temporal_attn

echo ""
echo "Job finished at: $(date)"
