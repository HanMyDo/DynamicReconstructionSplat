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
#SBATCH --exclude=essen,koblenz
#SBATCH --time=16:00:00

# Override scratch paths
export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

# Create log directory on host
mkdir -p slurm_logs

echo "=============================================="
echo "Training Temporal Gaussian Head - Initial Run"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

# Extract one dataset sequence to fast local /tmp (auto-cleaned after job)
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
ls /tmp/bonn_data/rgbd_bonn_dataset/${DATASET_SEQUENCE}/
echo ""

# Remove old container if exists
enroot remove -f train_temporal_gs 2>/dev/null || true

# Create container from sqsh file
enroot create --name train_temporal_gs ~/anysplat.sqsh

# Start container — mount both /mnt and /tmp so the extracted data is accessible
enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp train_temporal_gs bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  echo 'Current directory:' \$(pwd)
  echo 'Python version:' \$(python --version)
  nvidia-smi
  echo ''

  python train_temporal_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name rgbd_bonn_crowd3 \
    --output_dir output_finetune_initial \
    --num_epochs 20 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --num_frames 4 \
    --temporal_weight 0.1 \
    --intrinsics bonn
"

# Cleanup container
enroot remove -f train_temporal_gs

echo ""
echo "Job finished at: $(date)"
