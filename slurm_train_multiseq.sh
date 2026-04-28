#!/bin/sh
#SBATCH --job-name=train_multiseq
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/train_multiseq_%j.out
#SBATCH --error=slurm_logs/train_multiseq_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=heidelberg,muenchen,koblenz
#SBATCH --time=23:59:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

mkdir -p slurm_logs

echo "=============================================="
echo "Multi-Sequence Training - Cross-Seq Experiment"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

# 8 training sequences — held-out for eval: rgbd_bonn_balloon2
# Covers: crowds, people, balloons, synchronous objects, box kidnapping
TRAIN_SEQUENCES="rgbd_bonn_crowd3 rgbd_bonn_crowd2 rgbd_bonn_balloon rgbd_bonn_balloon_tracking rgbd_bonn_synchronous rgbd_bonn_synchronous2 rgbd_bonn_person_tracking rgbd_bonn_kidnapping_box"

echo "Extracting training sequences to /tmp/bonn_data/ ..."
mkdir -p /tmp/bonn_data
python3 -c "
import zipfile
sequences = '${TRAIN_SEQUENCES}'.split()
with zipfile.ZipFile('/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip', 'r') as zf:
    all_members = zf.namelist()
    for seq in sequences:
        prefix = f'rgbd_bonn_dataset/{seq}/'
        members = [m for m in all_members if m.startswith(prefix)]
        print(f'  {seq}: {len(members)} files')
        zf.extractall('/tmp/bonn_data/', members)
print('Extraction done.')
"
echo ""

enroot remove -f train_multiseq 2>/dev/null || true
enroot create --name train_multiseq ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp train_multiseq bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  export CUDA_VISIBLE_DEVICES=0
  echo 'Current directory:' \$(pwd)
  python --version
  nvidia-smi
  echo ''

  python train_temporal_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_names rgbd_bonn_crowd3,rgbd_bonn_crowd2,rgbd_bonn_balloon,rgbd_bonn_balloon_tracking,rgbd_bonn_synchronous,rgbd_bonn_synchronous2,rgbd_bonn_person_tracking,rgbd_bonn_kidnapping_box \
    --output_dir output_finetune_multiseq_12f \
    --num_epochs 10 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --num_frames 12 \
    --temporal_weight 0.1 \
    --intrinsics bonn
"

enroot remove -f train_multiseq

echo ""
echo "Job finished at: $(date)"
