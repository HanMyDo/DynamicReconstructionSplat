#!/bin/sh
#SBATCH --job-name=train_vggt4d_opp
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --output=slurm_logs/train_vggt4d_opp_%j.out
#SBATCH --error=slurm_logs/train_vggt4d_opp_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=heidelberg,muenchen,koblenz
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

mkdir -p slurm_logs

echo "=============================================="
echo "Multi-Sequence Training - VGGT4D Opportunistic"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

# 4 training sequences — held-out for eval: rgbd_bonn_crowd
# Covers: crowds, balloon, synchronous objects
TRAIN_SEQUENCES="rgbd_bonn_crowd3 rgbd_bonn_crowd2 rgbd_bonn_balloon rgbd_bonn_synchronous"

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

VGGT4D_CKPT="/mnt/home/hanmydo/DynamicReconstructionSplat/ckpts/vggt4d_model_tracker_fixed_e20.pt"
if [ ! -f "$VGGT4D_CKPT" ]; then
  echo "Downloading VGGT4D pretrained weights..."
  mkdir -p "$(dirname "$VGGT4D_CKPT")"
  wget -c "https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt" \
    -O "$VGGT4D_CKPT"
  if [ $? -ne 0 ] || [ ! -s "$VGGT4D_CKPT" ]; then
    echo "ERROR: Failed to download VGGT4D weights. Aborting."
    exit 1
  fi
  echo "Download complete: $(du -sh "$VGGT4D_CKPT" | cut -f1)"
else
  echo "VGGT4D weights already present: $(du -sh "$VGGT4D_CKPT" | cut -f1)"
fi
echo ""

enroot remove -f train_vggt4d_opp 2>/dev/null || true
enroot create --name train_vggt4d_opp ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp train_vggt4d_opp bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  export CUDA_VISIBLE_DEVICES=0
  echo 'Current directory:' \$(pwd)
  python --version
  nvidia-smi
  echo ''

  echo 'Installing open3d for Stage 3 dynamic mask refinement...'
  pip install open3d --quiet
  echo ''

  python train_temporal_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_names rgbd_bonn_crowd3,rgbd_bonn_crowd2,rgbd_bonn_balloon,rgbd_bonn_synchronous \
    --output_dir output_finetune_vggt4d \
    --num_epochs 20 \
    --batch_size 1 \
    --learning_rate 1e-6 \
    --num_frames 12 \
    --temporal_weight 0.1 \
    --intrinsics bonn \
    --vggt4d_weights_path /mnt/home/hanmydo/DynamicReconstructionSplat/ckpts/vggt4d_model_tracker_fixed_e20.pt \
    --resume /mnt/home/hanmydo/DynamicReconstructionSplat/output_finetune_vggt4d/checkpoint_latest.pt
"

enroot remove -f train_vggt4d_opp

echo ""
echo "Job finished at: $(date)"
