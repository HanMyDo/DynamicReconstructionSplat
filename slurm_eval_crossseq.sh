#!/bin/sh
#SBATCH --job-name=eval_crossseq
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/eval_crossseq_%j.out
#SBATCH --error=slurm_logs/eval_crossseq_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=bonn,heidelberg,muenchen,stuttgart,koblenz
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

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

echo "Downloading VGGT4D pretrained weights..."
mkdir -p /mnt/home/hanmydo/DynamicReconstructionSplat/ckpts
if [ ! -f /mnt/home/hanmydo/DynamicReconstructionSplat/ckpts/vggt4d_model_tracker_fixed_e20.pt ]; then
  wget -q -c "https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt?download=true" \
    -O /mnt/home/hanmydo/DynamicReconstructionSplat/ckpts/vggt4d_model_tracker_fixed_e20.pt
  echo "Download complete."
else
  echo "Weights already cached."
fi
echo ""

enroot remove -f eval_crossseq 2>/dev/null || true
enroot create --name eval_crossseq ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp eval_crossseq bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  export CUDA_VISIBLE_DEVICES=0
  echo 'Current directory:' \$(pwd)
  python --version
  nvidia-smi
  echo ''

  echo 'Installing open3d for Stage 3 dynamic mask refinement...'
  pip install open3d --quiet
  echo ''

  echo '=============================================='
  echo '[1/3] VGGT original baseline...'
  echo '=============================================='
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${HELD_OUT} \
    --intrinsics bonn \
    --num_frames 12 \
    --split all \
    --no_vggt4d \
    --output_dir output_crossseq_vggt_baseline

  echo '=============================================='
  echo '[2/3] VGGT4D token suppression (VGGT-1B weights)...'
  echo '=============================================='
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${HELD_OUT} \
    --intrinsics bonn \
    --num_frames 12 \
    --split all \
    --output_dir output_crossseq_vggt4d_tokensuppress

  echo '=============================================='
  echo '[3/3] VGGT4D token suppression (pretrained VGGT4D weights)...'
  echo '=============================================='
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${HELD_OUT} \
    --intrinsics bonn \
    --num_frames 12 \
    --split all \
    --vggt4d_weights_path ckpts/vggt4d_model_tracker_fixed_e20.pt \
    --output_dir output_crossseq_vggt4d_pretrained
"

enroot remove -f eval_crossseq

echo ""
echo "=============================================="
echo "Cross-sequence results on ${HELD_OUT}:"
echo "  [1] VGGT original:                output_crossseq_vggt_baseline/metrics.json"
echo "  [2] VGGT4D + VGGT-1B weights:     output_crossseq_vggt4d_tokensuppress/metrics.json"
echo "  [3] VGGT4D + pretrained weights:  output_crossseq_vggt4d_pretrained/metrics.json"
echo "=============================================="
echo "Job finished at: $(date)"
