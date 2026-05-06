#!/bin/sh
#SBATCH --job-name=eval_multiseq
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/eval_multiseq_%j.out
#SBATCH --error=slurm_logs/eval_multiseq_%j.err
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
echo "Multi-Sequence Eval: VGGT vs VGGT4D"
echo "=============================================="
echo "Job started on node: $(hostname)"
echo "Time: $(date)"
echo ""

# One additional held-out sequence (crowd scene, high dynamics)
SEQ="rgbd_bonn_crowd"

echo "Extracting ${SEQ} to /tmp/bonn_data/ ..."
mkdir -p /tmp/bonn_data
python3 -c "
import zipfile, sys
seq = '${SEQ}'
zf_path = '/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip'
with zipfile.ZipFile(zf_path, 'r') as zf:
    prefix = f'rgbd_bonn_dataset/{seq}/'
    members = [m for m in zf.namelist() if m.startswith(prefix)]
    if not members:
        print(f'ERROR: {seq} not found in zip', flush=True)
        sys.exit(1)
    print(f'Extracting {len(members)} files for {seq}...', flush=True)
    zf.extractall('/tmp/bonn_data/', members)
print('Extraction done.', flush=True)
"
echo ""

enroot remove -f eval_multiseq 2>/dev/null || true
enroot create --name eval_multiseq ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp eval_multiseq bash -c "
  cd /mnt/home/hanmydo/DynamicReconstructionSplat
  export CUDA_VISIBLE_DEVICES=0
  echo 'Current directory:' \$(pwd)
  python --version
  nvidia-smi
  echo ''

  echo 'Installing open3d ...'
  pip install open3d --quiet
  echo ''

  echo '=============================================='
  echo '[1/2] VGGT baseline on ${SEQ}...'
  echo '=============================================='
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${SEQ} \
    --intrinsics bonn \
    --num_frames 12 \
    --split all \
    --no_vggt4d \
    --output_dir output_eval_multiseq/${SEQ}_vggt_baseline

  echo '=============================================='
  echo '[2/2] VGGT4D token suppression on ${SEQ}...'
  echo '=============================================='
  python eval_gaussian_head.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name ${SEQ} \
    --intrinsics bonn \
    --num_frames 12 \
    --split all \
    --output_dir output_eval_multiseq/${SEQ}_vggt4d_tokensuppress
"

enroot remove -f eval_multiseq

echo ""
echo "=============================================="
echo "Results for ${SEQ}:"
echo "  [1] VGGT baseline:         output_eval_multiseq/${SEQ}_vggt_baseline/metrics.json"
echo "  [2] VGGT4D tokensuppress:  output_eval_multiseq/${SEQ}_vggt4d_tokensuppress/metrics.json"
echo "=============================================="
echo "Job finished at: $(date)"
