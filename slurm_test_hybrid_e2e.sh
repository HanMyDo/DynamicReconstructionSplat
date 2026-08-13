#!/bin/sh
#SBATCH --job-name=test_hyb_e2e
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/test_hyb_e2e_%j.out
#SBATCH --error=slurm_logs/test_hyb_e2e_%j.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:40:00

# =============================================================================
# END-TO-END GATE for hybrid voxelization — 3 batches, FROZEN head, ~20 min.
# Decides whether the 16-20h retrain is safe to launch. Two things can ONLY
# fail on real data, and both would otherwise surface hours into training:
#
#   1. MEMORY. Six fused static sets instead of one. On synthetic data the
#      fusion ratio was 0.264, so expect ~6*0.264 = 1.6x the current static
#      Gaussian count. nf8 already OOM'd on 24GB, so this is the number that
#      decides whether nf6 fits. Watch "[hybrid] ... gaussians" and peak mem.
#
#   2. THE LABELS SURVIVING THE FULL PATH. voxelize_static_hybrid is proven
#      LOO-exact in isolation (tests/test_hybrid_fusion.py, ALL PASS). What is
#      NOT yet proven is that gaussian_only_view survives padding, the gs-prune
#      step and the decoder gate. A silent leak would hide exactly there, and it
#      would inflate STATIC psnr — where our entire +1.51 dB gain sits.
#
# READ THE PSNR AS A SANITY BOUND, NOT A RESULT. This is the FROZEN head, which
# was never trained under fusion, so its scales are calibrated for V redundant
# copies. Fusing them without retraining REMOVES copies without enlarging the
# splats, so a LOWER psnr here is EXPECTED and is not a failure of the hybrid.
# What would be a real failure:
#   * OOM                          -> lower --num_frames or raise voxel_size
#   * psnr near 0 / black renders  -> the static background got double-dropped
#                                     (LOO applied to pre-fused sets - a bug)
#   * "[hybrid]" never printed     -> flag or masks not reaching the encoder
#
# USAGE:  sbatch slurm_test_hybrid_e2e.sh [SEQ]
# CHECK:  grep -E "hybrid|peak|psnr" slurm_logs/test_hyb_e2e_<jobid>.out
# =============================================================================

EVAL_SEQ=${1:-rgbd_bonn_removing_obstructing_box}
# $2 VOXEL_SIZE: fusion voxel edge. The default 0.001 EQUALS the frozen Gaussian scale
# p50 (0.00095) = the point spacing, so every point got its own voxel and the measured
# fusion ratio was ~0.9 (4.86M gaussians vs 1.2M per-pixel: 4x the cost, zero benefit).
# The v per-target-view sets cost v x the fused count, so the hybrid only breaks even
# when fusion merges ~v:1, i.e. ratio ~1/v = 0.167 at v=6. Sweep upward: 0.002 0.005 0.01.
VOXEL_SIZE=${2:-0.001}
# $3 SCALE_MULT (diagnostic): enlarge splats at render time. At voxel_size 0.005 the
# fused point spacing is ~5x the frozen head's scale p50 (0.00095), so frozen splats
# cover a small fraction of each surface. Recovery under scale_mult => the collapse is
# COVERAGE and training fixes it; no recovery => GEOMETRIC (fusion averaged depths that
# disagree) and no head fine-tuning can repair it. Never report a scale_mult number.
SCALE_MULT=${3:-1.0}
SEQ_TAG=$(echo ${EVAL_SEQ} | sed 's/rgbd_bonn_//')
REPO="/mnt/home/hanmydo/DynamicReconstructionSplat"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
MASKS="output_dyn_masks_precomputed_cs16_r518_st3_fs0"
VTAG=$(echo ${VOXEL_SIZE} | tr '.' 'p')
STAG=""
[ "${SCALE_MULT}" != "1.0" ] && STAG="_sm$(echo ${SCALE_MULT} | tr '.' 'p')"
OUT_DIR="output_test_hybrid_e2e_${SEQ_TAG}_vox${VTAG}${STAG}_$(date +%Y%m%d)"

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH slurm_logs

BONN_DATA=/tmp/bonn_hyb_${SLURM_JOB_ID}
mkdir -p ${BONN_DATA}
python3 -c "
import zipfile
prefix = 'rgbd_bonn_dataset/${EVAL_SEQ}/'
with zipfile.ZipFile('/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip', 'r') as zf:
    zf.extractall('${BONN_DATA}/', [m for m in zf.namelist() if m.startswith(prefix)])
print('Extraction done.')
"

CONTAINER=test_hyb_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash -c "
  cd ${REPO}
  export CUDA_VISIBLE_DEVICES=0
  pip install open3d --quiet

  python eval_gaussian_head.py \
    --data_dir ${BONN_DATA}/rgbd_bonn_dataset \
    --dataset_name ${EVAL_SEQ} \
    --intrinsics bonn \
    --num_frames 6 \
    --split all \
    --image_batch_start 400 \
    --max_image_batches 3 \
    --vggt4d_weights_path ${VGGT4D_CKPT} \
    --dyn_mask_dir ${MASKS} \
    --hybrid_voxelize \
    --voxel_size ${VOXEL_SIZE} \
    --scale_mult ${SCALE_MULT} \
    --eval_loo \
    --output_dir ${OUT_DIR}

  echo ''
  echo '--- PEAK GPU MEMORY (the number that decides if the retrain fits) ---'
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv
"
STATUS=$?

enroot remove -f ${CONTAINER}
rm -rf ${BONN_DATA}

echo ""
echo "=============================================="
echo "exit=${STATUS}   metrics -> ${OUT_DIR}/metrics.json"
echo "PASS if: [hybrid] printed a gaussian count, no OOM, psnr is a real number."
echo "A LOWER psnr than the frozen baseline is EXPECTED here (frozen head was"
echo "never trained under fusion) — it is NOT a reason to abandon the hybrid."
echo "Job finished at: $(date)"
echo "=============================================="
exit ${STATUS}
