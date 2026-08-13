#!/bin/sh
#SBATCH --job-name=test_hybrid
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/test_hybrid_%j.out
#SBATCH --error=slurm_logs/test_hybrid_%j.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:10:00

# =============================================================================
# Unit tests for hybrid static-fusion (EncoderAnySplat.voxelize_static_hybrid).
# No dataset, no checkpoint, ~seconds of compute — this is a correctness gate,
# not an experiment.
#
# THE TEST THAT MATTERS IS #3 (LOO EXACT).
#   Fusing static points across frames means the decoder can no longer drop
#   view j's own Gaussians by frame index (it drops on fidx == j, and a fused
#   voxel has no single source frame). If the fusion still SAW view j, then j's
#   own depth estimate leaks into j's own render — project->unproject->project,
#   the self-reprojection shortcut — and static PSNR inflates for a trivial
#   reason. Static carries our entire measured gain (+1.51 dB), so a leak here
#   would silently invalidate the headline result.
#   Test 3 corrupts the excluded frame's points, features AND confidence, then
#   demands bit-identical output. Anything other than True = do not proceed.
#
# USAGE:  sbatch slurm_test_hybrid_fusion.sh
# CHECK:  cat slurm_logs/test_hybrid_<jobid>.out    (want "ALL PASS")
# =============================================================================

REPO="/mnt/home/hanmydo/DynamicReconstructionSplat"

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH
mkdir -p slurm_logs

CONTAINER=test_hybrid_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash -c "
  cd ${REPO}
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH=${REPO}:\$PYTHONPATH
  python tests/test_hybrid_fusion.py
"
STATUS=$?

enroot remove -f ${CONTAINER}

echo ""
echo "=============================================="
if [ ${STATUS} -eq 0 ]; then
  echo "ALL PASS -> hybrid fusion is LOO-exact, safe to build steps 2-4 on it."
else
  echo "FAILED (exit ${STATUS}) -> do NOT proceed; read test 3 above."
fi
echo "Job finished at: $(date)"
echo "=============================================="
exit ${STATUS}
