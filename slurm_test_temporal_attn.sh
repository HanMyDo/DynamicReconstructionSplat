#!/bin/sh
#SBATCH --job-name=test_tattn
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --output=slurm_logs/test_tattn_%j.out
#SBATCH --error=slurm_logs/test_tattn_%j.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --requeue

# =============================================================================
# Correctness gate for TemporalAttentionBlock. No dataset, no checkpoint, ~1 min.
#
# THE TEST THAT MATTERS IS #1: identity at initialisation.
#   output_scale starts at 0 so the block is meant to be a no-op until it learns.
#   The original code added the residual in DOWNSAMPLED space and returned
#   upsample(x_down), throwing away the full-resolution input -- so it returned
#   upsample(avgpool(x)) regardless of output_scale, i.e. it LOW-PASS FILTERED the
#   DPT features unconditionally. That produced -0.54 psnr / -0.59 static /
#   -0.28 dyn on 3/3 sequences and a scene-INDEPENDENT LPIPS penalty
#   (+0.0260/+0.0274/+0.0264), which is a fixed blur, not attention.
#   The bug PASSED at ds=1 and only appeared at ds>1 -- this test checks 1, 2 and 4.
#
# Also checks: high-frequency preservation, that the block is NOT inert once
# output_scale != 0, and that frame ORDER changes the output (the criterion that
# separates a temporal mechanism from a multi-view one).
#
# Safe on opportunistic: 1 min, no output to lose, --requeue on preemption.
#
# USAGE:  sbatch slurm_test_temporal_attn.sh
# CHECK:  cat slurm_logs/test_tattn_<jobid>.out     (want "ALL PASS")
# =============================================================================

REPO="/mnt/home/hanmydo/DynamicReconstructionSplat"

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH slurm_logs

CONTAINER=test_tattn_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash -c "
  cd ${REPO}
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH=${REPO}:\$PYTHONPATH
  python tests/test_temporal_attention.py
"
STATUS=$?

enroot remove -f ${CONTAINER}

echo ""
echo "=============================================="
if [ ${STATUS} -eq 0 ]; then
  echo "ALL PASS -> the block is identity at init; the temporal re-run is meaningful."
else
  echo "FAILED (exit ${STATUS}) -> do NOT re-run training; read test 1 above."
fi
echo "Job finished at: $(date)"
echo "=============================================="
exit ${STATUS}
