#!/bin/sh
#SBATCH --job-name=rank_seqs
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/rank_seqs_20260805_%j.out
#SBATCH --error=slurm_logs/rank_seqs_20260805_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=bonn,heidelberg,muenchen,stuttgart,koblenz
#SBATCH --time=00:20:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

# =============================================================================
# Rank Bonn sequences by object-motion / camera-motion DECOUPLING — 2026-08-05
# -----------------------------------------------------------------------------
# Bonn is monocular, so temporal distance and viewpoint change are coupled: a
# held-out frame far in time is also far in viewpoint. That confound flattened
# the stride-8 LOO result (static degraded as much as dynamic). Sequences with a
# SLOW camera but LOTS of moving content break the coupling — held-out frames
# then differ mainly because things MOVED, which is what we want to measure.
#
# CPU-only work (numpy over groundtruth.txt); extracts just the small pose/rgb
# index files, not the images. Runs in ~seconds.
#
# USAGE:  sbatch slurm_rank_sequences_20260805.sh [MASK_DIR]
#   $1 MASK_DIR  optional precomputed-mask parent for the dynamic fraction column,
#                e.g. output_dyn_masks_precomputed_cs16_r518_st3_fs0
#                (omit -> ranks by camera speed only)
#
# READ THE TABLE: pick eval sequences with LOW cam_m/s + HIGH dyn% (top of the
# ranking). Use 3-5 of them for evaluation, disjoint from the training set.
# =============================================================================

MASK_DIR=${1:-}
REPO="/mnt/home/hanmydo/DynamicReconstructionSplat"

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH slurm_logs

GT_DATA=/tmp/bonn_gt_${SLURM_JOB_ID}
mkdir -p ${GT_DATA}

echo "Extracting groundtruth.txt for ALL sequences (small text files only)..."
python3 -c "
import zipfile
with zipfile.ZipFile('/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip') as z:
    members = [m for m in z.namelist()
               if m.endswith('groundtruth.txt') or m.endswith('rgb.txt')]
    z.extractall('${GT_DATA}', members)
    print(f'  extracted {len(members)} index files')
"
echo ""

CONTAINER=rank_seqs_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

if [ -n "${MASK_DIR}" ]; then
  MASK_FLAG="--mask_dir ${MASK_DIR}"
else
  MASK_FLAG=""
fi

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash -c "
  cd ${REPO}
  python rank_sequences.py \
    --data_dir ${GT_DATA}/rgbd_bonn_dataset \
    ${MASK_FLAG}
"

enroot remove -f ${CONTAINER}
rm -rf ${GT_DATA}

echo ""
echo "Done. Pick 3-5 EVAL sequences from the TOP of the ranking"
echo "(low cam_m/s, high dyn%), disjoint from the training sequences."
