#!/bin/bash
#SBATCH --job-name=eval_battery
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/eval_battery_%j.out
#SBATCH --error=slurm_logs/eval_battery_%j.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:59:00
#SBATCH --nodelist=bonn,heidelberg,muenchen,stuttgart,koblenz
# =============================================================================
# The WHOLE re-validation battery in ONE job.
#
# WHY. students_normal runs one job at a time and allows two submitted, but the
# battery is nine evaluations -- so queueing them individually needs babysitting
# all day. Chaining them inside a single allocation means one sbatch and
# everything is done by evening. Extraction and the container are set up once
# instead of nine times, which also saves ~15 min.
#
# WHAT IT RUNS (all on the SAME masks, ~25 min each except synchronous2 ~10):
#   removing_obstructing_box : control, flow, rigid-4, ft-control, ft-flow
#   placing_obstructing_box  : control, flow
#   synchronous2             : control, flow
# The control is repeated per sequence ON PURPOSE: a new mask changes both the
# dyn/static split and which Gaussians the flow displaces, so an old control is
# not a valid baseline for a new flow number.
#
# USAGE: sbatch slurm_eval_battery.sh [MASK_DIR] [DATE_TAG]
# =============================================================================

MASK_DIR=${1:-output_dyn_masks_precomputed_cs64_r518_st3_fs1_m6}
DATE_TAG=${2:-m6}
NF=6

REPO="/mnt/home/hanmydo/DynamicReconstructionSplat"
VGGT4D_CKPT="${REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt"
cd ${REPO}
mkdir -p slurm_logs

CKPT=$(ls -t ${REPO}/output_*anc1p0*/checkpoint_best_ep5_*.pt 2>/dev/null | head -1)
echo "anchor checkpoint: [${CKPT:-NONE FOUND -- ft runs will be skipped}]"

BASE="--eval_loo --frame_stride 8 --max_image_batches 0 --dyn_mask_dir ${MASK_DIR}"
FLOW="--track_dynamic --dyn_motion_knn 8 --dyn_motion_strict --dyn_motion_pred_bandwidth 1.5"
RIGID="--track_dynamic --dyn_motion_groups 4"

# seq | mode | extra flags | flag tag
RUNS=(
"rgbd_bonn_removing_obstructing_box|frozen||_loo_s8_pcm_nf6"
"rgbd_bonn_removing_obstructing_box|frozen|${FLOW}|_loo_s8_flow8sb1p5_pcm_nf6"
"rgbd_bonn_removing_obstructing_box|frozen|${RIGID}|_loo_s8_trk4_pcm_nf6"
"rgbd_bonn_removing_obstructing_box|ft||_loo_s8_pcm_nf6"
"rgbd_bonn_removing_obstructing_box|ft|${FLOW}|_loo_s8_flow8sb1p5_pcm_nf6"
"rgbd_bonn_placing_obstructing_box|frozen||_loo_s8_pcm_nf6"
"rgbd_bonn_placing_obstructing_box|frozen|${FLOW}|_loo_s8_flow8sb1p5_pcm_nf6"
"rgbd_bonn_synchronous2|frozen||_loo_s8_pcm_nf6"
"rgbd_bonn_synchronous2|frozen|${FLOW}|_loo_s8_flow8sb1p5_pcm_nf6"
)

# --- masks must exist for every sequence BEFORE we start: a missing mask does not
# --- fail the eval, it silently falls back to live detection and reports a number
# --- computed under the wrong protocol. Better to refuse now than 3 h from now.
for SEQ in rgbd_bonn_removing_obstructing_box rgbd_bonn_placing_obstructing_box rgbd_bonn_synchronous2; do
  N=$(ls "${MASK_DIR}/${SEQ}/masks"/*.png 2>/dev/null | wc -l)
  echo "masks ${SEQ}: ${N}"
  [ "${N}" -eq 0 ] && { echo "ERROR: no masks for ${SEQ} under ${MASK_DIR}"; exit 1; }
done

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH

BONN_DATA=/tmp/bonn_battery_${SLURM_JOB_ID}
mkdir -p ${BONN_DATA}
echo "Extracting all three eval sequences ..."
python3 -c "
import zipfile
seqs = ['rgbd_bonn_removing_obstructing_box','rgbd_bonn_placing_obstructing_box','rgbd_bonn_synchronous2']
with zipfile.ZipFile('/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip','r') as zf:
    members = [m for m in zf.namelist() if any(m.startswith(f'rgbd_bonn_dataset/{s}/') for s in seqs)]
    print(f'{len(members)} files'); zf.extractall('${BONN_DATA}/', members)
print('done')"

# Build the in-container script on /tmp (mounted), so the run list survives the
# quoting boundary intact rather than being re-expanded inside bash -c.
BODY=/tmp/battery_body_${SLURM_JOB_ID}.sh
{
  echo "set -u"
  echo "cd ${REPO}"
  echo "export CUDA_VISIBLE_DEVICES=0"
  echo "pip install open3d --quiet"
  i=0
  for R in "${RUNS[@]}"; do
    i=$((i+1))
    SEQ=$(echo "$R" | cut -d'|' -f1)
    MODE=$(echo "$R" | cut -d'|' -f2)
    XFLAGS=$(echo "$R" | cut -d'|' -f3)
    FTAG=$(echo "$R" | cut -d'|' -f4)
    [ "${MODE}" = "ft" ] && [ -z "${CKPT}" ] && continue
    CKPT_FLAG=""; [ "${MODE}" = "ft" ] && CKPT_FLAG="--checkpoint ${CKPT}"
    SEQ_TAG=$(echo ${SEQ} | sed 's/rgbd_bonn_//')
    OUT="output_eval_${MODE}${FTAG}_${SEQ_TAG}_${DATE_TAG}"
    echo "echo '===== [${i}/${#RUNS[@]}] ${MODE} ${SEQ_TAG} ${XFLAGS:-no-motion} -> ${OUT} ('\$(date)')'"
    # keep going if one config fails; the others are still worth having
    echo "python eval_gaussian_head.py --data_dir ${BONN_DATA}/rgbd_bonn_dataset \
--dataset_name ${SEQ} --intrinsics bonn --num_frames ${NF} --split all \
--vggt4d_weights_path ${VGGT4D_CKPT} --output_dir ${OUT} ${CKPT_FLAG} ${BASE} ${XFLAGS} || echo \"FAILED: ${OUT}\""
  done
} > ${BODY}
echo "--- battery plan ---"; grep -c "^python" ${BODY}; echo "--------------------"

CONTAINER=eval_battery_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh
enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash ${BODY}

enroot remove -f ${CONTAINER}
rm -rf ${BONN_DATA} ${BODY}

echo "=============================================="
echo "Battery done at $(date). Results:"
ls -d output_eval_*_${DATE_TAG} 2>/dev/null
echo "=============================================="
