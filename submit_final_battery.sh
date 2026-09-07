#!/bin/bash
# =============================================================================
# Submit the FULL comparison battery, unattended.
#
# Per sequence, three jobs chained by slurm dependency:
#   1. otsu2 mask precompute   (SKIPPED if that sequence's masks already exist)
#   2. VANILLA  AnySplat + VGGT, no dynamics handling      <- the "before"
#   3. OURS     VGGT4D + flow-gated compositing + 4D PLY   <- the "after"
#
# Both evals cover the WHOLE sequence (one render per window) and use the same
# masks, so every metric column is comparable and the frames line up 1:1 for
# make_comparison_figure. The PLY window defaults to the middle of each
# sequence, which is valid whatever its length -- do not hard-code an index
# here, a window outside the range silently writes no PLY.
#
# The mechanism is training-free (frozen head), so there is no train/test
# contamination and ANY sequence is a valid eval.
#
# USAGE:  ./submit_final_battery.sh [SEQ ...]
#         EVAL_DATE=final ./submit_final_battery.sh          # default 4 sequences
# =============================================================================
set -uo pipefail
REPO="${HOME}/DynamicReconstructionSplat"; cd "${REPO}"
M2="${HOME}/data/mask_out/output_dyn_masks_precomputed_cs64_r518_st3_fs1_m6_otsu2"
F="--track_dynamic --dyn_motion_knn 8 --dyn_motion_strict --dyn_motion_pred_bandwidth 1.5 --dyn_motion_tracker raft --dyn_motion_max_disp_mult 3.0"
VID="--image_batch_start 0 --max_image_batches 99999 --image_views 0"
export EVAL_DATE="${EVAL_DATE:-final}"

if [ "$#" -gt 0 ]; then SEQS="$*"; else
  SEQS="rgbd_bonn_balloon rgbd_bonn_synchronous2 rgbd_bonn_removing_obstructing_box rgbd_bonn_placing_obstructing_box"
fi

for SEQ in ${SEQS}; do
  [ -d "${HOME}/data/bonn/rgbd_bonn_dataset/${SEQ}/rgb" ] || { echo "SKIP ${SEQ}: no rgb/"; continue; }
  N=$(ls "${M2}/${SEQ}/masks"/*.png 2>/dev/null | wc -l)
  DEP=""
  if [ "${N}" -eq 0 ]; then
    JID=$(sbatch --parsable slurm_precompute_masks_hex.sh "${SEQ}" 64 518 3 1 6 per_frame mean 64 attention 2)
    echo "${SEQ}: masks -> job ${JID}"
    DEP="--dependency=afterok:${JID}"
  else
    echo "${SEQ}: ${N} masks already present"
  fi

  A=$(sbatch --parsable ${DEP} slurm_eval_hex.sh baseline \
        "--no_vggt4d --frame_stride 4 --dyn_mask_dir ${M2} ${VID}" "${SEQ}" 16)
  B=$(sbatch --parsable ${DEP} slurm_eval_hex.sh baseline \
        "--frame_stride 4 --dyn_mask_dir ${M2} --per_frame_dynamic ${F} ${VID} --ply_per_frame" "${SEQ}" 16)
  echo "${SEQ}: vanilla -> job ${A} | ours -> job ${B}"
done

TAG=$(echo ${SEQS} | tr ' ' '\n' | sed 's/rgbd_bonn_//' | tr '\n' ' ')
cat <<TXT

Submitted. When they finish, build each figure with:
  for S in ${TAG}; do
    sbatch slurm_compare_hex.sh \\
      output_eval_frozen_vggt_s4_pcm_nf16_\${S}_${EVAL_DATE} \\
      output_eval_frozen_pfd_s4_flow8raftsb1p5_cl3p0_pcm_nf16_\${S}_${EVAL_DATE} \\
      cmp_\${S} "AnySplat + VGGT (vanilla)" "ours: VGGT4D + flow-gated" 1
  done
TXT
