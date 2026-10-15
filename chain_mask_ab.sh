#!/bin/bash
# =============================================================================
# MASK A/B: the adopted render config `F`, unchanged, against two mask sets.
#
# ONE VARIABLE. Both arms of every pair share F byte-for-byte, the same chunk
# size, the same window set and the same sequence; only --dyn_mask_dir differs.
# That is the whole point -- Oct 15 produced four other changes (the moved%
# diagnostic, per_window.json, --global_flow_thresh, a directory rename) and
# none of them may enter this comparison. Three are inert by construction and
# --global_flow_thresh was measured and REJECTED, so it is not passed.
#
# CHUNK IS MATCHED PER PAIR, not fixed globally: the flow detector host-OOMs
# (`Killed`) at chunk 512 past ~950 frames, so long sequences must run at 256
# (128 for TUM halfsphere, whose cs256 attempt was killed). Pairing a 512 arm
# against a 256 arm would make chunk size a second variable.
#
# WHY A PNG GUARD. Nine zero-count mask directories exist under ~/data/mask_out
# from failed runs, and the eval only checks that a directory EXISTS -- on
# Oct 15 two box-sequence evals died in one second against Sep 25 stubs. A short
# directory here aborts its own pair and leaves the rest of the chain alone,
# rather than being discovered in the morning.
#
# USAGE:  EVAL_DATE=mab ./chain_mask_ab.sh            # submit
#         DRY=1 ./chain_mask_ab.sh                     # print and check only
# =============================================================================
set -uo pipefail

TAG="${EVAL_DATE:-mab}"
DRY="${DRY:-0}"
REPO="${REPO:-${HOME}/DynamicReconstructionSplat}"
MO="${HOME}/data/mask_out"   # --dyn_mask_dir takes the TOP-LEVEL dir;
                              # eval appends /<sequence>/masks itself.
BONN="${HOME}/data/bonn/rgbd_bonn_dataset"
TUM="${HOME}/data/tum"
cd "${REPO}"

# The adopted config, copied verbatim from REPRODUCE_FIGURES.md "ADOPTED Oct 14
# 2026". Do NOT rebuild it from memory: a from-memory version once dropped
# --dyn_conf_opacity 4.0, which is worth +0.52 psnr.
F="--per_frame_dynamic --dyn_conf_opacity 4.0 --track_dynamic --dyn_motion_knn 8 \
--dyn_motion_strict --dyn_motion_pred_bandwidth 1.5 --dyn_motion_tracker raft \
--dyn_motion_max_disp_mult 3.0 --dyn_opacity_comp 1.0 --bg_color 1 1 1 \
--eval_loo --image_views 8 --dyn_unsupported static --dyn_opacity_gain 4 \
--dyn_unsup_gain 0.25 --render_max_scale_frac 0.011"

# Window SAMPLING, not render config: starts 0,5,10..., which is exactly the set
# MoVieS/NeoVerse slide through, and ~5x cheaper than step 1. Identical in both
# arms, so it cannot shift the comparison.
SAMPLE="--batch_stride 5"

NF=16          # 16-frame window, as every published ladder rung
STRIDE=8       # the 4 s rung (baseline seconds = STRIDE / 2)

# seq | chunk | old mask set | new mask set | data root | intrinsics
ROWS=(
"rgbd_bonn_balloon|512|output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_otsu2_glob|output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid|${BONN}|bonn"
"rgbd_bonn_synchronous2|512|output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_otsu2_glob|output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid|${BONN}|bonn"
"rgbd_bonn_crowd|256|output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_otsu2_glob|output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16|${BONN}|bonn"
"rgbd_bonn_removing_obstructing_box|256|output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_otsu2_glob|output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16|${BONN}|bonn"
"rgbd_dataset_freiburg3_walking_xyz|256|output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_otsu2_glob|output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16|${TUM}|tum_fr3"
"rgbd_dataset_freiburg3_walking_halfsphere|128|output_dyn_masks_precomputed_cs128_r518_st3_fs1_m6_otsu2_glob|output_dyn_masks_precomputed_cs128_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16|${TUM}|tum_fr3"
)

n_ok=0; n_skip=0; SKIPPED=""
for ROW in "${ROWS[@]}"; do
  IFS='|' read -r SEQ CHUNK OLDM NEWM ROOT INTR <<< "${ROW}"
  NFRAMES=$(ls "${ROOT}/${SEQ}/rgb"/*.png "${ROOT}/${SEQ}/rgb"/*.jpg 2>/dev/null | wc -l)
  N_OLD=$(ls "${MO}/${OLDM}/${SEQ}/masks"/*.png 2>/dev/null | wc -l)
  N_NEW=$(ls "${MO}/${NEWM}/${SEQ}/masks"/*.png 2>/dev/null | wc -l)

  # Both arms must cover the sequence. Unequal counts are as bad as missing
  # ones: the two arms would then be scored over different frames.
  if [ "${NFRAMES}" -eq 0 ] || [ "${N_OLD}" -lt "${NFRAMES}" ] || [ "${N_NEW}" -lt "${NFRAMES}" ]; then
    echo "SKIP ${SEQ}: frames=${NFRAMES} old=${N_OLD} new=${N_NEW}  (need both >= frames)"
    SKIPPED="${SKIPPED} ${SEQ}"; n_skip=$((n_skip + 1)); continue
  fi
  echo "OK   ${SEQ}: frames=${NFRAMES} old=${N_OLD} new=${N_NEW} chunk=${CHUNK} intr=${INTR}"
  n_ok=$((n_ok + 1))
  [ "${DRY}" = "1" ] && continue

  # Both arms in ONE allocation so they share the model load and cannot drift
  # apart through a cluster change between two separate jobs.
  EVAL_DATE="${TAG}" sbatch --job-name=mab --dependency=singleton \
    --time=08:00:00 --gres=gpu:1 \
    --export=ALL,DATA_ROOT="${ROOT}",INTRINSICS="${INTR}" \
    slurm_probe_hex.sh baseline "${MO}/${OLDM}" "${SEQ}" "${NF}" "${STRIDE}" 0 99999 \
      "old=${F} ${SAMPLE} --dyn_mask_dir ${MO}/${OLDM}" \
      "new=${F} ${SAMPLE} --dyn_mask_dir ${MO}/${NEWM}"
done

echo "----"
if [ "${DRY}" = "1" ]; then
  echo "DRY RUN -- nothing submitted. ${n_ok} pair(s) would run, ${n_skip} skipped:${SKIPPED:-none}"
else
  echo "submitted ${n_ok} pair(s), skipped ${n_skip}:${SKIPPED:-none}"
fi
echo "tag=${TAG}  ->  output_probe_frozen_{old,new}_<seq>_${TAG}/"
echo "metrics.json for the aggregate, per_window.json to re-slice to busy windows."
