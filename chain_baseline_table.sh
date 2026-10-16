#!/bin/bash
# =============================================================================
# THE DELIVERABLE: ours vs MoVieS vs NeoVerse across time horizons.
#
# Only OUR arm is re-rendered. The baselines' outputs are rendered RGB
# predictions and do not depend on our masks -- masks enter only at SCORING
# time, through compare_masked's --mask_dir -- so re-running them would cost
# hours and change nothing. They are reused from ~/MoVieS/out and ~/NeoVerse/out.
#
# WHY OUR ARM MUST BE RE-RENDERED. Every published row predates two changes:
# --render_max_scale_frac 0.011 (adopted Oct 14) and the SAM-video-global masks
# (adopted Oct 15, worth +0.54 psnr held-out over four sequences).
# REPRODUCE_FIGURES says not to mix pre/post-ceiling rows in one table.
#
# ⚠️ WINDOW STEP IS NOT UNIFORM. The baselines used --window_step 2 at strides
# 16 and 20 on balloon and synchronous2, because those sequences are too short
# for step 5 to leave enough windows; everything else used 5. Our --batch_stride
# must match per stride or the pairing silently drops 4 of every 5 frames (the
# "paired 26 of 126" failure of the Oct 13 ladder). The baseline image counts
# confirm it: balloon s16 = 100 = (439-240)/2.
#
# ⚠️ BOTH METHODS ARE SCORED UNDER THE NEW MASKS. psnr_dynamic/psnr_static are
# not comparable across mask sets, so one mask set has to be chosen for both
# arms; the new one describes where the person actually is. This means the
# BASELINES' dynamic/static numbers also move relative to the published table.
# Say so in the writeup rather than letting a reader discover it.
#
# USAGE:  EVAL_DATE=bt ./chain_baseline_table.sh      # submit
#         DRY=1 ./chain_baseline_table.sh             # check only
# =============================================================================
set -uo pipefail

TAG="${EVAL_DATE:-bt}"
DRY="${DRY:-0}"
REPO="${REPO:-${HOME}/DynamicReconstructionSplat}"
MO="${HOME}/data/mask_out"
BONN="${HOME}/data/bonn/rgbd_bonn_dataset"
TUM="${HOME}/data/tum"
cd "${REPO}"

# Verbatim from REPRODUCE_FIGURES.md "ADOPTED Oct 14 2026". Do NOT rebuild from
# memory: a from-memory version once dropped --dyn_conf_opacity 4.0 (+0.52 psnr).
F="--per_frame_dynamic --dyn_conf_opacity 4.0 --track_dynamic --dyn_motion_knn 8 \
--dyn_motion_strict --dyn_motion_pred_bandwidth 1.5 --dyn_motion_tracker raft \
--dyn_motion_max_disp_mult 3.0 --dyn_opacity_comp 1.0 --bg_color 1 1 1 \
--eval_loo --image_views 8 --dyn_unsupported static --dyn_opacity_gain 4 \
--dyn_unsup_gain 0.25 --render_max_scale_frac 0.011"

A512="output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid"
A256="output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16"

# seq | baseline key | dense? | stride:batch_stride,... | data root | intrinsics | mask set
ROWS=(
"rgbd_bonn_balloon|balloon|dense|4:5,8:5,12:5,16:2,20:2|${BONN}|bonn|${A512}"
"rgbd_bonn_synchronous2|sync2|dense|4:5,8:5,12:5,16:2,20:2|${BONN}|bonn|${A512}"
"rgbd_bonn_crowd|crowd|plain|4:5,12:5,20:5,30:5,40:5|${BONN}|bonn|${A256}"
"rgbd_dataset_freiburg3_walking_xyz|tum_xyz|plain|4:5,8:5,12:5,16:5,20:5|${TUM}|tum_fr3|${A256}"
)

IDS=""; n_sub=0; n_skip=0
for ROW in "${ROWS[@]}"; do
  IFS='|' read -r SEQ KEY DENSE PAIRS ROOT INTR MASK <<< "${ROW}"
  NFRAMES=$(ls "${ROOT}/${SEQ}/rgb"/*.png "${ROOT}/${SEQ}/rgb"/*.jpg 2>/dev/null | wc -l)
  N_MASK=$(ls "${MO}/${MASK}/${SEQ}/masks"/*.png 2>/dev/null | wc -l)
  if [ "${NFRAMES}" -eq 0 ] || [ "${N_MASK}" -lt "${NFRAMES}" ]; then
    echo "SKIP ${SEQ}: frames=${NFRAMES} masks=${N_MASK}"; n_skip=$((n_skip+1)); continue
  fi

  for P in ${PAIRS//,/ }; do
    S="${P%%:*}"; STEP="${P##*:}"
    [ "${DENSE}" = "dense" ] && BDIR="loo_${KEY}_dense_s${S}" || BDIR="loo_${KEY}_s${S}"
    # Refuse a stride whose baselines are missing: a half-populated table is
    # worse than a short one, because the gap is invisible once it is a figure.
    NM=$(ls "${HOME}/MoVieS/out/${BDIR}"/pred_*.png 2>/dev/null | wc -l)
    NN=$(ls "${HOME}/NeoVerse/out/${BDIR}"/pred_*.png 2>/dev/null | wc -l)
    if [ "${NM}" -eq 0 ] || [ "${NN}" -eq 0 ]; then
      echo "SKIP ${KEY} s${S}: movies=${NM} neoverse=${NN} (${BDIR})"
      n_skip=$((n_skip+1)); continue
    fi
    echo "OK   ${KEY} s${S}  step=${STEP}  movies=${NM} neoverse=${NN}  (${S}/2 = $(echo "scale=1; ${S}/2" | bc) s)"
    n_sub=$((n_sub+1))
    [ "${DRY}" = "1" ] && continue

    J=$(EVAL_DATE="${TAG}" sbatch --parsable --job-name=bt --dependency=singleton \
      --time=08:00:00 --gres=gpu:1 \
      --export=ALL,DATA_ROOT="${ROOT}",INTRINSICS="${INTR}" \
      slurm_probe_hex.sh baseline "${MO}/${MASK}" "${SEQ}" 16 "${S}" 0 99999 \
        "ours_s${S}=${F} --batch_stride ${STEP} --dyn_mask_dir ${MO}/${MASK}")
    IDS="${IDS}:${J}"
  done
done

echo "----"
if [ "${DRY}" = "1" ]; then
  echo "DRY RUN -- nothing submitted. ${n_sub} our-arm job(s) would run, ${n_skip} skipped."
  exit 0
fi
echo "submitted ${n_sub} our-arm job(s), skipped ${n_skip}"

# Scoring runs CPU-only, once, after every render finishes. afterany rather than
# afterok so one failed stride does not discard the whole table; the scorer skips
# what is missing and says which rows are absent.
if [ -n "${IDS}" ]; then
  SJ=$(EVAL_DATE="${TAG}" sbatch --parsable --dependency="afterany${IDS}" \
       score_baseline_table.sh)
  echo "scoring job ${SJ} queued behind them -> baseline_table_${TAG}.md"
fi
