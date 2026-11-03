#!/bin/bash
# =============================================================================
# Two lanes, both finish overnight, never more than two GPUs.
#
# LANE 1 (pcd) -- THE PERFECT-MASK ORACLE, the question we currently cannot
# answer at all and the one a reviewer will ask first. Dynamic Replica ships GT
# foreground masks; our own masks score IoU 0.50-0.81 against them, so there is
# real headroom to measure. Two scenes chosen by GT travel so the result is not
# a single point and not confounded by travel: 0cde48 at 1103 px (where
# relocation error is also largest) and 216ba3 at 274 px as the control.
# 06dcf6 is excluded -- 52 px of travel and 0.003 mean area, nothing to see.
#
# The GT masks must be compared against the ADOPTED mask recipe, not the old
# attention masks that happen to exist for these scenes: the adopted ones are
# already worth +0.54 over those, so using attention would overstate the
# headroom. Hence a precompute first, then the A/B behind it.
#
# PREDICTION, recorded before the run: GT masks improve STATIC and reduce
# FLOATS, and do NOT complete the person -- masks decide WHICH gaussians are
# treated as dynamic, not whether relocating them yields a coherent person.
# Read our dyn_fraction_mean against the GT area of 0.089: far above it means
# false positives, and the static gain is measuring those.
#
# LANE 2 (ns5) -- replicate --dyn_nearest_source 2 across five sequences.
# On balloon it was the best PROFILE of its family: overall tied (-0.003),
# static +0.712, and ssim AND lpips both better, which src_window_dyn 2 never
# managed. But src_window 2 also looked good on balloon and then went 1/10 on
# overall across five sequences, so this is a replication, not an adoption.
# ⛔ It does NOT fix the person -- measured, the arms stay unresolved.
#
# USAGE:  ./chain_tonight.sh        # submit
#         DRY=1 ./chain_tonight.sh  # print only
# =============================================================================
set -uo pipefail

DRY="${DRY:-0}"
REPO="${REPO:-${HOME}/DynamicReconstructionSplat}"
MO="${HOME}/data/mask_out"
BONN="${HOME}/data/bonn/rgbd_bonn_dataset"
TUM="${HOME}/data/tum"
ADOPTED="output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid"
cd "${REPO}"
mkdir -p slurm_logs

# The adopted config, verbatim. Swept flags are appended per arm.
F="--per_frame_dynamic --dyn_conf_opacity 4.0 --track_dynamic --dyn_motion_knn 8 \
--dyn_motion_strict --dyn_motion_pred_bandwidth 1.5 --dyn_motion_tracker raft \
--dyn_motion_max_disp_mult 3.0 --dyn_opacity_comp 1.0 --bg_color 1 1 1 \
--eval_loo --image_views 8 --dyn_unsupported static --dyn_opacity_gain 4 \
--dyn_unsup_gain 0.25 --render_max_scale_frac 0.011 --image_size 392 518 \
--src_window_dyn 4 --batch_stride 5"

# ---------------------------------------------------------------- LANE 1 ----
IDS=""
for S in dynrep_0cde48 dynrep_216ba3; do
  if [ ! -d "${BONN}/${S}/rgb" ]; then echo "SKIP ${S}: no rgb/"; continue; fi
  if [ ! -d "${MO}/gtmask/${S}/masks" ]; then
    echo "SKIP ${S}: no GT mask symlink at ${MO}/gtmask/${S}/masks"; continue; fi
  echo "precompute ${S} (chunk 512, 300 frames)"
  [ "${DRY}" = "1" ] && continue
  J=$(sbatch --parsable --job-name=pcd --dependency=singleton \
      --export=ALL,DATA_ROOT="${BONN}" \
      slurm_precompute_masks_hex.sh "${S}" 512 518 3 1 6 per_frame mean 64 \
        flow 2 0 0 0 0 1 0 2 0 1 8)
  IDS="${IDS}:${J}"
done

if [ "${DRY}" != "1" ] && [ -n "${IDS}" ]; then
  for S in dynrep_0cde48 dynrep_216ba3; do
    [ -d "${MO}/gtmask/${S}/masks" ] || continue
    # afterany, not afterok: if one precompute dies the other scene's A/B should
    # still run. The probe refuses a mask directory shorter than the sequence.
    EVAL_DATE=gtm sbatch --job-name=pcd --dependency="afterany${IDS},singleton" \
      --time=04:00:00 --gres=gpu:1 \
      --export=ALL,DATA_ROOT="${BONN}",INTRINSICS=bonn \
      slurm_probe_hex.sh baseline "${MO}/${ADOPTED}" "${S}" 16 8 10 25 \
        "ours=${F} --dyn_mask_dir ${MO}/${ADOPTED}" \
        "gt=${F} --dyn_mask_dir ${MO}/gtmask" >/dev/null
    echo "  A/B for ${S} queued behind the precomputes -> *_${S}_gtm"
  done
fi

# ---------------------------------------------------------------- LANE 2 ----
# seq | mask set | data root | intrinsics
ROWS=(
"rgbd_bonn_balloon|cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid|${BONN}|bonn"
"rgbd_bonn_synchronous2|cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid|${BONN}|bonn"
"rgbd_bonn_crowd|cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16|${BONN}|bonn"
"rgbd_dataset_freiburg3_walking_xyz|cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16|${TUM}|tum_fr3"
"rgbd_dataset_freiburg3_walking_halfsphere|cs128_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16|${TUM}|tum_fr3"
)
n=0
for ROW in "${ROWS[@]}"; do
  IFS='|' read -r SEQ MASK ROOT INTR <<< "${ROW}"
  NF_SEQ=$(ls "${ROOT}/${SEQ}/rgb"/*.png "${ROOT}/${SEQ}/rgb"/*.jpg 2>/dev/null | wc -l)
  N_MASK=$(ls "${MO}/output_dyn_masks_precomputed_${MASK}/${SEQ}/masks"/*.png 2>/dev/null | wc -l)
  if [ "${NF_SEQ}" -eq 0 ] || [ "${N_MASK}" -lt "${NF_SEQ}" ]; then
    echo "SKIP ${SEQ}: frames=${NF_SEQ} masks=${N_MASK}"; continue; fi
  echo "ns replication ${SEQ}"
  n=$((n + 1))
  [ "${DRY}" = "1" ] && continue
  EVAL_DATE=ns5 sbatch --job-name=ns5 --dependency=singleton \
    --time=04:00:00 --gres=gpu:1 \
    --export=ALL,DATA_ROOT="${ROOT}",INTRINSICS="${INTR}" \
    slurm_probe_hex.sh baseline "${MO}/output_dyn_masks_precomputed_${MASK}" \
      "${SEQ}" 16 8 10 25 \
      "k0=${F}" "k2=${F} --dyn_nearest_source 2" >/dev/null
done

echo "----"
if [ "${DRY}" = "1" ]; then echo "DRY RUN -- nothing submitted."; exit 0; fi
echo "lane pcd: 2 precomputes then 2 mask A/Bs   |   lane ns5: ${n} replication job(s)"
echo
echo "in the morning:"
echo "  for A in ours gt; do for S in dynrep_0cde48 dynrep_216ba3; do python3 -c \"import json,sys;m=json.load(open(sys.argv[1]+'/metrics.json'));print(sys.argv[1][21:].ljust(28),'psnr %.3f dyn %.3f stat %.3f ssim %.4f lpips %.4f'%(m['psnr'],m['psnr_dynamic'],m['psnr_static'],m['ssim'],m['lpips']))\" output_probe_frozen_\${A}_\${S}_gtm; done; done"
echo "  for A in k0 k2; do for S in balloon synchronous2 crowd rgbd_dataset_freiburg3_walking_xyz rgbd_dataset_freiburg3_walking_halfsphere; do python3 -c \"import json,sys;m=json.load(open(sys.argv[1]+'/metrics.json'));print(sys.argv[1][21:].ljust(46),'psnr %.3f dyn %.3f stat %.3f ssim %.4f lpips %.4f'%(m['psnr'],m['psnr_dynamic'],m['psnr_static'],m['ssim'],m['lpips']))\" output_probe_frozen_\${A}_\${S}_ns5 2>/dev/null; done; done"
