#!/bin/bash
# =============================================================================
# --src_window_dyn 2 vs 4, five sequences x two horizons, ONE variable.
#
# WHY. Oct 23 on balloon, 25 windows: 2 beat the adopted 4 on FOUR of five
# metrics (psnr 18.933 v 18.623, static 20.476 v 19.602, ssim 0.7621 v 0.7516,
# lpips 0.3063 v 0.3194), costing dynamic 14.782 v 15.418 -- and the renders
# agreed: the background is visibly sharper, and much of what read as a "soft
# person" was soft EVERYTHING. At 1 the person goes transparent, so the useful
# range is narrow and 2 is the candidate.
#
# That measurement was taken WITH --dyn_motion_fill on and on ONE sequence, and
# the fill has since been rejected on renders. So this re-runs the comparison
# clean: fill OFF, F otherwise byte-identical to the adopted config, both arms
# in one allocation so they cannot drift apart through a cluster change.
#
# It is the only positive from a day in which four separate coverage mechanisms
# each raised dynamic PSNR and each lost on the renders -- and notably it comes
# from the opposite direction, using LESS information per mover rather than more.
#
# Each arm is ~1 minute for 25 windows, so the whole sweep is well under an
# hour. One lane (job name sw2 + singleton), which also queues it behind the
# TUM d2/d4 jobs already submitted under that name.
#
# USAGE:  ./chain_srcwin.sh          # submit
#         DRY=1 ./chain_srcwin.sh    # print only
# =============================================================================
set -uo pipefail

TAG="${EVAL_DATE:-sw2}"
DRY="${DRY:-0}"
REPO="${REPO:-${HOME}/DynamicReconstructionSplat}"
MO="${HOME}/data/mask_out/output_dyn_masks_precomputed"
BONN="${HOME}/data/bonn/rgbd_bonn_dataset"
TUM="${HOME}/data/tum"
cd "${REPO}"
mkdir -p slurm_logs

# The adopted config MINUS --src_window_dyn, which is the swept variable, and
# WITHOUT --dyn_motion_fill (measured and rejected on renders, Oct 23).
F="--per_frame_dynamic --dyn_conf_opacity 4.0 --track_dynamic --dyn_motion_knn 8 \
--dyn_motion_strict --dyn_motion_pred_bandwidth 1.5 --dyn_motion_tracker raft \
--dyn_motion_max_disp_mult 3.0 --dyn_opacity_comp 1.0 --bg_color 1 1 1 \
--eval_loo --image_views 8 --dyn_unsupported static --dyn_opacity_gain 4 \
--dyn_unsup_gain 0.25 --render_max_scale_frac 0.011 --image_size 392 518 \
--batch_stride 5"

# seq | mask set (the ADOPTED Oct 15 recipe) | data root | intrinsics
ROWS=(
"rgbd_bonn_balloon|cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid|${BONN}|bonn"
"rgbd_bonn_synchronous2|cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid|${BONN}|bonn"
"rgbd_bonn_crowd|cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16|${BONN}|bonn"
"rgbd_dataset_freiburg3_walking_xyz|cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16|${TUM}|tum_fr3"
"rgbd_dataset_freiburg3_walking_halfsphere|cs128_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16|${TUM}|tum_fr3"
)

IDS=""; n=0
for ROW in "${ROWS[@]}"; do
  IFS='|' read -r SEQ MASK ROOT INTR <<< "${ROW}"
  NF_SEQ=$(ls "${ROOT}/${SEQ}/rgb"/*.png "${ROOT}/${SEQ}/rgb"/*.jpg 2>/dev/null | wc -l)
  N_MASK=$(ls "${MO}_${MASK}/${SEQ}/masks"/*.png 2>/dev/null | wc -l)
  if [ "${NF_SEQ}" -eq 0 ] || [ "${N_MASK}" -lt "${NF_SEQ}" ]; then
    echo "SKIP ${SEQ}: frames=${NF_SEQ} masks=${N_MASK}"; continue
  fi
  for S in 4 8; do
    # Start at window 10 and take 25: at batch_stride 5 that is frames 50..170,
    # plus the window's own span (15*S), so the longest case here needs 290 of
    # synchronous2's 358. Starting at 0 would risk an empty room on balloon.
    echo "OK   ${SEQ} s${S}  frames=${NF_SEQ} masks=${N_MASK}"
    n=$((n + 1))
    [ "${DRY}" = "1" ] && continue
    J=$(EVAL_DATE="${TAG}" sbatch --parsable --job-name=sw2 --dependency=singleton \
        --time=04:00:00 --gres=gpu:1 \
        --export=ALL,DATA_ROOT="${ROOT}",INTRINSICS="${INTR}" \
        slurm_probe_hex.sh baseline "${MO}_${MASK}" "${SEQ}" 16 "${S}" 10 25 \
          "d2_s${S}=${F} --src_window_dyn 2" \
          "d4_s${S}=${F} --src_window_dyn 4")
    IDS="${IDS}:${J}"
  done
done

echo "----"
if [ "${DRY}" = "1" ]; then
  echo "DRY RUN -- nothing submitted. ${n} job(s) would run (2 arms each)."
  exit 0
fi
echo "submitted ${n} job(s)${IDS}"

# A one-page verdict waiting in the morning, rather than 20 metrics.json files.
# `.` not `source`: --wrap runs under sh, where source is not a builtin.
SJ=$(sbatch --parsable --job-name=sw2sum --dependency="afterany${IDS}" \
     --partition=gpu --cpus-per-task=2 --time=00:20:00 \
     --output=slurm_logs/sw2sum_%j.out --error=slurm_logs/sw2sum_%j.err \
     --wrap ". /opt/miniforge3/etc/profile.d/conda.sh && conda activate dynrec && cd ${REPO} && python tools/summarize_srcwin.py --tag ${TAG}")
echo "summary job ${SJ} -> srcwin_${TAG}.md"
echo
echo "in the morning:  cat ${REPO}/srcwin_${TAG}.md"
