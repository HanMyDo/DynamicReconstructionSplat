#!/bin/bash
# =============================================================================
# THE HORIZON LADDER AT nf32, MATCHED INPUTS.
#
# WHY. The only baseline claim the thesis has is horizon-dependent: we lose at
# short spans and cross beyond ~6 s. The Nov 27 matched-input table measured a
# single rung -- nf32 stride 4, a 4.1 s span -- and we lost to NeoVerse 1/7.
# That is the rung where we were ALREADY KNOWN to lose, so it tests nothing.
# This walks the rungs that actually bracket the crossing.
#
# ⚠️ AT nf32 A WINDOW SPANS 31*S FRAMES, i.e. ~S seconds at 30 Hz -- NOT S/2 as
# in the old 16-frame ladder. Every "seconds" figure in the older notes has to
# be re-derived before it is compared with anything here.
#
# ⚠️ SEQUENCE LENGTH CAPS THE RUNG (counts from sequence-lengths memo):
#   sync2 358 -> 8 | balloon 439 -> 12 | crowd 928, removing 963, TUM -> 16+
# A row whose sequence is too short for a rung is skipped, not silently run on
# two windows.
#
# ⚠️ BASELINES ARE CAPPED AT --max_windows 60. We need ~50 paired frames per
# row, not the whole sequence; uncapped runs cost hours and change no number.
#
# ⛔ MoVieS gets no Dynamic Replica arm (no correct intrinsics preset) and its
# runner MUST have --poses vggt (the gt default is the 13.4 -> 23.4 dB trap)
# and MUST run from ~/MoVieS (its sys.path.append(".")).
#
# USAGE:  ./chain_nf32_ladder.sh          # submit
#         DRY=1 ./chain_nf32_ladder.sh    # print the plan only
#         RUNGS="8 12" ./chain_nf32_ladder.sh
# =============================================================================
set -uo pipefail
DRY="${DRY:-0}"
RUNGS="${RUNGS:-8 12 16}"
REPO="${REPO:-${HOME}/DynamicReconstructionSplat}"
MO="${HOME}/data/mask_out"
BONN="${HOME}/data/bonn/rgbd_bonn_dataset"
TUM="${HOME}/data/tum"
A512="output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid"
A256="output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16"
A128="output_dyn_masks_precomputed_cs128_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16"
cd "${REPO}"; mkdir -p slurm_logs

# The adopted config minus the two flags that scale with window size.
F="--per_frame_dynamic --dyn_conf_opacity 4.0 --track_dynamic --dyn_motion_knn 8 \
--dyn_motion_strict --dyn_motion_pred_bandwidth 1.5 --dyn_motion_tracker raft \
--dyn_motion_max_disp_mult 3.0 --dyn_opacity_comp 1.0 --bg_color 1 1 1 \
--eval_loo --dyn_unsupported static --dyn_opacity_gain 4 --dyn_unsup_gain 0.25 \
--render_max_scale_frac 0.011 --image_size 392 518 --batch_stride 5 \
--dyn_nearest_source 4 --image_views 16 --src_window_dyn 8"

# seq | key | root | intrinsics | mask | max rung
ROWS=(
"rgbd_bonn_crowd|crowd|${BONN}|bonn|${A256}|16"
"rgbd_bonn_removing_obstructing_box|removing|${BONN}|bonn|${A256}|16"
"rgbd_dataset_freiburg3_walking_xyz|tum_xyz|${TUM}|tum_fr3|${A256}|16"
"rgbd_dataset_freiburg3_walking_halfsphere|tum_halfsphere|${TUM}|tum_fr3|${A128}|16"
"rgbd_bonn_balloon|balloon|${BONN}|bonn|${A512}|12"
"rgbd_bonn_synchronous2|sync2|${BONN}|bonn|${A512}|8"
)

sub () { [ "${DRY}" = "1" ] && { echo "    DRY: $*" >&2; echo "0"; return; }; sbatch --parsable "$@"; }

for S in ${RUNGS}; do
  IDS=""; n=0
  echo "=== rung stride ${S}  (span $(awk -v s=${S} 'BEGIN{printf "%.1f", 31*s/30}') s)"
  for ROW in "${ROWS[@]}"; do
    IFS='|' read -r SEQ KEY ROOT INTR MASK MAXS <<< "${ROW}"
    [ "${S}" -gt "${MAXS}" ] && { echo "  skip ${KEY}: rung ${S} > max ${MAXS}"; continue; }
    NF_SEQ=$(ls "${ROOT}/${SEQ}/rgb"/*.png "${ROOT}/${SEQ}/rgb"/*.jpg 2>/dev/null | wc -l)
    NMASK=$(ls "${MO}/${MASK}/${SEQ}/masks"/*.png 2>/dev/null | wc -l)
    if [ "${NF_SEQ}" -eq 0 ] || [ "${NMASK}" -lt "${NF_SEQ}" ]; then
      echo "  SKIP ${KEY}: frames=${NF_SEQ} masks=${NMASK}"; continue; fi
    # Need the window span plus room for ~50 sliding starts.
    if [ "$(( 31 * S + 250 ))" -gt "${NF_SEQ}" ]; then
      echo "  SKIP ${KEY} s${S}: needs $(( 31 * S + 250 )) frames, has ${NF_SEQ}"; continue; fi
    echo "  ${KEY} s${S}  frames=${NF_SEQ}"
    n=$((n + 1))

    J=$(EVAL_DATE="lad${S}" sub --job-name=lad --dependency=singleton --time=08:00:00 \
        --gres=gpu:2 --export=ALL,DATA_ROOT="${ROOT}",INTRINSICS="${INTR}" \
        slurm_probe_hex.sh baseline "${MO}/${MASK}" "${SEQ}" 32 "${S}" 10 250 "ours=${F}")
    IDS="${IDS}:${J}"

    J=$(sub --job-name=lad --dependency=singleton --partition=gpu --gres=gpu:2 \
        --cpus-per-task=8 --time=04:00:00 \
        --output="slurm_logs/nv_${KEY}_s${S}_%j.out" --error="slurm_logs/nv_${KEY}_s${S}_%j.out" \
        --wrap ". /opt/miniforge3/etc/profile.d/conda.sh; conda activate neoverse; \
cd ${REPO}; . ./slurm_gpu_env.sh; python tools/neoverse_bonn_loo.py \
--seq_dir ${ROOT}/${SEQ} --out \${HOME}/NeoVerse/out/nf32_${KEY}_s${S} \
--frames 32 --stride ${S} --window_step 5 --max_windows 60")
    IDS="${IDS}:${J}"

    case "${SEQ}" in dynrep_*) ;; *)
      J=$(sub --job-name=lad --dependency=singleton --partition=gpu --gres=gpu:2 \
          --cpus-per-task=8 --time=04:00:00 \
          --output="slurm_logs/mv_${KEY}_s${S}_%j.out" --error="slurm_logs/mv_${KEY}_s${S}_%j.out" \
          --wrap ". /opt/miniforge3/etc/profile.d/conda.sh; conda activate movies; \
. ${REPO}/slurm_gpu_env.sh; cd \${HOME}/MoVieS; python ${REPO}/tools/movies_bonn_loo.py \
--seq_dir ${ROOT}/${SEQ} --out \${HOME}/MoVieS/out/nf32_${KEY}_s${S} \
--frames 32 --stride ${S} --window_step 5 --max_windows 60 --poses vggt --intrinsics ${INTR}")
      IDS="${IDS}:${J}" ;;
    esac
  done

  if [ "${DRY}" != "1" ] && [ -n "${IDS}" ]; then
    # afterany, not afterok: one dead sequence should not cost the whole rung.
    sub --job-name=ladscore --dependency="afterany${IDS}" --partition=gpu --gres=gpu:1 \
      --cpus-per-task=8 --time=03:00:00 \
      --output="slurm_logs/score_lad${S}_%j.out" --error="slurm_logs/score_lad${S}_%j.out" \
      --wrap ". /opt/miniforge3/etc/profile.d/conda.sh; conda activate dynrec; \
cd ${REPO}; . ./slurm_gpu_env.sh; TAG=lad${S} S=${S} ./chain_score_nf32.sh" >/dev/null
    echo "  -> ${n} sequence(s), scoring queued -> nf32_table_lad${S}.md"
  fi
done

echo "----"
[ "${DRY}" = "1" ] && { echo "DRY RUN -- nothing submitted."; exit 0; }
echo "in the morning:  cat ${REPO}/nf32_table_lad*.md"
