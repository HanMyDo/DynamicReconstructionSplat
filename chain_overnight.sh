#!/bin/bash
# =============================================================================
# ⛔ MEASURED AND REJECTED, Oct 20 2026. Run as `bt5` over all five sequences:
# it FIXED tum_halfsphere (mean dynamic fraction 0.436 -> 0.169, whose masks
# really were broken) but REGRESSED balloon (0.109 -> 0.343, renders visibly
# worse: a mottled white wall where background gaussians took the dynamic path)
# and tum_xyz (0.192 -> 0.323). The adopted masks remain the Oct 15 set, which
# is what chain_baseline_table.sh and score_baseline_table.sh DEFAULT to; this
# script only overrides them through the environment, so the default path is
# unaffected and nothing here needs reverting.
#
# WHY IT CANNOT BE FIXED BY PICKING A BETTER STATISTIC. MAD measures the bulk
# and is blind to the tail by construction -- which is exactly what made it
# immune to real motion and fixed halfsphere. But balloon's false positives LIVE
# in that tail: a textureless white wall gives unreliable depth, so the
# camera-induced flow computed through it is wrong and the residual is large
# with nothing moving. No statistic over the raw residual separates "moving
# person" from "depth error on a blank wall". Balloon got 2.000 px where Otsu
# said 4.387. A fix would have to normalise the residual by the LOCAL
# camera-induced flow, which is what the false positives scale with.
#
# AND THE TOLERANCE IS GONE ANYWAY: completion multiplies seeds into final area
# by 2.7x (halfsphere), 3.8x (crowd), 5.7x (balloon), so a 2x threshold error
# becomes a 3-6x mask error. That is the argument for replacing the completion
# stage, and the reason tuning the threshold above it cannot work.
#
# Kept runnable because the precompute wiring is correct and reusable; set a
# different recipe before using it.
# =============================================================================

# =============================================================================
# ONE NIGHT: new masks on all five sequences -> the 25-arm table -> the figure.
#
# Stage 1, five mask precomputes. The recipe is the adopted one plus the two
# things measured on Oct 19:
#   --mask_thresh_mad 4   multi-Otsu assumes a multi-modal histogram; the flow
#                         residual is unimodal and heavy-tailed, so its split
#                         landed anywhere (0.41..14.97 across one sequence's
#                         chunks) and usually too high, discarding most of the
#                         person. median + 4*MAD held 1.98..2.75 instead and cut
#                         empty seeds 35.8% -> 8.2%.
#   --mask_thresh_min 2   a fast object CLOSE to the camera carries a halo of
#                         intermediate residual; where a chunk's background is
#                         quiet the adaptive threshold drops under it and the
#                         seed spills onto the wall (crowd's quiet chunks, 1.12
#                         and 1.22, while the same K is correct at 2.30 on a
#                         close walker). The floor is RAFT's own error.
#   --sam_support_frac    drop a propagated object-frame that was seeded nowhere
#                         nearby; worth 0.2447 -> 0.1920 mean area on halfsphere.
#
# ONE LANE, and that is deliberate. The first attempt ran two lanes and the two
# chunk-512 sequences (balloon, synchronous2) landed on the node together: both
# were Killed inside three minutes by the HOST OOM killer, not the GPU. The
# adopted recipe had only ever run these one at a time, so the limit had never
# been hit. Host memory scales with chunk size, and we cannot predict the
# headroom for a 512 beside a 256, so the lane is serial: five hours instead of
# three, against a failed night costing a day. The second GPU stays free.
#
# Stage 2 is a tiny CPU job that waits for all five and then runs
# chain_baseline_table.sh with the new mask sets exported. It runs INSIDE a job
# so the night does not need anyone awake; the table chain then queues its own
# scoring and figure jobs behind itself. chain_baseline_table.sh refuses a
# sequence whose mean dynamic fraction exceeds MAX_AREA, so a mask set that runs
# away costs that row and not the other four.
#
# USAGE:  ./chain_overnight.sh          # submit
#         DRY=1 ./chain_overnight.sh    # print what would be submitted
# =============================================================================
set -uo pipefail

TAG="${EVAL_DATE:-bt5}"
DRY="${DRY:-0}"
REPO="${REPO:-${HOME}/DynamicReconstructionSplat}"
MO="${HOME}/data/mask_out"
BONN="${HOME}/data/bonn/rgbd_bonn_dataset"
TUM="${HOME}/data/tum"
cd "${REPO}"
mkdir -p slurm_logs

SUF="_mad4_sup0p03w10_fl2p0"          # what args 25..28 below name the output

# seq | chunk | anchors | data root
# Anchors follow the adopted recipe: 8 (the default, so no name token) for the
# two short Bonn sequences, 16 for everything over ~500 frames -- a 900-frame
# sequence prompted with 8 is prompted half as densely.
ROWS=(
"rgbd_bonn_balloon|512|8|${BONN}"
"rgbd_bonn_synchronous2|512|8|${BONN}"
"rgbd_bonn_crowd|256|16|${BONN}"
"rgbd_dataset_freiburg3_walking_xyz|256|16|${TUM}"
"rgbd_dataset_freiburg3_walking_halfsphere|128|16|${TUM}"
)

IDS=""; n=0
for ROW in "${ROWS[@]}"; do
  IFS='|' read -r SEQ CHUNK ANCH ROOT <<< "${ROW}"
  [ -d "${ROOT}/${SEQ}/rgb" ] || { echo "SKIP ${SEQ}: no rgb/ under ${ROOT}"; continue; }
  n=$((n+1))
  echo "precompute ${SEQ}  chunk=${CHUNK} anchors=${ANCH}"
  [ "${DRY}" = "1" ] && continue
  J=$(sbatch --parsable --job-name=pcn --dependency=singleton \
      --export=ALL,DATA_ROOT="${ROOT}" \
      slurm_precompute_masks_hex.sh "${SEQ}" "${CHUNK}" 518 3 1 6 per_frame mean 64 \
        flow 2 0 0 0 0 1 0 2 0 1 "${ANCH}" 0 0 0 4 10 0.03 2.0)
  #                                                    ^ --save_residual OFF. The
  # panels settled the threshold question today, and at chunk 512 they hold a
  # full-chunk copy of both the residual and the depth on the host (~400MB each),
  # which helped push the first attempt into the OOM killer. Re-run one sequence
  # with arg 24 = 1 if a mask set looks wrong in the morning.
  IDS="${IDS}:${J}"
done

# The three mask sets chain_baseline_table.sh and score_baseline_table.sh read.
# Same three-way split as the adopted recipe (512 / 256 / 128), so overriding
# these three names is the whole change -- the chains themselves are unmodified
# except for the area guard and the figure job.
A512="output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid${SUF}"
A256="output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16${SUF}"
A128="output_dyn_masks_precomputed_cs128_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16${SUF}"

echo "----"
if [ "${DRY}" = "1" ]; then
  echo "DRY RUN -- nothing submitted."
  echo "  A512=${A512}"
  echo "  A256=${A256}"
  echo "  A128=${A128}"
  exit 0
fi
echo "submitted ${n} precompute(s)${IDS}"

# `.` not `source`: --wrap runs under sh, where source is not a builtin. That
# silently skipped conda activation in every Track4World job until it was caught.
DJ=$(sbatch --parsable --job-name=btdrv --dependency="afterany${IDS}" \
     --partition=gpu --cpus-per-task=2 --time=00:30:00 \
     --output=slurm_logs/btdrv_%j.out --error=slurm_logs/btdrv_%j.err \
     --wrap ". /opt/miniforge3/etc/profile.d/conda.sh && conda activate dynrec && cd ${REPO} && A512='${A512}' A256='${A256}' A128='${A128}' EVAL_DATE='${TAG}' ./chain_baseline_table.sh")
echo "driver job ${DJ} waits for them, then submits the table (tag ${TAG}),"
echo "which queues scoring and the five-column figure behind itself."
echo
echo "morning check:"
echo "  cat ${REPO}/baseline_table_${TAG}.md"
echo "  ls ~/figs_${TAG}.tgz"
echo "  grep -h dyn_fraction_mean ${MO}/*${SUF}/*/meta.json"
