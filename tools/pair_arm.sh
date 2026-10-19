#!/bin/bash
# Pair ONE of our arms against both baselines at ONE stride, for a figure.
#
# score_baseline_table.sh does this for the whole table; this is the one-off
# version, for arms that are not the adopted config -- e.g. the plain
# AnySplat + vanilla VGGT baseline, which has no dynamics handling at all.
#
# ⚠️ --ours_offset 8*STRIDE is REQUIRED: our renders are named by WINDOW-START
# frame while both baselines hold out frames//2 = start + 8*stride. Omitting it
# pairs 0 frames, or worse, a handful of accidental index collisions that still
# produce a plausible-looking figure.
#
# USAGE: ./tools/pair_arm.sh <probe_tag> <arm_name> <stride> <out_tag>
#   ./tools/pair_arm.sh van vanilla 4 cmp_van
set -uo pipefail
TAG=${1:?probe EVAL_DATE}; ARM=${2:?arm name}; S=${3:?stride}; OUT=${4:?output dir}
MO="${HOME}/data/mask_out"; BONN="${HOME}/data/bonn/rgbd_bonn_dataset"; TUM="${HOME}/data/tum"
A512="${MO}/output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid"
A256="${MO}/output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16"
A128="${MO}/output_dyn_masks_precomputed_cs128_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16"
# seq | baseline key | dense? | data root | mask set
ROWS=(
"rgbd_bonn_balloon|balloon|dense|${BONN}|${A512}"
"rgbd_bonn_synchronous2|sync2|dense|${BONN}|${A512}"
"rgbd_bonn_crowd|crowd|plain|${BONN}|${A256}"
"rgbd_dataset_freiburg3_walking_xyz|tum_xyz|plain|${TUM}|${A256}"
"rgbd_dataset_freiburg3_walking_halfsphere|tum_halfsphere|plain|${TUM}|${A128}"
)
mkdir -p "${OUT}"
for ROW in "${ROWS[@]}"; do
  IFS='|' read -r SEQ KEY DENSE ROOT MASK <<< "${ROW}"
  SHORT="${SEQ#rgbd_bonn_}"
  OURS="output_probe_frozen_${ARM}_${SHORT}_${TAG}/images"
  [ -d "${OURS}" ] || { echo "MISSING ${OURS}"; continue; }
  [ "${DENSE}" = "dense" ] && BD="loo_${KEY}_dense_s${S}" || BD="loo_${KEY}_s${S}"
  for B in movies neoverse; do
    [ "${B}" = "movies" ] && SRC="${HOME}/MoVieS/out/${BD}" || SRC="${HOME}/NeoVerse/out/${BD}"
    [ -d "${SRC}" ] || { echo "MISSING baseline ${SRC}"; continue; }
    D="${OUT}/${KEY}_${B}_s${S}"; rm -rf "${D}"
    echo "=== ${KEY} s${S} vs ${B}"
    python tools/build_movies_comparison.py --movies_dir "${SRC}" \
      --ours_images "${OURS}" --bonn_rgb "${ROOT}/${SEQ}/rgb" \
      --ours_offset $(( 8 * S )) --mask_dir "${MASK}/${SEQ}/masks" \
      --out "${D}" 2>&1 | grep -E "paired|indexed" || true
  done
done
