#!/bin/bash
# Pair ONE of our arms against both baselines at ONE stride, for a figure.
#
# score_baseline_table.sh does this for the whole table; this is the one-off
# version, for arms that are not the adopted config -- e.g. the plain
# AnySplat + vanilla VGGT baseline, which has no dynamics handling at all.
#
# ⚠️ --ours_offset HALF*STRIDE is REQUIRED: our renders are named by WINDOW-START
# frame while both baselines hold out frames//2 = start + HALF*stride. Omitting it
# pairs 0 frames, or worse, a handful of accidental index collisions that still
# produce a plausible-looking figure.
#
# ⚠️ HALF IS frames//2 AND IT IS NOT ALWAYS 8. It was 8 while every run used
# 16-frame windows. At the nf32 operating point adopted Nov 2026 it is 16, and a
# stale 8 silently pairs nothing. Set it explicitly whenever --frames is not 16.
#
# USAGE: ./tools/pair_arm.sh <probe_tag> <arm_name> <stride> <out_tag>
#   ./tools/pair_arm.sh van vanilla 4 cmp_van
#   HALF=16 BD_FMT='nf32_%s_s4' ./tools/pair_arm.sh rb ours 4 cmp_nf32
#
# HALF    frames//2 of the BASELINE windows (default 8, i.e. 16-frame windows).
# BD_FMT  printf pattern for the baseline output dir, %s = the baseline key
#         (balloon, sync2, crowd, tum_xyz, tum_halfsphere). Default keeps the
#         historical loo_<key>[_dense]_s<stride> naming.
set -uo pipefail
TAG=${1:?probe EVAL_DATE}; ARM=${2:?arm name}; S=${3:?stride}; OUT=${4:?output dir}
HALF="${HALF:-8}"; BD_FMT="${BD_FMT:-}"
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
"dynrep_0cde48|dynrep_0cde48|plain|${BONN}|${A512}"
"dynrep_216ba3|dynrep_216ba3|plain|${BONN}|${A512}"
)
mkdir -p "${OUT}"
for ROW in "${ROWS[@]}"; do
  IFS='|' read -r SEQ KEY DENSE ROOT MASK <<< "${ROW}"
  SHORT="${SEQ#rgbd_bonn_}"
  OURS="output_probe_frozen_${ARM}_${SHORT}_${TAG}/images"
  [ -d "${OURS}" ] || { echo "MISSING ${OURS}"; continue; }
  if [ -n "${BD_FMT}" ]; then BD=$(printf "${BD_FMT}" "${KEY}")
  elif [ "${DENSE}" = "dense" ]; then BD="loo_${KEY}_dense_s${S}"
  else BD="loo_${KEY}_s${S}"; fi
  for B in movies neoverse; do
    [ "${B}" = "movies" ] && SRC="${HOME}/MoVieS/out/${BD}" || SRC="${HOME}/NeoVerse/out/${BD}"
    [ -d "${SRC}" ] || { echo "MISSING baseline ${SRC}"; continue; }
    D="${OUT}/${KEY}_${B}_s${S}"; rm -rf "${D}"
    echo "=== ${KEY} s${S} vs ${B}"
    python tools/build_movies_comparison.py --movies_dir "${SRC}" \
      --ours_images "${OURS}" --bonn_rgb "${ROOT}/${SEQ}/rgb" \
      --ours_offset $(( HALF * S )) --mask_dir "${MASK}/${SEQ}/masks" \
      --out "${D}" 2>&1 | grep -E "paired|indexed" || true
  done
done
