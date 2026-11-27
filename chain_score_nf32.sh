#!/bin/bash
# =============================================================================
# Pair and score OUR nf32 arm against MoVieS and NeoVerse at the matched
# operating point (32 frames, stride 4, window_step 5).
#
# WHY A NEW SCRIPT. score_baseline_table.sh walks a STRIDE LADDER and hardcodes
# two things that are wrong here:
#   * --ours_offset 8*stride. That is frames//2 for 16-frame windows. At nf32 it
#     is 16*stride = 64. A stale 8 does not fail, it pairs NOTHING (measured:
#     0 frames vs 3 on the same smoke data), or pairs accidental collisions that
#     still draw a plausible table.
#   * loo_<key>[_dense]_s<stride> directory names. The nf32 baselines are in
#     nf32_<key>_s4.
#
# ⚠️ DYNAMIC REPLICA HAS NO MoVieS ARM ON PURPOSE. MoVieS needs an intrinsics
# preset and only has bonn and tum_fr3; Dynamic Replica is neither, and the
# runner's own help warns the wrong preset "silently handicaps the baseline".
# NeoVerse is pose-free and takes no intrinsics, so it covers all seven.
#
# ⚠️ BOTH ARMS ARE SCORED UNDER OUR MASKS. psnr_dynamic/psnr_static are not
# comparable across mask sets, so one set has to serve both; ours describes
# where the person actually is. Say so in the writeup.
#
# USAGE:  ./chain_score_nf32.sh            # tag rb2
#         TAG=rb3 ./chain_score_nf32.sh
# =============================================================================
set -uo pipefail
TAG="${TAG:-rb2}"
S=4
HALF=16                      # frames//2 at nf32 -- see above
REPO="${REPO:-${HOME}/DynamicReconstructionSplat}"
MO="${HOME}/data/mask_out"
BONN="${HOME}/data/bonn/rgbd_bonn_dataset"
TUM="${HOME}/data/tum"
A512="output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid"
A256="output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16"
A128="output_dyn_masks_precomputed_cs128_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16"
cd "${REPO}"; mkdir -p slurm_logs "cmp_${TAG}"

# seq | baseline key | data root | mask set
ROWS=(
"rgbd_bonn_balloon|balloon|${BONN}|${A512}"
"rgbd_bonn_synchronous2|sync2|${BONN}|${A512}"
"rgbd_bonn_crowd|crowd|${BONN}|${A256}"
"rgbd_dataset_freiburg3_walking_xyz|tum_xyz|${TUM}|${A256}"
"rgbd_dataset_freiburg3_walking_halfsphere|tum_halfsphere|${TUM}|${A128}"
"dynrep_0cde48|dynrep_0cde48|${BONN}|${A512}"
"dynrep_216ba3|dynrep_216ba3|${BONN}|${A512}"
)

for ROW in "${ROWS[@]}"; do
  IFS='|' read -r SEQ KEY ROOT MASK <<< "${ROW}"
  SHORT="${SEQ#rgbd_bonn_}"
  OURS="output_probe_frozen_ours_${SHORT}_${TAG}/images"
  [ -d "${OURS}" ] || { echo "MISSING ours: ${OURS}"; continue; }
  for B in neoverse movies; do
    [ "${B}" = "movies" ] && SRC="${HOME}/MoVieS/out/nf32_${KEY}_s4" \
                          || SRC="${HOME}/NeoVerse/out/nf32_${KEY}_s4"
    [ "${B}" = "movies" ] && LABEL="MoVieS" || LABEL="NeoVerse"
    [ -d "${SRC}" ] || { echo "MISSING baseline: ${SRC}"; continue; }
    OUT="cmp_${TAG}/${KEY}_${B}_s${S}"
    rm -rf "${OUT}"           # a half-paired run leaves files behind
    echo "=== PAIR ${KEY} vs ${LABEL}"
    python tools/build_movies_comparison.py \
      --movies_dir "${SRC}" --ours_images "${OURS}" \
      --bonn_rgb "${ROOT}/${SEQ}/rgb" \
      --ours_offset $(( HALF * S )) \
      --mask_dir "${MO}/${MASK}/${SEQ}/masks" \
      --out "${OUT}" 2>&1 | grep -E "paired|SKIPPED|indexed" \
        | tee "slurm_logs/nf32_${KEY}_${B}_s${S}.txt" || true
    # NOTE the pair step WRITES the log and the score step APPENDS to it: the
    # "paired N" count lives in the pairing output, and the table needs it in the
    # same file it parses, or every row reports n=?.
    echo "=== SCORE ${KEY} vs ${LABEL}"
    python tools/compare_masked.py \
      --gt_dir "${OUT}/gt" --a "${OUT}/ours" --a_panels 2 --a_label ours \
      --b "${OUT}/movies" --b_label "${LABEL}" \
      --mask_dir "${OUT}/mask" --lpips \
      2>&1 | tee -a "slurm_logs/nf32_${KEY}_${B}_s${S}.txt"
  done
done

# ---- one table, parsed from the logs so it cannot disagree with them --------
python - "${TAG}" <<'PY'
import glob, re, sys
tag = sys.argv[1]
row_re = re.compile(r'^(overall|dynamic|static|LPIPS|SSIM)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+([+-]\d+\.\d+)')
pair_re = re.compile(r'paired (\d+) frames')
recs = {}
for f in sorted(glob.glob("slurm_logs/nf32_*_s4.txt")):
    key = f.split("nf32_", 1)[1].rsplit("_s4.txt", 1)[0]
    seq, base = key.rsplit("_", 1)
    d = recs.setdefault((seq, base), {})
    for line in open(f):
        m = row_re.match(line.strip())
        if m:
            d[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        p = pair_re.search(line)
        if p:
            d["n"] = int(p.group(1))
out = ["# Ours vs baselines at nf32 stride 4 (" + tag + ")", "",
       "Deltas are OURS MINUS BASELINE; for LPIPS, negative favours us.",
       "⚠️ both arms scored under OUR masks. ⚠️ MoVieS has no Dynamic Replica arm",
       "(no correct intrinsics preset).", "",
       "| sequence | baseline | n | overall | dynamic | static | SSIM | LPIPS |",
       "|---|---|---|---|---|---|---|---|"]
for (seq, base), d in sorted(recs.items()):
    if not d.get("overall"):
        continue
    c = [f"{d[k][2]:+.2f}" if k in d else "—" for k in ("overall", "dynamic", "static")]
    c += [f"{d[k][2]:+.4f}" if k in d else "—" for k in ("SSIM", "LPIPS")]
    out.append(f"| {seq} | {base} | {d.get('n','?')} | " + " | ".join(c) + " |")
out += ["", "⛔ Do not quote a row with n < 50: below that the LOO sample has been",
        "shown to flip sign (the sparse 6 s row read -0.48 and became +0.70 dense)."]
open(f"nf32_table_{tag}.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
PY
echo "wrote nf32_table_${TAG}.md"
