#!/bin/bash
#SBATCH --job-name=btscore
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/btscore_%j.out
#SBATCH --error=slurm_logs/btscore_%j.err
# =============================================================================
# Phase 2 of the baseline table: pair our renders against each baseline's, score
# them under the SAME masks, and assemble one table plus render strips.
#
# A GPU is requested only because compare_masked's --lpips runs a perceptual net;
# everything else here is CPU. It is one short job, not one per row.
#
# Both baselines write pred_<idx>_<stem>.png, so build_movies_comparison serves
# both; its output subdirectory is always called "movies" whichever one it paired,
# which is why the label is passed separately to compare_masked.
# =============================================================================
set -uo pipefail

TAG="${EVAL_DATE:-bt}"
REPO="${REPO:-${HOME}/DynamicReconstructionSplat}"
MO="${HOME}/data/mask_out"
BONN="${HOME}/data/bonn/rgbd_bonn_dataset"
TUM="${HOME}/data/tum"
cd "${REPO}"
mkdir -p "cmp_${TAG}" slurm_logs

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate dynrec

A512="${A512:-output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid}"
A256="${A256:-output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16}"
A128="${A128:-output_dyn_masks_precomputed_cs128_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16}"

ROWS=(
"rgbd_bonn_balloon|balloon|dense|4 8 12 16 20|${BONN}|${A512}"
"rgbd_bonn_synchronous2|sync2|dense|4 8 12 16 20|${BONN}|${A512}"
"rgbd_bonn_crowd|crowd|plain|4 12 20 30 40|${BONN}|${A256}"
"rgbd_dataset_freiburg3_walking_xyz|tum_xyz|plain|4 8 12 16 20|${TUM}|${A256}"
"rgbd_dataset_freiburg3_walking_halfsphere|tum_halfsphere|plain|4 8 12 16 20|${TUM}|${A128}"
)

for ROW in "${ROWS[@]}"; do
  IFS='|' read -r SEQ KEY DENSE STRIDES ROOT MASK <<< "${ROW}"
  for S in ${STRIDES}; do
    # The probe strips the rgbd_bonn_ prefix but keeps TUM names whole.
    SHORT="${SEQ#rgbd_bonn_}"
    OURS="output_probe_frozen_ours_s${S}_${SHORT}_${TAG}/images"
    if [ ! -d "${OURS}" ]; then     # fall back to a glob if the spelling differs
      OURS=$(ls -d output_probe_frozen_ours_s${S}_*_${TAG}/images 2>/dev/null | head -1)
    fi
    if [ -z "${OURS}" ] || [ ! -d "${OURS}" ]; then
      echo "MISSING ours: ${KEY} s${S} (no output_probe_frozen_ours_s${S}_*_${TAG}/images)"
      continue
    fi
    [ "${DENSE}" = "dense" ] && BDIR="loo_${KEY}_dense_s${S}" || BDIR="loo_${KEY}_s${S}"

    for B in movies neoverse; do
      [ "${B}" = "movies" ] && SRC="${HOME}/MoVieS/out/${BDIR}"  || SRC="${HOME}/NeoVerse/out/${BDIR}"
      [ "${B}" = "movies" ] && LABEL="MoVieS"                     || LABEL="NeoVerse"
      [ -d "${SRC}" ] || { echo "MISSING baseline: ${SRC}"; continue; }

      OUT="cmp_${TAG}/${KEY}_${B}_s${S}"
      rm -rf "${OUT}"          # start clean: a half-paired run leaves files behind
      echo "=== PAIR ${KEY} s${S} vs ${LABEL}"
      # --ours_offset IS REQUIRED and is 8*stride. Our renders are named by their
      # WINDOW-START frame (b{start}_v08), while both baselines hold out frames//2 = view
      # 8, i.e. absolute frame start + 8*stride. Without the offset the two index spaces
      # do not overlap and the pairing silently reports `paired 0` -- or worse, a handful
      # of ACCIDENTAL collisions: measured, crowd s20 paired 94 (= (625-160)/5+1) and
      # crowd s40 paired 2 (= {320,325}), which still produced a plausible-looking table.
      python tools/build_movies_comparison.py \
        --movies_dir "${SRC}" --ours_images "${OURS}" \
        --bonn_rgb "${ROOT}/${SEQ}/rgb" \
        --ours_offset $(( 8 * S )) \
        --mask_dir "${MO}/${MASK}/${SEQ}/masks" \
        --out "${OUT}" 2>&1 | grep -E "paired|SKIPPED|indexed" || true

      echo "=== SCORE ${KEY} s${S} vs ${LABEL}   [seconds=$(awk -v s=${S} 'BEGIN{printf "%.1f", s/2}')]"
      python tools/compare_masked.py \
        --gt_dir "${OUT}/gt" --a "${OUT}/ours" --a_panels 2 --a_label ours \
        --b "${OUT}/movies" --b_label "${LABEL}" \
        --mask_dir "${OUT}/mask" --lpips \
        2>&1 | tee "slurm_logs/bt_${KEY}_${B}_s${S}.txt"
    done
  done
done

# ---- assemble one table -----------------------------------------------------
# Parsed from compare_masked's printed rows (overall/dynamic/static at 2dp,
# LPIPS/SSIM at 4dp) rather than re-derived, so the table cannot disagree with
# the logs it came from. LPIPS and SSIM are on the DYNAMIC CROP, not full frame.
python - "${TAG}" <<'PY'
import glob, re, sys, os
tag = sys.argv[1]
row_re = re.compile(r'^(overall|dynamic|static|LPIPS|SSIM)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+([+-]\d+\.\d+)')
recs = {}
for f in sorted(glob.glob("slurm_logs/bt_*_s*.txt")):
    m = re.match(r'slurm_logs/bt_(.+)_(movies|neoverse)_s(\d+)\.txt$', f)
    if not m:
        continue
    key, base, s = m.group(1), m.group(2), int(m.group(3))
    vals = {}
    for line in open(f):
        r = row_re.match(line.strip())
        if r:
            vals[r.group(1)] = (float(r.group(2)), float(r.group(3)), float(r.group(4)))
    if vals:
        recs[(key, base, s)] = vals

out = [f"# Baseline table ({tag})", "",
       "Ours vs MoVieS and NeoVerse. 16-frame window, leave-one-out on every view,",
       "both arms scored under the SAME (new) masks. Seconds = stride / 2 at 30 Hz.",
       "LPIPS and SSIM are on the dynamic-region crop. `A - B` is ours minus theirs,",
       "so positive is better for PSNR and SSIM, negative is better for LPIPS.", "",
       "⚠️ tum_halfsphere's masks have known false positives (people covered, but the",
       "ceiling and sometimes the table also marked). The comparison is still fair --",
       "both arms are scored under the same masks -- and overall/LPIPS/SSIM are",
       "mask-independent, but read its dynamic and static columns with that in mind.", ""]
for base, label in (("neoverse", "NeoVerse"), ("movies", "MoVieS")):
    out += [f"## vs {label}", "",
            "| sequence | s | sec | overall | dynamic | static | LPIPS | SSIM |",
            "|---|---|---|---|---|---|---|---|"]
    for (key, b, s) in sorted(recs, key=lambda t: (t[0], t[2])):
        if b != base:
            continue
        v = recs[(key, b, s)]
        def d(n, p=2):
            return f"{v[n][2]:+.{p}f}" if n in v else "—"
        out.append(f"| {key} | {s} | {s/2:.1f} | {d('overall')} | {d('dynamic')} | "
                   f"{d('static')} | {d('LPIPS',4)} | {d('SSIM',4)} |")
    out.append("")
out += [f"Rows parsed: {len(recs)}. Any stride absent above had no log, i.e. its render "
        f"or its baseline was missing — check the job output before reading the trend.", "",
        "## Pairing counts (read these FIRST)", "",
        "A row is only meaningful if it paired close to the baseline's frame count.",
        "", "| row | paired |", "|---|---|"]
import subprocess
# NEWEST log only. Globbing all of them replays every earlier attempt, so a failed
# run's "paired 0" rows reappear beside the good ones in the same table.
for f in sorted(glob.glob("slurm_logs/btscore_*.out"), key=os.path.getmtime)[-1:]:
    for line in open(f, errors="replace"):
        if line.startswith("=== PAIR"):
            cur = line.split("=== PAIR ")[1].strip()
        elif "paired" in line and "frames" in line:
            n = re.search(r'paired (\d+) frames', line)
            if n:
                out.append(f"| {cur} | {n.group(1)} |")
out.append("")
path = f"baseline_table_{tag}.md"
open(path, "w").write("\n".join(out))
print("\n".join(out))
print(f"\nwrote {path}")
PY

echo "=============================================="
echo "table -> baseline_table_${TAG}.md   |   per-row logs -> slurm_logs/bt_*.txt"
echo "render strips: compare_methods.py --gt_dir cmp_${TAG}/<key>_<base>_s<S>/gt \\"
echo "  --a cmp_${TAG}/<key>_<base>_s<S>/ours --a_panels 2 --a_label ours \\"
echo "  --b cmp_${TAG}/<key>_<base>_s<S>/movies --b_label <Label> --out fig_<key>_s<S> --fig_stride 4"
echo "Finished at: $(date)"
