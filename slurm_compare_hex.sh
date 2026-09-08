#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/compare_%j.out
#SBATCH --error=slurm_logs/compare_%j.err
# =============================================================================
# Compare two FINISHED eval runs: metrics table + side-by-side figure + tarball.
#
# CPU ONLY -- deliberately no --gres, so this never takes a card away from a
# training or eval job. It exists because the login node has no PIL (it lives in
# the conda env) and work is not supposed to run there anyway.
#
# USAGE: sbatch slurm_compare_hex.sh DIR_A DIR_B [OUT] [LABEL_A] [LABEL_B] [STRIDE]
#   sbatch slurm_compare_hex.sh output_eval_ft_pfd_s4_pcm_nf16_balloon_newvid \
#                               output_eval_ft_pfd_s4_pcm_nf16_balloon_otsu2 \
#                               cmp_otsu "otsu1 masks" "otsu2 masks"
#
# READ ONLY psnr AND lpips WHEN THE TWO RUNS USED DIFFERENT MASKS. The mask
# DEFINES the static/dynamic split, so psnr_dynamic / psnr_static / lpips_dynamic
# are computed over different pixels in the two runs and their difference is not
# a change in quality. The table prints that warning with the numbers.
# =============================================================================
set -uo pipefail
A=${1:?usage: sbatch slurm_compare_hex.sh DIR_A DIR_B [OUT] [LABEL_A] [LABEL_B] [STRIDE]}
B=${2:?missing DIR_B}; OUT=${3:-cmp}; LA=${4:-A}; LB=${5:-B}; STRIDE=${6:-8}

REPO="${HOME}/DynamicReconstructionSplat"; cd ${REPO}; mkdir -p slurm_logs
for d in "${A}" "${B}"; do
  [ -d "${d}/images" ] || { echo "ERROR: no ${d}/images -- did that eval write images?"; exit 1; }
done

source /opt/miniforge3/etc/profile.d/conda.sh; conda activate dynrec

python - "${A}" "${B}" "${LA}" "${LB}" <<'PY'
import json, os, sys
a_dir, b_dir, la, lb = sys.argv[1:5]
ks = ["psnr", "lpips", "psnr_static", "psnr_dynamic", "lpips_dynamic"]
rows = []
for lbl, d in ((la, a_dir), (lb, b_dir)):
    p = os.path.join(d, "metrics.json")
    if not os.path.exists(p):
        print("MISSING", p)
        continue
    rows.append((lbl, json.load(open(p))))
cell = lambda m, k: f"{m[k]:15.4f}" if m.get(k) is not None else f"{'-':>15s}"
print(f"{'run':24s}" + "".join(f"{k:>15s}" for k in ks))
for lbl, m in rows:
    print(f"{lbl:24s}" + "".join(cell(m, k) for k in ks))
if len(rows) == 2:
    a, b = rows[0][1], rows[1][1]
    print(f"{'delta (B-A)':24s}" + "".join(
        f"{b[k]-a[k]:+15.4f}" if a.get(k) is not None and b.get(k) is not None
        else f"{'-':>15s}" for k in ks))
    print("\nnote: if the two runs used DIFFERENT masks, only psnr and lpips are comparable.")
    print("      the mask defines the static/dynamic split, so the other three columns")
    print("      measure different pixels in each run.")
PY

# Build into a temp dir and swap at the end. Deleting ${OUT} up front means a
# rerun that is cancelled midway leaves a PARTIAL directory where a complete one
# used to be -- and the frame count looks plausible, so it is only noticed later
# when the video turns out to be a quarter of the sequence.
rm -rf "${OUT}.tmp"
python make_comparison_figure.py --a "${A}/images" --b "${B}/images" --out "${OUT}.tmp" \
  --label_a "${LA}" --label_b "${LB}" || { echo "ERROR: figure build failed"; rm -rf "${OUT}.tmp"; exit 1; }
rm -rf "${OUT}" && mv "${OUT}.tmp" "${OUT}"

N=$(ls "${OUT}"/*.png 2>/dev/null | wc -l)
echo "figure frames: ${N}"
SEL=$(ls "${OUT}"/*.png 2>/dev/null | awk "(NR-1) % ${STRIDE} == 0" | head -40)
[ -z "${SEL}" ] && { echo "ERROR: no frames matched (are the filenames shared between runs?)"; exit 1; }
tar czf "${OUT}.tgz" ${SEL}
echo "wrote ${OUT}.tgz ($(echo "${SEL}" | wc -l) of ${N} frames)"
echo
echo "scp hanmydo@172.21.192.113:~/DynamicReconstructionSplat/${OUT}.tgz ."
