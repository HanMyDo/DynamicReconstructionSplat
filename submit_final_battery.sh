#!/bin/bash
# =============================================================================
# Submit the FULL comparison battery, unattended.
#
# Per sequence, three jobs chained by slurm dependency:
#   1. otsu2 mask precompute   (SKIPPED if that sequence's masks already exist)
#   2. VANILLA  AnySplat + VGGT, no dynamics handling      <- the "before"
#   3. OURS     VGGT4D + flow-gated compositing + 4D PLY   <- the "after"
#
# Both evals cover the WHOLE sequence (one render per window) and use the same
# masks, so every metric column is comparable and the frames line up 1:1 for
# make_comparison_figure. The PLY window defaults to the middle of each
# sequence, which is valid whatever its length -- do not hard-code an index
# here, a window outside the range silently writes no PLY.
#
# The mechanism is training-free (frozen head), so there is no train/test
# contamination and ANY sequence is a valid eval.
#
# SERIAL=1 runs the whole battery on ONE GPU: every job gets a shared name and
# --dependency=singleton, so slurm starts at most one at a time. Note you CANNOT
# get this by cancelling jobs -- whenever you are under the QOS cap slurm just
# promotes the next queued job into the free slot. It has to be set at submit.
#
# USAGE:  ./submit_final_battery.sh [SEQ ...]
#         EVAL_DATE=final ./submit_final_battery.sh            # 2 GPUs, ~5-6 h
#         SERIAL=1 EVAL_DATE=final ./submit_final_battery.sh   # 1 GPU,  ~10-12 h
# =============================================================================
set -uo pipefail
REPO="${HOME}/DynamicReconstructionSplat"; cd "${REPO}"
# WHOLE-SEQUENCE detection (chunk 512 = one pass, one KMeans, one Otsu), which is
# parity with demo_vggt4d.process_scene. The old cs64 masks were our chunking
# deviation. Measured on balloon, 30 windows, same flags both sides:
#   cs64  psnr 20.62 dyn 16.71 static 21.91 lpips_dyn 0.3731 dynfrac 0.141
#   cs512 psnr 20.73 dyn 17.27 static 23.81 lpips_dyn 0.3260 dynfrac 0.290
# Better on every metric at DOUBLE the mask fraction -- under-masking is the
# expensive error, not over-masking. Static gains most (+1.90) because with the
# person fully covered none of them ghosts into the static bucket, while a wrongly
# masked chair only renders own-frame instead of V times: thinner, but in the right
# place.
M2="${HOME}/data/mask_out/output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_otsu2_glob"
F="--track_dynamic --dyn_motion_knn 8 --dyn_motion_strict --dyn_motion_pred_bandwidth 1.5 --dyn_motion_tracker raft --dyn_motion_max_disp_mult 3.0 --dyn_opacity_comp 1.0"
# ADOPTED Sep 2026 (probe on balloon, 30 windows, vs the same config without them):
#   --dyn_opacity_comp 1.0   +0.87 psnr / +1.70 dyn / -0.027 lpips_dyn; rendered
#     alpha inside the dynamic mask 0.909 -> 0.968 against 0.979 static. The gate
#     removes contributors the pretrained head sized its opacities for; this puts
#     the alpha back. Needs --per_frame_dynamic, so it rides in F, not BG.
#   --bg_color 1 1 1         +0.43 psnr / +0.84 static / better on BOTH lpips
#     columns, -0.54 on dyn psnr. Upstream AnySplat renders on white and the head
#     was fitted against it. MUST be on BOTH arms or the comparison is decided by
#     fill colour: at oc0 the same swap costs 2.24 dB of dynamic psnr, and it only
#     becomes a net win once compensation has shrunk the transmittance it fills.
BG="--bg_color 1 1 1"
VID="--image_batch_start 0 --max_image_batches 99999 --image_views 0"
# --ply_own_frame_only: one crisp copy per timestamp. Without it the file holds all
# V copies stacked along the trajectory, which reads as overlapping ghosts and no
# defined person. Oversized splats are dropped by export_ply now (1.9% of gaussians
# carried 31% of the visible haze), so no post-hoc prune_ply pass is needed.
PLY="--ply_per_frame --ply_own_frame_only"
export EVAL_DATE="${EVAL_DATE:-final}"

# One GPU at a time: singleton is per (user, job name), so a shared name serialises
# everything. Combined with afterok via a comma, which slurm ANDs.
SERIAL="${SERIAL:-0}"
NAME=""
[ "${SERIAL}" = "1" ] && NAME="--job-name=chain"
mkdep () {   # $1 = job id to wait for, or empty
  local d=""
  [ -n "${1:-}" ] && d="afterok:$1"
  if [ "${SERIAL}" = "1" ]; then
    if [ -n "${d}" ]; then d="singleton,${d}"; else d="singleton"; fi
  fi
  [ -n "${d}" ] && echo "--dependency=${d}"
}

if [ "$#" -gt 0 ]; then SEQS="$*"; else
  SEQS="rgbd_bonn_balloon rgbd_bonn_synchronous2 rgbd_bonn_removing_obstructing_box rgbd_bonn_placing_obstructing_box"
fi

for SEQ in ${SEQS}; do
  [ -d "${HOME}/data/bonn/rgbd_bonn_dataset/${SEQ}/rgb" ] || { echo "SKIP ${SEQ}: no rgb/"; continue; }
  # PARTIAL masks are worse than none: eval does not fail on a missing frame, it
  # silently falls back to LIVE detection, so the sequence is scored under a
  # different protocol than the others and the comparison is quietly invalid.
  # Require one mask per rgb frame, not merely "some masks exist".
  NRGB=$(ls "${HOME}/data/bonn/rgbd_bonn_dataset/${SEQ}/rgb"/*.png 2>/dev/null | wc -l)
  N=$(ls "${M2}/${SEQ}/masks"/*.png 2>/dev/null | wc -l)
  JID=""
  if [ "${N}" -lt "${NRGB}" ]; then
    [ "${N}" -gt 0 ] && echo "${SEQ}: INCOMPLETE masks (${N}/${NRGB}) -> regenerating"
    JID=$(sbatch --parsable ${NAME} $(mkdep) slurm_precompute_masks_hex.sh \
            "${SEQ}" 64 518 3 1 6 per_frame mean 64 attention 2)
    echo "${SEQ}: masks -> job ${JID}"
  else
    echo "${SEQ}: ${N}/${NRGB} masks already present"
  fi

  A=$(sbatch --parsable ${NAME} $(mkdep "${JID}") slurm_eval_hex.sh baseline \
        "--no_vggt4d --frame_stride 4 --dyn_mask_dir ${M2} ${BG} ${VID}" "${SEQ}" 16)
  B=$(sbatch --parsable ${NAME} $(mkdep "${JID}") slurm_eval_hex.sh baseline \
        "--frame_stride 4 --dyn_mask_dir ${M2} --per_frame_dynamic ${F} ${BG} ${VID} ${PLY}" "${SEQ}" 16)
  echo "${SEQ}: vanilla -> job ${A} | ours -> job ${B}"
done

TAG=$(echo ${SEQS} | tr ' ' '\n' | sed 's/rgbd_bonn_//' | tr '\n' ' ')
cat <<TXT

Submitted. When they finish, build each figure with:
  for S in ${TAG}; do
    sbatch slurm_compare_hex.sh \\
      output_eval_frozen_vggt_s4_bg111_pcm_nf16_\${S}_${EVAL_DATE} \\
      output_eval_frozen_pfd_s4_flow8raftsb1p5_cl3p0_bg111_oc1p0_pcm_nf16_\${S}_${EVAL_DATE} \\
      cmp_\${S} "AnySplat + VGGT (vanilla)" "ours: VGGT4D + flow-gated" 1
  done
TXT
