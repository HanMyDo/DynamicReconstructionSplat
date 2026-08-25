#!/bin/sh
# =============================================================================
# Build the PRESENTATION FIGURES: one Gaussian .ply per (sequence x model), all
# from a SINGLE window so vanilla and ours are the same 6 frames, same poses,
# same masks -- only the model differs. Anything visible is then the model.
#
# Runs 6 single-window evals (~2 min each). The QOS allows only 2 submitted jobs
# at a time, so this submits ONE AT A TIME and waits, rather than failing with
# QOSMaxSubmitJobPerUserLimit. Total ~15 min.
#
# The .ply always comes from the LAST processed window, so --max_image_batches 1
# at --image_batch_start N makes the export exactly window N.
#
# USAGE:  nohup sh make_ply_figures.sh > ply_figs.log 2>&1 &   <-- survives a dropped
#           connection; plain `sh make_ply_figures.sh` dies with the shell (the already
#           submitted slurm job keeps running, but the loop stops). Re-running is safe:
#           finished outputs are skipped.
#         sh make_ply_figures.sh          (submits everything, waits, prints scp)
#         sh make_ply_figures.sh --dry    (print what it would do)
# =============================================================================

DRY=0
[ "$1" = "--dry" ] && DRY=1

CKPT=output_train_testbed_dw0p0_lr2e-5_pcm_nf6_loo_sr0_tw0_shr0p0_lpips0p05_dh_dc1p0_anc1p0_20260706/checkpoint_best_ep5_20260820-0325_psnr27.87dB.pt
MASKS=output_dyn_masks_precomputed_cs16_r518_st3_fs0
TAG=20260825fig
COMMON="--eval_loo --frame_stride 8 --gain_correct --dyn_mask_dir ${MASKS} --max_image_batches 1"

if [ ${DRY} -eq 0 ] && [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  echo "(run this ON THE CLUSTER; --dry works anywhere)"
  exit 1
fi

# sequence : window : why that window
#   synchronous2             200  two people moving, dyn pixel fraction ~0.086
#   removing_obstructing_box 500  mid-sequence, person carrying the box
#   placing_obstructing_box  500  the flat cardboard face -- a PLANE, so per-frame
#                                 depth disagreement is unmistakable (best slide)
SEQS="synchronous2:200 removing_obstructing_box:500 placing_obstructing_box:500"

submit_and_wait() {
  _seq=$1; _win=$2; _mode=$3      # _mode: ours | vanilla
  if [ "${_mode}" = "vanilla" ]; then
    _ck="baseline"; _extra="--no_vggt4d"
  else
    _ck="${CKPT}";  _extra=""
  fi
  _flags="${_extra} ${COMMON} --image_batch_start ${_win}"
  if [ "${_mode}" = "vanilla" ]; then
    _out="output_eval_frozen_vggt_loo_s8_gc_pcm_nf6_${_seq}_${TAG}"
  else
    _out="output_eval_ft_loo_s8_gc_pcm_nf6_${_seq}_${TAG}"
  fi
  echo ">>> ${_mode} / ${_seq} @ window ${_win}"
  # IDEMPOTENT: a finished job leaves gaussians.ply. Re-running the script after an
  # interruption (dropped VPN kills the foreground loop, though not the slurm jobs)
  # then only does the work that is actually missing.
  if [ -f "${_out}/gaussians.ply" ]; then
    echo "    already done -> ${_out}/gaussians.ply (skipping)"
    return
  fi
  if [ ${DRY} -eq 1 ]; then
    echo "    EVAL_DATE=${TAG} sbatch slurm_eval_compositing_20260711.sh \"${_ck}\" \"${_flags}\" rgbd_bonn_${_seq} 6"
    return
  fi
  _jid=$(EVAL_DATE=${TAG} sbatch --parsable --qos=students_normal \
           slurm_eval_compositing_20260711.sh "${_ck}" "${_flags}" rgbd_bonn_${_seq} 6)
  if [ -z "${_jid}" ]; then echo "    submit FAILED"; return; fi
  echo "    job ${_jid} ... waiting"
  while squeue -j "${_jid}" -h 2>/dev/null | grep -q .; do sleep 15; done
  echo "    done: $(sacct -j ${_jid} --format=State -n | head -1)"
}

for entry in ${SEQS}; do
  s=$(echo "${entry}" | cut -d: -f1)
  w=$(echo "${entry}" | cut -d: -f2)
  submit_and_wait "${s}" "${w}" vanilla
  submit_and_wait "${s}" "${w}" ours
done

echo
echo "=============================================================="
echo "DONE. Pull them locally with DISTINCT names (identical file sizes"
echo "make a mixed-up pair invisible in a viewer -- md5sum to be sure):"
echo
echo "  mkdir -p ~/Downloads/ply_figs && cd ~/Downloads/ply_figs"
echo "  R=/mnt/home/hanmydo/DynamicReconstructionSplat"
echo "  H=hanmydo@131.159.11.60"
for entry in ${SEQS}; do
  s=$(echo "${entry}" | cut -d: -f1); w=$(echo "${entry}" | cut -d: -f2)
  echo "  scp \$H:\$R/output_eval_frozen_vggt_loo_s8_gc_pcm_nf6_${s}_${TAG}/gaussians.ply ./vanilla_${s}_b${w}.ply"
  echo "  scp \$H:\$R/output_eval_ft_loo_s8_gc_pcm_nf6_${s}_${TAG}/gaussians.ply          ./ours_${s}_b${w}.ply"
done
echo "  md5sum *.ply    # all six must differ"
echo "=============================================================="
