# =============================================================================
# Which GPU this job runs on. Sourced by every slurm_*_hex.sh launcher.
#
# WHY THIS FILE EXISTS. Each launcher used to choose its own card:
#
#     PICK=$(nvidia-smi ... | sort -t, -k2 -nr | head -1 | cut -d, -f1)
#     export CUDA_VISIBLE_DEVICES=${PICK}
#
# -- "take whichever card has the most free memory right now". That RACES.
# Free memory at pick time reserves nothing, so two jobs starting within a few
# seconds of each other both see the same emptiest card and both take it, and
# whichever allocates second dies. It killed the entire ns5b run on 26 Nov 2026
# (6/6 arms OOM, "Process 4027929 has 29.79 GiB memory in use" on a card our
# own job had just measured as free) and is the likely cause of the earlier
# xyz n41s3 OOM as well.
#
# Slurm does not hand the same gres/gpu to two jobs. Its allocation is
# therefore the only choice that cannot collide, and the job here is to honour
# it rather than second-guess it.
#
# A BUSY ALLOCATED CARD IS NOW A HARD ERROR, not a reason to move. Moving is
# precisely what caused the collision: it makes one job trespass on a card
# another job was given. Callers keep their own MiB thresholds and messages.
#
# Sets for the caller:
#   CUDA_VISIBLE_DEVICES  the allocated card (left as slurm set it)
#   PICK                  index or list, for the caller's log line
#   FREE                  MiB free on the card torch will actually open
#
# FREE comes from torch.cuda.mem_get_info rather than nvidia-smi because it is
# read THROUGH CUDA_VISIBLE_DEVICES, so it describes the card the job will
# really use under every renumbering scheme, cgroup-isolated or not, which
# nvidia-smi's physical indices do not. Requires the conda env to be active --
# true in all six launchers, which activate before sourcing this.
# =============================================================================
if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Slurm exports one of these; first non-empty wins. A list ("0,1") keeps its
  # order and torch opens the first entry as device 0.
  _GPU_ENV="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-}}}}"
  if [ -n "${_GPU_ENV}" ]; then
    export CUDA_VISIBLE_DEVICES="${_GPU_ENV}"
    PICK="${_GPU_ENV}"
    echo "[gpu] honouring slurm allocation: CUDA_VISIBLE_DEVICES=${PICK}"
  else
    PICK="cuda-default"
    echo "[gpu] slurm job ${SLURM_JOB_ID} exported no gpu variable; leaving selection to cuda"
  fi
  unset _GPU_ENV
else
  # No scheduler, so nothing else is reserving cards and the emptiest-card pick
  # is still the right behaviour for an interactive run.
  PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES=${PICK}
  echo "[gpu] no scheduler: picked emptiest card ${PICK}"
fi

FREE=$(python -c "import torch;print(int(torch.cuda.mem_get_info(0)[0])//1048576)" 2>/dev/null)
case "${FREE}" in ''|*[!0-9]*) FREE="" ;; esac
if [ -z "${FREE}" ]; then
  # No torch / no driver: fall back to the SMALLEST free figure among visible
  # cards. Conservative on purpose -- it may refuse a run that would have fit,
  # which is cheaper than an OOM three hours in.
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
         | sort -n | head -1 | tr -d ' ')
  case "${FREE}" in ''|*[!0-9]*) FREE=0 ;; esac
  echo "[gpu] torch unavailable; nvidia-smi floor across visible cards: ${FREE} MiB"
fi
