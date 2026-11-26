# =============================================================================
# Which GPU this job runs on. Sourced by every slurm_*_hex.sh launcher.
#
# WHAT THIS NODE ACTUALLY DOES (measured 26 Nov 2026, job 5114):
#   CVD=[0]  JOB_GPUS=[1]  device_count()==1
# hex-4gpu cgroup-isolates GPUs per job. A gpu:1 job sees exactly ONE card,
# renumbered to 0 whatever its physical index. Two consequences:
#
#   1. The old pick -- "take whichever card nvidia-smi says is emptiest" --
#      was a NO-OP. It enumerated the single visible card and chose it. It was
#      never selecting between cards, so it also never raced for one.
#   2. A job CANNOT route around a busy card. The alternatives are invisible.
#
# AND YET A CARD CAN BE SHARED. Slurm gave job 5114 physical GPU 1 while
# alexander's job 5104 was already resident on it (41782 MiB, 95% util, pid
# 4073126, envs/sgmmor). Slurm's gres accounting and the devices the cgroup
# exposes do not agree on this node, so "allocated to us" does not mean "ours
# alone". This is what killed the ns5b run: at 02:06 the co-resident process
# held 29.79 GiB, our 12 GiB guard passed on the ~17 GiB that were left, the
# other process grew to 41 GiB, and all 6 arms died at
# anysplat.py:1456 pad_tensor_list. Nothing we did caused it and no choice of
# card inside a gpu:1 job can avoid it.
#
# SO THERE ARE EXACTLY TWO DEFENCES, and this file implements both:
#
#   A. Measure free memory on the card we will REALLY open, before the run, and
#      refuse to start if it is short. torch.cuda.mem_get_info reads through
#      CUDA_VISIBLE_DEVICES, so it is right under any renumbering; nvidia-smi's
#      physical indices are not. A co-resident process that GROWS can still
#      kill us later -- the guard only catches one that is already large.
#   B. Ask for more than one card (sbatch --gres=gpu:2) and use the emptiest OF
#      OUR OWN ALLOCATION. This is the only way to dodge a squatter, and it is
#      race-free because every candidate is already ours. Never look outside
#      the allocation: that WOULD be a race, and it would trespass on another
#      job's card.
#
# Sets for the caller:
#   CUDA_VISIBLE_DEVICES  the chosen card
#   PICK                  a label for the caller's log line
#   FREE                  MiB free on the card torch will actually open
#
# Requires the conda env to be active -- true in all six launchers, which
# activate before sourcing this.
# =============================================================================
if [ -n "${SLURM_JOB_ID:-}" ]; then
  _GPU_ENV="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-}}}}"
  if [ -n "${_GPU_ENV}" ]; then
    export CUDA_VISIBLE_DEVICES="${_GPU_ENV}"
    PICK="slurm ${_GPU_ENV}"
  else
    PICK="cuda-default"
  fi
  unset _GPU_ENV
  echo "[gpu] allocation: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} (SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset})"
else
  # No scheduler: nothing is reserving cards, so choosing one is ours to do.
  PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES=${PICK}
  PICK="manual ${PICK}"
  echo "[gpu] no scheduler: picked emptiest card ${CUDA_VISIBLE_DEVICES}"
fi

# Defence B. One python call reports the visible count and the emptiest index,
# so a single CUDA init covers both. Indices are relative to the allocation we
# already hold, which is exactly what re-exporting CUDA_VISIBLE_DEVICES selects.
_SEL=$(python -c "import torch;n=torch.cuda.device_count();b=max(range(n),key=lambda i:torch.cuda.mem_get_info(i)[0]) if n else 0;print(n,b,int(torch.cuda.mem_get_info(b)[0])//1048576 if n else 0)" 2>/dev/null)
# NOT `set -- ${_SEL}`: this file is sourced at top level, and the launchers
# rely on their own positional parameters (slurm_probe_hex.sh does `shift 7`
# and then iterates "$@" for the arm specs), so clobbering $@ here would
# silently drop every arm of every probe.
_NGPU=$(echo "${_SEL}" | cut -d' ' -f1)
_BEST=$(echo "${_SEL}" | cut -d' ' -f2)
_FREE=$(echo "${_SEL}" | cut -d' ' -f3)
case "${_NGPU}" in ''|*[!0-9]*) _NGPU=0 ;; esac
if [ "${_NGPU}" -gt 1 ]; then
  export CUDA_VISIBLE_DEVICES="${_BEST}"
  PICK="${PICK} -> ${_BEST} (emptiest of ${_NGPU} allocated)"
  echo "[gpu] ${_NGPU} cards allocated; using the emptiest of ours: ${_BEST}"
fi
FREE="${_FREE}"
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
unset _SEL _NGPU _BEST _FREE
