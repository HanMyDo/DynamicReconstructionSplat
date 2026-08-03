#!/bin/sh
#SBATCH --job-name=orig_vggt4d
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --output=slurm_logs/orig_vggt4d_20260802_%j.out
#SBATCH --error=slurm_logs/orig_vggt4d_20260802_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=bonn,heidelberg,muenchen,stuttgart,koblenz
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Han-My.Do@tum.de

# =============================================================================
# GROUND-TRUTH TEST — run the PURE, UNMODIFIED original VGGT4D demo_vggt4d.py on a
# strided box clip, to compare its dynamic masks against OUR precompute's masks.
#   original masks ALSO bad  -> confirms a VGGT4D METHOD limitation on Bonn (our
#                               Stage-1 port is faithful, so this is the finding)
#   original masks GOOD      -> a hidden runtime/numerical bug remains in our path
#
# This is the "output" check that the line-by-line code audit can't give on its own.
#
# PREREQ — copy the ORIGINAL repo to the cluster ONCE (from your LOCAL machine):
#   scp -r ~/Dokumente/MA_WS2526/Masterthesis/Repos/VGGT4D hanmydo@head:~/VGGT4D
#
# USAGE:  sbatch slurm_original_vggt4d_demo_20260802.sh [SEQUENCE] [N_FRAMES]
#   $1 SEQUENCE  default rgbd_bonn_moving_nonobstructing_box
#   $2 N_FRAMES  strided frames fed to the original AT ONCE (default 32; spans the
#                whole seq. If it OOMs at 518, rerun with 24 or 16.)
# =============================================================================

SEQUENCE=${1:-rgbd_bonn_moving_nonobstructing_box}
N_FRAMES=${2:-32}
VGGT4D_REPO=${VGGT4D_REPO:-$HOME/VGGT4D}
OUR_REPO=/mnt/home/hanmydo/DynamicReconstructionSplat
OUR_CKPT=${OUR_REPO}/ckpts/vggt4d_model_tracker_fixed_e20.pt

export ENROOT_RUNTIME_PATH=/tmp/$USER/runtime
export ENROOT_CACHE_PATH=/tmp/$USER/cache
export ENROOT_DATA_PATH=/tmp/$USER/data
export TMPDIR=/tmp
mkdir -p $ENROOT_RUNTIME_PATH $ENROOT_CACHE_PATH $ENROOT_DATA_PATH slurm_logs

echo "=============================================="
echo "ORIGINAL VGGT4D demo — ${SEQUENCE}  (${N_FRAMES} strided frames)"
echo "repo: ${VGGT4D_REPO}   node: $(hostname)   $(date)"
echo "=============================================="

if [ ! -d "$VGGT4D_REPO" ]; then
  echo "ERROR: original repo not found at $VGGT4D_REPO"
  echo "  scp -r <local>/VGGT4D hanmydo@head:~/VGGT4D"
  exit 1
fi

# --- checkpoint where the demo hardcodes it (./ckpts/model_tracker_fixed_e20.pt) ---
mkdir -p ${VGGT4D_REPO}/ckpts
ln -sf ${OUR_CKPT} ${VGGT4D_REPO}/ckpts/model_tracker_fixed_e20.pt

# --- extract the sequence's rgb frames ---
BONN=/tmp/bonn_orig_${SLURM_JOB_ID}
mkdir -p ${BONN}
python3 -c "
import zipfile
prefix='rgbd_bonn_dataset/${SEQUENCE}/rgb/'
with zipfile.ZipFile('/mnt/projects/theses/dynrecsplat/rgbd_bonn_dataset.zip') as z:
    m=[x for x in z.namelist() if x.startswith(prefix)]
    z.extractall('${BONN}', m); print('extracted', len(m), 'rgb files')
"

# --- build the strided input dir  <input>/<scene>/*.png  spanning the whole seq ---
INPUT=/tmp/orig_input_${SLURM_JOB_ID}
SCENE_IN=${INPUT}/${SEQUENCE}
mkdir -p ${SCENE_IN}
python3 -c "
import glob, os, shutil
src=sorted(glob.glob('${BONN}/rgbd_bonn_dataset/${SEQUENCE}/rgb/*.png'))
n=len(src); k=${N_FRAMES}; stride=max(1, n//k)
pick=src[::stride][:k]
for p in pick: shutil.copy(p, os.path.join('${SCENE_IN}', os.path.basename(p)))
print(f'{n} frames -> {len(pick)} strided (stride {stride}) spanning the whole sequence')
"

OUT=${OUR_REPO}/output_original_vggt4d_demo/${SEQUENCE}_n${N_FRAMES}
mkdir -p ${OUT}

CONTAINER=orig_vggt4d_${SLURM_JOB_ID}
enroot remove -f ${CONTAINER} 2>/dev/null || true
enroot create --name ${CONTAINER} ~/anysplat.sqsh

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp ${CONTAINER} bash -c "
  cd ${VGGT4D_REPO}
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH=${VGGT4D_REPO}:\$PYTHONPATH
  # Container already has torch/cv2/open3d-ish deps via our requirements.txt; the
  # original repo only additionally needs open3d + evo (evo = pose-saving in store.py,
  # unused for masks but a top-level import). Same inline-install pattern as our slurm.
  pip install open3d evo --quiet
  python --version && nvidia-smi
  echo ''
  echo '=== running ORIGINAL demo_vggt4d.py (unmodified) ==='
  python demo_vggt4d.py --input_dir ${INPUT} --output_dir ${OUT}

  echo '=== building red overlays (mask on strided RGB) for easy comparison ==='
  python3 - <<PY
import glob, os, cv2, numpy as np
scene='${SEQUENCE}'
outdir=os.path.join('${OUT}', scene)
frames=sorted(glob.glob(os.path.join('${SCENE_IN}','*.png')))
masks=sorted(glob.glob(os.path.join(outdir,'dynamic_mask_*.png')))
ov=os.path.join(outdir,'overlays'); os.makedirs(ov, exist_ok=True)
for i,mp in enumerate(masks):
    if i>=len(frames): break
    img=cv2.imread(frames[i]); m=cv2.imread(mp,0)
    if img is None or m is None: continue
    m=cv2.resize(m,(img.shape[1],img.shape[0]))
    red=img.copy(); red[m>127]=(0,0,255)
    cv2.imwrite(os.path.join(ov,os.path.basename(frames[i])), cv2.addWeighted(img,0.6,red,0.4,0))
print('wrote', len(masks), 'overlays ->', ov)
PY
"

enroot remove -f ${CONTAINER}
rm -rf ${BONN} ${INPUT}

echo ""
echo "=============================================="
echo "ORIGINAL VGGT4D masks -> ${OUT}/${SEQUENCE}/dynamic_mask_*.png"
echo "ORIGINAL overlays     -> ${OUT}/${SEQUENCE}/overlays/"
echo "Compare against OUR overlays. Original ALSO bad => VGGT4D method limit on Bonn."
echo "Finished: $(date)"
echo "=============================================="
