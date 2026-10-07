# Reproducing the figures

Every figure below comes from the SAME code and the SAME masks. What differs is
the flags. Nothing here needs a fine-tuned checkpoint: `baseline` means the
frozen head, which is what every reported number uses.

Shared setup (cluster):

```
cd ~/DynamicReconstructionSplat && git pull
D=~/data/mask_out/output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_otsu2_glob
F="--per_frame_dynamic --track_dynamic --dyn_motion_knn 8 --dyn_motion_strict \
   --dyn_motion_pred_bandwidth 1.5 --dyn_motion_tracker raft \
   --dyn_motion_max_disp_mult 3.0 --dyn_opacity_comp 1.0 --bg_color 1 1 1"
```

`D` is the adopted mask set: whole-sequence detection, chunk 512 (one pass =
parity with `demo_vggt4d.process_scene`), `--global_post`, otsu 2. The two
~1000-frame box sequences do not fit at 512 and use chunk 256; `global_post`
still thresholds over the whole sequence, so the part that mattered survives.
Protocol: one pass where the sequence fits, largest feasible chunk otherwise.

`--gres=gpu:2` on any sbatch below is only needed while a stray process is
holding GPU 0. It lets the script's "pick the freest card" avoid the blocked one.

## 1. Render comparison (GT | vanilla | ours)

The battery writes both arms, then one CPU-only job builds the panels.

```
SERIAL=1 EVAL_DATE=final3 ./submit_final_battery.sh
sbatch slurm_compare_hex.sh \
  output_eval_frozen_vggt_s4_bg111_pcm_nf16_balloon_final3 \
  output_eval_frozen_pfd_s4_flow8raftsb1p5_cl3p0_bg111_oc1p0_pcm_nf16_balloon_final3 \
  cmp_balloon_final3 "AnySplat + VGGT (vanilla)" "ours: VGGT4D + flow-gated" 1
```

`make_comparison_figure.py` shares the GT panel between runs, so the three
columns are frame-aligned by construction. The full set of frames is in
`cmp_balloon_final3/`; the `.tgz` is only a strided sample.

## 2. Per-timestamp point cloud (ONE person per timestamp)

```
B="$F --ply_per_frame --ply_own_frame_only --ply_dyn_keep_frac 0.25 --image_size 392 518"
EVAL_DATE=native sbatch slurm_probe_hex.sh baseline "$D" rgbd_bonn_balloon 6 4 12 6 "native=$B"
```

-> `CORRECT_FINAL_1WINDOW_native/`. The person comes from frame j alone, so it is
a 2.5D shell: clean and easy to read, but NOT what the renderer does.

## 3. Accumulated point cloud (what the renderer actually uses)

```
B="$F --ply_per_frame --ply_dyn_keep_frac 0.25 --image_size 392 518"
EVAL_DATE=real4d sbatch slurm_probe_hex.sh baseline "$D" rgbd_bonn_balloon 6 12 12 6 "real4d=$B"
```

-> `real4d/`. Dropping `--ply_own_frame_only` relocates EVERY frame's dynamic
gaussians onto timestamp j by scene flow and drops the ones tracking cannot
place, which is the rule the renderer follows. Fuller object, but measurably more
spread: concentration (fraction of dynamic gaussians within 0.3 world of their
median) is 15.5% here against 31.7% for (2), because relocated copies carry
displacement error. Stride 12 rather than 4 so the six timestamps span ~2.4 s and
the motion is visible; at stride 4 they span 0.7 s and nothing moves.

## 4. Baseline comparison vs MoVieS (the horizon curve)

Parameterised by sequence, so it runs unchanged on any Bonn-layout sequence --
including Dynamic Replica once `tools/dynrep_to_bonn.py` has converted it.

**Prerequisite -- the `movies` env** (once; validated by their own DAVIS clip
scoring 29.12 self-recon / 26.10 LOO through this exact install, so a low Bonn
number is Bonn, not the install):

- torch **2.11.0+cu128** -- their pinned 2.5.1+cu124 cannot target Blackwell sm_120.
- gsplat **1.5.3 prebuilt** -- 1.5.0 as their `setup.sh` pins cannot build here
  (system nvcc 13.4 against torch's 12.8).
- `from __future__ import annotations` at the top of `kiui/op.py` (NameError
  otherwise).
- `bg_color = bg_color[0]` in `gs_util.py`: gsplat 1.5.3 derives `image_dims` from
  `means2d.shape[:-2]`, which is EMPTY on the single-camera path, so the background
  must be a bare `(3,)`.
- torch-scatter must be rebuilt with `--no-binary` (the wheel is a stale ABI).

Neither patch affects numerics.

**Step 1 -- our renders.** Already produced by the `final4` battery in section 1:
nf16 with `--image_views 0`, which writes one `GT|pred` panel per frame. The
comparison indexes these by frame number, so this must cover the sequence.
⚠️ On synchronous2 it covers frames 0-296 of 358; held-out frames past the end are
reported as `SKIPPED` and dropped from both sides.

**Step 2 -- MoVieS leave-one-out at matched horizons.** Cluster, env `movies`:

```
SEQ=synchronous2
cd ~/MoVieS
for S in 4 8 12; do
  sed "s|rgbd_bonn_dataset/rgbd_bonn_[a-z0-9_]*|rgbd_bonn_dataset/rgbd_bonn_$SEQ|; \
       s|--stride [0-9]*|--stride $S|; s|--window_step [0-9]*|--window_step 5|; \
       s|out/loo_[A-Za-z0-9_]*|out/loo_${SEQ}_dense_s$S|; \
       s|--gres=gpu:2|--gres=gpu:1|" run_loo.sh > run_${SEQ}_$S.sh
done
grep -h -e gres -e "\-\-stride" -e window_step -e "out/" run_${SEQ}_*.sh
P=""
for S in 4 8 12; do P=$(sbatch --parsable ${P:+--dependency=afterany:$P} run_${SEQ}_$S.sh); echo "stride $S -> $P"; done
```

**Chained on purpose**: `--dependency=afterany` means exactly one job runs at a
time, so this never occupies more than one GPU. The cluster is shared. `afterany`
rather than `afterok` because the later strides do not depend on the earlier ones
succeeding, only on the GPU freeing up. Each stride takes ~2-3 min (48 windows at
~2.7 s), so the whole sweep is ~10 min on one card.

**Step 3 -- pair and score.** CPU only, env `dynrec` (needs `cv2`):

```
cd ~/DynamicReconstructionSplat
SEQ=synchronous2
O=output_eval_frozen_pfd_s4_flow8raftsb1p5_cl3p0_bg111_oc1p0_pcm_nf16_${SEQ}_final4/images
M=/data/hanmydo/mask_out/output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_otsu2_glob/rgbd_bonn_${SEQ}/masks
for S in 4 8 12; do
  python tools/build_movies_comparison.py --movies_dir ~/MoVieS/out/loo_${SEQ}_dense_s$S \
    --ours_images $O --bonn_rgb /data/hanmydo/bonn/rgbd_bonn_dataset/rgbd_bonn_${SEQ}/rgb \
    --mask_dir $M --out cmp_${SEQ}_clean_s$S | grep -e paired -e SKIPPED
  python tools/compare_masked.py --gt_dir cmp_${SEQ}_clean_s$S/gt \
    --a cmp_${SEQ}_clean_s$S/ours --a_panels 2 --a_label ours \
    --b cmp_${SEQ}_clean_s$S/movies --b_label MoVieS --mask_dir cmp_${SEQ}_clean_s$S/mask
done
```

**Step 4 -- the qualitative strips** (GT | ours | MoVieS, labelled):

```
python compare_methods.py --gt_dir cmp_${SEQ}_clean_s12/gt \
  --a cmp_${SEQ}_clean_s12/ours --a_panels 2 --a_label ours \
  --b cmp_${SEQ}_clean_s12/movies --b_label MoVieS --out fig_${SEQ}_s12 --fig_stride 4
```

### Which protocol row is which (read this before quoting a number)

Two independent things must match between the methods, and each was wrong once:

**1. HORIZON must match.** `submit_final_battery.sh` hardcodes `--frame_stride 4`,
so the `final*` renders are a ~2 s window NO MATTER what stride MoVieS runs at.
Pairing those against MoVieS at stride 8 or 12 hands MoVieS a 2-3x longer
reconstruction window than we take ourselves. Measured cost of that mistake: a
balloon row read `+0.70` overall unmatched and `-0.50` matched -- a sign flip.
⇒ Use the per-stride ladders, which are config-identical to `final4` and differ
ONLY in `frame_stride`:

```
balloon      output_probe_frozen_h04|h08|h12|h16|h20_balloon_h{NN}
synchronous2 output_probe_frozen_g4|g8|g12_synchronous2_g{N}
```

⚠️ Check a config by reading `eval_config.json["argv"]`, the NESTED dict. A
top-level `.get("frame_stride")` returns `None` even when it is set, which will
fool you into thinking a run is unconfigured.

**2. HELD-OUT PROTOCOL must match.** `movies_bonn_loo.py` holds out `frames//2`
and reconstructs it from the other 15. Our eval only does the equivalent with
`--eval_loo`, and the ladders above do NOT use it. Report these as separate rows,
never mixed:

| row | ours | MoVieS | `--ours_offset` |
|---|---|---|---|
| self-recon (symmetric, weaker) | `h*`/`g*` ladder, no `--eval_loo` | `--self_recon` | `0` |
| leave-one-out (headline) | `--eval_loo --image_views 8` | default | `$((8 * STRIDE))` |
| strict no-look-at-j (control) | piecewise-rigid mode, no `--track_dynamic` | default | per above |

`--image_views 8` rather than the launcher's hardcoded `0` is REQUIRED for the LOO
row: view 0 is the window's FIRST frame, so holding it out means extrapolating from
one side while MoVieS interpolates from both. Spec flags are appended after `${WIN}`
in `slurm_probe_hex.sh`, so a later `--image_views 8` overrides it. Middle-view
coverage also aligns with MoVieS's held-out range, which fixes thin samples
(synchronous2 at 6 s: 17 -> ~36 paired frames).

**Asymmetries to STATE rather than fix**, both in our favour:
- Under `--eval_loo` the scene-flow path still displaces Gaussians using the tracks'
  OBSERVED position at frame j (`decoder_splatting_cuda.py` "SCENE FLOW", and the
  protocol note in `dyn_motion.py`). Appearance and source geometry come only from
  other frames, but POSITION reads frame j. This is the standard monocular
  dynamic-NVS convention ("motion fitted on the full video, appearance held out");
  MoVieS instead predicts forward from 15 frames. The piecewise-rigid mode is the
  strict variant and is the third row above.
- `--eval_loo` drops view j's Gaussians, but the backbone still SAW frame j.

Launching the LOO ladder (chained = one GPU at a time):

```
M2=~/data/mask_out/output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_otsu2_glob
F="--per_frame_dynamic --dyn_conf_opacity 4.0 --track_dynamic --dyn_motion_knn 8 \
   --dyn_motion_strict --dyn_motion_pred_bandwidth 1.5 --dyn_motion_tracker raft \
   --dyn_motion_max_disp_mult 3.0 --dyn_opacity_comp 1.0 --bg_color 1 1 1 \
   --eval_loo --image_views 8"
P=""
for S in 4 8 12; do P=$(EVAL_DATE=loo$S sbatch --parsable --time=04:00:00 \
  ${P:+--dependency=afterany:$P} slurm_probe_hex.sh baseline "$M2" \
  rgbd_bonn_synchronous2 16 $S 0 99999 "L$S=$F"); echo "stride $S -> $P"; done
```

⚠️ Always confirm with `squeue` that the jobs exist. Two submissions in one session
silently never ran, and `sacct --starttime today` was what revealed it.

### Why these settings

- **`--window_step 5`, not the default 13.** At 13 the paired counts collapse to
  7-21 frames and the deltas are noisy *and biased high*: synchronous2 read
  +2.61/+4.43 at 4 s sparse versus +1.85/+3.18 dense. Treat any sparse LOO row as
  an upper bound.
- **Fresh output directories per sweep.** Re-running a different `--window_step`
  into an existing dir leaves the old grid's files in place, and the pairing step
  silently unions them (visible as off-grid held-out indices like 77, 90, 109).
  It changed the answer by <0.08 dB, but the directory stops being reproducible
  from the script.
- **`--poses vggt` is mandatory.** Bonn's `groundtruth.txt` costs MoVieS 10 dB
  because the mocap marker frame is not the camera optical frame; see
  `baseline-movies-bonn-poses` in memory.
- **One mask set scores both methods.** The mask selects an evaluation region and
  is not an input -- neither method sees it. This is what makes `psnr_dynamic`
  comparable between the two columns; comparing it across runs with *different*
  masks is the trap documented in `mask-round-sep24-sam-negative`.
- **Window-normalised time is faithful, not an artefact.** We pass
  `linspace(0, 1, frames)`; their own loader does
  `(ts - ts.min())/(ts.max() - ts.min())` (`~/MoVieS/src/data/spring_dataset.py:38`),
  which is identical for evenly spaced indices. Stride is therefore invisible to
  their model by construction.
- **Check the strips for frame alignment.** `compare_methods.py` pairs by sorted
  order, which is why `movies_bonn_loo.py` names outputs by *source* frame index.
  If GT/ours/MoVieS look misaligned, the pairing is wrong and the metrics are
  meaningless.

## Why these settings and not others (all measured, all on balloon)

- `--image_size 392 518`, NOT 672x896. AnySplat infers at 448x448; 672x896 is 3x
  that density, so splats overlap by sqrt(3) and the cloud reads as a painterly
  smear. 392x518 = 203k px ~ 448^2, both divisible by 14, aspect ~4:3. Measured
  per-copy coverage at 672x896: 1.71.
- `--num_frames 6`, NOT 16. Every surface carries V overlapping copies from V
  slightly different poses; 16 misaligned translucent layers blur, 6 blur less.
  nf16 gave 6.96M gaussians at opacity 0.060 and 34.9% renderable, against a
  historical good reference of 400k / 0.195 / 68.9%.
- NO scale multiplier. `--ply_dyn_scale_mult -1` (sqrt(V)) is RETRACTED, see
  d0fed35: the coverage measurement behind it sampled nearest neighbours within a
  subsample and inflated spacing ~3.5x.
- NOT `--ply_single_frame`. It also drops the other frames' STATIC gaussians,
  which empties the background behind the person.
- FROZEN, not the anchor checkpoint. With splat sizing equalised the anchor's
  dynamic object is less concentrated (21.8% vs 31.7%), and it shrinks the scene
  diagonal from 1.78 to 1.28 -- a ~40% metric scale change that PSNR cannot see.
