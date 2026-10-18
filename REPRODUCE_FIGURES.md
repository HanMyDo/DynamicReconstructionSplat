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

## ⭐ ADOPTED Oct 14 2026 — `--render_max_scale_frac 0.011`

The full current recipe, which is what the horizon ladders use (the `F` above predates the
compositing knobs and is kept only for the figures in sections 1-3):

```
F="--per_frame_dynamic --dyn_conf_opacity 4.0 --track_dynamic --dyn_motion_knn 8 \
   --dyn_motion_strict --dyn_motion_pred_bandwidth 1.5 --dyn_motion_tracker raft \
   --dyn_motion_max_disp_mult 3.0 --dyn_opacity_comp 1.0 --bg_color 1 1 1 \
   --eval_loo --image_views 8 --dyn_unsupported static --dyn_opacity_gain 4 \
   --dyn_unsup_gain 0.25 --render_max_scale_frac 0.011"
```

`--render_max_scale_frac 0.011` is the ONLY survivor of the Oct 14 seven-arm sweep (crowd @ 6 s,
removing @ 20 s, TUM walking_xyz @ 6 s). It was never harmful, gained **+0.88 psnr / +1.59 static**
on crowd, and is the only modified arm that looked more defined rather than less. It applies the
ceiling `ply_export.py` has used since September to the RENDER path, which had none.

⛔ Rejected in the same sweep: `--src_window` (costs coverage: alpha 0.970 -> 0.885, TUM static
-1.89), `--dyn_unsupported drop` and low `--dyn_unsup_gain` (white holes -- the masks flag static
furniture as dynamic, so dropping unrelocatable Gaussians deletes background), and every
combination of them.

⚠️ **Every number reported before Oct 14 was produced WITHOUT this flag.** Do not mix old and new
rows in one table; re-run the arm you are quoting, or report the ceiling as an ablation.

## ⭐⭐⭐ ADOPTED Oct 18 2026 — `--src_window_dyn 4`

The largest single improvement measured this week. Add `--src_window_dyn 4` to `F`
(it works alone; `--src_window` stays 0, i.e. static keeps every contributor).

**What it fixes.** Far-frame DYNAMIC gaussians splatter across the BACKGROUND. A global
`--src_window` removes them, which is why it looked like a sharpness-for-coverage trade
and was rejected in October; clipping ONLY the movers gets the same benefit with static
coverage intact. Proof it is the movers and not static-vs-static disagreement: protecting
the movers while clipping static made psnr_static WORSE, 15.53 -> 13.74.

| | psnr | ssim | lpips | dyn | static | alpha_static |
|---|---|---|---|---|---|---|
| crowd, full window | 14.123 | 0.5938 | 0.4783 | 14.016 | 14.773 | 0.907 |
| **crowd, dyn 4** | **15.315** | **0.6481** | **0.4179** | 13.166 | **17.433** | 0.881 |
| TUM xyz, full | 16.468 | 0.6335 | 0.3530 | 13.737 | 17.109 | 0.946 |
| **TUM xyz, dyn 4** | **16.625** | **0.6460** | **0.3379** | 13.492 | **17.514** | 0.942 |

2/2 sequences, two datasets, same signs; TUM is ~1/5 the magnitude, the usual transfer
ratio. Renders on crowd prefer `dyn 4` over `dyn 8` and over the full window. Cost is
dynamic −0.85 (crowd) / −0.25 (TUM), not visible in the renders.
⚠️ `--src_window_dyn 8` is the conservative alternative: crowd psnr +1.09, static +1.60,
dynamic FLAT (−0.045). Use it if a figure must not touch the movers at all.

## ⭐⭐ ADOPTED Oct 18 2026 — `--image_size 392 518` (the aspect squash)

`--image_size` defaults to 448 448, but Bonn and TUM are 640x480 = 4:3, so the default
grid squashes the scene 25% horizontally AND is off VGGT's training distribution — the
help for that flag says so. An off-distribution input degrades the depth and poses every
frame contributes, which is what blurs STATIC content, since static sharpness depends on
15 frames' geometry agreeing.

`392 518` is 4:3, divisible by 14 and the SAME pixel count as 448², isolating the squash
at zero cost. Crowd @ stride 8, 3 windows, `F` otherwise verbatim:

| | psnr | ssim | lpips | static | alpha_static |
|---|---|---|---|---|---|
| 448x448 | 14.214 | 0.5566 | 0.5101 | 14.884 | 0.884 |
| **392x518** | 14.193 | **0.5912** | **0.4793** | **15.025** | **0.9054** |

+0.035 ssim / −0.031 lpips at flat psnr — the signature of a sharper image, and as large
as the whole SAM2-mask adoption. ⚠️ The RENDERS show little visible change, so this is
adopted as a CORRECTNESS fix with perceptual support, not a visible win.
⚠️⚠️ `--image_size` CHANGES EVERY METRIC: the 5-sequence baseline table predates it and
cannot absorb it. Adopting means re-running OUR 25 arms; the baselines' renders are fixed
and do not need redoing.

## ⭐⭐ ADOPTED Oct 15 2026 — the MASK recipe (`F` is unchanged)

`F` above is still the render config, unchanged. What changed is how the masks are
made. The adopted mask set is now the **flow detector completed by SAM 2 video
propagation over the whole sequence**:

```
sbatch --export=ALL,DATA_ROOT=$HOME/data/bonn/rgbd_bonn_dataset --gres=gpu:1 \
  slurm_precompute_masks_hex.sh <SEQ> 256 518 3 1 6 per_frame mean 64 flow 2 \
  0 0 0 0 1 0 2 0 1 16
#                   ^method ^otsu            ^glob ^sam=2 ^gvid ^anchors
# -> output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_flow_otsu2_glob_samvid_gvid_an16
```

**Why:** the old attention masks mark the table and every chair as dynamic; these mark
one chair back. Measured on crowd at a window where people actually move (held-out
304-333), new vs old masks win on all three mask-independent metrics under BOTH the
neutral and the adopted config: **+0.19 / +0.13 psnr, +0.003 / +0.014 ssim,
-0.009 / -0.027 lpips**. The renders are the stronger evidence -- people are solid
instead of faint and scattered. Full record and the two traps that nearly hid this
result are in the `result-masks-validated-crowd` memory note.

⚠️ **`psnr_dynamic` and `psnr_static` are NOT comparable across mask sets** -- different
masks define different pixel populations. Only `psnr`, `ssim` and `lpips` compare.

⚠️ **Every number reported before Oct 15 used the OLD attention masks.**

## ⭐⭐ MEASUREMENT NOISE FLOOR — 0.006 dB (Oct 15 2026)

Two byte-identical configs, crowd, 30 windows at stride 8, `--eval_loo`:

| | psnr | ssim | lpips |
|---|---|---|---|
| r1 | 13.8437 | 0.5753 | 0.4944 |
| r2 | 13.8373 | 0.5755 | 0.4944 |
| **Δ** | **0.0064** | **0.0002** | **0.0000** |

RAFT is not bit-deterministic, so repeated runs are not identical, but at 30 windows the
aggregate spread is **~0.006 dB**. Consequences: the Oct 15 mask win (+0.19 / +0.13 psnr)
is 20-30x the noise and is REAL; `--dyn_conf_opacity` (+0.52) and `--dyn_opacity_comp`
(+0.87) were far above noise; and anything claimed below ~0.05 dB on a single sequence
needs this control re-run beside it. ⚠️ The floor is for the 30-window AGGREGATE -- the
per-window `[DynFlow] moved %` varies several points run to run (72.4 vs 71.1 over 3
windows), so do not read a single window's diagnostic as exact.

Two fixes landed with it, both correcting a whole-sequence decision that was being taken
per chunk -- the same class of error `--global_post` fixed for the attention path:
- `--sam_video_global` (arg 20): propagation ran PER CHUNK, so crowd's four chunks came
  out at mean masked area 0.060 / 0.319 / 0.179 / 0.116. Balloon's 439 frames fit one
  chunk of 512, which is why its masks were already good. Empty frames 60.3% -> 5.0%.
- ⛔ `--global_flow_thresh` (arg 22): **MEASURED AND REJECTED Oct 15.** The flow
  residual's multi-Otsu threshold is per chunk (3.281 / 6.998 / **40.034** / 6.406 on
  crowd) and making it global gives **38.401** -- next to the outlier, not the cluster,
  because the residual is heavy-tailed and a whole-sequence Otsu is dominated by the
  tail. Seed coverage 4.39% -> **0.48%**, frames with no seed 60.3% -> **98.6%**, and
  SAM then grows **0.388** mean area from 13 anchors against 0.183. Per-chunk Otsu is
  acting as useful LOCAL NORMALISATION. The flag stays in the code, off.
  ⚠️ Per-chunk Otsu is still unstable (12x thresholds at similar medians: 0.29 / 0.48 /
  0.42 / 0.31), so the principled fix is a ROBUST statistic -- a residual percentile, or
  median + k*MAD -- not a global Otsu. Not attempted.

⛔ Still rejected: `--dyn_unsupported drop`. But note the Oct 14 sweep rejected it
*because* "the masks flag static furniture as dynamic, so dropping unrelocatable
Gaussians deletes background" -- a cause these masks remove. Re-test as its own
single-variable experiment.

`--sam_anchors` scales with sequence length: 8 was tuned on balloon's 439 frames, 16 for
~930. Chunk 512 host-OOMs (`Killed`) at 963 frames; 256 fits and takes ~28 min for 928.

### Superseded (pre-Oct-15 mask set)

`D` was: whole-sequence detection, chunk 512 (one pass =
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
