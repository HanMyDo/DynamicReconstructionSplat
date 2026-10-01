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
