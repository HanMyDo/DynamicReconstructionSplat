#!/usr/bin/env python3
"""Precompute VGGT4D dynamic masks over LONG temporal windows, once per sequence.

WHY (see memory: next-dynamic-mask-precompute-plan):
    In-pipeline detection ran on 12-frame windows (~0.4s at 30fps) — far too little
    motion for the attention-based detector, so the mask found ~5% of pixels and
    missed the moving objects. The ORIGINAL VGGT4D runs detection over the whole
    clip. This script decouples DETECTION (large temporal window, cheap: backbone
    attention only, NO Gaussians / NO rendering) from RECONSTRUCTION (12-frame
    windows, memory-bound). It processes a sequence in CHUNKS of `--chunk_size`
    frames (bounded memory on the 24g GPU), computes a per-frame mask for each,
    and caches them to disk. Train/eval then LOAD these instead of recomputing.
    This does NOT break "feed-forward" (mask = a forward pass, not optimization;
    the cache is only a training-time speedup).

STAGES (`--stages`, default 3 = full original VGGT4D pipeline):
    Stage 1 = attention -> coarse mask. Stage 2 = re-run the backbone WITH the mask
    (token suppression) -> refined poses + depth. Stage 3 = geometric refinement of
    the coarse mask using those refined poses+depth (open3d), and this is the mask
    the original SAVES. `--stages 1` stops at the coarse mask (debug/ablation).
    NOTE: Stage 2 needs a second aggregator pass + the depth head, so `--stages 3`
    uses more memory than Stage-1-only -> the fittable chunk_size is smaller.
    This mirrors EncoderAnySplat.forward's detection path minus the Gaussian head.

MATCHES the original here: preprocessing = 518 long-edge ASPECT-PRESERVED crop
    (load_and_preprocess_images mode="crop"), and the Stage-1 detection functions
    are VGGT4D's own (extract_dyn_map / cluster / adaptive_multiotsu on the
    UPSAMPLED map — the Otsu-order fix already landed in anysplat.py).

Output: <output_dir>/<dataset_name>/masks/<frame_stem>.png     (binary, 0/255)
        <output_dir>/<dataset_name>/overlays/<frame_stem>.png   (red overlay, to eyeball)
        <output_dir>/<dataset_name>/meta.json                   (settings + dyn fraction)
Masks are at the DETECTION resolution (518 x aspect-preserved height); integration
into train/eval must resample them to the 448x448 reconstruction grid on load.

Example:
  python precompute_dyn_masks.py \
    --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
    --dataset_name rgbd_bonn_moving_nonobstructing_box \
    --output_dir dyn_masks_precomputed \
    --vggt4d_weights_path ckpts/vggt4d_model_tracker_fixed_e20.pt \
    --chunk_size 32 --save_overlays
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_temporal_gaussian_head import create_model, TrainingConfig
from src.model.encoder.anysplat import _AMP_DTYPE
from src.model.encoder.vggt4d.masks.dynamic_mask import adaptive_multiotsu_variance
from src.model.encoder.vggt.utils.load_fn import load_and_preprocess_images
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.model.encoder.dyn_flow_mask import flow_residual_map
from src.model.encoder.dyn_mask_post import complete_masks, motion_gate_masks
from src.model.encoder.vggt4d.masks import cluster_attention_maps


def gather_frame_paths(seq_dir: Path):
    """All RGB frames of a sequence, in temporal (filename) order.

    Keyed later by filename stem, so this must match how train/eval reference frames.
    """
    rgb_dir = seq_dir / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"No rgb/ dir under {seq_dir}")
    paths = sorted(
        p for p in rgb_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not paths:
        raise FileNotFoundError(f"No frames in {rgb_dir}")
    return paths


def chunk_ranges(n_frames: int, chunk_size: int, min_frames: int = 6):
    """Contiguous, non-overlapping [start, end) ranges covering all frames.

    Every frame is detected inside a window of ~chunk_size frames (cross-frame
    attention spans the whole chunk), so it gets the temporal context the 12-frame
    path lacked. A tiny final chunk is absorbed into the previous one.
    """
    ranges = []
    start = 0
    while start < n_frames:
        end = min(start + chunk_size, n_frames)
        ranges.append([start, end])
        start = end
    if len(ranges) >= 2 and (ranges[-1][1] - ranges[-1][0]) < min_frames:
        ranges[-2][1] = ranges[-1][1]
        ranges.pop()
    return ranges


def build_passes(n_frames: int, chunk_size: int, stride: int = 1, min_frames: int = 6,
                 margin: int = 0):
    """Group frame INDICES into detection passes (each pass = one backbone forward).

    Returns a list of (process_indices, emit_indices). A pass runs the backbone on
    `process_indices` but only WRITES masks for `emit_indices`.

    WHY `margin`. Every extractor in the original detector compares a reference frame
    against IN-PASS offsets [-6,-4,-2,2,4,6]. With contiguous, non-overlapping passes
    the frames near a pass boundary lose half of that window -- at chunk_size 16 only
    positions 6..9 keep all six neighbours, so 12 of 16 masks are computed from a
    partial window. `margin` M makes passes OVERLAP by M on each side and emits only
    the interior, so every emitted frame has its full window. Costs compute, not
    memory: the pass is still <= chunk_size frames, there are just more of them
    (emit block = chunk_size - 2M, so runtime scales by chunk_size/(chunk_size-2M)).

    stride == 1: contiguous windows via chunk_ranges (original behavior). Each pass is
      chunk_size CONSECUTIVE frames -> spans only ~chunk_size frames of time, so in a
      slow sequence the moving object barely displaces (weak dynamic signal).

    stride  > 1: for each offset k in [0, stride), take frames k, k+stride, k+2*stride, ...
      and split that strided list into blocks of <= chunk_size. Each pass then holds
      <= chunk_size frames spaced `stride` apart, spanning up to chunk_size*stride frames
      of the sequence -> the object travels FAR across a pass while GPU memory stays at
      chunk_size frames. Every frame lands in exactly one pass (frame j -> offset j%stride),
      so the saved masks still tile the whole sequence with no gaps or overlaps.
    """
    if stride <= 1:
        if margin <= 0:
            return [(list(range(s, e)), list(range(s, e)))
                    for s, e in chunk_ranges(n_frames, chunk_size, min_frames)]
        emit_size = max(1, chunk_size - 2 * margin)
        out = []
        for s, e in chunk_ranges(n_frames, emit_size, min_frames=1):
            ps, pe = max(0, s - margin), min(n_frames, e + margin)
            out.append((list(range(ps, pe)), list(range(s, e))))
        return out
    passes = []
    for k in range(stride):
        idxs = list(range(k, n_frames, stride))
        blocks = [idxs[s:s + chunk_size] for s in range(0, len(idxs), chunk_size)]
        # Absorb a tiny tail block into the previous one (per offset) — mirrors
        # chunk_ranges. A 1-frame pass makes cross-frame dynamic detection degenerate
        # (all-NaN score -> Otsu crashes on [nan, nan]), so never emit one.
        if len(blocks) >= 2 and len(blocks[-1]) < min_frames:
            blocks[-2] = blocks[-2] + blocks[-1]
            blocks.pop()
        passes.extend(blocks)
    # Safety net (stride close to n_frames can leave a lone undersized pass): merge any
    # remaining sub-min pass into the previous, so no pass has < min_frames.
    if len(passes) >= 2:
        merged = [passes[0]]
        for p in passes[1:]:
            if len(p) < min_frames:
                merged[-1] = merged[-1] + p
            else:
                merged.append(p)
        passes = merged
    return [(p, p) for p in passes]


def save_mask_png(mask_hw: np.ndarray, path: Path):
    """mask_hw: float/bool [H, W] in {0,1} -> binary PNG (0/255)."""
    arr = (np.asarray(mask_hw) > 0.5).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)


def save_overlay_png(img_chw: torch.Tensor, mask_hw: np.ndarray, path: Path):
    """Red overlay of the mask on the RGB frame, for quick visual inspection."""
    img = (img_chw.detach().float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    m = (np.asarray(mask_hw) > 0.5)
    over = img.copy()
    over[m] = (0.5 * over[m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(over, mode="RGB").save(path)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Precompute VGGT4D Stage-1 dynamic masks over long temporal windows.")
    ap.add_argument("--data_dir", required=True, help="Root containing <dataset_name>/rgb/")
    ap.add_argument("--dataset_name", required=True, help="Sequence dir, e.g. rgbd_bonn_moving_nonobstructing_box")
    ap.add_argument("--output_dir", required=True, help="Where cached masks are written")
    ap.add_argument("--vggt4d_weights_path", default=None, help="VGGT4D weights (.pt); omit to init from VGGT-1B")
    ap.add_argument("--chunk_size", type=int, default=32,
                    help="Frames per detection window (bigger = more motion, more memory; attention ~O(N^2)).")
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="0 = AUTO full-sequence span (VALIDATED RECIPE): stride=ceil(n_frames/chunk_size) "
                         "so every pass spans the WHOLE sequence (matches the original VGGT4D's whole-clip "
                         "context) while every frame still gets a mask. 1 = consecutive windows. >1 = each "
                         "pass takes every STRIDE-th frame, spanning up to chunk_size*STRIDE frames. Use 0 "
                         "with --det_resolution 518; that combo reproduced the original's mask quality.")
    ap.add_argument("--mask_otsu_level", type=int, default=1,
                    help="Which multi-Otsu split separates dynamic from static, counting down "
                         "from the highest. 1 (original) keeps ONLY the topmost class, which on a "
                         "partially-moving object is just its fastest part -- the arm, with the "
                         "torso in the class immediately below being discarded. 2 keeps the top "
                         "two classes so the whole object survives. This, not the score "
                         "aggregation, is what controls coverage: the threshold is adaptive, so "
                         "rescaling cluster scores merely moves the split with them.")
    ap.add_argument("--no_stream_qk", action="store_true",
                    help="Keep the whole Q/K capture resident on the GPU during extraction, "
                         "as the original literally writes it. That is the single largest "
                         "allocation in the precompute (17.6 GiB for global_tok_k alone at "
                         "192 frames) and it caps chunk size at ~128 on a 47 GB card, even "
                         "though the loop only ever touches 7 frames at a time. Streaming "
                         "is on by default and computes identical values.")
    ap.add_argument("--global_post", action="store_true",
                    help="FAITHFUL TO THE ORIGINAL: cluster and threshold over the WHOLE "
                         "sequence instead of per chunk. demo_vggt4d.process_scene loads "
                         "every frame at once, so it runs ONE KMeans and ONE multi-Otsu "
                         "over all frames; chunking gives each chunk its own, so a quiet "
                         "stretch gets a lower bar than a busy one (background blobs there, "
                         "a missed person here). This keeps the expensive attention pass "
                         "chunked -- only the post-processing goes global -- at the cost of "
                         "running stage 1 twice. Needs host RAM for the encoder features of "
                         "the whole sequence (~1 GB per 500 frames at 518).")
    ap.add_argument("--mask_motion_gate", type=float, default=0.0,
                    help="Drop mask COMPONENTS whose flow residual does not exceed this "
                         "multiple of the frame's own static-region residual. Attention "
                         "over-fires on static structure BESIDE a moving object (the desk, "
                         "the chair) because it responds to attention dissimilarity there, "
                         "not motion; geometry is ~0 on anything static however close it "
                         "sits. Gating per component, not per pixel, keeps the person whole "
                         "where the residual is weak instead of re-eroding what the shape "
                         "completion joined. 3.0 is a reasonable start; 0 = off. Needs "
                         "--stages 3 (uses Stage-1 depth and Stage-2 poses) and costs a "
                         "RAFT pass over the chunk.")
    ap.add_argument("--mask_close", type=int, default=0,
                    help="Morphological closing radius (px) to BRIDGE the gaps between parts "
                         "of one object -- the detector fires on limbs and outlines and misses "
                         "the torso, so the mask arrives as disconnected pieces of one person.")
    ap.add_argument("--mask_fill", action="store_true",
                    help="Fill interior holes, turning a ring of moving edges into a solid body. "
                         "Runs AFTER closing, which has to connect the outline first.")
    ap.add_argument("--mask_min_area", type=int, default=0,
                    help="Delete connected components smaller than this many pixels. These specks "
                         "are the false positives that make per-frame compositing tear real "
                         "background out of the scene (measured -2.96 dB on one placing cluster).")
    ap.add_argument("--mask_dilate", type=int, default=0,
                    help="Final dilation radius (px). A mask edge slightly inside the object "
                         "leaves a rim of it behind, which then ghosts.")
    ap.add_argument("--mask_method", default="attention", choices=["attention", "flow", "union"],
                    help="Which signal defines the dynamic mask. 'attention' is VGGT4D's own: "
                         "attention dissimilarity between a frame and its neighbours, which "
                         "responds to how FAST a pixel moves and so finds a swinging arm but "
                         "misses the torso. 'flow' is geometric: predict each pixel's motion from "
                         "depth and pose, subtract it from measured RAFT flow, and flag the "
                         "residual -- a slow torso still moves differently from the wall behind "
                         "it, so whole objects are covered. 'union' takes both.")
    ap.add_argument("--mask_aggregate", default="mean", choices=["mean", "max", "p90"],
                    help="How a feature cluster inherits its dynamic score. 'mean' (original) "
                         "dilutes a partially-moving object: on a walking person only the fast "
                         "parts score, so averaging drops the cluster below threshold and the mask "
                         "covers an arm rather than a person -- which per-frame compositing then "
                         "cannot protect, so the rest of them ghosts. 'p90'/'max' propagate the "
                         "moving part's score across its cluster, masking the OBJECT.")
    ap.add_argument("--mask_n_clusters", type=int, default=64,
                    help="KMeans clusters for the refinement. Fewer clusters group a person into "
                         "one region (so p90/max can recruit all of them); more keeps boundaries "
                         "tight. 64 is the original.")
    ap.add_argument("--mask_normalize", default="per_frame", choices=["per_frame", "global"],
                    help="How cluster scores are rescaled before thresholding. 'per_frame' is the "
                         "original: every frame is stretched to [0,1], so a frame with nothing "
                         "moving still contributes its brightest patches to a globally-thresholded "
                         "mask -- a steady false-positive source on sequences that are mostly "
                         "quiet. 'global' normalises once across the pass, so a quiet frame's "
                         "scores stay low and can be rejected entirely.")
    ap.add_argument("--pass_margin", type=int, default=0,
                    help="Overlap passes by this many frames on each side and write masks only "
                         "for the interior. The detector compares each frame against in-pass "
                         "offsets +-2/4/6, so without a margin the frames at a pass boundary are "
                         "computed from a truncated window (at chunk_size 16, 12 of 16 frames). "
                         "6 gives every emitted frame its full window. Costs runtime, not memory: "
                         "emit block = chunk_size - 2*margin, so runtime scales by "
                         "chunk_size/(chunk_size-2*margin) -- 4x at chunk 16 margin 6, 2x at "
                         "margin 4. Only applies to --frame_stride 1.")
    ap.add_argument("--det_resolution", type=int, default=518,
                    help="NATIVE detection long-edge (target_size passed to load_and_preprocess_images). "
                         "518 = original VGGT4D (faithful). Lower (e.g. 448/378) -> fewer tokens -> less "
                         "host+GPU memory -> fits more frames per chunk. Masks come out at this resolution "
                         "and are upsampled to the reconstruction grid on load.")
    ap.add_argument("--stages", type=int, default=3, choices=[1, 3],
                    help="1 = Stage-1 coarse attention mask only. 3 = full original VGGT4D pipeline: Stage 1 "
                         "(coarse) -> Stage 2 (re-run backbone with mask -> refined poses/depth) -> Stage 3 "
                         "(geometric refinement). 3 uses more memory (extra aggregator pass + depth head) so "
                         "the fittable chunk_size is smaller. Needs open3d for Stage 3.")
    ap.add_argument("--preprocess_mode", default="crop", choices=["crop", "pad"],
                    help="Original VGGT4D preprocessing. 'crop' = 518 wide, aspect-preserved (matches demo).")
    ap.add_argument("--save_overlays", action="store_true", help="Also write red mask-on-RGB overlays.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seq_dir = Path(args.data_dir) / args.dataset_name
    out_dir = Path(args.output_dir) / args.dataset_name
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = gather_frame_paths(seq_dir)
    # stride 0 = AUTO full-sequence span: ceil(n_frames / chunk_size) so each pass
    # covers the whole sequence (the validated 518+full-span recipe).
    stride = args.frame_stride
    if stride == 0:
        stride = max(1, (len(frame_paths) + args.chunk_size - 1) // args.chunk_size)
    passes = build_passes(len(frame_paths), args.chunk_size, stride,
                          margin=(args.pass_margin if stride <= 1 else 0))
    span_hint = f", span up to ~{args.chunk_size * stride} frames/pass" if stride > 1 else ""
    auto_hint = " (auto full-span)" if args.frame_stride == 0 else ""
    print(f"Sequence: {args.dataset_name}  |  {len(frame_paths)} frames  |  "
          f"{len(passes)} pass(es) of <= {args.chunk_size}  |  stride {stride}{auto_hint}{span_hint}")

    print("Creating model (VGGT4D backbone + dynamic detection)...")
    config = TrainingConfig(
        use_vggt4d=True,
        enable_dynamic_detection=True,
        vggt4d_weights_path=args.vggt4d_weights_path,
        dyn_mask_normalize=args.mask_normalize,
        dyn_mask_aggregate=args.mask_aggregate,
        dyn_mask_otsu_level=args.mask_otsu_level,
        dynamic_n_clusters=args.mask_n_clusters,
    )
    model = create_model(config).to(device).eval()
    encoder = model.encoder

    # ---------------------------------------------------------------- PASS A
    # Collect the raw attention maps and encoder features for EVERY frame, then
    # cluster and threshold once over all of them -- the order and the operations
    # of demo_vggt4d.process_scene, which never chunks. Only the attention pass
    # stays chunked, because that is the part that does not fit; nothing about
    # KMeans or Otsu needs the frames to be resident on the GPU together.
    coarse_by_pass = None
    if args.global_post:
        print(f"[GlobalPost] pass A: collecting attention maps over {len(passes)} chunk(s)")
        _feats, _dyns, _rows = [], [], []      # _rows: (pass_index, n_frames, emitted mask)
        for ci, (idxs, emit_idxs) in enumerate(passes):
            chunk_paths = [frame_paths[i] for i in idxs]
            images = load_and_preprocess_images(
                [str(p) for p in chunk_paths], mode=args.preprocess_mode,
                target_size=args.det_resolution).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=_AMP_DTYPE):
                _t, _ps, qk_dict, enc_feat = encoder.aggregator(
                    images.to(_AMP_DTYPE), dyn_masks=None)
            # Pass A only needs Q/K and the encoder features. The aggregated token
            # list is several GB (one tensor per layer) and is used by stages 2-3,
            # which run in the MAIN loop, not here -- holding it across the
            # extraction is what pushed chunk 2 into OOM while chunk 1 fitted.
            # extract_dyn_map then moves query, key and camera-query to the GPU
            # together (~9 GB at chunk 96), so this headroom is exactly what it needs.
            del _t, _ps
            if device.type == "cuda":
                torch.cuda.empty_cache()
            dyn_maps, feat_map = encoder.attention_dyn_score_parts(
                images, qk_dict, enc_feat, stream_qk=not args.no_stream_qk)
            _feats.append(feat_map.float().cpu())
            _dyns.append(dyn_maps.float().cpu())
            H_full, W_full = images.shape[-2], images.shape[-1]
            _emit = torch.tensor([i in set(emit_idxs) for i in idxs], dtype=torch.bool)
            _rows.append((ci, len(idxs), _emit))
            del qk_dict, enc_feat, dyn_maps, feat_map, images
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  [pass A {ci+1}/{len(passes)}] {len(idxs)} frames collected", flush=True)

        # ONE KMeans over every frame's patches (the original clusters the whole scene).
        all_feat = torch.cat(_feats, dim=0)
        all_dyn = torch.cat(_dyns, dim=0)
        del _feats, _dyns
        print(f"[GlobalPost] clustering {all_feat.shape[0]} frames "
              f"({all_feat.shape[0] * all_feat.shape[1] * all_feat.shape[2]} patches, "
              f"C={all_feat.shape[-1]}) with k={args.mask_n_clusters} -- this is the slow step")
        norm_map, _ = cluster_attention_maps(
            all_feat, all_dyn, n_clusters=args.mask_n_clusters,
            normalize=args.mask_normalize, aggregate=args.mask_aggregate)
        del all_feat, all_dyn

        # Upsample FIRST, then threshold -- same order as the original, and the same
        # reason: bilinear smoothing lowers peaks, so a threshold taken on the patch
        # map is systematically too high for the full-resolution one.
        _up = []
        for a in range(0, norm_map.shape[0], 32):        # chunked only to bound peak RAM
            u = F.interpolate(norm_map[a:a + 32].unsqueeze(1).float(),
                              size=(H_full, W_full),
                              mode="bilinear", align_corners=False).squeeze(1)
            _up.append(u)
        upsampled = torch.cat(_up, dim=0)
        del _up, norm_map

        # ONE threshold, from the EMITTED frames only: margin frames appear in two
        # passes, and letting them vote twice would tilt the split toward whatever
        # happens to sit at a chunk boundary.
        emit_flags = torch.cat([e for (_, _, e) in _rows], dim=0)
        thr = adaptive_multiotsu_variance(
            upsampled[emit_flags].numpy(), level=args.mask_otsu_level)
        frac = float((upsampled[emit_flags] > thr).float().mean())
        print(f"[GlobalPost] ONE threshold for the whole sequence: {thr:.4f} "
              f"-> dynamic pixels {100 * frac:.1f}%", flush=True)

        coarse_by_pass, _o = [], 0
        for (_ci, _n, _e) in _rows:
            coarse_by_pass.append((upsampled[_o:_o + _n] > thr).float().unsqueeze(0))
            _o += _n
        del upsampled
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_frame_fraction = {}
    for ci, (idxs, emit_idxs) in enumerate(passes):
        chunk_paths = [frame_paths[i] for i in idxs]
        # Load NATIVELY at the detection long-edge (aspect-preserved crop, /14). Default
        # 518 == the ORIGINAL VGGT4D (our fork's shared default is 448 for reconstruction;
        # we pass target_size here so the precompute matches the original, no post-hoc
        # interpolation). Lower --det_resolution to fit more frames within fixed RAM.
        images = load_and_preprocess_images(
            [str(p) for p in chunk_paths], mode=args.preprocess_mode,
            target_size=args.det_resolution)
        images = images.unsqueeze(0).to(device)  # [1, N, 3, H, W]
        n = images.shape[1]
        span = (idxs[-1] - idxs[0]) if len(idxs) > 1 else 0
        print(f"[pass {ci+1}/{len(passes)}] {n} frames  idx {idxs[0]}..{idxs[-1]}  "
              f"span {span} (stride {stride})  res={tuple(images.shape[-2:])}")

        # STAGE 1: backbone (NO mask) -> Q/K + tokens. Coarse mask from attention; and
        # Stage-1 depth+intrinsic from the tokens (the original feeds THESE to Stage 3).
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=_AMP_DTYPE):
            tokens1, patch_start1, qk_dict, enc_feat = encoder.aggregator(
                images.to(_AMP_DTYPE), dyn_masks=None)
        if coarse_by_pass is not None:
            dyn_mask = coarse_by_pass[ci].to(images.device)   # global threshold, pass A
        else:
            dyn_mask, _ = encoder.compute_attention_dynamic_mask(images, qk_dict, enc_feat)  # [1, N, H, W]
        del qk_dict, enc_feat
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.stages >= 3:
            # FAITHFUL to the ORIGINAL VGGT4D (demo_vggt4d.py): Stage 3 uses
            #   depth + intrinsic from STAGE 1 (predictions1), poses from STAGE 2 (predictions2).
            # Stage 2 suppresses dynamic tokens, so its depth on the MOVING object is degraded
            # — exactly the region Stage 3 must reason about — hence Stage-1 depth is used.
            with torch.amp.autocast("cuda", enabled=False):
                pose1 = encoder.camera_head(tokens1)
                _, intrinsic_s1 = pose_encoding_to_extri_intri(pose1[-1], images.shape[-2:])
                depth_s1, _ = encoder.depth_head(
                    tokens1, images=images, patch_start_idx=patch_start1)
            del tokens1
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # STAGE 2: re-run WITH the coarse mask (token suppression) -> refined poses.
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=_AMP_DTYPE):
                tokens2, _, _, _ = encoder.aggregator(
                    images.to(_AMP_DTYPE), dyn_masks=dyn_mask.to(images.device),
                    capture_qk=False)  # Stage 2 only needs poses; its Q/K is discarded
            with torch.amp.autocast("cuda", enabled=False):
                pose2 = encoder.camera_head(tokens2)
                extrinsic_s2, _ = pose_encoding_to_extri_intri(pose2[-1], images.shape[-2:])
            del tokens2
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # OPTIONAL: replace/augment the attention mask with a GEOMETRIC one.
            # The attention detector responds to how FAST a pixel moves, so it finds a
            # swinging arm and misses the torso. Flow residual instead asks whether a
            # pixel moves DIFFERENTLY from what the camera alone would produce, which a
            # slow torso does just as much as a fast arm -- so it covers whole objects.
            if args.mask_method in ("flow", "union"):
                res = flow_residual_map(
                    images[0], depth_s1[0].squeeze(-1), extrinsic_s2[0],
                    intrinsic_s1[0], depth_conf=None)
                # Threshold the same way the attention path does, so the two masks are
                # directly comparable and the multi-Otsu machinery is shared.
                thr = adaptive_multiotsu_variance(
                    res.cpu().numpy(), level=args.mask_otsu_level)
                flow_mask = (res > thr).float().unsqueeze(0)
                print(f"[FlowMask] residual threshold={thr:.3f}, "
                      f"dynamic pixels={flow_mask.mean()*100:.1f}% "
                      f"(attention gave {dyn_mask.mean()*100:.1f}%)", flush=True)
                dyn_mask = (torch.maximum(dyn_mask, flow_mask) if args.mask_method == "union"
                            else flow_mask)

            # STAGE 3: geometric refinement using Stage-1 depth + Stage-1 intrinsic + Stage-2 poses.
            # refine_dynamic_mask takes EXTRINSIC (world2cam) and inverts it to cam2world
            # internally, matching the original's predictions2["cam2world"].
            with torch.amp.autocast("cuda", enabled=False):
                dyn_mask = encoder.refine_dynamic_mask(
                    images, depth_s1, extrinsic_s2, intrinsic_s1, dyn_mask)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            del tokens1

        dyn_mask = dyn_mask.float().cpu()

        # Gate BEFORE completion. Completion does not only join an object to itself
        # -- close=4 + dilate=2 also bridges a person to the desk they stand beside,
        # and the gate then judges ONE component containing both. Measured: it either
        # passed (person AND desk marked) or failed (person gone entirely), which is
        # the same bug showing up two ways. On the raw detection those are separate
        # components, so the static ones can be removed and completion afterwards
        # grows only what survived.
        if args.mask_motion_gate > 0:
            if args.stages < 3:
                print("[MotionGate] needs --stages 3 (Stage-1 depth + Stage-2 poses); skipping")
            else:
                _res = flow_residual_map(images[0], depth_s1[0].squeeze(-1),
                                         extrinsic_s2[0], intrinsic_s1[0]).cpu().numpy()
                _before = float(dyn_mask.mean())
                _g = motion_gate_masks(dyn_mask[0].numpy(), _res,
                                       mult=args.mask_motion_gate)
                print(f"[MotionGate] dynamic pixels {100*_before:.1f}% -> "
                      f"{100*_g.mean():.1f}% (mult={args.mask_motion_gate}, "
                      f"residual median={float(np.median(_res)):.2f}px)", flush=True)
                dyn_mask = torch.from_numpy(_g).unsqueeze(0)

        # The detector returns PARTS of an object (limbs and outlines, not the
        # torso interior), and every downstream mechanism then splits the person:
        # masked parts get handled, unmasked parts stay and render from every
        # frame at once. Completing the shape first is what makes the mask
        # describe an OBJECT rather than the places motion was easiest to see.
        if (args.mask_close or args.mask_fill or args.mask_min_area or args.mask_dilate):
            _m = complete_masks(dyn_mask[0].numpy(), close=args.mask_close,
                                fill=args.mask_fill, min_area=args.mask_min_area,
                                dilate=args.mask_dilate)
            print(f"[MaskPost] dynamic pixels {dyn_mask.mean()*100:.1f}% -> "
                  f"{_m.mean()*100:.1f}% (close={args.mask_close} fill={args.mask_fill} "
                  f"min_area={args.mask_min_area} dilate={args.mask_dilate})", flush=True)
            dyn_mask = torch.from_numpy(_m).unsqueeze(0)

        emit_set = set(emit_idxs)
        for i, p in enumerate(chunk_paths):
            if idxs[i] not in emit_set:
                continue          # margin frame: context only, its own window is truncated
            m = dyn_mask[0, i].numpy()  # [H, W] in {0,1}
            save_mask_png(m, masks_dir / f"{p.stem}.png")
            if args.save_overlays:
                save_overlay_png(images[0, i], m, overlays_dir / f"{p.stem}.png")
            per_frame_fraction[p.name] = float((m > 0.5).mean())

    fracs = np.array(list(per_frame_fraction.values()))
    meta = {
        "dataset_name": args.dataset_name,
        "n_frames": len(frame_paths),
        "chunk_size": args.chunk_size,
        "frame_stride": stride,
        "frame_stride_arg": args.frame_stride,
        "n_passes": len(passes),
        "pass_margin": args.pass_margin,
        "mask_normalize": args.mask_normalize,
        "mask_aggregate": args.mask_aggregate,
        "mask_method": args.mask_method,
        "mask_otsu_level": args.mask_otsu_level,
        "global_post": args.global_post,
        "mask_motion_gate": args.mask_motion_gate,
        "mask_close": args.mask_close,
        "mask_fill": args.mask_fill,
        "mask_min_area": args.mask_min_area,
        "mask_dilate": args.mask_dilate,
        "mask_n_clusters": args.mask_n_clusters,
        "preprocess_mode": args.preprocess_mode,
        "det_resolution": args.det_resolution,
        "stages": args.stages,
        "dyn_fraction_mean": float(fracs.mean()) if len(fracs) else 0.0,
        "dyn_fraction_min": float(fracs.min()) if len(fracs) else 0.0,
        "dyn_fraction_max": float(fracs.max()) if len(fracs) else 0.0,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Done. Masks -> {masks_dir}")
    print(f"Dynamic fraction: mean {meta['dyn_fraction_mean']*100:.1f}%  "
          f"(min {meta['dyn_fraction_min']*100:.1f}%, max {meta['dyn_fraction_max']*100:.1f}%)")
    print("Now LOOK at the overlays: does the moving object light up?")
    print("=" * 60)


if __name__ == "__main__":
    main()
