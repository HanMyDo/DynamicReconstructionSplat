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

STAGE-1 ONLY (deliberate — see plan). The original does Stage 1 (attention mask)
    -> Stage 2 (refine poses) -> Stage 3 (geometric mask refine) and SAVES the
    Stage-3 mask. This script does STAGE 1 ONLY, because:
      (a) our hypothesis is that TEMPORAL CONTEXT (not Stage 3) was the dominant
          problem — the box experiment showed widening the window alone, with
          Stage 3 unchanged, moved the mask onto the object; and
      (b) our current Stage-3 integration ERODES the mask (~8-10% -> ~5-7%),
          the opposite of the original (likely because we don't feed it Stage-2
          refined poses). Copying it would re-break the mask.
    DECISION: validate the Stage-1 long-window mask first. If it covers the
    objects -> done (simpler than the original). If it's noisy/patchy -> then
    build the faithful Stage 2->3 (needs depth+poses, still no Gaussians).

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
from PIL import Image

from train_temporal_gaussian_head import create_model, TrainingConfig
from src.model.encoder.anysplat import _AMP_DTYPE
from src.model.encoder.vggt.utils.load_fn import load_and_preprocess_images


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
    ranges = chunk_ranges(len(frame_paths), args.chunk_size)
    print(f"Sequence: {args.dataset_name}  |  {len(frame_paths)} frames  |  "
          f"{len(ranges)} chunk(s) of ~{args.chunk_size}")

    print("Creating model (VGGT4D backbone + dynamic detection)...")
    config = TrainingConfig(
        use_vggt4d=True,
        enable_dynamic_detection=True,
        vggt4d_weights_path=args.vggt4d_weights_path,
    )
    model = create_model(config).to(device).eval()
    encoder = model.encoder

    per_frame_fraction = {}
    for ci, (s, e) in enumerate(ranges):
        chunk_paths = frame_paths[s:e]
        # Original preprocessing: 518 long edge, aspect-preserved -> [N, 3, H, W] float32 in [0,1]
        images = load_and_preprocess_images([str(p) for p in chunk_paths], mode=args.preprocess_mode)
        images = images.unsqueeze(0).to(device)  # [1, N, 3, H, W]
        n = images.shape[1]
        print(f"[chunk {ci+1}/{len(ranges)}] frames {s}..{e-1} ({n})  res={tuple(images.shape[-2:])}")

        # Pass 1 ONLY: backbone attention -> Q/K, then attention-based dynamic mask.
        # Mirrors EncoderAnySplat.forward's detection path; NO Gaussians, NO rendering,
        # NO Stage 2/3.
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=_AMP_DTYPE):
            _, _, qk_dict, enc_feat = encoder.aggregator(images.to(_AMP_DTYPE), dyn_masks=None)
        dyn_mask, _ = encoder.compute_attention_dynamic_mask(images, qk_dict, enc_feat)  # [1, N, H, W]
        dyn_mask = dyn_mask.float().cpu()
        del qk_dict, enc_feat
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for i, p in enumerate(chunk_paths):
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
        "preprocess_mode": args.preprocess_mode,
        "stage": "stage1_only (no Stage 2/3)",
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
