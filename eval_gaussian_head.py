"""
Evaluation script for Temporal Gaussian Head fine-tuning.

Runs in two modes:
  1. Baseline: fresh pretrained model, no checkpoint loaded
  2. Fine-tuned: loads a checkpoint from train_temporal_gaussian_head.py

Outputs per run:
  - metrics.json            : PSNR/SSIM overall + masked to dynamic regions
  - images/                 : GT | predicted comparison images for every frame
  - rgb.mp4                 : novel view synthesis video (interpolated predicted poses)
  - depth.mp4               : depth video
  - gaussians.ply           : 3D Gaussian point cloud (last batch)
  - dyn_mask/               : dynamic mask overlays (VGGT4D only)

Usage:
    # Baseline (VGGT4D)
    python eval_gaussian_head.py \
        --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
        --dataset_name rgbd_bonn_crowd3 \
        --output_dir output_eval_baseline

    # Fine-tuned
    python eval_gaussian_head.py \
        --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
        --dataset_name rgbd_bonn_crowd3 \
        --checkpoint output_finetune_initial/checkpoint_best.pt \
        --output_dir output_eval_finetuned

    # Original VGGT (no VGGT4D)
    python eval_gaussian_head.py \
        --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
        --dataset_name rgbd_bonn_crowd3 \
        --no_vggt4d \
        --output_dir output_eval_vggt
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.utils as vutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_temporal_gaussian_head import (
    VideoFrameDataset,
    create_model,
    compute_rendering_loss,
    load_precomputed_masks,
    INTRINSICS_PRESETS,
    TrainingConfig,
)
from src.evaluation.metrics import compute_psnr, compute_ssim
from src.misc.image_io import save_interpolated_video, save_image
from src.model.ply_export import export_ply


def load_model(checkpoint_path, config, device):
    model = create_model(config)
    model = model.to(device)

    if checkpoint_path is not None:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Only restore gaussian head weights — never the frozen VGGT4D backbone —
        # so the backbone always reflects the freshly loaded pretrained weights.
        saved = ckpt["model_state_dict"]
        current = model.state_dict()
        head_keys = {k: v for k, v in saved.items()
                     if "gaussian_param_head" in k or "gaussian_adapter" in k}
        current.update(head_keys)
        model.load_state_dict(current)
        epoch = ckpt.get("epoch", "?")
        step = ckpt.get("global_step", "?")
        print(f"  -> epoch {epoch}, step {step}, restored {len(head_keys)} gaussian head tensors")
    else:
        print("No checkpoint — running pretrained weights only")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def save_dynamic_mask_overlay(image, dyn_mask, path):
    """Save RGB image with dynamic mask as red overlay."""
    img_np = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    mask_np = dyn_mask.cpu().numpy()                # [H, W]

    overlay = img_np.copy()
    overlay[mask_np > 0.5] = overlay[mask_np > 0.5] * 0.5 + np.array([0.8, 0.1, 0.1]) * 0.5
    overlay = np.clip(overlay, 0, 1)

    Image.fromarray((overlay * 255).astype(np.uint8)).save(path)


@torch.no_grad()
def evaluate(model, dataloader, config, output_dir, device, max_image_batches=50, image_batch_start=0,
             per_frame_dynamic=False, leave_one_out=False, precomputed_mask_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    dyn_mask_dir = os.path.join(output_dir, "dyn_mask")
    os.makedirs(images_dir, exist_ok=True)
    if precomputed_mask_dir is not None:
        print(f"  dynamic masks: LOADING precomputed from {precomputed_mask_dir} "
              f"(overrides live detection for the dyn/static split)")
    n_precomp_hits = 0

    total_psnr, total_ssim = 0.0, 0.0
    total_psnr_dyn = 0.0
    total_psnr_static = 0.0
    total_dyn_pixel_fraction = 0.0
    n_dyn_frames = 0
    n_static_frames = 0
    n_frames = 0

    last_gaussians = None
    last_pred_pose = None
    last_h, last_w = None, None
    last_dyn_mask = None

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        if images.dim() == 4:
            images = images.unsqueeze(0)
        b, v, c, h, w = images.shape

        encoder_output = model.encoder(images, global_step=0)
        gaussians = encoder_output.gaussians
        infos = encoder_output.infos
        pred_pose = encoder_output.pred_context_pose

        # Always use predicted poses — GT poses are in Bonn world frame,
        # incompatible with VGGT4D's predicted world frame (Gaussians would project outside frustum).
        ext = pred_pose["extrinsic"]
        intr = pred_pose["intrinsic"].clone()
        intr = torch.stack([intr[:, :, 0] * w, intr[:, :, 1] * h, intr[:, :, 2]], dim=2)

        # Per-frame dynamic compositing / leave-one-out. Both default OFF, so the
        # stored baselines reproduce exactly.
        _, decoder_out = compute_rendering_loss(
            model, images, gaussians, ext, intr,
            gaussian_frame_idx=(infos.get("gaussian_frame_idx")
                                if (per_frame_dynamic or leave_one_out) else None),
            gaussian_dyn_flag=(infos.get("gaussian_dyn_flag")
                               if per_frame_dynamic else None),
            leave_one_out=leave_one_out,
        )
        pred_rgb = decoder_out.color  # [B, V, 3, H, W] in [0, 1]

        dyn_mask = infos.get("dyn_mask", None)  # [B, V, H, W] on CPU, or None

        # Override the live (per-window) detection with precomputed full-span masks.
        if precomputed_mask_dir is not None:
            raw_names = batch.get("frame_names")
            if raw_names is not None:
                # default_collate wraps each name in a 1-tuple at batch_size=1
                frame_names = [x[0] if isinstance(x, (list, tuple)) else x for x in raw_names]
                ds = batch.get("dataset_name")
                if isinstance(ds, (list, tuple)):
                    ds = ds[0]
                loaded = load_precomputed_masks(frame_names, precomputed_mask_dir, h, w, device, dataset_name=ds)
                if loaded is not None:
                    dyn_mask = loaded
                    n_precomp_hits += 1

        # --- Per-frame metrics and comparison images ---
        for v_idx in range(v):
            pred_frame = pred_rgb[0, v_idx].clamp(0, 1)   # [3, H, W]
            gt_frame = images[0, v_idx].clamp(0, 1)        # [3, H, W]

            psnr_val = compute_psnr(pred_frame.unsqueeze(0), gt_frame.unsqueeze(0)).mean().item()
            ssim_val = compute_ssim(pred_frame.unsqueeze(0), gt_frame.unsqueeze(0)).mean().item()
            total_psnr += psnr_val
            total_ssim += ssim_val
            n_frames += 1

            # Dynamic-masked metrics (PSNR only — masked SSIM is unreliable due to zero-padding bias)
            if dyn_mask is not None:
                mask = dyn_mask[0, v_idx].to(device)   # [H, W]
                n_total_px = mask.numel()
                n_px = mask.sum().item()
                total_dyn_pixel_fraction += n_px / n_total_px

                if n_px >= 10:
                    mask3 = mask.unsqueeze(0).expand(3, -1, -1)
                    mse_dyn = ((pred_frame * mask3 - gt_frame * mask3) ** 2).sum() / (3 * n_px)
                    total_psnr_dyn += -10 * torch.log10(mse_dyn + 1e-8).item()
                    n_dyn_frames += 1

                # Static-masked metrics (complement of dyn_mask)
                n_px_s = n_total_px - n_px
                if n_px_s >= 10:
                    static_mask3 = (1.0 - mask).clamp(0, 1).unsqueeze(0).expand(3, -1, -1)
                    mse_static = ((pred_frame * static_mask3 - gt_frame * static_mask3) ** 2).sum() / (3 * n_px_s)
                    total_psnr_static += -10 * torch.log10(mse_static + 1e-8).item()
                    n_static_frames += 1

            # Save GT | predicted comparison image for a window of batches
            if image_batch_start <= batch_idx < image_batch_start + max_image_batches:
                comparison = torch.cat([gt_frame, pred_frame], dim=2)  # side by side [3, H, 2W]
                save_image(comparison, os.path.join(images_dir, f"b{batch_idx:04d}_v{v_idx:02d}.png"))

                if dyn_mask is not None:
                    os.makedirs(dyn_mask_dir, exist_ok=True)
                    save_dynamic_mask_overlay(
                        gt_frame, dyn_mask[0, v_idx],
                        os.path.join(dyn_mask_dir, f"b{batch_idx:04d}_v{v_idx:02d}.png")
                    )

        # Keep last batch for video + PLY output
        last_gaussians = gaussians
        last_pred_pose = pred_pose
        last_h, last_w = h, w
        last_dyn_mask = dyn_mask  # [B, V, H, W] or None

    # --- Video output (interpolated predicted poses, last batch) ---
    if last_gaussians is not None and last_pred_pose is not None:
        print("Saving rgb.mp4 and depth.mp4...")
        save_interpolated_video(
            last_pred_pose["extrinsic"],
            last_pred_pose["intrinsic"],
            1, last_h, last_w,
            last_gaussians,
            output_dir,
            model.decoder,
        )

    # --- PLY export (last batch) ---
    if last_gaussians is not None:
        print("Saving gaussians.ply...")
        ply_path = os.path.join(output_dir, "gaussians.ply")
        # Flatten dynamic mask to match Gaussian layout [V*H*W] (assumes no voxelization)
        dyn_mask_flat = None
        if last_dyn_mask is not None:
            dyn_mask_flat = last_dyn_mask[0].cpu().numpy().reshape(-1).astype(np.float32)

        export_ply(
            last_gaussians.means[0],
            last_gaussians.scales[0],
            last_gaussians.rotations[0],
            last_gaussians.harmonics[0],
            last_gaussians.opacities[0],
            Path(ply_path),
            save_sh_dc_only=True,
            dyn_mask_flat=dyn_mask_flat,
            dyn_opacity_scale=0.5,  # dim dynamic Gaussians to 50% rather than near-invisible
        )

    # --- Metrics summary ---
    avg_dyn_pixel_frac = total_dyn_pixel_fraction / n_frames if n_frames > 0 else None
    metrics = {
        "psnr": total_psnr / n_frames if n_frames > 0 else 0.0,
        "ssim": total_ssim / n_frames if n_frames > 0 else 0.0,
        "psnr_dynamic": total_psnr_dyn / n_dyn_frames if n_dyn_frames > 0 else None,
        "psnr_static": total_psnr_static / n_static_frames if n_static_frames > 0 else None,
        "avg_dyn_pixel_fraction": avg_dyn_pixel_frac,
        "n_frames": n_frames,
        "n_dynamic_frames": n_dyn_frames,
        "n_static_frames": n_static_frames,
        "mask_source": ("precomputed" if precomputed_mask_dir is not None else "live_detection"),
        "precomputed_mask_dir": precomputed_mask_dir,
        "n_batches_with_precomputed_mask": n_precomp_hits,
    }

    print(f"\nResults:")
    print(f"  PSNR (overall):          {metrics['psnr']:.2f} dB")
    print(f"  SSIM (overall):          {metrics['ssim']:.4f}")
    if metrics["psnr_dynamic"] is not None:
        print(f"  PSNR (dynamic regions):  {metrics['psnr_dynamic']:.2f} dB")
        print(f"  PSNR (static  regions):  {metrics['psnr_static']:.2f} dB")
        print(f"  Avg dynamic pixel frac:  {metrics['avg_dyn_pixel_fraction']:.1%}")
    else:
        print(f"  PSNR (dynamic regions):  N/A")
        print(f"  PSNR (static  regions):  N/A")

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOutputs saved to {output_dir}/")
    print(f"  metrics.json, images/, rgb.mp4, depth.mp4, gaussians.ply"
          + (", dyn_mask/" if os.path.exists(dyn_mask_dir) else ""))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs fine-tuned Gaussian Head")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--intrinsics", type=str, default="bonn",
                        help="Intrinsics preset: 'bonn', 'tum_fr1', 'tum_fr3'")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned checkpoint. Omit for baseline.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Gap between the frames in a window. stride>1 spreads the SAME num_frames over a "
                             "longer time span (more object motion for the dynamic detector) at NO extra memory. "
                             "Use to test whether the weak dynamic mask is a temporal-context problem (0.4s window).")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "all"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_vggt4d", action="store_true",
                        help="Use original VGGT backbone instead of VGGT4D (no dynamic detection)")
    parser.add_argument("--vggt4d_weights_path", type=str, default=None,
                        help="Path to VGGT4D fine-tuned weights (.pt). If omitted, initializes from VGGT-1B.")
    parser.add_argument("--max_image_batches", type=int, default=50,
                        help="Save comparison images for N batches (avoids disk quota).")
    parser.add_argument("--per_frame_dynamic", action="store_true",
                        help="Render dynamic Gaussians ONLY into the frame they were unprojected from "
                             "(static ones still render into every frame). Removes the multi-frame ghosting "
                             "of moving objects. Requires VGGT4D dynamic detection. Off = original behaviour.")
    parser.add_argument("--eval_loo", action="store_true",
                        help="Leave-one-out: when rendering view j, drop ALL Gaussians that came from view j, "
                             "so j must be reconstructed from the OTHER frames. The honest control against "
                             "self-reprojection — a large LOO gap on dynamic regions is the expected result "
                             "(this architecture cannot model motion) and is itself reportable.")
    parser.add_argument("--image_batch_start", type=int, default=0,
                        help="First batch index to start saving images from. Use ~half total batches for mid-sequence.")
    parser.add_argument("--dyn_mask_dir", type=str, default=None,
                        help="Directory of PRECOMPUTED dynamic-mask PNGs (named by rgb frame stem), e.g. "
                             "output_dyn_masks_precomputed_cs16_r518_st3_fs49/<SEQ>/masks. When set, these "
                             "override the live per-window detection for the dynamic/static PSNR split — "
                             "use the validated 518+full-span masks instead of the weak in-eval detection.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode:     {'Fine-tuned' if args.checkpoint else 'Baseline'}")
    backbone_label = "VGGT (original)" if args.no_vggt4d else \
        f"VGGT4D (weights: {args.vggt4d_weights_path or 'init from VGGT-1B'})"
    print(f"Backbone: {backbone_label}")

    intrinsics = INTRINSICS_PRESETS[args.intrinsics]

    config = TrainingConfig(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        num_frames=args.num_frames,
        use_vggt4d=not args.no_vggt4d,
        enable_dynamic_detection=not args.no_vggt4d,
        vggt4d_weights_path=args.vggt4d_weights_path,
    )

    print(f"\nLoading {args.split} dataset...")
    dataset = VideoFrameDataset(
        args.data_dir,
        args.dataset_name,
        intrinsics=intrinsics,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        image_size=config.image_size,
        split=args.split,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"  {len(dataset)} sequences")

    print("\nLoading model...")
    model = load_model(args.checkpoint, config, device)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_config.json"), "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "dataset": f"{args.data_dir}/{args.dataset_name}",
            "split": args.split,
            "num_frames": args.num_frames,
            "backbone": "vggt" if args.no_vggt4d else "vggt4d",
            "mode": "finetuned" if args.checkpoint else "baseline",
            "per_frame_dynamic": args.per_frame_dynamic,
            "leave_one_out": args.eval_loo,
        }, f, indent=2)

    print(f"\nRunning evaluation on {args.split} split ({len(dataset)} batches)...")
    print(f"  per_frame_dynamic={args.per_frame_dynamic}  leave_one_out={args.eval_loo}")
    evaluate(model, dataloader, config, args.output_dir, device,
             max_image_batches=args.max_image_batches,
             image_batch_start=args.image_batch_start,
             per_frame_dynamic=args.per_frame_dynamic,
             leave_one_out=args.eval_loo,
             precomputed_mask_dir=args.dyn_mask_dir)


if __name__ == "__main__":
    main()
