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
        --use_temporal_attention \
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
        # Only restore temporal_attention weights — never the frozen backbone —
        # so the backbone always reflects the freshly loaded pretrained weights.
        saved = ckpt["model_state_dict"]
        current = model.state_dict()
        temporal_keys = {k: v for k, v in saved.items() if "temporal_attention" in k}
        current.update(temporal_keys)
        model.load_state_dict(current)
        epoch = ckpt.get("epoch", "?")
        step = ckpt.get("global_step", "?")
        print(f"  -> epoch {epoch}, step {step}, restored {len(temporal_keys)} temporal_attention tensors")
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
def evaluate(model, dataloader, config, output_dir, device, max_image_batches=50, image_batch_start=0):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    dyn_mask_dir = os.path.join(output_dir, "dyn_mask")
    os.makedirs(images_dir, exist_ok=True)

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

        _, decoder_out = compute_rendering_loss(model, images, gaussians, ext, intr)
        pred_rgb = decoder_out.color  # [B, V, 3, H, W] in [0, 1]

        dyn_mask = infos.get("dyn_mask", None)  # [B, V, H, W] on CPU, or None

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
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "all"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_temporal_attention", action="store_true",
                        help="Enable temporal attention (set if checkpoint was trained with it)")
    parser.add_argument("--no_vggt4d", action="store_true",
                        help="Use original VGGT backbone instead of VGGT4D (no dynamic detection)")
    parser.add_argument("--vggt4d_weights_path", type=str, default=None,
                        help="Path to VGGT4D fine-tuned weights (.pt). If omitted, initializes from VGGT-1B.")
    parser.add_argument("--max_image_batches", type=int, default=50,
                        help="Save comparison images for N batches (avoids disk quota).")
    parser.add_argument("--image_batch_start", type=int, default=0,
                        help="First batch index to start saving images from. Use ~half total batches for mid-sequence.")
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
        use_temporal_attention=args.use_temporal_attention,
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
        }, f, indent=2)

    print(f"\nRunning evaluation on {args.split} split ({len(dataset)} batches)...")
    evaluate(model, dataloader, config, args.output_dir, device,
             max_image_batches=args.max_image_batches,
             image_batch_start=args.image_batch_start)


if __name__ == "__main__":
    main()
