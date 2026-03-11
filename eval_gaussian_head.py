"""
Evaluation script for Temporal Gaussian Head fine-tuning.

Runs in two modes:
  1. Baseline: fresh pretrained model, no checkpoint loaded
  2. Fine-tuned: loads a checkpoint from train_temporal_gaussian_head.py

Computes PSNR/SSIM overall and masked to dynamic regions.
Saves side-by-side comparison images (GT | predicted).

Usage:
    # Baseline
    python eval_gaussian_head.py \
        --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
        --dataset_name rgbd_bonn_crowd3 \
        --output_dir output_eval_baseline

    # Fine-tuned
    python eval_gaussian_head.py \
        --data_dir /tmp/bonn_data/rgbd_bonn_dataset \
        --dataset_name rgbd_bonn_crowd3 \
        --checkpoint output_finetune_smoketest/checkpoint_best.pt \
        --output_dir output_eval_finetuned
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


def load_model(checkpoint_path, config, device):
    model = create_model(config)
    model = model.to(device)

    if checkpoint_path is not None:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        step = ckpt.get("global_step", "?")
        print(f"  -> epoch {epoch}, step {step}")
    else:
        print("No checkpoint provided — running baseline (pretrained weights only)")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


@torch.no_grad()
def evaluate(model, dataloader, intrinsics, config, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    total_psnr, total_ssim = 0.0, 0.0
    total_psnr_dyn, total_ssim_dyn = 0.0, 0.0
    n_dyn_batches = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        if images.dim() == 4:
            images = images.unsqueeze(0)
        b, v, c, h, w = images.shape

        encoder_output = model.encoder(images, global_step=0)
        gaussians = encoder_output.gaussians
        infos = encoder_output.infos

        # Use GT poses if available, else fall back to predicted
        if "gt_extrinsics" in batch:
            ext = batch["gt_extrinsics"].to(device)
            intr = batch["gt_intrinsics"].to(device)
            if ext.dim() == 3:
                ext = ext.unsqueeze(0)
                intr = intr.unsqueeze(0)
        else:
            pred = encoder_output.pred_context_pose
            ext = pred["extrinsic"]
            intr = pred["intrinsic"].clone()
            intr = torch.stack([intr[:, :, 0] * w, intr[:, :, 1] * h, intr[:, :, 2]], dim=2)

        mse_loss, decoder_out = compute_rendering_loss(model, images, gaussians, ext, intr)
        pred_rgb = decoder_out.color  # [B, V, 3, H, W]

        # Overall metrics (per-frame average)
        for v_idx in range(v):
            pred_frame = pred_rgb[0, v_idx]   # [3, H, W] in [0, 1]
            gt_frame = images[0, v_idx]        # [3, H, W] in [0, 1]

            psnr_val = compute_psnr(pred_frame.unsqueeze(0), gt_frame.unsqueeze(0)).mean().item()
            ssim_val = compute_ssim(pred_frame.unsqueeze(0), gt_frame.unsqueeze(0)).mean().item()
            total_psnr += psnr_val
            total_ssim += ssim_val

        # Dynamic-masked metrics
        dyn_mask = infos.get("dyn_mask", None)  # [B, V, H, W] on CPU
        if dyn_mask is not None:
            dyn_mask_gpu = dyn_mask.to(device)  # [B, V, H, W]
            for v_idx in range(v):
                mask = dyn_mask_gpu[0, v_idx]   # [H, W], 1=dynamic
                if mask.sum() < 10:
                    continue
                pred_frame = pred_rgb[0, v_idx]
                gt_frame = images[0, v_idx]
                # Apply mask: zero out static pixels
                mask3 = mask.unsqueeze(0).expand(3, -1, -1)  # [3, H, W]
                pred_masked = pred_frame * mask3
                gt_masked = gt_frame * mask3
                n_pixels = mask.sum().item()
                mse_dyn = ((pred_masked - gt_masked) ** 2).sum() / (3 * n_pixels)
                psnr_dyn = -10 * torch.log10(mse_dyn + 1e-8).item()
                ssim_dyn = compute_ssim(pred_masked.unsqueeze(0), gt_masked.unsqueeze(0)).mean().item()
                total_psnr_dyn += psnr_dyn
                total_ssim_dyn += ssim_dyn
                n_dyn_batches += 1

        num_batches += 1

        # Save side-by-side comparison images (first frame of each batch)
        gt_vis = images[0, 0].cpu().clamp(0, 1)
        pred_vis = pred_rgb[0, 0].cpu().clamp(0, 1)
        comparison = torch.stack([gt_vis, pred_vis], dim=0)  # [2, 3, H, W]
        vutils.save_image(comparison, os.path.join(images_dir, f"batch{batch_idx:04d}.png"), nrow=2)

    n_frames = num_batches * v
    metrics = {
        "psnr": total_psnr / n_frames,
        "ssim": total_ssim / n_frames,
        "psnr_dynamic": total_psnr_dyn / n_dyn_batches if n_dyn_batches > 0 else None,
        "ssim_dynamic": total_ssim_dyn / n_dyn_batches if n_dyn_batches > 0 else None,
        "n_batches": num_batches,
        "n_dynamic_frames_evaluated": n_dyn_batches,
    }

    print(f"\nResults:")
    print(f"  PSNR (overall):  {metrics['psnr']:.2f} dB")
    print(f"  SSIM (overall):  {metrics['ssim']:.4f}")
    if metrics["psnr_dynamic"] is not None:
        print(f"  PSNR (dynamic regions): {metrics['psnr_dynamic']:.2f} dB")
        print(f"  SSIM (dynamic regions): {metrics['ssim_dynamic']:.4f}")
    else:
        print(f"  PSNR (dynamic regions): N/A (no dynamic regions detected)")

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_dir}/metrics.json")
    print(f"Comparison images saved to {images_dir}/")

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
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_temporal_attention", action="store_true",
                        help="Enable temporal attention (set if checkpoint was trained with it)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {'Fine-tuned' if args.checkpoint else 'Baseline'}")

    intrinsics = INTRINSICS_PRESETS[args.intrinsics]

    config = TrainingConfig(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        num_frames=args.num_frames,
        use_temporal_attention=args.use_temporal_attention,
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
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
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
            "mode": "finetuned" if args.checkpoint else "baseline",
        }, f, indent=2)

    print(f"\nRunning evaluation on {args.split} split...")
    evaluate(model, dataloader, intrinsics, config, args.output_dir, device)


if __name__ == "__main__":
    main()
