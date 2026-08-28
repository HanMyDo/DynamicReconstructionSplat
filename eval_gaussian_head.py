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
from src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips
from src.misc.image_io import save_interpolated_video, save_image
from src.model.ply_export import export_ply


# Config fields that change the MODEL ARCHITECTURE, not just its behaviour. The eval
# model must be built with the same values the checkpoint was trained with, or the
# corresponding weights have nowhere to load and are silently dropped.
# This bit us: eval never set use_temporal_attention, so a temporally-trained checkpoint
# was evaluated on a model with NO temporal block -- its 8 tensors were skipped by the
# `k in current` filter, and the head weights (trained alongside that block) were loaded
# into an architecture missing it. Both temporal evals were therefore meaningless.
# Only fields that OWN SAVED PARAMETERS belong here. use_vggt4d / hybrid_voxelize /
# voxel_size change behaviour but add no weights, and use_vggt4d in particular is set
# explicitly on the CLI (--no_vggt4d) for the backbone ablation -- the checkpoint must
# not silently override that. Anything else that mismatches is caught by the orphan
# guard below rather than being second-guessed here.
_ARCH_FIELDS = (
    "use_temporal_attention",
    "temporal_spatial_downsample",
    "temporal_num_heads",
    "temporal_use_pe",
)


def _apply_checkpoint_arch(config, ckpt, override_keys=()):
    """Copy architecture-affecting fields from the checkpoint's stored config."""
    saved_cfg = ckpt.get("config") or {}
    if not saved_cfg:
        print("[arch] checkpoint has no stored config -- using CLI/default architecture",
              flush=True)
        return config
    changed = []
    for f in _ARCH_FIELDS:
        if f in override_keys or f not in saved_cfg:
            continue
        want = saved_cfg[f]
        if hasattr(config, f) and getattr(config, f) != want:
            changed.append(f"{f}: {getattr(config, f)} -> {want}")
            setattr(config, f, want)
    print(f"[arch] from checkpoint: {changed if changed else 'no changes needed'}", flush=True)
    return config


def load_model(checkpoint_path, config, device):
    ckpt = None
    if checkpoint_path is not None:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = _apply_checkpoint_arch(config, ckpt)

    model = create_model(config)
    model = model.to(device)

    if checkpoint_path is not None:
        # Only restore gaussian head weights — never the frozen VGGT4D backbone —
        # so the backbone always reflects the freshly loaded pretrained weights.
        saved = ckpt["model_state_dict"]
        current = model.state_dict()
        # Restore exactly the modules the TRAINING run saved. The checkpoint records
        # them ('saved_prefixes'), so eval can never drift from train -- the old
        # hardcoded pair silently discarded any other unfrozen module (e.g.
        # depth_head), leaving pretrained weights in place and producing plausible
        # but meaningless numbers. Older checkpoints lack the key -> same pair as before.
        # Restore EVERYTHING the training run saved, except the frozen backbone.
        # Relying on a name list was fragile: the training script builds THREE separate
        # checkpoint dicts (periodic, best, final) and only one of them carried
        # 'saved_prefixes', so a dh checkpoint silently restored just 64 head tensors
        # and evaluated PRETRAINED geometry. head_state_dict() already filters at save
        # time, so whatever is in the file is what was trained -- load all of it and
        # exclude only 'aggregator' (the backbone, which must stay at pretrained
        # weights and is never saved by the current code anyway).
        head_keys = {k: v for k, v in saved.items()
                     if k in current and "aggregator" not in k}
        # Any saved tensor with NO matching key in the model means the eval architecture
        # differs from the trained one. Previously these were dropped in silence, which
        # is how a temporally-trained checkpoint got evaluated without its temporal block.
        orphans = [k for k in saved
                   if k not in current and "aggregator" not in k]
        if orphans:
            raise RuntimeError(
                f"{len(orphans)} saved tensors have no matching module in the eval model "
                f"(e.g. {orphans[:4]}). The evaluation architecture does not match the "
                "trained one -- refusing to evaluate a different model than was trained.")
        prefixes = ckpt.get("saved_prefixes")
        groups = sorted({k.split(".")[1] if k.startswith("encoder.") else k.split(".")[0]
                         for k in head_keys})
        current.update(head_keys)
        model.load_state_dict(current)
        print(f"[ckpt] restored {len(head_keys)} tensors; modules={groups}", flush=True)
        if prefixes:
            print(f"[ckpt] training recorded saved_prefixes={prefixes}", flush=True)
            # Only a real failure if the module HAS parameters in this model but none
            # of them were restored. gaussian_adapter is parameter-free, so it
            # legitimately contributes zero tensors -- the first version of this check
            # raised on it and killed a valid eval.
            missing = [p for p in prefixes
                       if any(p in k for k in current)
                       and not any(p in k for k in head_keys)]
            if missing:
                raise RuntimeError(
                    f"checkpoint says it trained {missing} but no such tensors were "
                    "restored -- refusing to evaluate a partly-pretrained model")
        if not head_keys:
            raise RuntimeError(
                f"checkpoint contained no tensors matching {prefixes} -- refusing to "
                "evaluate a silently-pretrained model")
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


def optimal_gain(pred, gt):
    """Least-squares scalar g minimising ||g*pred - gt||^2, i.e. a pure EXPOSURE fix.

    Control for the following confound: the frozen head was trained by AnySplat under
    FULL compositing, where each pixel's own-frame Gaussian supplies most of the energy.
    Under leave-one-out that contribution is gone, so the frozen model systematically
    UNDER-renders. Fine-tuning can then win a large PSNR delta by simply turning the gain
    up -- with no structural improvement at all. Applying the optimal gain to the FROZEN
    model measures how much of the reported gain is merely this brightness mismatch.
    """
    num = (pred * gt).sum()
    den = (pred * pred).sum().clamp_min(1e-8)
    return (num / den).clamp(0.1, 10.0)


def umeyama_ate(pred_xyz, gt_xyz):
    """Sim(3)-aligned ATE (RMSE, metres) between two camera-centre trajectories.

    Bonn's GT poses live in a different world frame (and scale) than the predicted ones,
    so a similarity alignment is required before any comparison is meaningful -- this is
    the standard trajectory-evaluation procedure. Returns None if degenerate.
    """
    X = np.asarray(pred_xyz, dtype=np.float64).T          # 3 x N
    Y = np.asarray(gt_xyz, dtype=np.float64).T
    if X.shape[1] < 3:
        return None
    mx, my = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
    Xc, Yc = X - mx, Y - my
    var = (Xc ** 2).sum()
    if var < 1e-12:
        return None
    U, D, Vt = np.linalg.svd(Yc @ Xc.T / X.shape[1])
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / (var / X.shape[1])
    err = Y - (scale * R @ X + (my - scale * R @ mx))
    return float(np.sqrt((err ** 2).sum(0).mean()))


@torch.no_grad()
def evaluate(model, dataloader, config, output_dir, device, max_image_batches=50, image_batch_start=0,
             per_frame_dynamic=False, leave_one_out=False, precomputed_mask_dir=None,
             track_dynamic=False, gain_correct=False, scale_mult=1.0,
             image_save_every=1, batch_stride=1):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    dyn_mask_dir = os.path.join(output_dir, "dyn_mask")
    os.makedirs(images_dir, exist_ok=True)
    if precomputed_mask_dir is not None:
        print(f"  dynamic masks: LOADING precomputed from {precomputed_mask_dir} "
              f"(overrides live detection for the dyn/static split)")
    n_precomp_hits = 0
    n_group_motion = 0   # batches where the tracker-driven motion model was available
    n_knn_motion = 0     # batches where the scene-flow displacement field was available
    total_gain = 0.0; n_gain = 0            # mean applied exposure gain (diagnostic)
    total_ate = 0.0;  n_ate = 0             # per-window Sim(3)-aligned ATE

    total_psnr, total_ssim = 0.0, 0.0
    total_psnr_dyn = 0.0
    total_psnr_static = 0.0
    total_dyn_pixel_fraction = 0.0
    # LPIPS: perceptual, and unlike PSNR it heavily penalises GHOSTING/blur of moving
    # objects — the failure mode this thesis is about. lpips_dynamic is computed on the
    # dynamic-region bounding-box CROP (not a zero-masked image, which would inject
    # artificial black edges into the perceptual network).
    total_lpips = 0.0
    total_lpips_dyn = 0.0
    n_lpips_dyn_frames = 0
    n_dyn_frames = 0
    n_static_frames = 0
    n_frames = 0

    last_gaussians = None
    last_pred_pose = None
    last_h, last_w = None, None
    last_dyn_mask = None

    for batch_idx, batch in enumerate(dataloader):
        # SUBSAMPLE THE WINDOWS. Metrics are averaged over every processed window, and
        # the windows are heavily overlapping sliding windows, so evaluating every Nth
        # one still spans the whole sequence at a fraction of the cost.
        # Needed for long sequences: TUM fr2/desk_with_person has 3670 windows (22020
        # frames) vs ~950 for a Bonn sequence, so a full pass runs >2h and the 24g
        # watchdog cancels it for low GPU-memory utilisation (41.7% < 50% threshold --
        # an nf6 eval only uses ~10GB of a 24GB card).
        if batch_stride > 1 and (batch_idx % batch_stride) != 0:
            continue
        images = batch["images"].to(device)
        if images.dim() == 4:
            images = images.unsqueeze(0)
        b, v, c, h, w = images.shape

        # Load precomputed masks ONCE — used for BOTH the per-frame compositing gate
        # (passed into the encoder as dyn_mask_override -> gaussian_dyn_flag) AND the
        # dyn/static PSNR split below.
        precomp_mask = None
        if precomputed_mask_dir is not None:
            raw_names = batch.get("frame_names")
            if raw_names is not None:
                # default_collate wraps each name in a 1-tuple at batch_size=1
                frame_names = [x[0] if isinstance(x, (list, tuple)) else x for x in raw_names]
                ds = batch.get("dataset_name")
                if isinstance(ds, (list, tuple)):
                    ds = ds[0]
                precomp_mask = load_precomputed_masks(
                    frame_names, precomputed_mask_dir, h, w, device, dataset_name=ds)

        encoder_output = model.encoder(images, global_step=0, dyn_mask_override=precomp_mask)
        gaussians = encoder_output.gaussians
        # DIAGNOSTIC ONLY (--scale_mult): enlarge every Gaussian at render time.
        # Discriminates the two explanations for the static-PSNR collapse under
        # hybrid fusion. The frozen head's scales are sized for ~0.001 point
        # spacing; at voxel_size 0.005 the fused spacing is ~5x larger, so the
        # splats cover a small fraction of the surface. If PSNR RECOVERS when we
        # simply scale them up, the collapse is COVERAGE (which training fixes,
        # since scale is a head output). If it does NOT recover, the collapse is
        # GEOMETRIC -- fusion averaged depth estimates that disagree -- and no
        # head fine-tuning can repair it, because means come from the FROZEN
        # depth head. Never use this for a reported number.
        if batch_idx == 0:
            print(f"[scale_mult] effective={scale_mult} "
                  f"(1.0 = diagnostic OFF) scale_mean={float(gaussians.scales.mean()):.6f} "
                  f"covar_mean={float(gaussians.covariances.mean()):.9f}",
                  flush=True)
        if scale_mult != 1.0:
            gaussians.scales = gaussians.scales * scale_mult
            # THE COVARIANCES ARE WHAT ACTUALLY RENDER. decoder_splatting_cuda passes
            # covars=covar_i to gsplat.rasterization, and gsplat uses explicit covars
            # INSTEAD of scales/quats when they are supplied -- so scaling .scales
            # alone changes nothing (measured: metrics byte-identical). Covariance is
            # quadratic in linear size, hence scale_mult ** 2.
            if getattr(gaussians, "covariances", None) is not None:
                gaussians.covariances = gaussians.covariances * (scale_mult ** 2)
            if batch_idx == 0:
                print(f"[scale_mult] APPLIED x{scale_mult} -> "
                      f"scale_mean={float(gaussians.scales.mean()):.6f} "
                      f"covar_mean={float(gaussians.covariances.mean()):.9f} "
                      f"(covar scaled by {scale_mult ** 2})", flush=True)
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
                                if (per_frame_dynamic or leave_one_out or track_dynamic) else None),
            gaussian_only_view=infos.get("gaussian_only_view"),
            gaussian_dyn_flag=(infos.get("gaussian_dyn_flag")
                               if (per_frame_dynamic or track_dynamic) else None),
            leave_one_out=leave_one_out,
            per_frame_compositing=per_frame_dynamic,
            # Motion displacement of dynamic Gaussians (needs the per-frame centroids
            # the encoder computed). Off by default so baselines reproduce exactly.
            # Single-centroid (one rigid motion for ALL dynamic content) is only used
            # when groups are disabled. With groups requested, a tracker failure must
            # mean NO displacement — never a silent downgrade to the crude mechanism,
            # which would look like "tracking doesn't help".
            dyn_centroid=(infos.get("dyn_centroid")
                          if (track_dynamic and config.dyn_motion_groups == 0) else None),
            dyn_centroid_pred=(infos.get("dyn_centroid_pred")
                               if (track_dynamic and config.dyn_motion_groups == 0) else None),
            dyn_centroid_valid=(infos.get("dyn_centroid_valid")
                                if (track_dynamic and config.dyn_motion_groups == 0) else None),
            dyn_group_centroid=(infos.get("dyn_group_centroid") if track_dynamic else None),
            dyn_group_pred=(infos.get("dyn_group_pred") if track_dynamic else None),
            dyn_group_valid=(infos.get("dyn_group_valid") if track_dynamic else None),
            gaussian_group_idx=(infos.get("gaussian_group_idx") if track_dynamic else None),
            # Scene-flow displacement field (dyn_motion_knn > 0). Present in infos only
            # when the encoder ran phase A+B, so no extra config gate needed here.
            gaussian_disp=(infos.get("gaussian_disp") if track_dynamic else None),
            gaussian_disp_valid=(infos.get("gaussian_disp_valid") if track_dynamic else None),
        )
        if infos.get("dyn_group_pred") is not None:
            n_group_motion += 1
        if infos.get("gaussian_disp") is not None:
            n_knn_motion += 1
        pred_rgb = decoder_out.color  # [B, V, 3, H, W] in [0, 1]

        # --- per-window camera-trajectory error (Sim3-aligned ATE) -------------
        # VGGT4D's actual published contribution is pose robustness under dynamics, which
        # rendering PSNR barely reflects. This measures it directly.
        if "gt_extrinsics" in batch:
            try:
                gt_w2c = batch["gt_extrinsics"].to(device).float()
                if gt_w2c.dim() == 3:
                    gt_w2c = gt_w2c.unsqueeze(0)
                gt_c2w = torch.linalg.inv(gt_w2c[0])              # [V,4,4]
                a = umeyama_ate(ext[0][:, :3, 3].cpu().numpy(),
                                gt_c2w[:, :3, 3].cpu().numpy())
                if a is not None:
                    total_ate += a; n_ate += 1
            except Exception:
                pass

        # dyn/static metrics split: prefer the precomputed mask (the same one that drove
        # the compositing gate above), at render resolution.
        dyn_mask = infos.get("dyn_mask", None)  # [B, V, H, W], or None
        if precomp_mask is not None:
            dyn_mask = precomp_mask
            n_precomp_hits += 1

        # --- Per-frame metrics and comparison images ---
        for v_idx in range(v):
            pred_frame = pred_rgb[0, v_idx].clamp(0, 1)   # [3, H, W]
            gt_frame = images[0, v_idx].clamp(0, 1)        # [3, H, W]

            # EXPOSURE CONTROL: rescale the prediction by its optimal scalar before
            # scoring. Structure is untouched, so any PSNR this recovers was a pure
            # brightness mismatch -- not reconstruction quality.
            if gain_correct:
                g = optimal_gain(pred_frame, gt_frame)
                pred_frame = (pred_frame * g).clamp(0, 1)
                total_gain += float(g); n_gain += 1

            psnr_val = compute_psnr(pred_frame.unsqueeze(0), gt_frame.unsqueeze(0)).mean().item()
            ssim_val = compute_ssim(pred_frame.unsqueeze(0), gt_frame.unsqueeze(0)).mean().item()
            lpips_val = compute_lpips(gt_frame.unsqueeze(0), pred_frame.unsqueeze(0)).mean().item()
            total_psnr += psnr_val
            total_ssim += ssim_val
            total_lpips += lpips_val
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

                    # Perceptual quality WHERE THE MOVING OBJECT IS: crop both images to
                    # the mask's bounding box (padded, min 32px so VGG has enough support)
                    # and run LPIPS there. A crop keeps real image context — masking to
                    # black would create edges the perceptual net reacts to.
                    rows = torch.any(mask > 0.5, dim=1).nonzero()
                    cols = torch.any(mask > 0.5, dim=0).nonzero()
                    if rows.numel() > 0 and cols.numel() > 0:
                        H_f, W_f = mask.shape
                        y0, y1 = rows[0].item(), rows[-1].item() + 1
                        x0, x1 = cols[0].item(), cols[-1].item() + 1
                        pad = 8
                        y0, y1 = max(0, y0 - pad), min(H_f, y1 + pad)
                        x0, x1 = max(0, x0 - pad), min(W_f, x1 + pad)
                        if (y1 - y0) >= 32 and (x1 - x0) >= 32:
                            total_lpips_dyn += compute_lpips(
                                gt_frame[:, y0:y1, x0:x1].unsqueeze(0),
                                pred_frame[:, y0:y1, x0:x1].unsqueeze(0),
                            ).mean().item()
                            n_lpips_dyn_frames += 1

                # Static-masked metrics (complement of dyn_mask)
                n_px_s = n_total_px - n_px
                if n_px_s >= 10:
                    static_mask3 = (1.0 - mask).clamp(0, 1).unsqueeze(0).expand(3, -1, -1)
                    mse_static = ((pred_frame * static_mask3 - gt_frame * static_mask3) ** 2).sum() / (3 * n_px_s)
                    total_psnr_static += -10 * torch.log10(mse_static + 1e-8).item()
                    n_static_frames += 1

            # Save GT | predicted comparison image for a window of batches.
            # --image_save_every thins these out: the evaluated batches are CONSECUTIVE
            # sliding windows, so batch 400 and 401 differ by one frame and 50 of them
            # are near-duplicates (300 images of almost the same moment). Saving every
            # Nth batch spreads the figures over the evaluated span instead.
            # NOTE the movement you want to SEE is mostly WITHIN a batch: at
            # --frame_stride 8 the six views v00..v05 of one batch are 8 frames apart,
            # so they span ~48 frames (~1.5 s) — that is where an object visibly moves.
            _save_this = (
                image_batch_start <= batch_idx < image_batch_start + max_image_batches
                and ((batch_idx - image_batch_start) % max(image_save_every, 1) == 0)
            )
            if _save_this:
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
        "lpips": total_lpips / n_frames if n_frames > 0 else None,
        "lpips_dynamic": (total_lpips_dyn / n_lpips_dyn_frames) if n_lpips_dyn_frames > 0 else None,
        "psnr_dynamic": total_psnr_dyn / n_dyn_frames if n_dyn_frames > 0 else None,
        "psnr_static": total_psnr_static / n_static_frames if n_static_frames > 0 else None,
        "avg_dyn_pixel_fraction": avg_dyn_pixel_frac,
        "batch_stride": batch_stride,
        "n_frames": n_frames,
        "n_dynamic_frames": n_dyn_frames,
        "n_static_frames": n_static_frames,
        "gain_correct": gain_correct,
        "mean_applied_gain": (total_gain / n_gain) if n_gain else None,
        "ate_sim3_rmse_m": (total_ate / n_ate) if n_ate else None,
        "n_windows_with_ate": n_ate,
        "track_dynamic": track_dynamic,
        "dyn_motion_groups": config.dyn_motion_groups,
        "n_batches_with_group_motion": n_group_motion,
        "dyn_motion_knn": getattr(config, "dyn_motion_knn", 0),
        "dyn_motion_strict": getattr(config, "dyn_motion_strict", False),
        "n_batches_with_knn_motion": n_knn_motion,
        "mask_source": ("precomputed" if precomputed_mask_dir is not None else "live_detection"),
        "precomputed_mask_dir": precomputed_mask_dir,
        "n_batches_with_precomputed_mask": n_precomp_hits,
    }

    print(f"\nResults:")
    print(f"  PSNR (overall):          {metrics['psnr']:.2f} dB")
    print(f"  SSIM (overall):          {metrics['ssim']:.4f}")
    if metrics["lpips"] is not None:
        print(f"  LPIPS (overall, lower better): {metrics['lpips']:.4f}")
    if metrics["lpips_dynamic"] is not None:
        print(f"  LPIPS (dynamic crop):          {metrics['lpips_dynamic']:.4f}")
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
    parser.add_argument("--batch_stride", type=int, default=1,
                        help="Evaluate only every Nth window. Windows are heavily overlapping, so "
                             "this still spans the whole sequence for 1/N the runtime. Needed on long "
                             "sequences (TUM fr2 has 3670 windows) where a full pass exceeds 2h and "
                             "the 24g watchdog cancels it for low GPU-memory utilisation.")
    parser.add_argument("--image_save_every", type=int, default=1,
                        help="Save a comparison image every Nth evaluated batch. The evaluated "
                             "batches are CONSECUTIVE sliding windows (400,401,...), so they are "
                             "near-duplicates; N=10 gives ~5 well-separated moments instead of 50 "
                             "nearly identical ones. Does not change any metric.")
    parser.add_argument("--scale_mult", type=float, default=1.0,
                        help="DIAGNOSTIC: multiply all Gaussian scales at render time. Used to "
                             "separate a coverage collapse (training can fix: scale is a head "
                             "output) from a geometric one (training CANNOT fix: means come from "
                             "the frozen depth head). Never report a number produced with this.")
    parser.add_argument("--voxel_size", type=float, default=0.001,
                        help="Fusion voxel edge length. MUST exceed the inter-frame point "
                             "spacing or nothing merges: the default 0.001 equals the frozen "
                             "Gaussian scale p50 (0.00095), so every point got its own voxel "
                             "and the measured fusion ratio was ~0.9 instead of ~1/V.")
    parser.add_argument("--hybrid_voxelize", action="store_true",
                        help="Fuse STATIC pixels into shared voxels (one set per target view, "
                             "that view excluded so leave-one-out stays exact); dynamic pixels "
                             "stay per-pixel. Requires dynamic masks (--dyn_mask_dir).")
    parser.add_argument("--eval_loo", action="store_true",
                        help="Leave-one-out: when rendering view j, drop ALL Gaussians that came from view j, "
                             "so j must be reconstructed from the OTHER frames. The honest control against "
                             "self-reprojection — a large LOO gap on dynamic regions is the expected result "
                             "(this architecture cannot model motion) and is itself reportable.")
    parser.add_argument("--image_batch_start", type=int, default=0,
                        help="First batch index to start saving images from. Use ~half total batches for mid-sequence.")
    parser.add_argument("--dyn_motion_groups", type=int, default=1,
                        help="With --track_dynamic: number of independently-moving GROUPS to model "
                             "(tracker-driven piecewise-rigid motion). 1 = one rigid motion for all "
                             "dynamic content (the crude version that failed); 3-4 lets e.g. a person "
                             "and a box move differently. Needs VGGT4D (uses its point tracker).")
    parser.add_argument("--dyn_motion_knn", type=int, default=0,
                        help="With --track_dynamic: >0 enables TRACK-CORRESPONDENCE SCENE FLOW and "
                             "sets K (nearest tracks per Gaussian). Each dynamic Gaussian is displaced "
                             "toward target frame j by the inverse-distance-weighted OBSERVED flow of "
                             "its K nearest tracks (non-rigid, no extrapolation). Takes precedence "
                             "over --dyn_motion_groups. PROTOCOL: uses frame j's pixels for motion "
                             "geometry ('motion fitted on full video, appearance held out') — the "
                             "strict no-look variant is the groups mode; report both.")
    parser.add_argument("--dyn_motion_n_query", type=int, default=1024,
                        help="Scene-flow mode: total tracker query budget, split across query frames.")
    parser.add_argument("--dyn_motion_gate_mult", type=float, default=3.0,
                        help="Scene-flow mode: trust radius = mult x median track NN spacing; "
                             "Gaussians farther than this from every track do not move.")
    parser.add_argument("--dyn_motion_strict", action="store_true",
                        help="Scene-flow mode, HONEST CONTROL: predict each track's position at the "
                             "target frame from the OTHER frames (constant velocity) instead of "
                             "observing it, so frame j is never read. Isolates what the non-rigid "
                             "per-Gaussian interpolation contributes from what OBSERVING j "
                             "contributes. Report alongside the non-strict number.")
    parser.add_argument("--dyn_motion_query_first_only", action="store_true",
                        help="Scene-flow mode: sample tracker queries only from frame 0's dynamic "
                             "pixels (legacy behaviour) instead of from every frame.")
    parser.add_argument("--gain_correct", action="store_true",
                        help="CONTROL: rescale each rendered frame by its optimal least-squares "
                             "scalar before computing metrics (pure exposure fix, no structural "
                             "change). Run on the FROZEN baseline to see how much of a fine-tuned "
                             "gain is merely brightness matching.")
    parser.add_argument("--track_dynamic", action="store_true",
                        help="Displace dynamic Gaussians by the object's estimated motion when "
                             "rendering another timestamp (first-order rigid model from per-frame "
                             "dynamic centroids; target-frame centroid is fitted from the OTHER "
                             "frames only, so it is leave-one-out safe). Off = Gaussians stay at "
                             "their source-frame positions (the baseline).")
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
        hybrid_voxelize=args.hybrid_voxelize,
        voxel_size=args.voxel_size,
        vggt4d_weights_path=args.vggt4d_weights_path,
        dyn_motion_groups=(args.dyn_motion_groups
                           if (args.track_dynamic and args.dyn_motion_knn == 0) else 0),
        dyn_motion_knn=(args.dyn_motion_knn if args.track_dynamic else 0),
        dyn_motion_n_query=args.dyn_motion_n_query,
        dyn_motion_query_all=not args.dyn_motion_query_first_only,
        dyn_motion_gate_mult=args.dyn_motion_gate_mult,
        dyn_motion_strict=args.dyn_motion_strict,
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
            "track_dynamic": args.track_dynamic,
            "dyn_motion_knn": args.dyn_motion_knn if args.track_dynamic else 0,
            "dyn_motion_n_query": args.dyn_motion_n_query,
            "dyn_motion_gate_mult": args.dyn_motion_gate_mult,
            "dyn_motion_query_all": not args.dyn_motion_query_first_only,
            "dyn_motion_strict": args.dyn_motion_strict,
        }, f, indent=2)

    print(f"\nRunning evaluation on {args.split} split ({len(dataset)} batches)...")
    print(f"  per_frame_dynamic={args.per_frame_dynamic}  leave_one_out={args.eval_loo}")
    evaluate(model, dataloader, config, args.output_dir, device,
             scale_mult=args.scale_mult,
             image_save_every=args.image_save_every,
             batch_stride=args.batch_stride,
             max_image_batches=args.max_image_batches,
             image_batch_start=args.image_batch_start,
             per_frame_dynamic=args.per_frame_dynamic,
             leave_one_out=args.eval_loo,
             precomputed_mask_dir=args.dyn_mask_dir,
             track_dynamic=args.track_dynamic,
             gain_correct=args.gain_correct)


if __name__ == "__main__":
    main()
