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
  - gaussians.ply           : 3D Gaussian point cloud (middle window; --ply_batch to choose)
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
from src.model.encoder.dyn_motion import compensate_dyn_opacity


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
             image_save_every=1, batch_stride=1, images_only=False, image_views=None,
             ply_batch=None, ply_per_frame=False, ply_dyn_source=-1, ply_dyn_opacity=1.0,
             ply_own_frame_only=False, ply_max_scale_frac=0.011,
             image_error_map=False, image_error_gain=4.0):
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
    # Gate accounting, averaged over windows. radius_rejected is the MASK-QUALITY
    # read-out: it is dominated by dynamic Gaussians sitting far from any track,
    # i.e. mask false positives in 3D, so a mask change should move it and little
    # else will. Previously one printed line per window and nothing in metrics.json.
    flow_stat_sums, n_flow_stats = {}, 0
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
    # RENDERED ALPHA (coverage). The decoder already returns it and eval threw it
    # away, so the one quantity that says whether a region is actually COVERED was
    # never recorded. It is the direct read-out for --dyn_opacity_comp: the
    # compositing gate removes contributors inside the dynamic mask, so alpha there
    # should sit below the static alpha, and the compensation should close the gap.
    # Split dyn/static because a single mean is dominated by the static majority --
    # which is exactly how a previous alpha measurement (0.967, ungated, during
    # training) concluded coverage was fine.
    total_alpha = 0.0
    total_alpha_dyn = 0.0
    total_alpha_static = 0.0
    n_alpha_dyn_frames = 0
    n_alpha_static_frames = 0
    n_alpha_frames = 0
    n_lpips_dyn_frames = 0
    n_dyn_frames = 0
    n_static_frames = 0
    n_frames = 0

    last_gaussians = None
    last_infos = None
    if ply_batch is None:
        # Pick the middle of the windows this run will ACTUALLY process. Taking the
        # middle of the whole dataset regardless is a silent no-output bug: with
        # --images_only every batch outside the image range is skipped, and with
        # --batch_stride the non-multiples are, so the chosen window may never be
        # reached and no PLY is written at all.
        try:
            n_total = len(dataloader)
        except TypeError:
            n_total = None
        if n_total is not None:
            if images_only:
                lo = min(image_batch_start, max(n_total - 1, 0))
                hi = min(n_total, image_batch_start + max_image_batches)
                ply_batch = (lo + hi) // 2 if hi > lo else lo
            else:
                ply_batch = n_total // 2
            if batch_stride > 1:
                ply_batch -= ply_batch % batch_stride     # snap onto a processed batch
            print(f"[ply] no --ply_batch given; exporting window {ply_batch} "
                  f"(middle of the {'image range' if images_only else 'sequence'}, "
                  f"{n_total} windows) rather than the last, where the moving object "
                  f"has usually left the frame", flush=True)
    elif batch_stride > 1 and ply_batch % batch_stride != 0:
        print(f"[ply] WARNING: --ply_batch {ply_batch} is not a multiple of "
              f"--batch_stride {batch_stride}, so that window is never processed and "
              f"no PLY will be written.", flush=True)
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
        # FIGURES-ONLY FAST PATH. Saving images is gated to a batch window, but the
        # encoder still ran on all ~920 windows, so re-rendering one region for a figure
        # cost a full ~25 min eval. With --images_only every batch outside the window is
        # skipped BEFORE the forward pass, turning figure iteration into ~1 min.
        # The metrics that come out are then computed over the saved window ONLY and are
        # NOT comparable to a full run -- metrics.json records this as images_only.
        if images_only and not (image_batch_start <= batch_idx
                                < image_batch_start + max_image_batches):
            if batch_idx >= image_batch_start + max_image_batches:
                break                                   # window passed; nothing left to save
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
            # Opacity compensation for the contributors the compositing gate drops.
            # Only meaningful WITH the gate -- without it nothing was removed.
            dyn_opacity_comp=(getattr(config, "dyn_opacity_comp", 0.0)
                              if per_frame_dynamic else 0.0),
        )
        if infos.get("dyn_group_pred") is not None:
            n_group_motion += 1
        if infos.get("gaussian_disp") is not None:
            n_knn_motion += 1
        _fs = infos.get("dyn_flow_stats")
        if _fs:
            for k, val in _fs.items():
                if val is not None:
                    flow_stat_sums[k] = flow_stat_sums.get(k, 0.0) + float(val)
            n_flow_stats += 1
        pred_rgb = decoder_out.color  # [B, V, 3, H, W] in [0, 1]
        pred_alpha = getattr(decoder_out, "alpha", None)  # [B, V, H, W] or None
        # The decoder builds this with a bare .squeeze(), which collapses the view
        # axis too when V == 1. Only trust it at the expected rank -- a coverage
        # diagnostic must never be the thing that crashes an eval.
        if pred_alpha is not None and pred_alpha.dim() != 4:
            pred_alpha = None

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

            # --- rendered alpha (coverage), overall and split by the mask -----
            a_frame = None
            if pred_alpha is not None:
                a_frame = pred_alpha[0, v_idx].detach().float()
                if a_frame.dim() == 3:          # [H,W,1] -> [H,W]
                    a_frame = a_frame.squeeze(-1)
                total_alpha += float(a_frame.mean())
                n_alpha_frames += 1

            # Dynamic-masked metrics (PSNR only — masked SSIM is unreliable due to zero-padding bias)
            if dyn_mask is not None:
                mask = dyn_mask[0, v_idx].to(device)   # [H, W]
                n_total_px = mask.numel()
                n_px = mask.sum().item()
                total_dyn_pixel_fraction += n_px / n_total_px

                if a_frame is not None and a_frame.shape == mask.shape:
                    m = mask > 0.5
                    if int(m.sum()) >= 10:
                        total_alpha_dyn += float(a_frame[m].mean())
                        n_alpha_dyn_frames += 1
                    if int((~m).sum()) >= 10:
                        total_alpha_static += float(a_frame[~m].mean())
                        n_alpha_static_frames += 1

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
            # --image_views restricts WHICH views are written. Saving all V views of
            # every batch over a whole sequence is 5520 near-duplicate stills (~2.4 GB)
            # and is unwatchable. Saving ONE view per batch instead yields a continuous
            # video of the sequence: window i starts at frame i, so view 0 of batch i IS
            # frame i, and consecutive batches step one frame. View 0 is also where the
            # motion correction is LARGEST (furthest in time from the rest of the
            # window), so it is the view that shows the mechanism best.
            _save_this = (
                image_batch_start <= batch_idx < image_batch_start + max_image_batches
                and ((batch_idx - image_batch_start) % max(image_save_every, 1) == 0)
                and (image_views is None or v_idx in image_views)
            )
            if _save_this:
                panels = [gt_frame, pred_frame]
                # ERROR MAP. A sub-decibel PSNR gain concentrated on ~12% of pixels is
                # invisible in a side-by-side render -- the eye cannot difference two
                # images. |pred - GT|, amplified, makes it directly visible: the moving
                # object glows in the control and dims when its Gaussians are displaced
                # to the right place. Compare the SAME filename across two runs.
                if image_error_map:
                    err = (pred_frame - gt_frame).abs().mean(dim=0, keepdim=True)
                    panels.append((err * image_error_gain).clamp(0, 1).repeat(3, 1, 1))
                comparison = torch.cat(panels, dim=2)  # [3, H, N*W]
                save_image(comparison, os.path.join(images_dir, f"b{batch_idx:04d}_v{v_idx:02d}.png"))

                if dyn_mask is not None:
                    os.makedirs(dyn_mask_dir, exist_ok=True)
                    save_dynamic_mask_overlay(
                        gt_frame, dyn_mask[0, v_idx],
                        os.path.join(dyn_mask_dir, f"b{batch_idx:04d}_v{v_idx:02d}.png")
                    )

        # Keep a batch for video + PLY output. The default is the MIDDLE window, not
        # the last: the last window sits at the end of the sequence, where the moving
        # object has usually left the frame, so the exported PLY showed a static scene.
        # The middle is a better blind default; --ply_batch overrides it, and the
        # motion ranking computed from the mask PNGs picks a genuinely active window.
        if ply_batch is None or batch_idx == ply_batch:
            last_gaussians = gaussians
            last_pred_pose = pred_pose
            last_h, last_w = h, w
            last_dyn_mask = dyn_mask  # [B, V, H, W] or None
            last_infos = infos

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

    # --ply_dyn_source keeps ALL static Gaussians but only ONE source frame's dynamic
    # ones. WHY: every window holds V copies of a moving object, one per source frame.
    # Exported together they stack -- in the control as V crisp copies strung along the
    # path, and under motion as V nearly-coincident copies that merge into one THICK
    # blurry blob. The blob is closer to the truth yet looks worse than V sharp wrong
    # answers. Keeping a single source frame removes the stacking from BOTH, so the
    # comparison shows what actually differs: one person standing still (control) vs
    # one person moving to where they were at t (flow).
    def _ply_keep(n_gauss, dev):
        if ply_dyn_source < 0 or last_infos is None:
            return None
        fi = last_infos.get("gaussian_frame_idx")
        df = last_infos.get("gaussian_dyn_flag")
        if fi is None or df is None:
            return None
        return (df[0].to(dev) <= 0.5) | (fi[0].to(dev) == ply_dyn_source)

    def _sub(t, keep):
        return t if keep is None else t[keep]

    # --- PLY export (the window selected above; middle by default) ---
    if last_gaussians is None:
        print(f"[ply] WARNING: window {ply_batch} was never processed, so no PLY was "
              f"written. It is outside this run's batch range "
              f"(images_only={images_only}, start={image_batch_start}, "
              f"max={max_image_batches}, stride={batch_stride}).", flush=True)
    if last_gaussians is not None:
        print("Saving gaussians.ply...")
        ply_path = os.path.join(output_dir, "gaussians.ply")
        # Flatten dynamic mask to match Gaussian layout [V*H*W] (assumes no voxelization)
        dyn_mask_flat = None
        if last_dyn_mask is not None:
            dyn_mask_flat = last_dyn_mask[0].cpu().numpy().reshape(-1).astype(np.float32)

        _k = _ply_keep(last_gaussians.means[0].shape[0], last_gaussians.means.device)
        if _k is not None and dyn_mask_flat is not None:
            dyn_mask_flat = dyn_mask_flat[_k.cpu().numpy()]
        export_ply(
            _sub(last_gaussians.means[0], _k),
            _sub(last_gaussians.scales[0], _k),
            _sub(last_gaussians.rotations[0], _k),
            _sub(last_gaussians.harmonics[0], _k),
            _sub(last_gaussians.opacities[0], _k),
            Path(ply_path),
            save_sh_dc_only=True,
            dyn_mask_flat=dyn_mask_flat,
            dyn_opacity_scale=0.5,  # dim dynamic Gaussians to 50% rather than near-invisible
            max_scale_frac=ply_max_scale_frac,
        )

    # --- 4D PLY export: one file per timestamp, dynamic Gaussians DISPLACED --------
    # This is the artifact the scene-flow mechanism actually produces. In the static
    # PLY above, a moving object appears as V copies strung along its trajectory (the
    # ghosting). Here, for timestamp j, every dynamic Gaussian is moved by its own
    # scene-flow displacement toward j -- so those V copies COLLAPSE onto the object's
    # position at j. Stepping through gaussians_t00..t{V-1}.ply in a viewer shows the
    # object moving through a static scene, which is the mechanism made visible.
    # Static Gaussians are untouched, so the background is identical in every file.
    if ply_per_frame and last_gaussians is not None and last_infos is not None:
        disp = last_infos.get("gaussian_disp")
        dynf = last_infos.get("gaussian_dyn_flag")
        fidx = last_infos.get("gaussian_frame_idx")
        if disp is None or dynf is None or fidx is None:
            print("[ply] --ply_per_frame needs --track_dynamic --dyn_motion_knn N "
                  "(no displacement field in this run); skipping the 4D export")
        else:
            dvalid = last_infos.get("gaussian_disp_valid")
            means = last_gaussians.means[0]
            dev = means.device
            dyn_v = dynf[0].to(dev).float()
            fid_v = fidx[0].to(dev)
            dyn_flat = (last_dyn_mask[0].cpu().numpy().reshape(-1).astype(np.float32)
                        if last_dyn_mask is not None else None)
            n_views = disp.shape[2]
            print(f"Saving {n_views} per-timestamp PLYs (4D export)...")
            for j in range(n_views):
                # own-frame Gaussians are already at their correct place for j
                move = dyn_v * (fid_v != j).float()
                if dvalid is not None:
                    move = move * dvalid[0, :, j].to(dev).float()
                means_j = means + move.unsqueeze(-1) * disp[0, :, j].to(dev).float()
                # FLOW-GATED, same rule the renderer uses: an off-frame dynamic
                # Gaussian that tracking could NOT relocate is DROPPED, because the
                # only thing it can do here is sit at a stale position and ghost.
                # Without this the 4D file still shows the copies the render removed,
                # so the point cloud disagrees with the picture it is supposed to show.
                _k = _ply_keep(means.shape[0], dev)
                if ply_own_frame_only:
                    # ONE copy of the moving object, as observed at j. The relocated
                    # copies are each individually plausible but carry displacement
                    # error, so ~9 of them stacked spread into a diffuse shell rather
                    # than reinforcing -- that is the scatter. Frame j's own Gaussians
                    # are already a COMPLETE dense unprojection of the object at that
                    # instant, so dropping the rest costs coverage of surfaces j could
                    # not see, and buys a crisp object. Best artifact; it shows
                    # per-timestamp geometry rather than the scene-flow mechanism.
                    _own = (dyn_v <= 0.5) | (fid_v == j)
                    _k = _own if _k is None else (_k & _own)
                elif dvalid is not None:
                    _ok = ((dyn_v <= 0.5) | (fid_v == j)
                           | (dvalid[0, :, j].to(dev) > 0))
                    _k = _ok if _k is None else (_k & _ok)
                _df = dyn_flat[_k.cpu().numpy()] if (_k is not None and dyn_flat is not None) else dyn_flat

                # OPACITY COMPENSATION, the PLY counterpart of decoder (1b).
                # gaussians.ply keeps all V copies of a moving object, so the V-fold
                # stack the head sized its opacities for is still there and nothing
                # needs fixing. THIS file is different: it keeps one copy per
                # timestamp (own-frame only) or own-frame plus the relocated ones, so
                # the survivors carry the same alpha deficit the gated RENDER had --
                # and a viewer applies no compensation of its own, so without this the
                # 4D PLY shows a washed-out object while the render next to it does
                # not. Same helper, same strength, `_k` as the survivor set, so the
                # two artefacts agree by construction.
                _op = last_gaussians.opacities[0]
                _oc = getattr(config, "dyn_opacity_comp", 0.0)
                if _oc > 0.0 and _k is not None:
                    _op = compensate_dyn_opacity(
                        _op, dyn_v, _k.to(_op.dtype), n_views, _oc)

                export_ply(
                    _sub(means_j, _k),
                    _sub(last_gaussians.scales[0], _k),
                    _sub(last_gaussians.rotations[0], _k),
                    _sub(last_gaussians.harmonics[0], _k),
                    _sub(_op, _k),
                    Path(os.path.join(output_dir, f"gaussians_t{j:02d}.ply")),
                    save_sh_dc_only=True,
                    dyn_mask_flat=_df,
                    max_scale_frac=ply_max_scale_frac,
                    # 1.0 = no fade. The 4D export exists to SHOW the moving object;
                    # the old 0.5 halved exactly the Gaussians the viewer came to see.
                    dyn_opacity_scale=ply_dyn_opacity,
                )
            print(f"  -> {output_dir}/gaussians_t00..t{n_views-1:02d}.ply "
                  f"({int(dyn_v.sum())} dynamic gaussians move; the rest are identical)")

    # --- PLY statistics -------------------------------------------------------
    # PSNR/SSIM/LPIPS cannot see representation degeneracy -- a cloud of collapsed
    # or oversized Gaussians still renders correctly at the eval camera, which is
    # why three separate scale collapses were only ever caught by eye in a viewer.
    # inspect_ply.py answers this but has to be run by hand on a downloaded file.
    # Recording a few numbers here makes a PLY regression visible in the same diff
    # as a PSNR regression.
    ply_stats = None
    if last_gaussians is not None:
        try:
            _sc = last_gaussians.scales[0].detach().float()
            _op = last_gaussians.opacities[0].detach().float().flatten()
            _mu = last_gaussians.means[0].detach().float()
            _big = _sc.max(dim=-1).values
            _lo = torch.quantile(_mu[:: max(_mu.shape[0] // 100000, 1)], 0.01, dim=0)
            _hi = torch.quantile(_mu[:: max(_mu.shape[0] // 100000, 1)], 0.99, dim=0)
            _diag = float((_hi - _lo).norm())
            ply_stats = {
                "n_gaussians": int(_sc.shape[0]),
                "scene_diag": _diag,
                "opacity_median": float(_op.median()),
                "opacity_frac_above_0p5": float((_op > 0.5).float().mean()),
                "scale_max_axis_median": float(_big.median()),
                # the oversized tail export_ply now drops -- if this grows, the
                # viewer is about to fill with concentric-ring haze again
                "frac_oversized": float((_big > 0.011 * _diag).float().mean()),
                "scale_frac_below_1e4": float((_big < 1e-4).float().mean()),
            }
        except Exception as e:
            print(f"[ply] stats failed ({e})", flush=True)

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
        # COVERAGE. alpha_dynamic well below alpha_static means the compositing
        # gate removed contributors the head was counting on -- the moving object
        # renders see-through and the background shows through it. This is what
        # --dyn_opacity_comp is meant to close; compare the two runs on this row,
        # not only on psnr_dynamic (a faint object can still score well against a
        # background that is roughly the right colour).
        "alpha_mean": (total_alpha / n_alpha_frames) if n_alpha_frames else None,
        "alpha_dynamic": ((total_alpha_dyn / n_alpha_dyn_frames)
                          if n_alpha_dyn_frames else None),
        "alpha_static": ((total_alpha_static / n_alpha_static_frames)
                         if n_alpha_static_frames else None),
        "background_color": list(getattr(config, "background_color", (0.0, 0.0, 0.0))),
        "dyn_opacity_comp": (getattr(config, "dyn_opacity_comp", 0.0)
                             if per_frame_dynamic else 0.0),
        "batch_stride": batch_stride,
        # true = metrics cover only the saved image window, NOT the sequence
        "images_only": images_only,
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
        "dyn_motion_pred_bandwidth": getattr(config, "dyn_motion_pred_bandwidth", 0.0),
        "dyn_motion_clean_tokens": getattr(config, "dyn_motion_clean_tokens", False),
        "dyn_motion_track_iters": getattr(config, "dyn_motion_track_iters", 0),
        "dyn_motion_chain": getattr(config, "dyn_motion_chain", False),
        "dyn_motion_tracker": getattr(config, "dyn_motion_tracker", "vggt"),
        "dyn_motion_smooth": getattr(config, "dyn_motion_smooth", 0),
        "dyn_motion_min_travel": getattr(config, "dyn_motion_min_travel", 0.0),
        "n_batches_with_knn_motion": n_knn_motion,
        **({f"flow_{k}": v / n_flow_stats for k, v in flow_stat_sums.items()}
           if n_flow_stats else {}),
        "n_windows_with_flow_stats": n_flow_stats,
        "ply": ply_stats,
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
    if ply_stats is not None:
        print(f"  PLY: {ply_stats['n_gaussians']} gaussians  opacity med "
              f"{ply_stats['opacity_median']:.3f}  max-axis med "
              f"{ply_stats['scale_max_axis_median']:.5f}  oversized "
              f"{100 * ply_stats['frac_oversized']:.2f}% (dropped on export)")
    if metrics.get("flow_radius_rejected_frac") is not None:
        print(f"  Flow gates: moved {100 * metrics['flow_moved_frac']:.1f}%  |  "
              f"radius rejected {100 * metrics['flow_radius_rejected_frac']:.1f}%  "
              f"(mask-quality read-out)  |  visibility rejected "
              f"{100 * metrics['flow_vis_rejected_frac']:.1f}%")
    if metrics["alpha_mean"] is not None:
        _ad, _as = metrics["alpha_dynamic"], metrics["alpha_static"]
        print(f"  Rendered alpha (overall):{metrics['alpha_mean']:.4f}")
        if _ad is not None and _as is not None:
            print(f"    alpha dynamic:         {_ad:.4f}   "
                  f"static: {_as:.4f}   gap: {_ad - _as:+.4f}"
                  f"{'  <- moving objects under-covered' if (_ad - _as) < -0.02 else ''}")
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
    parser.add_argument("--ply_batch", type=int, default=None,
                        help="Which window's Gaussians to export as PLY. Default = the MIDDLE "
                             "window: the last one sits at the end of the sequence, where the "
                             "moving object has usually left the frame. Pair with --images_only "
                             "for a fast targeted export of a window you picked by motion.")
    parser.add_argument("--image_error_map", action="store_true",
                        help="Append a per-pixel |pred - GT| panel to each saved image. A "
                             "sub-decibel gain on ~12%% of pixels cannot be seen by comparing two "
                             "renders side by side; in the error map the moving object glows in the "
                             "control and dims under motion compensation. Compare the same filename "
                             "between a control run and a flow run.")
    parser.add_argument("--image_error_gain", type=float, default=4.0,
                        help="Brightness multiplier for --image_error_map (default 4).")
    parser.add_argument("--ply_dyn_source", type=int, default=-1,
                        help="Keep ALL static Gaussians but only the dynamic ones from THIS source "
                             "frame (default -1 = all). A window holds V copies of a moving object, "
                             "one per source frame; exported together they stack into a thick blob "
                             "under motion and V crisp copies without it, which makes the correct "
                             "result look worse. With e.g. 0, the control shows one person frozen at "
                             "frame 0 and the flow export shows that same person moving to where "
                             "they were at t -- the actual difference, unobscured.")
    parser.add_argument("--ply_own_frame_only", action="store_true",
                        help="4D EXPORT: at timestamp j keep ONLY frame j's own dynamic "
                             "Gaussians, not the relocated ones. The relocated copies each "
                             "carry displacement error, so stacking ~9 of them spreads the "
                             "object into a shell instead of reinforcing it. Frame j's own "
                             "are already a complete dense view of the object at that "
                             "instant. Gives the cleanest-looking 4D artifact, at the cost "
                             "of showing per-timestamp geometry rather than the scene-flow "
                             "mechanism -- say which one a figure is.")
    parser.add_argument("--ply_dyn_opacity", type=float, default=1.0,
                        help="Opacity multiplier for DYNAMIC gaussians in the 4D export. "
                             "1.0 = untouched (the moving object is the point of the file). "
                             "Lower fades it, which is only useful to see through it.")
    parser.add_argument("--ply_per_frame", action="store_true",
                        help="4D EXPORT: also write gaussians_t00..tNN.ply, one per timestamp, with "
                             "each dynamic Gaussian displaced by its scene-flow motion to that "
                             "timestamp. Without motion a moving object appears as V ghost copies "
                             "along its path; here they collapse onto its position at t, so "
                             "stepping through the files shows the object moving. Needs "
                             "--track_dynamic --dyn_motion_knn N.")
    parser.add_argument("--image_views", type=str, default="all",
                        help="Which views to write images for: 'all' (default) or a comma-separated "
                             "list, e.g. '0'. Use '0' with --image_batch_start 0 --max_image_batches "
                             "99999 to render the WHOLE sequence as one frame per batch (view 0 of "
                             "batch i IS frame i), i.e. a continuous video, instead of V near-"
                             "duplicate stills per window. View 0 is also where the motion "
                             "correction is largest.")
    parser.add_argument("--images_only", action="store_true",
                        help="FIGURES ONLY: skip every batch outside the --image_batch_start / "
                             "--max_image_batches window before the forward pass, so rendering a "
                             "chosen region takes ~1 min instead of a full ~25 min eval. The "
                             "resulting metrics cover ONLY that window and are NOT comparable to a "
                             "full run (metrics.json records images_only=true). Use a separate "
                             "output dir (EVAL_DATE=...) so it cannot overwrite a real result.")
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
    parser.add_argument("--dyn_motion_max_disp_mult", type=float, default=0.0,
                        help="Cap each Gaussian's displacement at this multiple of the MEDIAN "
                             "observed track motion for the frame pair. Guards the strict "
                             "predictor, which fits a velocity from the frames nearest the "
                             "target and extrapolates it unbounded, so one bad tracker hop can "
                             "throw a Gaussian near the camera where it washes the frame. "
                             "0 = off (reproduces every measured result); 3.0 is generous.")
    parser.add_argument("--dyn_motion_gate_mult", type=float, default=3.0,
                        help="Scene-flow mode: trust radius = mult x median track NN spacing; "
                             "Gaussians farther than this from every track do not move.")
    parser.add_argument("--dyn_motion_strict", action="store_true",
                        help="Scene-flow mode, HONEST CONTROL: predict each track's position at the "
                             "target frame from the OTHER frames (constant velocity) instead of "
                             "observing it, so frame j is never read. Isolates what the non-rigid "
                             "per-Gaussian interpolation contributes from what OBSERVING j "
                             "contributes. Report alongside the non-strict number.")
    parser.add_argument("--dyn_motion_tracker", type=str, default="vggt", choices=["vggt", "raft"],
                        help="Point tracker feeding the scene flow. 'vggt' = the backbone's "
                             "TrackHead, measured to recover only ~20%% of the motion on Bonn "
                             "(13.6 px of a 67 px shift) regardless of iterations, features, "
                             "chaining or query count. 'raft' = chained dense optical flow from "
                             "torchvision's pretrained RAFT, whose per-hop motion here (~13 px) is "
                             "well inside its range; visibility via forward-backward consistency.")
    parser.add_argument("--dyn_motion_chain", action="store_true",
                        help="Track by chaining one frame at a time, re-querying at each newly "
                             "found position, instead of matching every frame against the query "
                             "frame. The tracker stalls after roughly one frame-step on large "
                             "motion (13.6 px recovered of a 67 px shift); chaining keeps each hop "
                             "small. Costs S-1 tracker calls per query frame.")
    parser.add_argument("--dyn_motion_track_iters", type=int, default=0,
                        help="Refinement iterations for the point tracker (0 = its default, 4). "
                             "Tracks initialise at the query position and each iteration takes one "
                             "correlation-guided step, so a small budget cannot reach a large "
                             "displacement: measured tracks travel ~13.6 px where the object moves "
                             "~94 px. Try 12-20.")
    parser.add_argument("--dyn_motion_clean_tokens", action="store_true",
                        help="Run the point tracker on UNSUPPRESSED features. VGGT4D damps dynamic "
                             "tokens in layers 0-4 -- that is its mechanism -- and the tracker "
                             "otherwise reads those same tokens, i.e. it follows the moving object "
                             "using features where that object was suppressed. Costs one extra "
                             "aggregator pass (tracking only; geometry is unchanged).")
    parser.add_argument("--dyn_motion_pred_bandwidth", type=float, default=0.0,
                        help="With --dyn_motion_strict: >0 estimates each track's velocity from a "
                             "LOCALLY weighted fit (frames nearest the target count most, sigma in "
                             "frames) instead of one fit over the whole window. At stride 8 a "
                             "6-frame window spans ~1.3 s, longer than a person stays linear. "
                             "0 = the uniform global fit that produced the measured +0.55 dB.")
    parser.add_argument("--dyn_motion_query_first_only", action="store_true",
                        help="Scene-flow mode: sample tracker queries only from frame 0's dynamic "
                             "pixels (legacy behaviour) instead of from every frame.")
    parser.add_argument("--dyn_motion_smooth", type=int, default=0,
                        help="Temporal MEDIAN filter width on each track's 3D trajectory "
                             "(0/1/2 = off, try 3 or 5). A track's position at frame v comes "
                             "from frame v's OWN predicted depth, so the displacement is a "
                             "difference of two independent depth errors and they ADD. The "
                             "relocated copies then land at slightly different wrong depths "
                             "and spread into a shell instead of reinforcing -- the scatter "
                             "around the moving object. Median, not mean: the errors are "
                             "outliers (a track sampling background past a silhouette), and "
                             "a mean would drag the neighbourhood with it.")
    parser.add_argument("--dyn_motion_min_travel", type=float, default=0.0,
                        help="Drop tracks whose 3D travel is below this fraction of the "
                             "window's MEDIAN track travel (0 = off, try 0.2-0.3). Query "
                             "points are seeded on dynamic-mask pixels, so every mask false "
                             "positive seeds a track on static background; those tracks are "
                             "never invalid, they just do not move, and the kNN average pulls "
                             "nearby real displacements toward zero. Relative to the median "
                             "because absolute motion is a property of the sequence.")
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
    parser.add_argument("--allow_partial_masks", action="store_true",
                        help="Run even though --dyn_mask_dir does not cover every frame. "
                             "The uncovered frames are treated as fully static, so their "
                             "moving object ghosts and lands in the static PSNR bucket -- "
                             "not comparable to a fully-covered run.")
    parser.add_argument("--ply_max_scale_frac", type=float, default=0.011,
                        help="PLY only: drop Gaussians whose largest axis exceeds this "
                             "fraction of the scene's p1-p99 diagonal. 0 disables. A tiny "
                             "number of enormous, faint splats carry most of the visible "
                             "haze -- measured on balloon, 50 of them span 6%% of the scene "
                             "each and carry 9.4%% of all opacity-weighted area; the default "
                             "removes 1.9%% of the Gaussians and 31%% of the haze. Affects the "
                             "exported file ONLY, never a rendered metric.")
    parser.add_argument("--bg_color", type=float, nargs=3, default=None,
                        metavar=("R", "G", "B"),
                        help="RENDER BACKGROUND, default black (0 0 0) as this repo has "
                             "always used. Upstream AnySplat renders on WHITE (1 1 1) and the "
                             "pretrained head's opacities were fitted against it: splatting "
                             "ends at C = sum(c_i a_i T_i) + T_final*bg, so every pixel where "
                             "the head leaves transmittance was TRAINED to be filled white. "
                             "Over the fused background alpha is near 1 and this barely shows; "
                             "inside the dynamic mask --per_frame_dynamic drives alpha down, so "
                             "the mismatch lands on exactly the moving objects. Try `1 1 1`.")
    parser.add_argument("--dyn_opacity_comp", type=float, default=0.0,
                        help="OPACITY COMPENSATION for the contributors per-frame compositing "
                             "removes (decoder (1b)). The head sized each dynamic Gaussian to "
                             "carry ~1/V of a surface's alpha because AnySplat renders all V "
                             "into every view; the gate leaves only own-frame + relocated ones, "
                             "so the survivors under-cover. Raises their opacity to match the "
                             "alpha V would have produced: 0 = off (the measured behaviour), "
                             "1 = full correction. SWEEP IT (0.25/0.5/1.0) -- over-correction "
                             "trades silhouette blur for a hard opaque edge in the wrong place, "
                             "which lpips_dynamic catches. Needs --per_frame_dynamic.")
    parser.add_argument("--dyn_mask_dir", type=str, default=None,
                        help="Directory of PRECOMPUTED dynamic-mask PNGs (named by rgb frame stem), e.g. "
                             "output_dyn_masks_precomputed_cs16_r518_st3_fs49/<SEQ>/masks. When set, these "
                             "override the live per-window detection for the dynamic/static PSNR split — "
                             "use the validated 518+full-span masks instead of the weak in-eval detection.")
    args = parser.parse_args()

    # --dyn_opacity_comp only compensates for what the COMPOSITING GATE removed, so
    # without the gate there is nothing to compensate and the flag silently does
    # nothing. Refuse rather than report a 'no effect' result that was never run.
    if args.dyn_opacity_comp > 0.0 and not (args.per_frame_dynamic
                                            or args.ply_own_frame_only
                                            or args.ply_dyn_source >= 0):
        parser.error("--dyn_opacity_comp needs a gate to compensate for: "
                     "--per_frame_dynamic (render), or --ply_own_frame_only / "
                     "--ply_dyn_source (PLY). With every copy kept, the V-fold stack "
                     "the head sized its opacities for is still there and this is a no-op.")

    # A mask directory that does not cover every RGB frame is the worst failure mode
    # here, because nothing fails: frames without a mask get an all-zero one, their
    # moving object is scored as static, and the run looks complete. submit_final_
    # battery.sh already guards this for its own submissions; a direct invocation or
    # slurm_probe_hex.sh did not. Counting files is enough -- the mask stems mirror
    # the rgb stems by construction.
    if args.dyn_mask_dir is not None:
        import glob as _glob
        _mdir = os.path.join(args.dyn_mask_dir, args.dataset_name, "masks")
        _rgb = len(_glob.glob(os.path.join(args.data_dir, args.dataset_name, "rgb", "*.png")))
        _msk = len(_glob.glob(os.path.join(_mdir, "*.png")))
        if _msk == 0:
            raise SystemExit(
                f"ERROR: --dyn_mask_dir given but no masks found at {_mdir}. Eval would "
                f"silently fall back to LIVE in-window detection, which is a different "
                f"protocol and not comparable to your other runs.")
        if _rgb and _msk < _rgb and not args.allow_partial_masks:
            raise SystemExit(
                f"ERROR: incomplete masks at {_mdir}: {_msk}/{_rgb} frames. The missing "
                f"ones would be treated as fully static, so their moving object ghosts "
                f"and is scored in the STATIC bucket. Finish the precompute, or pass "
                f"--allow_partial_masks if you know this is what you want.")
        print(f"[dyn_mask] {_msk}/{_rgb} frames covered at {_mdir}", flush=True)

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
        dyn_motion_max_disp_mult=args.dyn_motion_max_disp_mult,
        dyn_motion_strict=args.dyn_motion_strict,
        dyn_motion_pred_bandwidth=args.dyn_motion_pred_bandwidth,
        dyn_motion_clean_tokens=args.dyn_motion_clean_tokens,
        dyn_motion_track_iters=args.dyn_motion_track_iters,
        dyn_motion_chain=args.dyn_motion_chain,
        dyn_motion_tracker=args.dyn_motion_tracker,
        dyn_motion_smooth=args.dyn_motion_smooth,
        dyn_motion_min_travel=args.dyn_motion_min_travel,
        background_color=(tuple(args.bg_color) if args.bg_color is not None
                          else TrainingConfig.background_color),
        dyn_opacity_comp=args.dyn_opacity_comp,
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
            "bg_color": list(config.background_color),
            "dyn_opacity_comp": args.dyn_opacity_comp,
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
            "dyn_motion_max_disp_mult": args.dyn_motion_max_disp_mult,
            "dyn_motion_query_all": not args.dyn_motion_query_first_only,
            "dyn_motion_strict": args.dyn_motion_strict,
            "dyn_motion_pred_bandwidth": args.dyn_motion_pred_bandwidth,
            "dyn_motion_clean_tokens": args.dyn_motion_clean_tokens,
            "dyn_motion_track_iters": args.dyn_motion_track_iters,
            "dyn_motion_chain": args.dyn_motion_chain,
            "dyn_motion_tracker": args.dyn_motion_tracker,
        }, f, indent=2)

    print(f"\nRunning evaluation on {args.split} split ({len(dataset)} batches)...")
    print(f"  per_frame_dynamic={args.per_frame_dynamic}  leave_one_out={args.eval_loo}")
    evaluate(model, dataloader, config, args.output_dir, device,
             scale_mult=args.scale_mult,
             image_save_every=args.image_save_every,
             batch_stride=args.batch_stride,
             images_only=args.images_only,
             ply_batch=args.ply_batch,
             ply_per_frame=args.ply_per_frame,
             ply_dyn_source=args.ply_dyn_source,
             ply_dyn_opacity=args.ply_dyn_opacity,
             ply_own_frame_only=args.ply_own_frame_only,
             ply_max_scale_frac=args.ply_max_scale_frac,
             image_error_map=args.image_error_map,
             image_error_gain=args.image_error_gain,
             image_views=(None if args.image_views.strip().lower() == "all"
                          else {int(x) for x in args.image_views.split(",") if x.strip() != ""}),
             max_image_batches=args.max_image_batches,
             image_batch_start=args.image_batch_start,
             per_frame_dynamic=args.per_frame_dynamic,
             leave_one_out=args.eval_loo,
             precomputed_mask_dir=args.dyn_mask_dir,
             track_dynamic=args.track_dynamic,
             gain_correct=args.gain_correct)


if __name__ == "__main__":
    main()
