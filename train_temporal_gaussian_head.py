"""
Fine-tuning script for the Gaussian Head on top of a frozen VGGT4D backbone.

Trains the AnySplat gaussian_param_head and gaussian_adapter on the features
produced by the frozen VGGT4D backbone. Loss: rendering (MSE) + temporal
consistency on Gaussian parameters + scale regularization.

Supports TUM-format datasets (Bonn RGB-D, TUM RGB-D Dynamic Scenes) with
ground truth camera poses from groundtruth.txt.

Usage:
    python train_temporal_gaussian_head.py --data_dir /path/to/data --output_dir /path/to/output

Requirements:
    - Ordered video sequences (frames must be temporally ordered)
    - groundtruth.txt in TUM format (timestamp tx ty tz qx qy qz qw)
    - rgb.txt timestamp index file
"""

import argparse
import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# Optional: wandb for real-time training-metrics dashboard. Soft import so the
# script still runs in environments without wandb installed.
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.model.anysplat import AnySplat
from src.model.encoder.anysplat import EncoderAnySplatCfg, OpacityMappingCfg
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from src.model.encoder.visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from src.evaluation.metrics import compute_psnr, compute_ssim


# ============================================================================
# Camera intrinsics for known datasets
# ============================================================================

# Bonn RGB-D Dynamic Dataset (640x480)
BONN_INTRINSICS = {
    "fx": 542.822841, "fy": 542.576870,
    "cx": 315.593520, "cy": 237.756098,
    "width": 640, "height": 480,
}

# TUM RGB-D fr3 sequences (Kinect v1, 640x480)
TUM_FR3_INTRINSICS = {
    "fx": 535.4, "fy": 539.2,
    "cx": 320.1, "cy": 247.6,
    "width": 640, "height": 480,
}

# TUM RGB-D fr1 sequences
TUM_FR1_INTRINSICS = {
    "fx": 517.3, "fy": 516.5,
    "cx": 318.6, "cy": 255.3,
    "width": 640, "height": 480,
}

INTRINSICS_PRESETS = {
    "bonn": BONN_INTRINSICS,
    "tum_fr1": TUM_FR1_INTRINSICS,
    "tum_fr3": TUM_FR3_INTRINSICS,
}


# ============================================================================
# TUM format pose parsing
# ============================================================================

def parse_tum_groundtruth(filepath: str) -> dict:
    """
    Parse a TUM-format groundtruth.txt file.

    Each line: timestamp tx ty tz qx qy qz qw
    Poses are camera-to-world.

    Returns:
        dict mapping timestamp (float) -> 4x4 camera-to-world numpy matrix
    """
    poses = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue

            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

            # Normalize quaternion
            n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            if n < 1e-10:
                continue
            qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

            # Quaternion to rotation matrix
            R = np.array([
                [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),       1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2)]
            ])

            T_c2w = np.eye(4)
            T_c2w[:3, :3] = R
            T_c2w[:3, 3] = [tx, ty, tz]
            poses[timestamp] = T_c2w

    return poses


def parse_tum_rgb_index(filepath: str) -> list:
    """
    Parse a TUM-format rgb.txt index file.

    Each line: timestamp filename

    Returns:
        List of (timestamp, filename) tuples, sorted by timestamp.
    """
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            timestamp = float(parts[0])
            filename = parts[1]
            entries.append((timestamp, filename))

    entries.sort(key=lambda x: x[0])
    return entries


def associate_poses_to_frames(
    rgb_entries: list,
    gt_poses: dict,
    max_dt: float = 0.02,
) -> list:
    """
    Associate each RGB frame with the nearest ground truth pose by timestamp.

    Args:
        rgb_entries: List of (timestamp, filename) from rgb.txt
        gt_poses: Dict of timestamp -> 4x4 matrix from groundtruth.txt
        max_dt: Maximum allowed time difference (seconds) for association

    Returns:
        List of (filename, 4x4_matrix) for frames that have a matching pose,
        or (filename, None) if no pose is close enough.
    """
    gt_timestamps = sorted(gt_poses.keys())
    gt_timestamps_np = np.array(gt_timestamps)

    associations = []
    for rgb_ts, rgb_filename in rgb_entries:
        # Find nearest GT timestamp
        idx = np.argmin(np.abs(gt_timestamps_np - rgb_ts))
        dt = abs(gt_timestamps_np[idx] - rgb_ts)

        if dt <= max_dt:
            associations.append((rgb_filename, gt_poses[gt_timestamps[idx]]))
        else:
            associations.append((rgb_filename, None))

    return associations


def build_intrinsic_matrix(intrinsics: dict, target_size: tuple) -> np.ndarray:
    """
    Build a 3x3 intrinsic matrix, scaled for a target image size.

    Args:
        intrinsics: Dict with fx, fy, cx, cy, width, height
        target_size: (target_h, target_w)

    Returns:
        3x3 intrinsic matrix
    """
    orig_w, orig_h = intrinsics["width"], intrinsics["height"]
    target_h, target_w = target_size

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    K = np.array([
        [intrinsics["fx"] * scale_x, 0.0, intrinsics["cx"] * scale_x],
        [0.0, intrinsics["fy"] * scale_y, intrinsics["cy"] * scale_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return K


# ============================================================================
# Config
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    # Data
    data_dir: str = "examples/vrnerf"
    dataset_name: str = "rgbd_bonn_crowd3"
    num_frames: int = 8  # Number of frames per training sample
    frame_stride: int = 1  # Stride between sampled frames
    image_size: tuple = (448, 448)  # Resize images to this size (must be divisible by 14)
    intrinsics_preset: str = "bonn"  # "bonn", "tum_fr1", "tum_fr3", or path to JSON

    # Model
    use_vggt4d: bool = True
    enable_dynamic_detection: bool = True
    vggt4d_weights_path: str = None

    # Training
    # LR schedule follows VGGT-Ω §6: linear warmup -> cosine decay over full cycle.
    # Ω recommends 10-15% warmup ratio for non-reconstruction downstream tasks
    # (Gaussian splatting head qualifies). Peak LR kept small for fine-tuning.
    batch_size: int = 1
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_ratio: float = 0.10  # Fraction of total optimizer steps used for linear warmup.
    warmup_steps: int = 0       # If > 0, overrides warmup_ratio with an absolute step count.
    gradient_clip: float = 1.0
    accumulate_grad_batches: int = 4  # Effective batch size = batch_size * accumulate

    # Loss weights
    mse_weight: float = 1.0
    temporal_consistency_weight: float = 0.1
    scale_reg_weight: float = 0.01  # L1 penalty on Gaussian scales to prevent size collapse
    sh_reg_weight: float = 0.01     # L1 penalty on SH DC magnitude — keeps f_dc bounded when fine-tuning the GS head on OOD color distributions
    dynamic_loss_downweight: float = 0.9  # Fraction to reduce dynamic-pixel MSE weight (0=uniform, 1=fully masked)

    # Pose handling
    use_gt_poses: bool = False  # Use predicted poses — GT poses are in Bonn world frame, incompatible with VGGT4D's predicted world frame

    # Checkpointing
    # save_every_n_steps tuned so at least one save lands inside the cluster's
    # 24h wallclock: at ~30s/batch × accumulate_grad_batches=4, 200 steps = ~6.7h
    # per save → ~3 saves per wallclock window → at most ~6.7h of work lost if
    # SLURM hard-kills mid-batch.
    output_dir: str = "output_finetune"
    save_every_n_steps: int = 200
    log_every_n_steps: int = 25  # wandb metric cadence: ~78 points per ~2K-batch epoch.
    val_every_epochs: int = 5    # Validate at epoch 1, then every N epochs, then at the end. Lower = denser val curve (set to 1 for sweeps).

    # Telemetry (wandb)
    use_wandb: bool = True
    wandb_project: str = "dynrecsplat"
    wandb_run_name: Optional[str] = None  # If None, wandb auto-generates.

    # Device
    device: str = "cuda"
    mixed_precision: bool = True


# ============================================================================
# Dataset
# ============================================================================

class VideoFrameDataset(Dataset):
    """
    Dataset for loading ordered video frames with GT poses from TUM-format datasets.

    Supports both Bonn RGB-D Dynamic and TUM RGB-D Dynamic datasets.

    Expects directory structure:
        data_dir/dataset_name/
            rgb/             # Color images
            depth/           # Depth images (optional)
            rgb.txt          # Timestamp index for RGB frames
            groundtruth.txt  # GT poses in TUM format
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        intrinsics: dict,
        num_frames: int = 8,
        frame_stride: int = 1,
        image_size: tuple = (518, 518),
        split: str = "train",
    ):
        self.data_dir = Path(data_dir) / dataset_name
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.split = split

        # Build intrinsic matrix scaled for target image size
        self.intrinsic_matrix = build_intrinsic_matrix(intrinsics, image_size)

        # Load frame index and GT poses
        rgb_txt = self.data_dir / "rgb.txt"
        gt_txt = self.data_dir / "groundtruth.txt"

        if rgb_txt.exists() and gt_txt.exists():
            # TUM-format dataset: use rgb.txt + groundtruth.txt
            rgb_entries = parse_tum_rgb_index(str(rgb_txt))
            gt_poses = parse_tum_groundtruth(str(gt_txt))
            associations = associate_poses_to_frames(rgb_entries, gt_poses)

            # Keep only frames that have a valid GT pose
            self.frame_data = [
                (self.data_dir / filename, pose)
                for filename, pose in associations
                if pose is not None
            ]
            print(f"  Associated {len(self.frame_data)}/{len(rgb_entries)} frames with GT poses")
        else:
            # Fallback: just load RGB files sorted, no GT poses
            print(f"  WARNING: No rgb.txt/groundtruth.txt found, loading without GT poses")
            frame_paths = sorted([
                p for p in self.rgb_dir.iterdir()
                if p.suffix.lower() in ['.png', '.jpg', '.jpeg']
            ])
            self.frame_data = [(p, None) for p in frame_paths]

        if len(self.frame_data) == 0:
            raise ValueError(f"No frames found in {self.data_dir}")

        # Split on frame indices first so train and val windows are fully disjoint.
        # Splitting on window start indices (the old approach) causes the last training
        # windows to overlap with the first val windows by up to (num_frames - 1) frames.
        n_total = len(self.frame_data)
        split_frame = int(n_total * 0.8)
        if split == "train":
            self.window_frames = self.frame_data[:split_frame]
        elif split == "val":
            self.window_frames = self.frame_data[split_frame:]
        else:  # "all" — use every frame, for held-out sequence evaluation
            self.window_frames = self.frame_data

        total_span = (num_frames - 1) * frame_stride + 1
        self.valid_starts = list(range(0, len(self.window_frames) - total_span + 1))

        if len(self.valid_starts) == 0:
            raise ValueError(
                f"Not enough frames for split='{split}'. Have {len(self.window_frames)}, "
                f"need at least {total_span} for {num_frames} frames with stride {frame_stride}"
            )

        has_poses = any(pose is not None for _, pose in self.window_frames)
        print(f"[{split}] {len(self.valid_starts)} sequences, "
              f"{len(self.window_frames)} frames, GT poses: {has_poses}")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]

        images = []
        extrinsics = []  # World-to-camera (4x4)
        has_all_poses = True

        for i in range(self.num_frames):
            frame_idx = start_idx + i * self.frame_stride
            frame_path, c2w_pose = self.window_frames[frame_idx]

            # Load image, squash-resize to target size, output [0, 1]
            # Using squash resize (not aspect-ratio crop) so intrinsics
            # scaling in build_intrinsic_matrix remains correct.
            pil_img = Image.open(str(frame_path)).convert('RGB')
            if self.image_size is not None:
                pil_img = pil_img.resize(
                    (self.image_size[1], self.image_size[0]),  # PIL wants (w, h)
                    Image.BILINEAR,
                )
            img = TF.to_tensor(pil_img)  # [3, H, W] in [0, 1]
            images.append(img)

            # Convert camera-to-world to world-to-camera (extrinsic)
            if c2w_pose is not None:
                w2c = np.linalg.inv(c2w_pose)
                extrinsics.append(torch.from_numpy(w2c.astype(np.float32)))
            else:
                has_all_poses = False

        result = {
            "images": torch.stack(images, dim=0),  # [V, 3, H, W]
        }

        if has_all_poses and len(extrinsics) == self.num_frames:
            result["gt_extrinsics"] = torch.stack(extrinsics, dim=0)  # [V, 4, 4]
            # Intrinsics are the same for all frames (same camera)
            K = torch.from_numpy(self.intrinsic_matrix)  # [3, 3]
            result["gt_intrinsics"] = K.unsqueeze(0).expand(self.num_frames, -1, -1)  # [V, 3, 3]

        return result


# ============================================================================
# Model creation and freezing
# ============================================================================

def create_model(config: TrainingConfig) -> AnySplat:
    """Create the AnySplat model with temporal attention enabled."""
    encoder_cfg = EncoderAnySplatCfg(
        name="anysplat",
        anchor_feat_dim=83,
        voxel_size=0.001,
        n_offsets=2,
        d_feature=32,
        add_view=False,
        num_monocular_samples=32,
        backbone=None,
        visualizer=EncoderVisualizerEpipolarCfg(
            num_samples=8,
            min_resolution=256,
            export_ply=False,
        ),
        gaussian_adapter=GaussianAdapterCfg(
            gaussian_scale_min=0.5,
            gaussian_scale_max=15.0,
            sh_degree=4,
        ),
        apply_bounds_shim=True,
        opacity_mapping=OpacityMappingCfg(
            initial=0.0,
            final=0.0,
            warm_up=1,
        ),
        gaussians_per_pixel=1,
        num_surfaces=1,
        gs_params_head_type="dpt_gs",
        pose_free=True,
        pred_head_type="depth",
        # VGGT4D settings
        use_vggt4d=config.use_vggt4d,
        vggt4d_weights_path=config.vggt4d_weights_path,
        enable_dynamic_detection=config.enable_dynamic_detection,
        dynamic_mask_threshold=None,
        dynamic_n_clusters=64,
        suppress_dynamic_gaussians=False,  # Bonn task: reconstruct dynamic objects, not suppress them
        use_temporal_attention=False,
    )

    decoder_cfg = DecoderSplattingCUDACfg(
        name="splatting_cuda",
        background_color=[0.0, 0.0, 0.0],
        make_scale_invariant=False,
    )

    model = AnySplat(encoder_cfg, decoder_cfg)

    # Load pretrained GS head weights from the official AnySplat checkpoint.
    print("Loading pretrained GS head weights from lhjiang/anysplat ...")
    pretrained = AnySplat.from_pretrained("lhjiang/anysplat")
    gs_head_result = model.encoder.gaussian_param_head.load_state_dict(
        pretrained.encoder.gaussian_param_head.state_dict(), strict=False
    )
    adapter_result = model.encoder.gaussian_adapter.load_state_dict(
        pretrained.encoder.gaussian_adapter.state_dict(), strict=False
    )
    print(f"  gaussian_param_head: missing={gs_head_result.missing_keys}, "
          f"unexpected={gs_head_result.unexpected_keys}")
    print(f"  gaussian_adapter:    missing={adapter_result.missing_keys}, "
          f"unexpected={adapter_result.unexpected_keys}")
    del pretrained
    torch.cuda.empty_cache()

    return model


def freeze_backbone(model: AnySplat):
    """Freeze VGGT4D backbone; train only the Gaussian head (gaussian_param_head + gaussian_adapter)."""
    for param in model.encoder.aggregator.parameters():
        param.requires_grad = False
    for param in model.encoder.camera_head.parameters():
        param.requires_grad = False
    if hasattr(model.encoder, 'depth_head'):
        for param in model.encoder.depth_head.parameters():
            param.requires_grad = False
    if hasattr(model.encoder, 'point_head'):
        for param in model.encoder.point_head.parameters():
            param.requires_grad = False

    for param in model.encoder.gaussian_param_head.parameters():
        param.requires_grad = True
    for param in model.encoder.gaussian_adapter.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_count:,} ({100*trainable_count/total_params:.2f}%)")
    print(f"Frozen parameters: {total_params - trainable_count:,}")

    return model


# ============================================================================
# Loss computation
# ============================================================================

def compute_rendering_loss(
    model: AnySplat,
    images: torch.Tensor,
    gaussians,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    dyn_mask: Optional[torch.Tensor] = None,
    dynamic_loss_downweight: float = 0.0,
) -> tuple:
    """
    Compute MSE rendering loss by rendering predicted Gaussians with given poses.

    Args:
        model: AnySplat model (for the decoder)
        images: Input images [B, V, 3, H, W] in [0, 1]
        gaussians: Predicted Gaussians
        extrinsics: 4x4 world-to-camera matrices [B, V, 4, 4]
        intrinsics: 3x3 intrinsic matrices [B, V, 3, 3]
        dyn_mask: Optional binary dynamic mask [B, V, H, W] (1=dynamic, 0=static)
        dynamic_loss_downweight: Fraction to reduce dynamic-pixel loss weight.
            0.0 = uniform loss, 0.9 = dynamic pixels get 0.1× weight, 1.0 = fully masked.

    Returns:
        (mse_loss, decoder_output)
    """
    b, v, c, h, w = images.shape
    device = images.device

    # Normalize intrinsics to [0, 1] range as expected by the decoder
    intrinsics_norm = intrinsics.clone()
    intrinsics_norm = torch.stack([
        intrinsics_norm[:, :, 0] / w,
        intrinsics_norm[:, :, 1] / h,
        intrinsics_norm[:, :, 2],
    ], dim=2)

    output = model.decoder.forward(
        gaussians,
        extrinsics,
        intrinsics_norm,
        torch.ones(b, v, device=device) * 0.01,  # near
        torch.ones(b, v, device=device) * 100.0,  # far
        (h, w),
        "depth",
    )

    pred_rgb = output.color  # [B, V, 3, H, W]
    gt_rgb = images  # Already in [0, 1]

    if dyn_mask is not None and dynamic_loss_downweight > 0.0:
        # Static pixels keep weight 1.0; dynamic pixels are downweighted.
        # Unsqueeze over channel dim so weights broadcast to [B, V, 3, H, W].
        weights = 1.0 - dynamic_loss_downweight * dyn_mask.float().to(pred_rgb.device)
        weights = weights.unsqueeze(2)
        mse_loss = (F.mse_loss(pred_rgb, gt_rgb, reduction='none') * weights).mean()
    else:
        mse_loss = F.mse_loss(pred_rgb, gt_rgb)

    return mse_loss, output


def compute_temporal_loss(
    infos: dict,
) -> torch.Tensor:
    """Compute temporal consistency loss from per-frame Gaussian parameters."""
    per_frame = infos.get('per_frame_gaussians', None)
    if per_frame is None:
        return torch.tensor(0.0)

    device = per_frame['opacity'].device
    dyn_mask = infos.get('dyn_mask', None)

    total_loss = torch.tensor(0.0, device=device)
    num_components = 0

    for key in ['opacity', 'scales', 'rotations']:
        # 'sh' intentionally excluded: SH coefficients are view-dependent; including them in
        # the temporal loss collapses f_dc to a near-constant "mean scene color".
        if key not in per_frame:
            continue

        params = per_frame[key]
        if params.shape[1] < 2:
            continue

        # Temporal difference between adjacent frames
        params_t = params[:, :-1]
        params_t1 = params[:, 1:]
        diff = (params_t - params_t1).abs()

        # Apply dynamic mask weighting (only penalize static regions)
        if dyn_mask is not None:
            static_mask = (1.0 - dyn_mask[:, 1:].float()).to(diff.device)

            # Expand mask dimensions to match diff
            while static_mask.dim() < diff.dim():
                static_mask = static_mask.unsqueeze(-1)

            # Interpolate if spatial dims don't match
            if static_mask.shape[2:4] != diff.shape[2:4]:
                static_mask = F.interpolate(
                    static_mask.flatten(0, 1),
                    size=diff.shape[2:4],
                    mode='nearest'
                ).view(*static_mask.shape[:2], *diff.shape[2:4], *([1] * (diff.dim() - 4)))

            diff = diff * static_mask
            # Denominator must count the same elements the numerator sums over;
            # static_mask is shape [..., 1] and broadcasts across the channel dim,
            # so without expand_as opacity (C=1) reads correctly but scales (C=3) /
            # rotations (C=4) are inflated by their channel count.
            denom = static_mask.expand_as(diff).sum().clamp_min(1e-8)
            if static_mask.sum() > 0:
                loss = diff.sum() / denom
            else:
                loss = diff.mean()
        else:
            loss = diff.mean()

        total_loss = total_loss + loss
        num_components += 1

    if num_components > 0:
        total_loss = total_loss / num_components

    return total_loss


# ============================================================================
# Training loop
# ============================================================================

def wandb_log(metrics: dict, step: Optional[int] = None) -> None:
    """No-op if wandb isn't initialized; logs scalars otherwise."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics, step=step)


def check_gaussian_health(gaussians, loss, step) -> bool:
    """
    Check for divergence in Gaussian parameters. Returns True if training should stop.
    Thresholds:
      - f_dc abs max > 5  → warning  (pretrained baseline peaks at ~2.25)
      - f_dc abs max > 25 → critical, stop training (raised from 15 since output_conv2 is now trainable)
      - scale max > 0.5   → warning
      - scale max > 2.0   → critical, stop training
      - loss NaN/inf      → stop immediately
    """
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\n[HEALTH] CRITICAL at step {step}: loss is {loss.item():.4f} — stopping training.")
        return True

    with torch.no_grad():
        f_dc = gaussians.harmonics[:, :, :, 0]  # DC SH coefficients [B, N, 3]
        f_dc_absmax = f_dc.abs().max().item()
        scale_max = gaussians.scales.max().item()
        scale_mean = gaussians.scales.mean().item()

    if f_dc_absmax > 25.0:
        print(f"\n[HEALTH] CRITICAL at step {step}: f_dc abs_max={f_dc_absmax:.2f} (>25) — SH diverged, stopping training.")
        return True
    if scale_max > 2.0:
        print(f"\n[HEALTH] CRITICAL at step {step}: scale_max={scale_max:.4f} (>2.0) — scales exploded, stopping training.")
        return True
    if f_dc_absmax > 5.0:
        print(f"\n[HEALTH] WARNING at step {step}: f_dc abs_max={f_dc_absmax:.2f} (>5) — SH drifting, watch closely.")
    if scale_max > 0.5:
        print(f"\n[HEALTH] WARNING at step {step}: scale_max={scale_max:.4f} (>0.5), scale_mean={scale_mean:.4f}")

    return False


def train_epoch(
    model: AnySplat,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
) -> tuple:
    """Train for one epoch."""
    model.train()
    device = torch.device(config.device)

    total_loss = 0.0
    total_mse_loss = 0.0
    total_temporal_loss = 0.0
    total_scale_reg = 0.0
    total_sh_reg = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for batch_idx, batch in enumerate(pbar):
        images = batch["images"].to(device)  # [B, V, 3, H, W]

        # Add batch dimension if needed
        if images.dim() == 4:
            images = images.unsqueeze(0)

        b, v, c, h, w = images.shape

        # Forward pass with mixed precision
        with autocast(enabled=config.mixed_precision):
            # Run encoder (uses predicted poses internally for depth unprojection)
            encoder_output = model.encoder(images, global_step=global_step)
            gaussians = encoder_output.gaussians
            pred_context_pose = encoder_output.pred_context_pose
            infos = encoder_output.infos

            # Choose poses for rendering loss
            if config.use_gt_poses and "gt_extrinsics" in batch:
                # Use ground truth poses for clean supervision
                gt_ext = batch["gt_extrinsics"].to(device)
                gt_int = batch["gt_intrinsics"].to(device)
                if gt_ext.dim() == 3:
                    gt_ext = gt_ext.unsqueeze(0)
                    gt_int = gt_int.unsqueeze(0)
                render_extrinsics = gt_ext
                render_intrinsics = gt_int
            else:
                # Fall back to predicted poses
                render_extrinsics = pred_context_pose['extrinsic']
                render_intrinsics = pred_context_pose['intrinsic']
                # Un-normalize intrinsics (decoder expects raw, compute_rendering_loss normalizes)
                render_intrinsics = render_intrinsics.clone()
                render_intrinsics = torch.stack([
                    render_intrinsics[:, :, 0] * w,
                    render_intrinsics[:, :, 1] * h,
                    render_intrinsics[:, :, 2],
                ], dim=2)

            # Compute losses
            mse_loss, _ = compute_rendering_loss(
                model, images, gaussians, render_extrinsics, render_intrinsics,
                dyn_mask=infos.get('dyn_mask', None),
                dynamic_loss_downweight=config.dynamic_loss_downweight,
            )

            temporal_loss = compute_temporal_loss(infos)

            # Scale regularization: L1 penalty on mean Gaussian scale to prevent size collapse
            scale_reg = gaussians.scales.mean()

            # SH DC regularization: L2 penalty on f_dc keeps the DC color term
            # bounded when fine-tuning on color distributions that differ from
            # AnySplat's pretraining mix. L2 (vs L1) specifically targets outlier
            # Gaussians — gradient is proportional to f_dc itself, so a few large
            # values get pulled back hard while well-behaved Gaussians barely feel
            # the penalty. L1 mean was too weak against heavy-tailed distributions
            # (max/mean ratio ~9× observed in v3).
            sh_reg = gaussians.harmonics[:, :, :, 0].pow(2).mean()

            # Total loss
            loss = (
                config.mse_weight * mse_loss +
                config.temporal_consistency_weight * temporal_loss +
                config.scale_reg_weight * scale_reg +
                config.sh_reg_weight * sh_reg
            )

        # Health check before backward (use unscaled loss)
        if check_gaussian_health(gaussians, loss, global_step):
            return total_loss / max(num_batches, 1), global_step, True

        # Scale for gradient accumulation
        loss = loss / config.accumulate_grad_batches

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (batch_idx + 1) % config.accumulate_grad_batches == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        # Logging
        total_loss += loss.item() * config.accumulate_grad_batches
        total_mse_loss += mse_loss.item()
        total_temporal_loss += temporal_loss.item()
        total_scale_reg += scale_reg.item()
        total_sh_reg += sh_reg.item()
        num_batches += 1

        last_lrs = scheduler.get_last_lr()
        pbar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'mse': f'{total_mse_loss/num_batches:.4f}',
            'temporal': f'{total_temporal_loss/num_batches:.4f}',
            'scale': f'{total_scale_reg/num_batches:.4f}',
            'sh_reg': f'{total_sh_reg/num_batches:.4f}',
            'lr': f'{last_lrs[0]:.2e}',
        })

        # wandb per-batch logging at log_every_n_steps cadence.
        # Log raw mse for an immediate, batch-local read; tqdm running averages
        # are deliberately not logged (they'd lag and smear epoch boundaries).
        # f_dc_absmax and scale_max are the actual quantities the health watchdog
        # checks, so they appear on the dashboard as a leading divergence signal.
        if global_step > 0 and batch_idx % config.log_every_n_steps == 0:
            batch_mse = mse_loss.item()
            with torch.no_grad():
                f_dc_absmax = gaussians.harmonics[:, :, :, 0].abs().max().item()
                scale_max = gaussians.scales.max().item()
            wandb_log({
                'train/loss': loss.item() * config.accumulate_grad_batches,
                'train/mse': batch_mse,
                'train/psnr_proxy_db': -10.0 * np.log10(max(batch_mse, 1e-8)),
                'train/temporal': temporal_loss.item(),
                'train/scale_reg': scale_reg.item(),
                'train/sh_reg': sh_reg.item(),
                'train/f_dc_absmax': f_dc_absmax,
                'train/scale_max': scale_max,
                'train/lr': last_lrs[0],
                'train/epoch_frac': epoch + batch_idx / max(len(dataloader), 1),
            }, step=global_step)

        # Save checkpoint periodically
        if global_step > 0 and global_step % config.save_every_n_steps == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, config)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    if num_batches > 0:
        avg_mse = total_mse_loss / num_batches
        wandb_log({
            'train_epoch/loss': avg_loss,
            'train_epoch/mse': avg_mse,
            'train_epoch/psnr_proxy_db': -10.0 * np.log10(max(avg_mse, 1e-8)),
            'train_epoch/temporal': total_temporal_loss / num_batches,
            'train_epoch/scale_reg': total_scale_reg / num_batches,
            'train_epoch/sh_reg': total_sh_reg / num_batches,
            'train_epoch/epoch': epoch + 1,
        }, step=global_step)
    return avg_loss, global_step, False


def validate(
    model: AnySplat,
    dataloader: DataLoader,
    config: TrainingConfig,
    global_step: int,
) -> dict:
    """Run validation."""
    model.eval()
    device = torch.device(config.device)

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_psnr_static = 0.0
    n_static_frames = 0
    n_frames = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["images"].to(device)
            if images.dim() == 4:
                images = images.unsqueeze(0)

            b, v, c, h, w = images.shape

            encoder_output = model.encoder(images, global_step=global_step)
            gaussians = encoder_output.gaussians
            pred_context_pose = encoder_output.pred_context_pose
            infos = encoder_output.infos

            # Use GT poses for validation too
            if config.use_gt_poses and "gt_extrinsics" in batch:
                gt_ext = batch["gt_extrinsics"].to(device)
                gt_int = batch["gt_intrinsics"].to(device)
                if gt_ext.dim() == 3:
                    gt_ext = gt_ext.unsqueeze(0)
                    gt_int = gt_int.unsqueeze(0)
                render_extrinsics = gt_ext
                render_intrinsics = gt_int
            else:
                render_extrinsics = pred_context_pose['extrinsic']
                render_intrinsics = pred_context_pose['intrinsic']
                render_intrinsics = render_intrinsics.clone()
                render_intrinsics = torch.stack([
                    render_intrinsics[:, :, 0] * w,
                    render_intrinsics[:, :, 1] * h,
                    render_intrinsics[:, :, 2],
                ], dim=2)

            _, render_output = compute_rendering_loss(
                model, images, gaussians, render_extrinsics, render_intrinsics
            )

            # Match eval_gaussian_head.py: clamp to [0,1] then accumulate PSNR/SSIM/MSE
            # per frame, not per batch. Per-batch aggregation followed by log is biased
            # vs per-frame log followed by mean (PSNR is non-linear).
            pred_rgb = render_output.color  # [B, V, 3, H, W]
            gt_rgb = images                 # [B, V, 3, H, W]
            bv = pred_rgb.shape[0] * pred_rgb.shape[1]
            pred_flat = pred_rgb.view(bv, *pred_rgb.shape[2:]).clamp(0, 1)
            gt_flat   = gt_rgb.view(bv, *gt_rgb.shape[2:]).clamp(0, 1)

            # compute_psnr in metrics.py also clamps internally; using it here keeps
            # train-time val PSNR numerically identical to eval_gaussian_head.py output.
            psnr_per_frame = compute_psnr(gt_flat, pred_flat)  # [BV]
            ssim_per_frame = compute_ssim(gt_flat, pred_flat)  # [BV]
            mse_per_frame  = ((pred_flat - gt_flat) ** 2).mean(dim=[1, 2, 3])  # [BV]
            total_psnr += psnr_per_frame.sum().item()
            total_ssim += ssim_per_frame.sum().item()
            total_mse  += mse_per_frame.sum().item()
            n_frames   += bv

            # Static-region PSNR using the dynamic mask from the encoder
            dyn_mask = infos.get('dyn_mask', None)
            if dyn_mask is not None:
                pred_c = pred_rgb.clamp(0, 1)
                gt_c   = gt_rgb.clamp(0, 1)
                for bi in range(b):
                    for vi in range(v):
                        mask = dyn_mask[bi, vi].to(device)  # [H, W]
                        n_static = (mask.numel() - mask.sum()).item()
                        if n_static >= 10:
                            static_w = (1.0 - mask).clamp(0, 1).unsqueeze(0).expand(3, -1, -1)
                            mse_s = ((pred_c[bi, vi] * static_w - gt_c[bi, vi] * static_w) ** 2).sum() / (3 * n_static)
                            total_psnr_static += -10 * torch.log10(mse_s + 1e-8).item()
                            n_static_frames += 1

            num_batches += 1

    # Per-frame averaging (matches eval_gaussian_head.py).
    nf = max(n_frames, 1)
    metrics = {
        'val_mse':         total_mse  / nf,
        'val_psnr':        total_psnr / nf,
        'val_ssim':        total_ssim / nf,
        'val_psnr_static': total_psnr_static / n_static_frames if n_static_frames > 0 else None,
    }

    static_str = (f", PSNR-static: {metrics['val_psnr_static']:.2f} dB"
                  if metrics['val_psnr_static'] is not None else "")
    print(f"Validation - MSE: {metrics['val_mse']:.4f}, "
          f"PSNR: {metrics['val_psnr']:.2f} dB, "
          f"SSIM: {metrics['val_ssim']:.4f}"
          f"{static_str}")

    wandb_payload = {
        'val/mse': metrics['val_mse'],
        'val/psnr_db': metrics['val_psnr'],
        'val/ssim': metrics['val_ssim'],
    }
    if metrics['val_psnr_static'] is not None:
        wandb_payload['val/psnr_static_db'] = metrics['val_psnr_static']
    wandb_log(wandb_payload, step=global_step)

    return metrics


# ============================================================================
# Checkpointing
# ============================================================================

def _atomic_torch_save(obj, path):
    """Write a torch checkpoint atomically: save to a temp file in the same
    directory, then os.replace() it into place. os.replace is atomic on POSIX,
    so a preemption/kill mid-write leaves either the previous good file or the
    leftover temp file — never a half-written canonical checkpoint. This is what
    makes opportunistic (preemptible) + --requeue resumes robust."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, config):
    """Save training checkpoint."""
    os.makedirs(config.output_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config.__dict__,
    }

    latest_path = os.path.join(config.output_dir, 'checkpoint_latest.pt')

    # Remove previous step checkpoint to avoid accumulating large files on disk
    prev_step_glob = os.path.join(config.output_dir, 'checkpoint_step*.pt')
    import glob as _glob
    for old in _glob.glob(prev_step_glob):
        os.remove(old)

    path = os.path.join(config.output_dir, f'checkpoint_step{global_step}.pt')
    _atomic_torch_save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

    _atomic_torch_save(checkpoint, latest_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint.

    Only restores gaussian_param_head and gaussian_adapter weights — never the
    frozen VGGT4D backbone — so the backbone always reflects the freshly loaded
    pretrained weights regardless of what was serialised in an older checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    saved = checkpoint['model_state_dict']
    current = model.state_dict()
    head_keys = {k: v for k, v in saved.items()
                 if 'gaussian_param_head' in k or 'gaussian_adapter' in k}
    current.update(head_keys)
    model.load_state_dict(current)

    # Older 'best' checkpoints saved only model weights. Be permissive so they
    # can still serve as resume points — optimizer/scheduler simply start fresh
    # (no Adam moments, warmup restarts), which is acceptable for short top-ups.
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print(f"  warning: '{checkpoint_path}' has no optimizer state — Adam moments reset.")
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print(f"  warning: '{checkpoint_path}' has no scheduler state — LR schedule restarts at step 0.")

    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)

    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, step {global_step})")
    print(f"  Restored {len(head_keys)} gaussian head tensors; backbone left as freshly loaded.")
    return epoch, global_step


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Temporal Gaussian Head")
    parser.add_argument("--data_dir", type=str, default="examples/vrnerf",
                        help="Root directory containing datasets")
    parser.add_argument("--dataset_name", type=str, default="rgbd_bonn_crowd3",
                        help="Name of the dataset subdirectory")
    parser.add_argument("--intrinsics", type=str, default="bonn",
                        help="Intrinsics preset: 'bonn', 'tum_fr1', 'tum_fr3'")
    parser.add_argument("--output_dir", type=str, default="output_finetune",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.10,
                        help="Linear-warmup fraction of total steps (VGGT-Ω §6: 10-15%% for non-recon fine-tuning).")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Absolute warmup steps; overrides --warmup_ratio when > 0.")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Stride between sampled frames")
    parser.add_argument("--dataset_names", type=str, default=None,
                        help="Comma-separated list of dataset names for multi-sequence training. "
                             "Overrides --dataset_name if provided.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--temporal_weight", type=float, default=0.1,
                        help="Weight for temporal consistency loss on Gaussian parameters")
    parser.add_argument("--scale_reg_weight", type=float, default=0.01,
                        help="Weight for L1 scale regularization (prevents Gaussian size collapse)")
    parser.add_argument("--sh_reg_weight", type=float, default=0.01,
                        help="Weight for L1 regularization on f_dc (SH DC color). Damps SH drift on OOD color distributions.")
    parser.add_argument("--no_gt_poses", action="store_true",
                        help="Use predicted poses (required for VGGT4D — GT poses live in Bonn's "
                             "world frame which is incompatible with the predicted world frame).")
    parser.add_argument("--vggt4d_weights_path", type=str, default=None,
                        help="Path to pretrained VGGT4D weights (.pt). If omitted, initializes from VGGT-1B.")
    parser.add_argument("--dynamic_loss_downweight", type=float, default=0.9,
                        help="How much to reduce MSE weight for dynamic pixels (0=uniform, 0.9=10%% weight, 1=fully masked). "
                             "Requires VGGT4D dynamic detection.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging (otherwise enabled by default).")
    parser.add_argument("--wandb_project", type=str, default="dynrecsplat",
                        help="wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb run name; if omitted, wandb auto-generates one.")
    parser.add_argument("--log_every_n_steps", type=int, default=25,
                        help="Cadence of per-batch wandb metric logging (lower = denser curves).")
    parser.add_argument("--val_every_epochs", type=int, default=5,
                        help="Validate at epoch 1, then every N epochs, then at the end. Set to 1 for a dense val curve (sweeps).")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Max gradient norm for clipping. Lower = safer against parameter blow-up.")

    args = parser.parse_args()

    # Resolve intrinsics
    if args.intrinsics in INTRINSICS_PRESETS:
        intrinsics = INTRINSICS_PRESETS[args.intrinsics]
    else:
        # Try loading from JSON file
        with open(args.intrinsics, 'r') as f:
            intrinsics = json.load(f)

    config = TrainingConfig(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        intrinsics_preset=args.intrinsics,
        temporal_consistency_weight=args.temporal_weight,
        scale_reg_weight=args.scale_reg_weight,
        sh_reg_weight=args.sh_reg_weight,
        dynamic_loss_downweight=args.dynamic_loss_downweight,
        use_gt_poses=not args.no_gt_poses,
        vggt4d_weights_path=args.vggt4d_weights_path,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_every_n_steps=args.log_every_n_steps,
        val_every_epochs=args.val_every_epochs,
        gradient_clip=args.gradient_clip,
    )

    print("=" * 60)
    print("Fine-tuning Temporal Gaussian Head")
    print("=" * 60)
    print(f"Dataset: {config.data_dir}/{config.dataset_name}")
    print(f"Intrinsics: {config.intrinsics_preset}")
    print(f"GT poses: {config.use_gt_poses}")
    print(f"Temporal loss weight: {config.temporal_consistency_weight}")

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    # Initialize wandb (no-op if disabled or unavailable; offline mode if no API key).
    if config.use_wandb:
        if not WANDB_AVAILABLE:
            print("wandb requested but not installed; continuing without telemetry.")
        else:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__,
                dir=config.output_dir,
            )
            print(f"wandb: project={config.wandb_project}, "
                  f"run={wandb.run.name}, mode={wandb.run.settings.mode}")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create datasets
    print("\nLoading datasets...")
    if args.dataset_names:
        from torch.utils.data import ConcatDataset
        names = [n.strip() for n in args.dataset_names.split(",")]
        print(f"Multi-sequence training on: {names}")
        train_dataset = ConcatDataset([
            VideoFrameDataset(config.data_dir, name, intrinsics=intrinsics,
                              num_frames=config.num_frames, frame_stride=config.frame_stride,
                              image_size=config.image_size, split="train")
            for name in names
        ])
        val_dataset = ConcatDataset([
            VideoFrameDataset(config.data_dir, name, intrinsics=intrinsics,
                              num_frames=config.num_frames, frame_stride=config.frame_stride,
                              image_size=config.image_size, split="val")
            for name in names
        ])
    else:
        train_dataset = VideoFrameDataset(
            config.data_dir,
            config.dataset_name,
            intrinsics=intrinsics,
            num_frames=config.num_frames,
            frame_stride=config.frame_stride,
            image_size=config.image_size,
            split="train",
        )
        val_dataset = VideoFrameDataset(
            config.data_dir,
            config.dataset_name,
            intrinsics=intrinsics,
            num_frames=config.num_frames,
            frame_stride=config.frame_stride,
            image_size=config.image_size,
            split="val",
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    model = model.to(device)

    print("\nFreezing backbone...")
    model = freeze_backbone(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizer: {len(trainable_params)} trainable tensors")

    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler: VGGT-Ω §6 recipe — linear warmup -> cosine decay across full cycle.
    total_steps = len(train_loader) * config.num_epochs // config.accumulate_grad_batches
    if config.warmup_steps > 0:
        warmup_steps = config.warmup_steps
        warmup_source = "explicit"
    else:
        warmup_steps = max(1, int(config.warmup_ratio * total_steps))
        warmup_source = f"ratio={config.warmup_ratio:.0%}"
    print(f"LR schedule: peak={config.learning_rate:.1e}, total_steps={total_steps}, "
          f"warmup={warmup_steps} ({warmup_source}), eta_min=1e-6")

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    scaler = GradScaler(enabled=config.mixed_precision)

    # Resume from checkpoint. Be resilient to a corrupt checkpoint_latest.pt: a
    # job preempted mid-write (before atomic saves existed, or on an fs hiccup)
    # can leave a truncated zip that torch.load can't open. Try the requested
    # checkpoint first, then fall back to the newest step checkpoint (identical
    # state — written just before latest in the same save), then to best, then
    # start fresh. Without this a single bad save aborts the whole run.
    start_epoch = 0
    global_step = 0
    if args.resume:
        import glob as _glob
        resume_dir = os.path.dirname(args.resume) or '.'
        candidates = [args.resume]
        candidates += sorted(_glob.glob(os.path.join(resume_dir, 'checkpoint_step*.pt')),
                             key=os.path.getmtime, reverse=True)
        candidates.append(os.path.join(resume_dir, 'checkpoint_best.pt'))

        loaded = False
        seen = set()
        for cand in candidates:
            if cand in seen or not os.path.exists(cand):
                continue
            seen.add(cand)
            try:
                start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, cand)
                start_epoch += 1
                loaded = True
                if cand != args.resume:
                    print(f"[RESUME] '{args.resume}' was unusable; recovered from '{cand}'.")
                break
            except Exception as e:
                print(f"[RESUME] Could not load '{cand}' ({e}); trying next fallback.")
        if not loaded:
            print("[RESUME] No usable checkpoint found — starting fresh.")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    # Initialize the best-checkpoint guard from an existing checkpoint_best.pt
    # if one is present. Without this, a resumed run would happily overwrite
    # a strong best-checkpoint with the first (possibly worse) val it produces.
    best_val_psnr = 0.0
    existing_best_path = os.path.join(config.output_dir, 'checkpoint_best.pt')
    if os.path.exists(existing_best_path):
        try:
            existing = torch.load(existing_best_path, map_location='cpu', weights_only=False)
            prior_best = existing.get('val_psnr_static') or existing.get('val_psnr') or 0.0
            best_val_psnr = float(prior_best)
            print(f"Found existing checkpoint_best.pt with PSNR {best_val_psnr:.2f} dB — "
                  f"will only overwrite if a later val beats this.")
            del existing
        except Exception as e:
            print(f"Could not read existing checkpoint_best.pt ({e}); starting best_val_psnr at 0.")

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        avg_loss, global_step, diverged = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, epoch, global_step
        )
        if diverged:
            print("\n[HEALTH] Training stopped early due to divergence. Saving emergency checkpoint...")
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, config)
            break

        # Validate at epoch 1 (early sanity check), then every val_every_epochs / at end.
        if epoch == 0 or (epoch + 1) % config.val_every_epochs == 0 or epoch == config.num_epochs - 1:
            val_metrics = validate(model, val_loader, config, global_step)

            # Prefer static PSNR for selection (aligns with thesis goal of improving
            # static reconstruction). Falls back to overall PSNR when no dynamic
            # mask is available (e.g. --no_vggt4d runs).
            psnr_for_selection = val_metrics['val_psnr_static'] or val_metrics['val_psnr']
            if psnr_for_selection > best_val_psnr:
                best_val_psnr = psnr_for_selection
                best_path = os.path.join(config.output_dir, 'checkpoint_best.pt')
                _atomic_torch_save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config.__dict__,
                    'val_psnr': val_metrics['val_psnr'],
                    'val_psnr_static': val_metrics['val_psnr_static'],
                }, best_path)
                criterion = "PSNR-static" if val_metrics['val_psnr_static'] is not None else "PSNR"
                print(f"New best model: {criterion} {best_val_psnr:.2f} dB")

    # Save final
    save_checkpoint(model, optimizer, scheduler, config.num_epochs - 1, global_step, config)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Checkpoints: {config.output_dir}")
    print("=" * 60)

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
