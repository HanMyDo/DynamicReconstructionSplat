"""
Fine-tuning script for the Temporal Gaussian Head.

This script fine-tunes only the Gaussian head (with temporal attention) while keeping
the VGGT4D backbone frozen. This enables the model to learn temporal consistency
for dynamic scene handling.

Supports TUM-format datasets (Bonn RGB-D Dynamic, TUM RGB-D Dynamic) with
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

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.model.anysplat import AnySplat
from src.model.encoder.anysplat import EncoderAnySplatCfg, OpacityMappingCfg
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from src.model.encoder.visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg


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
    use_temporal_attention: bool = True
    temporal_num_heads: int = 4
    temporal_spatial_downsample: int = 4

    # Training
    batch_size: int = 1
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    accumulate_grad_batches: int = 4  # Effective batch size = batch_size * accumulate

    # Loss weights
    mse_weight: float = 1.0
    temporal_consistency_weight: float = 0.1

    # Pose handling
    use_gt_poses: bool = True  # Use GT poses for rendering loss (recommended)

    # Checkpointing
    output_dir: str = "output_finetune"
    save_every_n_steps: int = 1000
    log_every_n_steps: int = 100

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

        # Calculate valid starting indices for sequences
        total_span = (num_frames - 1) * frame_stride + 1
        self.valid_starts = list(range(0, len(self.frame_data) - total_span + 1))

        if len(self.valid_starts) == 0:
            raise ValueError(
                f"Not enough frames. Have {len(self.frame_data)}, "
                f"need at least {total_span} for {num_frames} frames with stride {frame_stride}"
            )

        # Split into train/val (80/20)
        split_idx = int(len(self.valid_starts) * 0.8)
        if split == "train":
            self.valid_starts = self.valid_starts[:split_idx]
        else:
            self.valid_starts = self.valid_starts[split_idx:]

        has_poses = any(pose is not None for _, pose in self.frame_data)
        print(f"[{split}] {len(self.valid_starts)} sequences, "
              f"{len(self.frame_data)} total frames, GT poses: {has_poses}")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]

        images = []
        extrinsics = []  # World-to-camera (4x4)
        has_all_poses = True

        for i in range(self.num_frames):
            frame_idx = start_idx + i * self.frame_stride
            frame_path, c2w_pose = self.frame_data[frame_idx]

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
        vggt4d_weights_path=None,
        enable_dynamic_detection=config.enable_dynamic_detection,
        dynamic_mask_threshold=None,
        dynamic_n_clusters=64,
        suppress_dynamic_gaussians=True,
        # Temporal attention settings
        use_temporal_attention=config.use_temporal_attention,
        temporal_num_heads=config.temporal_num_heads,
        temporal_dropout=0.0,
        temporal_spatial_downsample=config.temporal_spatial_downsample,
        temporal_use_pe=True,
        temporal_max_frames=32,
    )

    decoder_cfg = DecoderSplattingCUDACfg(
        name="splatting_cuda",
        background_color=[0.0, 0.0, 0.0],
        make_scale_invariant=False,
    )

    model = AnySplat(encoder_cfg, decoder_cfg)
    return model


def freeze_backbone(model: AnySplat):
    """Freeze backbone, only train Gaussian head + temporal attention."""
    # Freeze aggregator
    for param in model.encoder.aggregator.parameters():
        param.requires_grad = False

    # Freeze camera head
    for param in model.encoder.camera_head.parameters():
        param.requires_grad = False

    # Freeze depth head
    if hasattr(model.encoder, 'depth_head'):
        for param in model.encoder.depth_head.parameters():
            param.requires_grad = False

    # Freeze point head if exists
    if hasattr(model.encoder, 'point_head'):
        for param in model.encoder.point_head.parameters():
            param.requires_grad = False

    # Keep Gaussian head trainable
    trainable_names = []
    for name, param in model.encoder.gaussian_param_head.named_parameters():
        param.requires_grad = True
        trainable_names.append(f"gaussian_param_head.{name}")

    # Keep Gaussian adapter trainable
    for name, param in model.encoder.gaussian_adapter.named_parameters():
        param.requires_grad = True
        trainable_names.append(f"gaussian_adapter.{name}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = total_params - trainable_count

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_count:,} ({100*trainable_count/total_params:.2f}%)")
    print(f"Frozen parameters: {frozen_count:,} ({100*frozen_count/total_params:.2f}%)")

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
) -> tuple:
    """
    Compute MSE rendering loss by rendering predicted Gaussians with given poses.

    Args:
        model: AnySplat model (for the decoder)
        images: Input images [B, V, 3, H, W] in [-1, 1]
        gaussians: Predicted Gaussians
        extrinsics: 4x4 world-to-camera matrices [B, V, 4, 4]
        intrinsics: 3x3 intrinsic matrices [B, V, 3, 3]

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

    # Compute MSE loss
    pred_rgb = output.color  # [B, V, 3, H, W]
    gt_rgb = images  # Already in [0, 1]

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

    for key in ['opacity', 'scales', 'rotations', 'sh']:
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
            static_mask = 1.0 - dyn_mask[:, 1:].float()

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
            if static_mask.sum() > 0:
                loss = diff.sum() / (static_mask.sum() + 1e-8)
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
                model, images, gaussians, render_extrinsics, render_intrinsics
            )

            temporal_loss = compute_temporal_loss(infos)

            # Total loss
            loss = (
                config.mse_weight * mse_loss +
                config.temporal_consistency_weight * temporal_loss
            )

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
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'mse': f'{total_mse_loss/num_batches:.4f}',
            'temporal': f'{total_temporal_loss/num_batches:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
        })

        # Save checkpoint periodically
        if global_step > 0 and global_step % config.save_every_n_steps == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, config)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step


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

            mse_loss, _ = compute_rendering_loss(
                model, images, gaussians, render_extrinsics, render_intrinsics
            )

            total_mse += mse_loss.item()
            psnr = -10 * torch.log10(mse_loss + 1e-8).item()
            total_psnr += psnr
            num_batches += 1

    metrics = {
        'val_mse': total_mse / num_batches if num_batches > 0 else 0.0,
        'val_psnr': total_psnr / num_batches if num_batches > 0 else 0.0,
    }

    print(f"Validation - MSE: {metrics['val_mse']:.4f}, PSNR: {metrics['val_psnr']:.2f} dB")
    return metrics


# ============================================================================
# Checkpointing
# ============================================================================

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

    path = os.path.join(config.output_dir, f'checkpoint_step{global_step}.pt')
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

    latest_path = os.path.join(config.output_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']

    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, step {global_step})")
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
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Stride between sampled frames")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no_temporal_attention", action="store_true",
                        help="Disable temporal attention (for ablation)")
    parser.add_argument("--temporal_weight", type=float, default=0.1,
                        help="Weight for temporal consistency loss")
    parser.add_argument("--no_gt_poses", action="store_true",
                        help="Use predicted poses instead of GT (not recommended)")

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
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        intrinsics_preset=args.intrinsics,
        use_temporal_attention=not args.no_temporal_attention,
        temporal_consistency_weight=args.temporal_weight,
        use_gt_poses=not args.no_gt_poses,
    )

    print("=" * 60)
    print("Fine-tuning Temporal Gaussian Head")
    print("=" * 60)
    print(f"Dataset: {config.data_dir}/{config.dataset_name}")
    print(f"Intrinsics: {config.intrinsics_preset}")
    print(f"GT poses: {config.use_gt_poses}")
    print(f"Temporal attention: {config.use_temporal_attention}")
    print(f"Temporal loss weight: {config.temporal_consistency_weight}")

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create datasets
    print("\nLoading datasets...")
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

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler with warmup
    total_steps = len(train_loader) * config.num_epochs // config.accumulate_grad_batches
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_steps
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - config.warmup_steps), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.warmup_steps],
    )

    scaler = GradScaler(enabled=config.mixed_precision)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_psnr = 0.0

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        avg_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, epoch, global_step
        )

        # Validate every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == config.num_epochs - 1:
            val_metrics = validate(model, val_loader, config, global_step)

            if val_metrics['val_psnr'] > best_val_psnr:
                best_val_psnr = val_metrics['val_psnr']
                best_path = os.path.join(config.output_dir, 'checkpoint_best.pt')
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'val_psnr': best_val_psnr,
                }, best_path)
                print(f"New best model: PSNR {best_val_psnr:.2f} dB")

    # Save final
    save_checkpoint(model, optimizer, scheduler, config.num_epochs - 1, global_step, config)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Checkpoints: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
