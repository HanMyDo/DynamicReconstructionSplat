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
from datetime import datetime
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
from src.evaluation.metrics import compute_psnr, compute_ssim, get_lpips


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
    # L1 penalty on MEAN Gaussian scale. NOTE: this penalises LARGE scales, i.e. it
    # PUSHES SCALES DOWN -- it does NOT 'prevent size collapse' as previously commented,
    # it CAUSES it. Measured: it drove a 26x scale collapse (0.0041 -> 0.00016), which
    # then forces f_dc to inflate to compensate for the lost alpha (the recurring
    # 'f_dc runaway'). Set to 0 unless you specifically want smaller Gaussians.
    scale_reg_weight: float = 0.01
    sh_reg_weight: float = 0.01     # L1 penalty on SH DC magnitude — keeps f_dc bounded when fine-tuning the GS head on OOD color distributions
    dynamic_loss_downweight: float = 0.9  # Fraction to reduce dynamic-pixel MSE weight (0=uniform, 1=fully masked)
    # Fraction of iterations trained with the LEAVE-ONE-OUT renderer when train_loo is on.
    # 1.0 = always LOO (the head then only ever sees the "own-frame Gaussians absent"
    # regime, and compensates with brighter/more opaque Gaussians -- measured: f_dc mean
    # +0.810 vs +0.091 frozen, opacity 2x -- which is correct for LOO but over-bright
    # under NORMAL full compositing, e.g. in a viewer). <1.0 randomises the regime per
    # iteration so the SAME Gaussians must be valid both with and without their own
    # frame, removing that conditioning ambiguity. Costs no extra memory: each step still
    # renders exactly one regime. 0.5 = balanced.
    # Perceptual loss. AnySplat's own training uses [mse, lpips, depth_consis] with
    # lpips weight 0.05; our hand-rolled loop used MSE ONLY, which is why the fine-tuned
    # head improves PSNR while LPIPS gets WORSE (0.311 -> 0.324) — the classic L2-vs-
    # perceptual trade. Restoring the upstream term targets exactly that regression.
    # COVERAGE supervision, taken from AnySplat's own loss suite (src/loss/loss_opacity.py,
    # upstream weight 0.1): MSE(rendered_alpha, valid_mask). It penalises alpha BELOW the
    # valid mask, i.e. it punishes holes/transparency. This is the direct counter to the
    # scale collapse we measured three separate times (scale_reg, my alpha-weighted loss,
    # and LPIPS all shrank splats because with V redundant Gaussians per surface shrinking
    # is free): collapsed scales lower rendered alpha, and this term charges for that.
    # NOTE it deliberately does NOT exclude uncovered pixels -- excluding them is exactly
    # what let an earlier version abandon hard pixels and shrink.
    # HYBRID voxelization: fuse static pixels into shared voxels (one set per target
    # view, that view excluded so LOO stays exact), keep dynamic pixels per-pixel.
    # Removes the V-copies-per-surface redundancy that made shrinking free.
    unfreeze_depth_head: bool = False
    hybrid_voxelize: bool = False
    voxel_size: float = 0.001
    scale_mult: float = 1.0
    opacity_weight: float = 0.0
    lpips_weight: float = 0.0
    # LPIPS is differentiable, so its VGG activations are retained for backward. Scoring
    # all V views would blow the 24GB budget the rasterizer already strains, so score a
    # random subset each step (unbiased in expectation, bounded memory).
    lpips_views: int = 2
    train_loo_prob: float = 1.0
    # Leave-one-out training objective: render each view from the OTHER views only.
    # Off = the legacy self-reprojection objective (nearly free, cancels pose error,
    # does not require multi-view consistency) — the likely reason fine-tuning never
    # generalised to held-out views. Requires voxelize=False (needs gaussian_frame_idx).
    train_loo: bool = False
    # >0 = tracker-driven piecewise-rigid motion for dynamic Gaussians (K groups).
    dyn_motion_groups: int = 0
    dyn_mask_dir: Optional[str] = None  # If set, load PRECOMPUTED dynamic masks (by frame stem) and override the live per-window detection for BOTH the downweight loss and the temporal loss. Use the validated 518+full-span masks so fine-tuning is shaped by correct masks.

    # Static-first curriculum (schedule on the dynamic-pixel MSE downweight).
    # Rationale: the pretrained head is near-optimal on static Bonn geometry but
    # takes an OOD shock when moving people/objects are thrown at it from step 0,
    # and that shock feeds the late-epoch drift (see project memory). So keep
    # dynamic pixels heavily downweighted early (head consolidates static
    # reconstruction on the OOD domain), then linearly phase dynamic content back
    # in. This reuses the existing dyn_mask + downweight machinery — the only
    # change is that the effective downweight now varies by epoch:
    #   epochs [0, curriculum_static_epochs)                     -> curriculum_static_downweight (hi)
    #   epochs [static, static+curriculum_ramp_epochs)           -> linear ramp hi -> dynamic_loss_downweight (lo)
    #   epochs [static+ramp, end)                                -> dynamic_loss_downweight (lo, i.e. joint training)
    # Disabled by default so existing recipes are unchanged; enable with --static_first.
    static_first_curriculum: bool = False
    curriculum_static_epochs: int = 2       # epochs of pure static (dynamic held at curriculum_static_downweight)
    curriculum_ramp_epochs: int = 3         # epochs to linearly phase dynamic in down to dynamic_loss_downweight
    curriculum_static_downweight: float = 1.0  # dynamic-pixel downweight during the static phase (1.0 = fully masked)

    # Per-frame dynamic compositing.
    # The Gaussians form ONE merged cloud with no time axis, so a moving object's
    # Gaussians from all V frames are rendered into EVERY frame -> it appears at all
    # V of its past positions ("ghosting"), and the model's only recourse is to fade
    # it out. That is why fine-tuning DEGRADES dynamic PSNR below the frozen baseline
    # (20.58 -> 18.72 held-out) no matter how the loss is weighted.
    # With this on, a Gaussian on a moving object renders ONLY into the frame it was
    # unprojected from; static Gaussians still render into all frames (keeping their
    # multi-view fusion). Uses the dyn_mask the backbone already predicts, as a
    # STRUCTURAL signal rather than merely a loss weight. Off by default so all
    # existing baselines reproduce bit-for-bit.
    per_frame_dynamic: bool = False

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
    keep_best_n: int = 3         # Keep only the newest N dated checkpoint_best_ep*.pt (newest == highest full PSNR). 0 = keep all. checkpoint_best.pt/final/latest never pruned.
    # Debug / smoke-test caps: stop each epoch after this many train / val batches
    # (0 = no cap, the normal setting). Bounds wall-clock so a smoke test can reach
    # validation + checkpointing + "Training complete" in minutes instead of hours.
    max_train_batches: int = 0
    max_val_batches: int = 0

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
        self.dataset_name = dataset_name  # for locating per-sequence precomputed masks
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
        frame_names = []  # rgb stems, for aligning precomputed dynamic masks in eval
        has_all_poses = True

        for i in range(self.num_frames):
            frame_idx = start_idx + i * self.frame_stride
            frame_path, c2w_pose = self.window_frames[frame_idx]
            frame_names.append(frame_path.stem)

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
            "frame_names": frame_names,             # list[str], len V
            "dataset_name": self.dataset_name,      # for per-sequence precomputed masks
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
        hybrid_voxelize=config.hybrid_voxelize,
        # fusion voxel edge: must EXCEED point spacing or nothing merges (see --voxel_size)
        voxel_size=config.voxel_size,
        dynamic_mask_threshold=None,
        dynamic_n_clusters=64,
        dyn_motion_groups=config.dyn_motion_groups,
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


def freeze_backbone(model: AnySplat, unfreeze_depth_head: bool = False):
    """Freeze the VGGT4D backbone; train the Gaussian head (+ optionally depth_head).

    unfreeze_depth_head trains the module that produces Gaussian POSITIONS
    (means = origins + directions * depths). Rationale: inter-frame depth
    disagreement is the common root of three measured problems -- the PLY's
    ribbon artefact (same surface at different depths per frame), the hybrid
    fusion failure (averaging disagreeing depths misplaces the fused point), and
    the leave-one-out shrink incentive (a Gaussian is only ever graded from views
    where it is misplaced, so shrinking reduces its error contribution). AnySplat
    itself trains depth_head (freeze_backbone: false + depth_consis loss).

    It does NOT enable dynamics: no per-pixel head can place an object at a target
    timestamp because the target index never enters it -- confirmed empirically by
    the stride-8 control (real motion in training: static +0.44, dynamic +0.03).
    """
    global _TRAINABLE_PREFIXES
    for param in model.encoder.aggregator.parameters():
        param.requires_grad = False
    for param in model.encoder.camera_head.parameters():
        param.requires_grad = False
    if hasattr(model.encoder, 'depth_head'):
        for param in model.encoder.depth_head.parameters():
            param.requires_grad = bool(unfreeze_depth_head)
        if unfreeze_depth_head and 'depth_head' not in _TRAINABLE_PREFIXES:
            # MUST happen for the weights to survive save/load (see _TRAINABLE_PREFIXES)
            _TRAINABLE_PREFIXES.append('depth_head')
            print("[unfreeze] depth_head is TRAINABLE and will be saved/restored",
                  flush=True)
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

def curriculum_dynamic_downweight(epoch: int, config: TrainingConfig) -> tuple:
    """Effective dynamic-pixel MSE downweight for `epoch` under the static-first curriculum.

    Returns (downweight, phase) where phase is one of "static", "ramp", "joint"
    (or "off" when the curriculum is disabled). The schedule is a pure function
    of the epoch index, so it survives checkpoint resume with no extra state.

    Phase 1 (static): dynamic held at `curriculum_static_downweight` (hi).
    Phase 2 (ramp):   linear interpolation hi -> `dynamic_loss_downweight` (lo),
                      landing exactly on lo at the final ramp epoch.
    Phase 3 (joint):  fixed at `dynamic_loss_downweight` (lo).
    """
    if not config.static_first_curriculum:
        return config.dynamic_loss_downweight, "off"

    static_e = config.curriculum_static_epochs
    ramp_e = config.curriculum_ramp_epochs
    hi = config.curriculum_static_downweight
    lo = config.dynamic_loss_downweight

    if epoch < static_e:
        return hi, "static"
    if ramp_e > 0 and epoch < static_e + ramp_e:
        # +1 so the first ramp epoch already moves off hi and the last lands on lo.
        frac = min((epoch - static_e + 1) / ramp_e, 1.0)
        return hi + (lo - hi) * frac, "ramp"
    return lo, "joint"


def compute_rendering_loss(
    model: AnySplat,
    images: torch.Tensor,
    gaussians,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    dyn_mask: Optional[torch.Tensor] = None,
    dynamic_loss_downweight: float = 0.0,
    gaussian_frame_idx: Optional[torch.Tensor] = None,
    gaussian_dyn_flag: Optional[torch.Tensor] = None,
    gaussian_only_view=None,
    leave_one_out: bool = False,
    dyn_centroid: Optional[torch.Tensor] = None,
    dyn_centroid_pred: Optional[torch.Tensor] = None,
    dyn_centroid_valid: Optional[torch.Tensor] = None,
    dyn_group_centroid: Optional[torch.Tensor] = None,
    dyn_group_pred: Optional[torch.Tensor] = None,
    dyn_group_valid: Optional[torch.Tensor] = None,
    gaussian_group_idx: Optional[torch.Tensor] = None,
    per_frame_compositing: bool = False,
) -> tuple:
    """
    Compute MSE rendering loss by rendering predicted Gaussians with given poses.

    gaussian_frame_idx / gaussian_dyn_flag enable PER-FRAME DYNAMIC COMPOSITING:
    dynamic Gaussians are rendered only into the view they came from (removing the
    multi-frame ghosting of moving objects), while static Gaussians still render into
    every view. Pass None (the default) for the original all-Gaussians-everywhere
    behaviour, so existing baselines reproduce exactly.

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
        gaussian_frame_idx=gaussian_frame_idx,
        gaussian_dyn_flag=gaussian_dyn_flag,
        gaussian_only_view=gaussian_only_view,
        leave_one_out=leave_one_out,
        dyn_centroid=dyn_centroid,
        dyn_centroid_pred=dyn_centroid_pred,
        dyn_centroid_valid=dyn_centroid_valid,
        dyn_group_centroid=dyn_group_centroid,
        dyn_group_pred=dyn_group_pred,
        dyn_group_valid=dyn_group_valid,
        gaussian_group_idx=gaussian_group_idx,
        per_frame_compositing=per_frame_compositing,
    )

    pred_rgb = output.color  # [B, V, 3, H, W]
    gt_rgb = images  # Already in [0, 1]

    # NOTE: no coverage/alpha weighting here, deliberately. An earlier version weighted
    # the LOO loss by rendered alpha so that pixels no other view covers were not
    # punished. That opened the opposite loophole: uncovered pixels carry zero weight,
    # so the model could ABANDON difficult pixels for free and sharpen the rest -- which
    # is how the Gaussian scales collapsed 26x (0.0041 -> 0.00016) while PSNR still rose.
    # The evaluation metric is an UNWEIGHTED mean over all pixels, so training uses the
    # same unweighted loss: optimise exactly what is measured, no reweighting.
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


def _resolve_mask_path(mask_dir, dataset_name, stem):
    """Locate a precomputed mask PNG. Supports two layouts:
      - PARENT (multi-seq): <mask_dir>/<dataset_name>/masks/<stem>.png  (precompute's output)
      - FLAT (single-seq):  <mask_dir>/<stem>.png                      (point straight at .../masks)
    Returns the first existing path, or None.
    """
    if dataset_name:
        p = os.path.join(mask_dir, dataset_name, "masks", f"{stem}.png")
        if os.path.exists(p):
            return p
    p = os.path.join(mask_dir, f"{stem}.png")
    return p if os.path.exists(p) else None


def load_precomputed_masks(frame_names, mask_dir, H, W, device, dataset_name=None):
    """Load precomputed dynamic-mask PNGs by frame stem and resample to (H, W).

    The precompute writes full-frame masks at detection resolution (aspect-preserved,
    e.g. 392x518); train/eval render full-frame squash at (H, W), so a plain interpolate
    reproduces the same squash and aligns mask to render. Missing frames -> zeros.
    `dataset_name` selects the per-sequence subdir so ONE --dyn_mask_dir works across a
    multi-sequence ConcatDataset (see _resolve_mask_path).

    Returns [1, V, H, W] float in {0,1}, or None if NO frame had a mask file (caller
    then falls back to the live detection). Shared by train_epoch, validate and eval.
    """
    masks, found = [], 0
    for stem in frame_names:
        p = _resolve_mask_path(mask_dir, dataset_name, stem)
        if p is not None:
            m = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0  # [Hm, Wm]
            t = F.interpolate(torch.from_numpy(m)[None, None], size=(H, W), mode="nearest")[0, 0]
            masks.append((t > 0.5).float())
            found += 1
        else:
            masks.append(torch.zeros(H, W))
    if found == 0:
        return None
    return torch.stack(masks, dim=0).unsqueeze(0).to(device)  # [1, V, H, W]


def load_batch_dyn_masks(batch, images, config, device):
    """Load this batch's PRECOMPUTED dynamic masks (config.dyn_mask_dir), or None.

    Called BEFORE the encoder forward and passed as `dyn_mask_override=` so the
    encoder takes its FAST PATH: Pass-1 attention detection and Stage-3 open3d refine
    are skipped (they are exactly what the precomputed mask replaces — Stage 3 alone
    is ~10s+ CPU per batch), Pass-2 token suppression is conditioned on the GOOD mask
    (matching how evaluation runs the backbone), and infos['dyn_mask'] /
    gaussian_dyn_flag are derived from it — so the downweight loss, the temporal
    loss AND the compositing labels all see the same correct mask.
    Returns [1, V, H, W] on `device`, or None (→ encoder falls back to live detection).
    """
    if config.dyn_mask_dir is None or "frame_names" not in batch:
        return None
    raw = batch["frame_names"]
    frame_names = [x[0] if isinstance(x, (list, tuple)) else x for x in raw]
    ds = batch.get("dataset_name")
    if isinstance(ds, (list, tuple)):
        ds = ds[0]
    H, W = images.shape[-2:]
    return load_precomputed_masks(frame_names, config.dyn_mask_dir, H, W, device, dataset_name=ds)


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

    # Static-first curriculum: the dynamic-pixel downweight is constant within an
    # epoch (pure function of the epoch index), so resolve it once here.
    dyn_downweight, curriculum_phase = curriculum_dynamic_downweight(epoch, config)
    if config.static_first_curriculum:
        print(f"  [curriculum] phase={curriculum_phase} "
              f"dynamic_downweight={dyn_downweight:.3f} "
              f"(static_epochs={config.curriculum_static_epochs}, "
              f"ramp_epochs={config.curriculum_ramp_epochs}, "
              f"hi={config.curriculum_static_downweight}, lo={config.dynamic_loss_downweight})")

    total_loss = 0.0
    total_mse_loss = 0.0
    total_temporal_loss = 0.0
    total_scale_reg = 0.0
    total_sh_reg = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for batch_idx, batch in enumerate(pbar):
        # Smoke-test cap: process at most max_train_batches, then end the epoch
        # early (epoch-end logging + return below still run normally).
        if config.max_train_batches and batch_idx >= config.max_train_batches:
            break

        images = batch["images"].to(device)  # [B, V, 3, H, W]

        # Add batch dimension if needed
        if images.dim() == 4:
            images = images.unsqueeze(0)

        b, v, c, h, w = images.shape

        # Load precomputed masks BEFORE the forward and hand them to the encoder:
        # fast path (skips Pass-1 detection + Stage-3 refine — big per-batch speedup)
        # AND Pass-2/token-suppression + gaussian_dyn_flag + infos['dyn_mask'] are all
        # conditioned on the GOOD mask, exactly as in evaluation. A post-forward infos
        # override cannot do any of that (audit Bug 1).
        precomp_mask = load_batch_dyn_masks(batch, images, config, device)
        if precomp_mask is not None and batch_idx == 0:
            print(f"  [dyn_mask] using PRECOMPUTED masks from {config.dyn_mask_dir} (encoder fast path)")

        # Forward pass with mixed precision
        with autocast(enabled=config.mixed_precision):
            # Run encoder (uses predicted poses internally for depth unprojection)
            encoder_output = model.encoder(images, global_step=global_step,
                                           dyn_mask_override=precomp_mask)
            gaussians = encoder_output.gaussians
            # SCALE INIT for hybrid fusion. The head's pretrained scales are sized for
            # ~0.001 point spacing; under fusion at voxel_size 0.005 the spacing is ~5x
            # larger, so unmodified splats cover a small fraction of each surface and
            # training would start at ~10 dB and spend epochs merely growing them.
            # Measured on the frozen head: a 5x enlargement recovers static 10.20 ->
            # 18.19 dB, which is what identified the collapse as COVERAGE rather than
            # geometry. This constant multiplier just starts the optimiser in that
            # basin; the head still learns per-Gaussian scales from there (which the
            # global multiplier cannot do -- it wrongly inflates the per-pixel dynamic
            # Gaussians too, dyn 20.64 -> 13.91).
            if config.scale_mult != 1.0:
                gaussians.scales = gaussians.scales * config.scale_mult
                if getattr(gaussians, "covariances", None) is not None:
                    # covariance is quadratic in linear size; gsplat renders covars
                    gaussians.covariances = gaussians.covariances * (config.scale_mult ** 2)
                if global_step == 0:
                    print(f"[scale_mult] train init x{config.scale_mult} -> "
                          f"scale_mean={float(gaussians.scales.mean()):.6f} "
                          f"covar_mean={float(gaussians.covariances.mean()):.9f}", flush=True)
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

            # Compute losses.
            # train_loo: render each view WITHOUT its own Gaussians, so the head must
            # reconstruct it from the OTHER views. Without this the loss is
            # SELF-REPROJECTION (unproject pixel from view j -> project back into view j
            # returns the same pixel, cancelling pose error exactly), which is nearly
            # free and requires no multi-view consistency — i.e. the objective does not
            # ask for what evaluation measures. LOO aligns training with the held-out
            # protocol. Needs gaussian_frame_idx (requires voxelize=False).
            # Randomise the compositing regime for THIS step (see train_loo_prob).
            # One regime per step -> identical memory; the mix happens in expectation.
            use_loo = config.train_loo and (
                config.train_loo_prob >= 1.0
                or torch.rand(()).item() < config.train_loo_prob)

            mse_loss, render_out = compute_rendering_loss(
                model, images, gaussians, render_extrinsics, render_intrinsics,
                dyn_mask=infos.get('dyn_mask', None),
                dynamic_loss_downweight=dyn_downweight,
                gaussian_frame_idx=(infos.get('gaussian_frame_idx')
                                    if (config.per_frame_dynamic or config.train_loo) else None),
                gaussian_only_view=infos.get('gaussian_only_view'),
                gaussian_dyn_flag=(infos.get('gaussian_dyn_flag')
                                   if config.per_frame_dynamic else None),
                per_frame_compositing=config.per_frame_dynamic,
                leave_one_out=use_loo,
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

            # Coverage term: push rendered alpha up to the valid mask (see opacity_weight).
            opacity_loss = torch.zeros((), device=images.device)
            if config.opacity_weight > 0.0 and render_out is not None \
                    and getattr(render_out, "alpha", None) is not None:
                alpha = render_out.alpha
                vm = encoder_output.depth_dict.get("conf_valid_mask")
                vm = torch.ones_like(alpha) if vm is None else vm.float().to(alpha.device)
                if vm.shape != alpha.shape:
                    vm = torch.ones_like(alpha)
                opacity_loss = F.mse_loss(alpha, vm)
                # One-time proof the term is LIVE (alpha real, differentiable, shapes sane).
                # Every silent-inert bug we hit (compute_lpips under @no_grad, the empty
                # $MASKS dir) was invisible because nothing printed. This prints.
                if not getattr(train_epoch, "_opac_logged", False):
                    train_epoch._opac_logged = True
                    print(f"[opacity_cov] LIVE w={config.opacity_weight} | "
                          f"alpha{tuple(alpha.shape)} mean={alpha.mean().item():.4f} "
                          f"min={alpha.min().item():.4f} "
                          f"frac<0.5={(alpha < 0.5).float().mean().item():.3f} | "
                          f"target mean={vm.mean().item():.4f} | "
                          f"requires_grad={alpha.requires_grad} | "
                          f"loss={opacity_loss.item():.5f}", flush=True)

            # Perceptual term on a random subset of views (see lpips_views).
            lpips_loss = torch.zeros((), device=images.device)
            if config.lpips_weight > 0.0 and render_out is not None:
                pred_v = render_out.color[0].clamp(0, 1)      # [V, 3, H, W]
                gt_v = images[0].clamp(0, 1)
                k = min(config.lpips_views, pred_v.shape[0])
                sel = torch.randperm(pred_v.shape[0], device=pred_v.device)[:k]
                # NOT compute_lpips(): that helper is @torch.no_grad, so using it as a
                # loss would silently contribute ZERO gradient. Call the cached LPIPS
                # module directly so the term actually trains.
                lpips_loss = get_lpips(pred_v.device).forward(
                    gt_v[sel], pred_v[sel], normalize=True).mean()

            # Total loss
            loss = (
                config.mse_weight * mse_loss +
                config.lpips_weight * lpips_loss +
                config.opacity_weight * opacity_loss +
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
                'train/opacity_cov': float(opacity_loss),
                # THE health metric for this run: alpha_mean is the quantity that
                # collapsed (only 19.7% of Gaussians renderable vs 68.9% frozen).
                # It must rise/hold, not fall.
                'train/alpha_mean': (float(render_out.alpha.mean())
                                     if render_out is not None
                                     and getattr(render_out, 'alpha', None) is not None
                                     else 0.0),
                'train/f_dc_absmax': f_dc_absmax,
                'train/scale_max': scale_max,
                'train/lr': last_lrs[0],
                'train/dynamic_downweight': dyn_downweight,
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
    total_psnr_dynamic = 0.0
    n_static_frames = 0
    n_dynamic_frames = 0
    n_frames = 0
    num_batches = 0

    with torch.no_grad():
        for val_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            # Smoke-test cap: validate on at most max_val_batches windows.
            if config.max_val_batches and val_idx >= config.max_val_batches:
                break

            images = batch["images"].to(device)
            if images.dim() == 4:
                images = images.unsqueeze(0)

            b, v, c, h, w = images.shape

            # Same pre-forward mask handoff as training (fast path + consistent
            # Pass-2 conditioning), so val measures the model exactly as trained.
            precomp_mask = load_batch_dyn_masks(batch, images, config, device)
            encoder_output = model.encoder(images, global_step=global_step,
                                           dyn_mask_override=precomp_mask)
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

            # Validation stays ALWAYS leave-one-out (not randomised) so val PSNR remains
            # directly comparable to the held-out eval protocol across runs.
            # Validation must use the SAME compositing AND the same held-out protocol
            # as training, or val PSNR measures a different renderer/task than the one
            # being optimised (train_loo -> val must also be leave-one-out).
            _, render_output = compute_rendering_loss(
                model, images, gaussians, render_extrinsics, render_intrinsics,
                gaussian_frame_idx=(infos.get('gaussian_frame_idx')
                                    if (config.per_frame_dynamic or config.train_loo) else None),
                gaussian_only_view=infos.get('gaussian_only_view'),
                gaussian_dyn_flag=(infos.get('gaussian_dyn_flag')
                                   if config.per_frame_dynamic else None),
                per_frame_compositing=config.per_frame_dynamic,
                leave_one_out=config.train_loo,  # validation: ALWAYS LOO (comparable to the eval protocol)
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
                        # Dynamic-region PSNR (tracked for monitoring; NOT used for
                        # checkpoint selection). Mirrors eval_gaussian_head.py.
                        n_dyn = mask.sum().item()
                        if n_dyn >= 10:
                            dyn_w = mask.clamp(0, 1).unsqueeze(0).expand(3, -1, -1)
                            mse_d = ((pred_c[bi, vi] * dyn_w - gt_c[bi, vi] * dyn_w) ** 2).sum() / (3 * n_dyn)
                            total_psnr_dynamic += -10 * torch.log10(mse_d + 1e-8).item()
                            n_dynamic_frames += 1

            num_batches += 1

    # Per-frame averaging (matches eval_gaussian_head.py).
    nf = max(n_frames, 1)
    metrics = {
        'val_mse':          total_mse  / nf,
        'val_psnr':         total_psnr / nf,
        'val_ssim':         total_ssim / nf,
        'val_psnr_static':  total_psnr_static / n_static_frames if n_static_frames > 0 else None,
        'val_psnr_dynamic': total_psnr_dynamic / n_dynamic_frames if n_dynamic_frames > 0 else None,
    }

    static_str = (f", PSNR-static: {metrics['val_psnr_static']:.2f} dB"
                  if metrics['val_psnr_static'] is not None else "")
    dynamic_str = (f", PSNR-dynamic: {metrics['val_psnr_dynamic']:.2f} dB"
                   if metrics['val_psnr_dynamic'] is not None else "")
    print(f"Validation - MSE: {metrics['val_mse']:.4f}, "
          f"PSNR: {metrics['val_psnr']:.2f} dB, "
          f"SSIM: {metrics['val_ssim']:.4f}"
          f"{static_str}{dynamic_str}")

    wandb_payload = {
        'val/mse': metrics['val_mse'],
        'val/psnr_db': metrics['val_psnr'],
        'val/ssim': metrics['val_ssim'],
    }
    if metrics['val_psnr_static'] is not None:
        wandb_payload['val/psnr_static_db'] = metrics['val_psnr_static']
    if metrics['val_psnr_dynamic'] is not None:
        wandb_payload['val/psnr_dynamic_db'] = metrics['val_psnr_dynamic']
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


def head_state_dict(model):
    """Trainable-only state dict: the gaussian_param_head + gaussian_adapter keys.

    The frozen VGGT4D backbone is ~99% of a full state_dict (~3.2 GB) and is
    DISCARDED on load anyway (load_checkpoint restores only these keys — the
    backbone always comes from the freshly loaded pretrained weights). Saving just
    these keys drops each checkpoint to ~tens of MB. This filter is the exact
    counterpart of the one in load_checkpoint / eval_gaussian_head.py, so old
    full checkpoints and new head-only ones both load identically.
    """
    prefixes = trainable_prefixes()
    return {k: v for k, v in model.state_dict().items()
            if any(pfx in k for pfx in prefixes)}


# WHAT THE CHECKPOINT CONTAINS. This list is the SINGLE source of truth: it drives
# both what train saves and what eval restores (eval reads it back out of the
# checkpoint). Keeping it implicit is a silent-wrong-number trap -- unfreezing a
# module without adding it here trains the module, DISCARDS it at save time, and
# then evaluates with pretrained weights, producing plausible numbers that mean
# nothing. Set by freeze_backbone() according to what was actually unfrozen.
_TRAINABLE_PREFIXES = ['gaussian_param_head', 'gaussian_adapter']


def trainable_prefixes():
    return list(_TRAINABLE_PREFIXES)


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, config, epoch_completed=False):
    """Save training checkpoint.

    epoch_completed distinguishes a MID-epoch periodic save (False) from an
    END-of-epoch / final save (True). Resume uses it to decide whether to advance
    to the next epoch (completed) or RE-RUN the interrupted epoch (not completed),
    so a wall-clock kill mid-epoch never silently skips training — and, for the
    last epoch, never "finishes" the job without actually training it.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'epoch_completed': epoch_completed,
        'global_step': global_step,
        'model_state_dict': head_state_dict(model),  # head/adapter only (~tens of MB)
        'saved_prefixes': trainable_prefixes(),
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
    # Legacy checkpoints (pre-flag) default to True so they keep the old
    # advance-to-next-epoch behaviour rather than unexpectedly re-running.
    epoch_completed = checkpoint.get('epoch_completed', True)

    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, step {global_step}, "
          f"epoch_completed={epoch_completed})")
    print(f"  Restored {len(head_keys)} gaussian head tensors; backbone left as freshly loaded.")
    return epoch, global_step, epoch_completed


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
                             "Requires VGGT4D dynamic detection. With --static_first this is the curriculum's final (joint-phase) value.")
    parser.add_argument("--static_first", action="store_true",
                        help="Enable the static-first curriculum: hold dynamic pixels at --curriculum_static_downweight for "
                             "--curriculum_static_epochs, then linearly phase them in over --curriculum_ramp_epochs down to "
                             "--dynamic_loss_downweight. Reuses the dyn_mask machinery; requires VGGT4D dynamic detection.")
    parser.add_argument("--curriculum_static_epochs", type=int, default=2,
                        help="Epochs to keep dynamic pixels at --curriculum_static_downweight before phasing in (static_first only).")
    parser.add_argument("--curriculum_ramp_epochs", type=int, default=3,
                        help="Epochs over which to linearly ramp the dynamic downweight from the static value to "
                             "--dynamic_loss_downweight (static_first only). 0 = hard switch at the end of the static phase.")
    parser.add_argument("--curriculum_static_downweight", type=float, default=1.0,
                        help="Dynamic-pixel downweight during the static phase (static_first only). 1.0 = dynamic fully masked.")
    parser.add_argument("--per_frame_dynamic", action="store_true",
                        help="Per-frame dynamic compositing: render dynamic Gaussians ONLY into the frame they "
                             "were unprojected from (static ones still render into every frame). Removes the "
                             "multi-frame ghosting of moving objects, which is why fine-tuning currently DEGRADES "
                             "dynamic PSNR. Requires VGGT4D dynamic detection; off by default (baselines reproduce).")
    parser.add_argument("--scale_mult", type=float, default=1.0,
                        help="Multiply Gaussian scales (and covariances) at construction. Use ~5 "
                             "with --voxel_size 0.005: fused spacing is ~5x the pretrained scale, "
                             "and 5x measurably recovers static 10.20 -> 18.19 dB on the frozen "
                             "head. Starts training in the right basin instead of growing scales "
                             "for several epochs.")
    parser.add_argument("--voxel_size", type=float, default=0.001,
                        help="Fusion voxel edge length (see --hybrid_voxelize). Default 0.001 "
                             "equals the point spacing, so it merges nothing; sweep upward.")
    parser.add_argument("--unfreeze_depth_head", action="store_true",
                        help="Also train depth_head (Gaussian POSITIONS). Targets inter-frame "
                             "depth disagreement, the common root of the PLY ribbons, the fusion "
                             "failure and the LOO shrink incentive. Upstream AnySplat trains it. "
                             "Does NOT enable dynamics (no target-timestamp input).")
    parser.add_argument("--hybrid_voxelize", action="store_true",
                        help="Fuse STATIC pixels into shared voxels (one set per target view, "
                             "that view excluded so leave-one-out stays exact) and keep DYNAMIC "
                             "pixels per-pixel. Removes the redundancy that made scale collapse "
                             "free. Requires dynamic masks (--dyn_mask_dir).")
    parser.add_argument("--opacity_weight", type=float, default=0.0,
                        help="Coverage loss MSE(rendered_alpha, valid_mask) from AnySplat's own suite "
                             "(upstream 0.1). Penalises holes/transparency, so it is the direct counter "
                             "to scale collapse (small splats -> low alpha -> charged for).")
    parser.add_argument("--lpips_weight", type=float, default=0.0,
                        help="Perceptual (LPIPS) loss weight. AnySplat trains with 0.05; our loop used "
                             "MSE only, which is why PSNR improves while LPIPS regresses. Set 0.05 to "
                             "restore the upstream term.")
    parser.add_argument("--lpips_views", type=int, default=2,
                        help="Views per step to score LPIPS on (memory bound; LPIPS is differentiable "
                             "so its VGG activations are retained for backward).")
    parser.add_argument("--train_loo_prob", type=float, default=1.0,
                        help="With --train_loo: probability of using the leave-one-out renderer on a "
                             "given step (1.0 = always). Training ONLY under LOO teaches the head to "
                             "compensate for the missing own-frame Gaussians with brighter/more opaque "
                             "output, which is correct under LOO but over-bright under normal full "
                             "compositing (visible as a washed-out PLY). 0.5 randomises the regime so "
                             "the same Gaussians must be valid in both. No extra memory: one regime per step.")
    parser.add_argument("--train_loo", action="store_true",
                        help="LEAVE-ONE-OUT training objective: render each view WITHOUT its own "
                             "Gaussians, so it must be reconstructed from the OTHER views. Without "
                             "this the loss is self-reprojection (project->unproject->project returns "
                             "the same pixel and cancels pose error), which requires no multi-view "
                             "consistency and does not match how we evaluate. Validation follows the "
                             "same protocol automatically. The MSE is coverage-weighted by rendered "
                             "alpha (weighted mean, detached) so pixels no other view covers are not "
                             "punished — otherwise the head learns to inflate scales over the holes.")
    parser.add_argument("--dyn_mask_dir", type=str, default=None,
                        help="Directory of PRECOMPUTED dynamic-mask PNGs (named by rgb frame stem), e.g. "
                             "output_dyn_masks_precomputed_cs16_r518_st3_fs49/<SEQ>/masks. When set, these "
                             "OVERRIDE the live per-window detection for both the dynamic downweight loss and "
                             "the temporal loss — fine-tune with the validated 518+full-span masks. For "
                             "multi-sequence training, point at a dir whose masks cover all training sequences.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging (otherwise enabled by default).")
    parser.add_argument("--wandb_project", type=str, default="dynrecsplat",
                        help="wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb run name; if omitted, wandb auto-generates one.")
    parser.add_argument("--log_every_n_steps", type=int, default=25,
                        help="Cadence of per-batch wandb metric logging (lower = denser curves).")
    parser.add_argument("--save_every_n_steps", type=int, default=200,
                        help="Cadence of periodic checkpoint saves (optimizer steps). Lower to force an early save (smoke tests).")
    parser.add_argument("--max_train_batches", type=int, default=0,
                        help="Stop each epoch after this many train batches (0 = no cap). Smoke-test / debug knob to bound wall-clock.")
    parser.add_argument("--max_val_batches", type=int, default=0,
                        help="Validate on at most this many batches (0 = no cap). Smoke-test / debug knob to bound wall-clock.")
    parser.add_argument("--val_every_epochs", type=int, default=5,
                        help="Validate at epoch 1, then every N epochs, then at the end. Set to 1 for a dense val curve (sweeps).")
    parser.add_argument("--keep_best_n", type=int, default=3,
                        help="Keep only the newest N dated checkpoint_best_ep*.pt files (newest == highest full PSNR). 0 = keep all. Never prunes checkpoint_best/final/latest.")
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
        train_loo=args.train_loo,
        unfreeze_depth_head=args.unfreeze_depth_head,
        hybrid_voxelize=args.hybrid_voxelize,
        voxel_size=args.voxel_size,
        scale_mult=args.scale_mult,
        opacity_weight=args.opacity_weight,
        lpips_weight=args.lpips_weight,
        lpips_views=args.lpips_views,
        train_loo_prob=args.train_loo_prob,
        dyn_mask_dir=args.dyn_mask_dir,
        per_frame_dynamic=args.per_frame_dynamic,
        static_first_curriculum=args.static_first,
        curriculum_static_epochs=args.curriculum_static_epochs,
        curriculum_ramp_epochs=args.curriculum_ramp_epochs,
        curriculum_static_downweight=args.curriculum_static_downweight,
        use_gt_poses=not args.no_gt_poses,
        vggt4d_weights_path=args.vggt4d_weights_path,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_every_n_steps=args.log_every_n_steps,
        save_every_n_steps=args.save_every_n_steps,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        val_every_epochs=args.val_every_epochs,
        keep_best_n=args.keep_best_n,
        gradient_clip=args.gradient_clip,
    )

    print("=" * 60)
    print("Fine-tuning Temporal Gaussian Head")
    print("=" * 60)
    print(f"Dataset: {config.data_dir}/{config.dataset_name}")
    print(f"Intrinsics: {config.intrinsics_preset}")
    print(f"GT poses: {config.use_gt_poses}")
    print(f"Temporal loss weight: {config.temporal_consistency_weight}")
    if config.static_first_curriculum:
        print(f"Static-first curriculum: ON — static_epochs={config.curriculum_static_epochs}, "
              f"ramp_epochs={config.curriculum_ramp_epochs}, "
              f"dynamic downweight {config.curriculum_static_downweight} -> {config.dynamic_loss_downweight}")
    else:
        print(f"Static-first curriculum: OFF — dynamic downweight fixed at {config.dynamic_loss_downweight}")

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
    model = freeze_backbone(model, unfreeze_depth_head=config.unfreeze_depth_head)

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
                loaded_epoch, global_step, epoch_completed = load_checkpoint(
                    model, optimizer, scheduler, cand)
                # Advance past a COMPLETED epoch; RE-RUN an interrupted one so a
                # mid-epoch kill never skips training (and the final epoch never
                # "finishes" without running).
                start_epoch = loaded_epoch + 1 if epoch_completed else loaded_epoch
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
            # Init from FULL PSNR to match the full-PSNR selection criterion below.
            prior_best = existing.get('val_psnr') or 0.0
            best_val_psnr = float(prior_best)
            print(f"Found existing checkpoint_best.pt with PSNR {best_val_psnr:.2f} dB — "
                  f"will only overwrite if a later val beats this.")
            del existing
        except Exception as e:
            print(f"Could not read existing checkpoint_best.pt ({e}); starting best_val_psnr at 0.")

    # Instant-finish guard: resuming a checkpoint whose epoch is already the last
    # one makes range(start_epoch, num_epochs) empty, so training silently does
    # nothing and wandb logs an empty run. This is CORRECT for a genuinely finished
    # run, but it also happens when a preemption lands in the final epoch (resume
    # is epoch-granular: start_epoch = last_saved_epoch + 1). Say so out loud so an
    # empty run is never mistaken for a crash / data loss.
    if start_epoch >= config.num_epochs:
        print(f"\n[RESUME] Loaded checkpoint is already at/after the final epoch "
              f"(start_epoch={start_epoch}, num_epochs={config.num_epochs}) — "
              f"NOTHING TO TRAIN, so this run will finish immediately with no new "
              f"steps. To train more, delete the output dir ('{config.output_dir}') "
              f"to start fresh, or raise --num_epochs above {start_epoch}.")

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

            # Select on FULL PSNR (the headline metric). Static-only selection
            # rewarded the static-for-dynamic drift (the saved "best" ended up worse
            # on dynamic regions, the thesis target). Dynamic PSNR is tracked/logged
            # for monitoring but deliberately NOT used for selection.
            psnr_for_selection = val_metrics['val_psnr']
            if psnr_for_selection > best_val_psnr:
                best_val_psnr = psnr_for_selection
                ckpt = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': head_state_dict(model),  # head/adapter only (~tens of MB)
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config.__dict__,
                    'val_psnr': val_metrics['val_psnr'],
                    'val_psnr_static': val_metrics['val_psnr_static'],
                    'val_psnr_dynamic': val_metrics['val_psnr_dynamic'],
                }
                # Permanent, uniquely-named record (date + epoch + PSNR) that later
                # epochs can NEVER overwrite — this is exactly what was missing when
                # the epoch-1 best got clobbered and the manual backup was truncated.
                stamp = datetime.now().strftime("%Y%m%d-%H%M")
                dated_name = f"checkpoint_best_ep{epoch + 1}_{stamp}_psnr{best_val_psnr:.2f}dB.pt"
                _atomic_torch_save(ckpt, os.path.join(config.output_dir, dated_name))
                # Canonical checkpoint_best.pt kept for resume-guard + eval scripts.
                _atomic_torch_save(ckpt, os.path.join(config.output_dir, 'checkpoint_best.pt'))
                extra = ""
                if (val_metrics['val_psnr_static'] is not None
                        and val_metrics['val_psnr_dynamic'] is not None):
                    extra = (f" (static {val_metrics['val_psnr_static']:.2f}, "
                             f"dynamic {val_metrics['val_psnr_dynamic']:.2f})")
                print(f"New best model: PSNR {best_val_psnr:.2f} dB{extra} -> {dated_name}")

                # Prune old dated bests: keep only the newest `keep_best_n`. A new
                # best is saved only when full PSNR increases, so newest == highest
                # PSNR — this keeps the strongest checkpoints and never deletes the
                # one just written. checkpoint_best.pt / _final_ / _latest / _step
                # are NOT matched by this glob and are never touched.
                if config.keep_best_n and config.keep_best_n > 0:
                    import glob as _glob
                    dated = sorted(
                        _glob.glob(os.path.join(config.output_dir, 'checkpoint_best_ep*.pt')),
                        key=os.path.getmtime,
                    )
                    for old in dated[:-config.keep_best_n]:
                        try:
                            os.remove(old)
                            print(f"  Pruned old best: {os.path.basename(old)}")
                        except OSError as e:
                            print(f"  Could not prune {os.path.basename(old)}: {e}")

        # End-of-epoch checkpoint: durably capture every completed (expensive,
        # ~16h) epoch so a wall-clock kill never loses a finished epoch. Refreshes
        # checkpoint_latest.pt with epoch=N, so on --resume start_epoch=N+1 and the
        # next epoch begins cleanly. Runs regardless of the validation cadence.
        # (Mid-epoch periodic saves still bound in-epoch loss to save_every_n_steps.)
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, config,
                        epoch_completed=True)

    # Save final: canonical checkpoint_latest.pt/step (for resume) PLUS a permanent
    # dated copy so the end-of-run state is never lost to overwrite either.
    save_checkpoint(model, optimizer, scheduler, config.num_epochs - 1, global_step, config,
                    epoch_completed=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    _atomic_torch_save({
        'epoch': config.num_epochs - 1,
        'global_step': global_step,
        'model_state_dict': head_state_dict(model),  # head/adapter only (~tens of MB)
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config.__dict__,
    }, os.path.join(config.output_dir, f"checkpoint_final_{stamp}.pt"))

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Checkpoints: {config.output_dir}")
    print("=" * 60)

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
