import copy

# VGGT parts
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float
from src.dataset.shims.bounds_shim import apply_bounds_shim
from src.dataset.shims.normalize_shim import apply_normalize_shim
from src.dataset.shims.patch_shim import apply_patch_shim
from src.dataset.types import BatchedExample, DataShim
from src.geometry.projection import sample_image_grid

from src.model.encoder.heads.vggt_dpt_gs_head import VGGT_DPT_GS_Head
from src.model.encoder.vggt.utils.geometry import (
    batchify_unproject_depth_map_to_point_map,
    unproject_depth_map_to_point_map,
)
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.utils.geometry import get_rel_pos  # used for model hub
from torch import nn, Tensor
from torch_scatter import scatter_add, scatter_max

from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone

from .backbone.croco.misc import transpose_to_landscape
from .common.gaussian_adapter import (
    GaussianAdapter,
    GaussianAdapterCfg,
    UnifiedGaussianAdapter,
)
from .encoder import Encoder, EncoderOutput
from .heads import head_factory
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

root_path = os.path.abspath(".")
sys.path.append(root_path)
from src.model.encoder.heads.head_modules import TransformerBlockSelfAttn
from src.model.encoder.vggt.heads.dpt_head import DPTHead
from src.model.encoder.vggt.layers.mlp import Mlp
from src.model.encoder.vggt.models.vggt import VGGT
from src.model.encoder.vggt4d.models.vggt4d import VGGTFor4D
from src.model.encoder.vggt4d.utils import organize_qk_dict
from src.model.encoder.dyn_motion import compute_dyn_group_motion
from src.model.encoder.vggt4d.masks import (
    extract_dyn_map,
    cluster_attention_maps,
    adaptive_multiotsu_variance,
    RefineDynMask,
)

inf = float("inf")

# bfloat16 requires CUDA compute capability >= 8.0 (Ampere+); fall back to float16
_AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class GSHeadParams:
    dec_depth: int = 23
    patch_size: tuple[int, int] = (14, 14)
    enc_embed_dim: int = 2048
    dec_embed_dim: int = 2048
    feature_dim: int = 256
    depth_mode = ("exp", -inf, inf)
    conf_mode = True


@dataclass
class EncoderAnySplatCfg:
    name: Literal["anysplat"]
    anchor_feat_dim: int
    voxel_size: float
    n_offsets: int
    d_feature: int
    add_view: bool
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    pred_pose: bool = True
    gt_pose_to_pts: bool = False
    gs_prune: bool = False
    opacity_threshold: float = 0.001
    gs_keep_ratio: float = 1.0
    pred_head_type: Literal["depth", "point"] = "point"
    freeze_backbone: bool = False
    freeze_module: Literal[
        "all",
        "global",
        "frame",
        "patch_embed",
        "patch_embed+frame",
        "patch_embed+global",
        "global+frame",
        "None",
    ] = "None"
    distill: bool = False
    render_conf: bool = False
    opacity_conf: bool = False
    conf_threshold: float = 0.1
    intermediate_layer_idx: Optional[List[int]] = None
    voxelize: bool = False
    use_vggt4d: bool = False
    vggt4d_weights_path: Optional[str] = None
    # Dynamic mask extraction options
    enable_dynamic_detection: bool = False
    dynamic_mask_threshold: Optional[float] = None  # None = use adaptive threshold
    dyn_motion_groups: int = 0   # >0 enables tracker-driven piecewise-rigid motion (K groups)
    dynamic_n_clusters: int = 64  # Number of clusters for KMeans refinement
    suppress_dynamic_gaussians: bool = False
    # Temporal attention options for Gaussian head (Fix 2 for dynamic handling)
    use_temporal_attention: bool = False
    temporal_num_heads: int = 4
    temporal_dropout: float = 0.0
    temporal_spatial_downsample: int = 4  # Downsample factor for efficiency
    temporal_use_pe: bool = True  # Use learnable temporal positional encoding
    temporal_max_frames: int = 32  # Maximum number of frames supported


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


def predict_centroid_leave_one_out(centroids, valid, min_pts=3):
    """Predict the dynamic-content centroid at each frame j WITHOUT using frame j.

    centroids: [B, V, 3] per-frame 3D centroid of the dynamic points.
    valid:     [B, V] bool — frames with enough dynamic points to trust.

    For every target j we fit a straight line (constant velocity) to the centroids of
    the OTHER valid frames, as a function of frame index, and evaluate it at j. Frame
    j's own centroid is excluded, so this is usable under leave-one-out: the predicted
    position at the held-out timestamp is an extrapolation/interpolation of the motion
    seen in the source frames, never a read-out of the target frame.

    Fewer than `min_pts` usable source frames -> fall back to their mean (no motion
    estimate); none -> fall back to frame j's own centroid (displacement becomes 0).

    Returns [B, V, 3].
    """
    B, V, _ = centroids.shape
    idx = torch.arange(V, device=centroids.device, dtype=centroids.dtype)
    pred = centroids.clone()
    for b in range(B):
        for j in range(V):
            m = valid[b].clone()
            m[j] = False                      # exclude the target frame (no leak)
            k = int(m.sum())
            if k == 0:
                continue                      # keep own centroid -> zero displacement
            t = idx[m]
            c = centroids[b][m]               # [k, 3]
            if k < min_pts:
                pred[b, j] = c.mean(0)
                continue
            # least-squares line fit per axis: c ≈ a * t + b0
            t_mean = t.mean()
            c_mean = c.mean(0)
            dt = t - t_mean
            denom = (dt * dt).sum()
            if denom < 1e-8:
                pred[b, j] = c_mean
                continue
            slope = (dt.unsqueeze(-1) * (c - c_mean)).sum(0) / denom   # [3]
            pred[b, j] = c_mean + slope * (idx[j] - t_mean)
    return pred


class EncoderAnySplat(Encoder[EncoderAnySplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderAnySplatCfg) -> None:
        super().__init__(cfg)

        # Choose between VGGT and VGGT4D
        self.use_vggt4d = cfg.use_vggt4d
        if self.use_vggt4d:
            model_full = VGGTFor4D()
            if cfg.vggt4d_weights_path is not None:
                print(f"Loading VGGT4D weights from {cfg.vggt4d_weights_path}")
                ckpt = torch.load(cfg.vggt4d_weights_path, map_location="cpu", weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                model_full.load_state_dict(state_dict)
            else:
                print("Initializing VGGT4D from pretrained VGGT-1B weights")
                vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
                model_full.camera_head.load_state_dict(vggt_model.camera_head.state_dict())
                model_full.depth_head.load_state_dict(vggt_model.depth_head.state_dict())
                model_full.point_head.load_state_dict(vggt_model.point_head.state_dict())
                vggt_agg_state = vggt_model.aggregator.state_dict()
                model_full.aggregator.load_state_dict(vggt_agg_state, strict=False)
                del vggt_model
            print("Using VGGT4D aggregator for dynamic scene handling")
        else:
            model_full = VGGT.from_pretrained("facebook/VGGT-1B")

        self.aggregator = model_full.aggregator.to(_AMP_DTYPE)
        self.freeze_backbone = cfg.freeze_backbone
        self.distill = cfg.distill
        self.pred_pose = cfg.pred_pose

        self.camera_head = model_full.camera_head
        # Point tracker: weights come with the VGGT4D checkpoint (model_tracker_fixed).
        # Kept (frozen) so the dynamic-motion model can get temporal correspondence.
        self.track_head = getattr(model_full, 'track_head', None)
        if self.cfg.pred_head_type == "depth":
            self.depth_head = model_full.depth_head
        else:
            self.point_head = model_full.point_head

        # Storage for VGGT4D outputs (for later dynamic mask extraction, tbd)
        self.qk_dict = None
        self.enc_feat = None

        if self.distill:
            self.distill_aggregator = copy.deepcopy(self.aggregator)
            self.distill_camera_head = copy.deepcopy(self.camera_head)
            self.distill_depth_head = copy.deepcopy(self.depth_head)
            for module in [
                self.distill_aggregator,
                self.distill_camera_head,
                self.distill_depth_head,
            ]:
                for param in module.parameters():
                    param.requires_grad = False
                    param.data = param.data.cpu()

        if self.freeze_backbone:
            # Freeze backbone components
            if self.cfg.pred_head_type == "depth":
                for module in [self.aggregator, self.camera_head, self.depth_head]:
                    for param in module.parameters():
                        param.requires_grad = False
            else:
                for module in [self.aggregator, self.camera_head, self.point_head]:
                    for param in module.parameters():
                        param.requires_grad = False
        else:
            # aggregator freeze
            freeze_module = self.cfg.freeze_module
            if freeze_module == "None":
                pass

            elif freeze_module == "all":
                for param in self.aggregator.parameters():
                    param.requires_grad = False

            else:
                module_pairs = {
                    "patch_embed+frame": ["patch_embed", "frame"],
                    "patch_embed+global": ["patch_embed", "global"],
                    "global+frame": ["global", "frame"],
                }

                if freeze_module in module_pairs:
                    for name, param in self.aggregator.named_parameters():
                        if any(m in name for m in module_pairs[freeze_module]):
                            param.requires_grad = False
                else:
                    for name, param in self.named_parameters():
                        param.requires_grad = (
                            freeze_module not in name and "distill" not in name
                        )

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity
        self.voxel_size = cfg.voxel_size
        self.gs_params_head_type = cfg.gs_params_head_type
        # fake backbone for head parameters
        head_params = GSHeadParams()
        self.gaussian_param_head = VGGT_DPT_GS_Head(
            dim_in=2048,
            patch_size=head_params.patch_size,
            output_dim=self.raw_gs_dim + 1,
            activation="norm_exp",
            conf_activation="expp1",
            features=head_params.feature_dim,
            # Temporal attention parameters for cross-frame Gaussian feature fusion
            use_temporal_attention=cfg.use_temporal_attention,
            temporal_num_heads=cfg.temporal_num_heads,
            temporal_dropout=cfg.temporal_dropout,
            temporal_spatial_downsample=cfg.temporal_spatial_downsample,
            temporal_use_pe=cfg.temporal_use_pe,
            temporal_max_frames=cfg.temporal_max_frames,
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def normalize_pts3d(self, pts3ds, valid_masks, original_extrinsics=None):
        # normalize pts_all
        B = pts3ds.shape[0]
        pts3d_norms = []
        scale_factors = []
        for bs in range(B):
            pts3d, valid_mask = pts3ds[bs], valid_masks[bs]
            if original_extrinsics is not None:
                camera_c2w = original_extrinsics[bs]
                first_camera_w2c = (
                    camera_c2w[0].inverse().unsqueeze(0).repeat(pts3d.shape[0], 1, 1)
                )

                pts3d_homo = torch.cat(
                    [pts3d, torch.ones_like(pts3d[:, :, :, :1])], dim=-1
                )
                transformed_pts3d = torch.bmm(
                    first_camera_w2c, pts3d_homo.flatten(1, 2).transpose(1, 2)
                ).transpose(1, 2)[..., :3]
                scene_scale = torch.norm(
                    transformed_pts3d.flatten(0, 1)[valid_mask.flatten(0, 2).bool()],
                    dim=-1,
                ).mean()
            else:
                transformed_pts3d = pts3d[valid_mask]
                dis = transformed_pts3d.norm(dim=-1)
                scene_scale = dis.mean().clip(min=1e-8)
            # pts3d_norm[bs] = pts3d[bs] / scene_scale
            pts3d_norms.append(pts3d / scene_scale)
            scale_factors.append(scene_scale)
        return torch.stack(pts3d_norms, dim=0), torch.stack(scale_factors, dim=0)

    def align_pts_all_with_pts3d(
        self, pts_all, pts3d, valid_mask, original_extrinsics=None
    ):
        # align pts_all with pts3d
        B = pts_all.shape[0]

        # follow vggt's normalization implementation
        pts3d_norm, scale_factor = self.normalize_pts3d(
            pts3d, valid_mask, original_extrinsics
        )  # check if this is correct
        pts_all = pts_all * scale_factor.view(B, 1, 1, 1, 1)

        return pts_all

    def pad_tensor_list(self, tensor_list, pad_shape, value=0.0):
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype
                )
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def voxelize_static_hybrid(self, img_feat, pts3d, voxel_size, conf,
                               static_mask, exclude_frame: int = -1):
        """HYBRID fusion: fuse ONLY static pixels into shared voxels, optionally
        excluding one source frame.

        WHY HYBRID. With voxelize=False every pixel of every frame becomes its own
        Gaussian, so a surface seen by V frames carries V redundant copies. That
        redundancy is what makes SHRINKING FREE: any one copy can collapse and the
        others still cover the surface. We measured the consequence three separate
        times (scale_reg, an alpha-weighted loss, and LPIPS each drove a collapse),
        and the last one left only 19.7% of Gaussians large enough to render (vs
        68.9% frozen) -> the transparent, background-less PLY. Fusing static points
        removes the redundancy, so a Gaussian must actually cover its own patch.

        WHY NOT FUSE EVERYTHING. Fusion averages points from different frames into
        one anchor, which destroys the 1:1 Gaussian -> (frame, pixel) mapping. For
        static content that is exactly right (the surface IS shared across frames).
        For a MOVING point it is wrong twice over: the "shared" surface is at a
        different place in each frame, so the average is a smear, and we lose the
        per-frame identity that every temporal/dynamic mechanism needs. So dynamic
        pixels stay per-pixel with their frame index; only static content is fused.

        WHY exclude_frame. Fused voxels have no single source frame, so leave-one-out
        can no longer drop view j's own contribution (the decoder drops on
        `fidx == j`). A fused voxel would carry view j's OWN depth estimate into
        view j's render -- project->unproject->project, i.e. the self-reprojection
        shortcut that inflates static PSNR for a trivial reason. Since static is
        where our entire measured gain sits, that would silently invalidate the
        result. Passing exclude_frame=j rebuilds the static set from the OTHER
        frames only, keeping LOO exact.

        Args:
            img_feat:    [V, C, H, W] anchor features
            pts3d:       [V, 3, H, W] world points
            conf:        [V, H, W]    confidence (fusion weight)
            static_mask: [V, H, W]    True where the pixel is static AND conf-valid
            exclude_frame: frame index to leave out of the fusion (-1 = keep all)

        Returns:
            (voxel_pts [M,3], voxel_feats [M,C]) -- M = number of occupied voxels.
        """
        V, C, H, W = img_feat.shape
        keep = static_mask.clone()
        if exclude_frame >= 0:
            keep[exclude_frame] = False

        pts_f = pts3d.permute(0, 2, 3, 1).flatten(0, 2)[keep.flatten()]      # [N,3]
        feat_f = img_feat.permute(0, 2, 3, 1).flatten(0, 2)[keep.flatten()]  # [N,C]
        conf_f = conf.flatten()[keep.flatten()]                              # [N]
        if pts_f.numel() == 0:
            return (pts_f.new_zeros((0, 3)), feat_f.new_zeros((0, feat_f.shape[-1])))

        voxel_indices = (pts_f / voxel_size).round().int()
        _, inverse_indices, _ = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        )
        # Confidence-softmax weights within each voxel (same scheme as the
        # upstream fusion, just restricted to the kept points).
        conf_voxel_max, _ = scatter_max(conf_f, inverse_indices, dim=0)
        conf_exp = torch.exp(conf_f - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0)
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(-1)

        voxel_pts = scatter_add(pts_f * weights, inverse_indices, dim=0)
        voxel_feats = scatter_add(feat_f * weights, inverse_indices, dim=0)
        return voxel_pts, voxel_feats

    def voxelizaton_with_fusion(self, img_feat, pts3d, voxel_size, conf=None):
        # img_feat: B*V, C, H, W
        # pts3d: B*V, 3, H, W
        V, C, H, W = img_feat.shape
        pts3d_flatten = pts3d.permute(0, 2, 3, 1).flatten(0, 2)

        voxel_indices = (pts3d_flatten / voxel_size).round().int()  # [B*V*N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        )

        # Flatten confidence scores and features
        conf_flat = conf.flatten()  # [B*V*N]
        anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, ...]

        # Compute softmax weights per voxel
        conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)
        conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(
            conf_exp, inverse_indices, dim=0
        )  # [num_unique_voxels]
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(
            -1
        )  # [B*V*N, 1]

        # Compute weighted average of positions and features
        weighted_pts = pts3d_flatten * weights
        weighted_feats = anchor_feats_flat.squeeze(1) * weights

        # Aggregate per voxel
        voxel_pts = scatter_add(
            weighted_pts, inverse_indices, dim=0
        )  # [num_unique_voxels, 3]
        voxel_feats = scatter_add(
            weighted_feats, inverse_indices, dim=0
        )  # [num_unique_voxels, feat_dim]

        return voxel_pts, voxel_feats

    @torch.no_grad()
    def compute_attention_dynamic_mask(
        self,
        images: torch.Tensor,
        qk_dict: dict,
        enc_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute dynamic mask from VGGT4D attention patterns.

        Args:
            images: Input images [B, V, C, H, W]
            qk_dict: Q/K dictionary from VGGT4D aggregator
            enc_feat: Encoder features for clustering refinement

        Returns:
            dyn_mask: Binary dynamic mask [B, V, H, W]
            dyn_map: Continuous dynamic score map [B, V, h, w] (patch resolution)
        """
        b, v, c, h, w = images.shape

        # Reshape images for extraction: [B*V, C, H, W] -> [V, C, H, W] (assuming B=1)
        images_flat = images.view(b * v, c, h, w)

        # Organize Q/K dict for dynamic extraction
        organized_qk = organize_qk_dict(qk_dict, n_img=v)

        # Extract raw dynamic maps from attention patterns
        dyn_maps = extract_dyn_map(organized_qk, images_flat)  # [V, h//14, w//14]

        # Prepare encoder features for clustering (reshape from patch tokens)
        # enc_feat shape: [B*V, P, C] where P = (H/14) * (W/14)
        patch_h, patch_w = h // 14, w // 14
        enc_feat_reshaped = enc_feat.view(v, patch_h, patch_w, -1)  # [V, h, w, C]

        # Refine using KMeans clustering
        clustered_map, _ = cluster_attention_maps(
            enc_feat_reshaped,
            dyn_maps,
            n_clusters=self.cfg.dynamic_n_clusters
        )

        # Upsample the continuous score to full resolution FIRST, then threshold.
        # ORDER IS CRITICAL and must match the reference implementation
        # (VGGT4D/demo_vggt4d.py: interpolate -> adaptive_multiotsu_variance -> compare).
        #
        # BUG FIXED (July 2026): we used to compute the Otsu threshold on the 37x37
        # PATCH map and then apply it to the upsampled 518x518 map. Bilinear upsampling
        # SMOOTHS — it pulls values toward local means, so peaks in the full-res map are
        # lower than in the patch map. A threshold calibrated on the sharp patch map is
        # therefore systematically TOO HIGH for the smoothed map, so only the most
        # extreme patches survive. Result: the moving people (moderate attention) were
        # dropped while spurious static hot-spots were kept — the mask found ~5% of
        # pixels and did not cover the people at all. Since the mask drives the loss
        # downweighting, the psnr_dynamic/psnr_static split, the temporal loss's static
        # masking AND the compositing gate, this silently corrupted every experiment.
        dyn_score_full = F.interpolate(
            clustered_map.unsqueeze(1),  # [V, 1, patch_h, patch_w]
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)  # [V, H, W]

        # Determine threshold ON THE UPSAMPLED MAP (as the reference does).
        # NOTE: the old code also imposed a max_dynamic_fraction=0.30 cap that the
        # reference does NOT have. It is removed: with the threshold fixed, people
        # legitimately occupy ~30-40% of these frames, so that cap would now start
        # clipping *correct* detections.
        if self.cfg.dynamic_mask_threshold is not None:
            threshold = self.cfg.dynamic_mask_threshold
        else:
            threshold = adaptive_multiotsu_variance(dyn_score_full.cpu().numpy())

        dyn_mask_full = (dyn_score_full > threshold).float()
        print(f"[DynMask] threshold={threshold:.3f}, dynamic pixels={dyn_mask_full.mean()*100:.1f}%")

        # Reshape back to batch format
        dyn_mask = dyn_mask_full.view(b, v, h, w)
        dyn_map = clustered_map.view(b, v, patch_h, patch_w)

        return dyn_mask, dyn_map

    @torch.no_grad()
    def refine_dynamic_mask(
        self,
        images: torch.Tensor,
        depth_map: torch.Tensor,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        coarse_dyn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Refine coarse attention-based mask using 3D geometry (Stage 3).

        Args:
            images: [B, V, C, H, W]
            depth_map: [B, V, H, W]
            extrinsic: world2cam [B, V, 4, 4]
            intrinsic: [B, V, 3, 3]
            coarse_dyn_mask: float mask [B, V, H, W]

        Returns:
            Refined float mask [B, V, H, W]
        """
        if RefineDynMask is None:
            print("[DynMask] open3d not available — skipping Stage 3 refinement")
            return coarse_dyn_mask
        try:
            b, v, c, h, w = images.shape
            images_flat = images.view(b * v, c, h, w).float().cpu()
            depths_flat = depth_map.view(b * v, h, w).cpu()
            coarse_masks_flat = coarse_dyn_mask.view(b * v, h, w).bool().cpu()

            # extrinsic may be [B, V, 3, 4] — pad to [B*V, 4, 4] before inverting
            ext_flat = extrinsic.view(b * v, extrinsic.shape[-2], 4)
            if ext_flat.shape[1] == 3:
                bottom = torch.zeros(b * v, 1, 4, dtype=ext_flat.dtype, device=ext_flat.device)
                bottom[:, 0, 3] = 1.0
                ext_flat = torch.cat([ext_flat, bottom], dim=1)
            cam2world_flat = torch.inverse(ext_flat).cpu()

            intrinsics_flat = intrinsic.view(b * v, 3, 3).cpu()

            refiner = RefineDynMask(
                images_flat, depths_flat, coarse_masks_flat,
                cam2world_flat, intrinsics_flat, device=torch.device("cpu"),
            )
            refined = refiner.refine_masks()  # [B*V, H, W] bool, on cpu
            return refined.float().view(b, v, h, w)
        except Exception as e:
            print(f"[DynMask] Stage 3 failed ({e}) — falling back to coarse mask")
            return coarse_dyn_mask

    def apply_dynamic_mask_to_gaussians(
        self,
        opacity: torch.Tensor,
        pts_all: torch.Tensor,
        dyn_mask: torch.Tensor,
        conf_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply dynamic mask to Gaussian opacities.

        Args:
            opacity: Gaussian opacities [B, N]
            pts_all: 3D points [B, V, H, W, 3]
            dyn_mask: Dynamic mask [B, V, H, W]
            conf_valid_mask: Confidence validity mask [B, V, H, W]

        Returns:
            Modified opacity tensor with dynamic regions suppressed
        """
        b, v, h, w, _ = pts_all.shape

        # Flatten the dynamic mask to match the flattened point structure
        dyn_mask_flat = dyn_mask.view(b, v * h * w)  # [B, V*H*W]
        conf_flat = conf_valid_mask.view(b, v * h * w)  # [B, V*H*W]

        # Get the dynamic values for valid points only
        valid_dyn_mask = dyn_mask_flat[conf_flat.to(dyn_mask_flat.device)] 

        if self.cfg.suppress_dynamic_gaussians:
            suppression_factor = 0.1  # Reduce opacity to 10% for dynamic regions

            if not self.cfg.voxelize:
                dyn_weights = (1.0 - (1.0 - suppression_factor) * valid_dyn_mask.unsqueeze(0).float()).to(opacity.device)
                opacity = opacity * dyn_weights

        return opacity

    def forward(
        self,
        image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        dyn_mask_override: Optional[torch.Tensor] = None,
    ) -> Gaussians:
        # dyn_mask_override: PRECOMPUTED dynamic mask [B,V,H,W] or [V,H,W] (any resolution;
        # resampled to the Gaussian grid). When given, it REPLACES the internally-detected
        # mask for gaussian_dyn_flag (the per-frame compositing gate) and infos['dyn_mask'],
        # so the good full-span masks — not the weak in-forward detection — drive compositing.
        device = image.device
        b, v, _, h, w = image.shape
        distill_infos = {}
        if self.distill:
            distill_image = image.clone().detach()
            for module in [
                self.distill_aggregator,
                self.distill_camera_head,
                self.distill_depth_head,
            ]:
                for param in module.parameters():
                    param.data = param.data.to(device, non_blocking=True)

            with torch.no_grad():
                # Process with bfloat16 precision
                with torch.amp.autocast("cuda", enabled=True, dtype=_AMP_DTYPE,):
                    distill_aggregated_tokens_list, distill_patch_start_idx = (
                        self.distill_aggregator(
                            distill_image.to(_AMP_DTYPE),
                            intermediate_layer_idx=self.cfg.intermediate_layer_idx,
                        )
                    )

                # Process with default precision
                with torch.amp.autocast("cuda", enabled=False):
                    # Get camera pose information
                    distill_pred_pose_enc_list = self.distill_camera_head(
                        distill_aggregated_tokens_list
                    )
                    last_distill_pred_pose_enc = distill_pred_pose_enc_list[-1]
                    distill_extrinsic, distill_intrinsic = pose_encoding_to_extri_intri(
                        last_distill_pred_pose_enc, image.shape[-2:]
                    )

                    # Get depth information
                    distill_depth_map, distill_depth_conf = self.distill_depth_head(
                        distill_aggregated_tokens_list,
                        images=distill_image,
                        patch_start_idx=distill_patch_start_idx,
                    )

                    # Convert depth to 3D points
                    distill_pts_all = batchify_unproject_depth_map_to_point_map(
                        distill_depth_map, distill_extrinsic, distill_intrinsic
                    )
                # Store results
                distill_infos["pred_pose_enc_list"] = distill_pred_pose_enc_list
                distill_infos["pts_all"] = distill_pts_all
                distill_infos["depth_map"] = distill_depth_map

                conf_threshold = torch.quantile(
                    distill_depth_conf.flatten(2, 3), 0.3, dim=-1, keepdim=True
                )  # Get threshold for each view
                conf_mask = distill_depth_conf > conf_threshold.unsqueeze(-1)
                distill_infos["conf_mask"] = conf_mask

                for module in [
                    self.distill_aggregator,
                    self.distill_camera_head,
                    self.distill_depth_head,
                ]:
                    for param in module.parameters():
                        param.data = param.data.cpu()
                # Clean up to save memory
                del distill_aggregated_tokens_list, distill_patch_start_idx
                del distill_pred_pose_enc_list, last_distill_pred_pose_enc
                del distill_extrinsic, distill_intrinsic
                del distill_depth_map, distill_depth_conf
                torch.cuda.empty_cache()

        dyn_mask = None
        dyn_map = None

        if self.use_vggt4d and dyn_mask_override is not None:
            # FAST PATH: we already have the (good, precomputed) mask, so SKIP the entire
            # in-forward detection — Pass-1 attention AND Stage-3 refine are exactly what
            # the override replaces, and Stage 3's open3d/KMeans is the per-window
            # bottleneck (hours over a sequence). Feed the override straight into Pass-2
            # token suppression. Saves ~1 aggregator pass + the whole Stage-3 refine.
            ov = dyn_mask_override.to(image.device).float()
            if ov.dim() == 3:  # [V,H,W] -> [1,V,H,W]
                ov = ov.unsqueeze(0)
            if ov.shape[-2:] != (h, w):
                b_o, v_o = ov.shape[0], ov.shape[1]
                ov = F.interpolate(
                    ov.reshape(b_o * v_o, 1, ov.shape[-2], ov.shape[-1]),
                    size=(h, w), mode="nearest",
                ).reshape(b_o, v_o, h, w)
            dyn_mask = (ov > 0.5).float()
            self.dyn_mask = dyn_mask
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=_AMP_DTYPE,):
                    aggregated_tokens_list, patch_start_idx, _, _ = self.aggregator(
                        image.to(_AMP_DTYPE),
                        dyn_masks=dyn_mask.to(image.device),
                        capture_qk=False,  # no detection here -> skip ~96 GPU->CPU Q/K copies
                    )
            torch.cuda.empty_cache()
        elif self.use_vggt4d:
            # Pass 1: extract Q/K for dynamic mask computation
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=_AMP_DTYPE,):
                    _, _, qk_dict, enc_feat = self.aggregator(
                        image.to(_AMP_DTYPE),
                        dyn_masks=None,
                    )

            if self.cfg.enable_dynamic_detection:
                print("Computing dynamic mask from attention patterns...")
                dyn_mask, dyn_map = self.compute_attention_dynamic_mask(
                    image, qk_dict, enc_feat
                )
                self.dyn_mask = dyn_mask
                self.dyn_map = dyn_map

            del qk_dict, enc_feat
            torch.cuda.empty_cache()

            # Pass 2: backbone with token suppression in layers 0–4 using dynamic mask.
            # This is the core VGGT4D mechanism — dynamic tokens are masked out so the
            # backbone builds cleaner spatial features for static regions.
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=_AMP_DTYPE,):
                    aggregated_tokens_list, patch_start_idx, _, _ = self.aggregator(
                        image.to(_AMP_DTYPE),
                        dyn_masks=dyn_mask.to(image.device) if dyn_mask is not None else None,
                        capture_qk=False,  # Pass 2 output Q/K is discarded (detection used Pass 1)
                    )
            torch.cuda.empty_cache()
        else:
            # Original VGGT: single pass, no dynamic detection
            with torch.amp.autocast("cuda", enabled=True, dtype=_AMP_DTYPE,):
                aggregated_tokens_list, patch_start_idx = self.aggregator(
                    image.to(_AMP_DTYPE),
                    intermediate_layer_idx=self.cfg.intermediate_layer_idx,
                )

        with torch.amp.autocast("cuda", enabled=False):
            pred_pose_enc_list = self.camera_head(aggregated_tokens_list)
            last_pred_pose_enc = pred_pose_enc_list[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                last_pred_pose_enc, image.shape[-2:]
            )  # only for debug

            if self.cfg.pred_head_type == "point":
                pts_all, pts_conf = self.point_head(
                    aggregated_tokens_list,
                    images=image,
                    patch_start_idx=patch_start_idx,
                )
            elif self.cfg.pred_head_type == "depth":
                depth_map, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=image,
                    patch_start_idx=patch_start_idx,
                )
                pts_all = batchify_unproject_depth_map_to_point_map(
                    depth_map, extrinsic, intrinsic
                )
                # Stage 3: geometric refinement of coarse dynamic mask.
                # Skipped when dyn_mask_override is given (the precomputed mask already IS
                # the full 3-stage result — re-refining it is the wasted open3d/KMeans cost).
                if (dyn_mask is not None and self.cfg.enable_dynamic_detection
                        and dyn_mask_override is None):
                    print("Refining dynamic mask with Stage 3 (geometric)...")
                    dyn_mask = self.refine_dynamic_mask(
                        image, depth_map, extrinsic, intrinsic, dyn_mask
                    )
                    self.dyn_mask = dyn_mask
            else:
                raise ValueError(f"Invalid pred_head_type: {self.cfg.pred_head_type}")

            if self.cfg.render_conf:
                conf_valid = torch.quantile(
                    depth_conf.flatten(0, 1), self.cfg.conf_threshold
                )
                conf_valid_mask = depth_conf > conf_valid
            else:
                conf_valid_mask = torch.ones_like(depth_conf, dtype=torch.bool)

        # dpt style gs_head input format
        out = self.gaussian_param_head(
            aggregated_tokens_list,
            pts_all.flatten(0, 1).permute(0, 3, 1, 2),
            image,
            patch_start_idx=patch_start_idx,
            image_size=(h, w),
        )

        # --- piecewise-rigid motion of the dynamic content (tracker-driven) -------
        # Must run while the aggregated tokens still exist (the tracker consumes them).
        self._dyn_group_map = None
        self._dyn_group_motion = None
        if (dyn_mask is not None and self.cfg.enable_dynamic_detection
                and getattr(self, "track_head", None) is not None
                and getattr(self.cfg, "dyn_motion_groups", 0) > 0):
            _m = compute_dyn_group_motion(
                self.track_head, aggregated_tokens_list, image, patch_start_idx,
                pts_all, dyn_mask, conf_valid_mask,
                n_groups=self.cfg.dyn_motion_groups,
            )
            if _m is not None:
                self._dyn_group_motion = _m[:3]
                self._dyn_group_map = _m[3]
        # -------------------------------------------------------------------------
        del aggregated_tokens_list, patch_start_idx
        torch.cuda.empty_cache()

        # Override the internally-detected mask with the PRECOMPUTED one (good full-span
        # masks), resampled to the Gaussian-grid resolution (== conf_valid_mask HxW).
        # Everything downstream — gaussian_dyn_flag (compositing gate) and infos['dyn_mask']
        # — then uses the good mask instead of the weak in-forward detection.
        if dyn_mask_override is not None:
            H_g, W_g = conf_valid_mask.shape[-2], conf_valid_mask.shape[-1]
            ov = dyn_mask_override.to(conf_valid_mask.device).float()
            if ov.dim() == 3:  # [V,H,W] -> [1,V,H,W]
                ov = ov.unsqueeze(0)
            b_o, v_o = ov.shape[0], ov.shape[1]
            ov = F.interpolate(
                ov.reshape(b_o * v_o, 1, ov.shape[-2], ov.shape[-1]),
                size=(H_g, W_g), mode="nearest",
            ).reshape(b_o, v_o, H_g, W_g)
            dyn_mask = (ov > 0.5).float()
            self.dyn_mask = dyn_mask

        pts_flat = pts_all.flatten(2, 3)
        scene_scale = pts_flat.norm(dim=-1).mean().clip(min=1e-8)

        anchor_feats, conf = out[:, :, : self.raw_gs_dim], out[:, :, self.raw_gs_dim]

        neural_feats_list, neural_pts_list = [], []
        # Per-Gaussian (source frame, dynamic) labels, used for PER-FRAME DYNAMIC
        # COMPOSITING in the decoder (static Gaussians render into every view;
        # dynamic ones only into the view they were unprojected from, which removes
        # the multi-frame "ghosting" of moving objects).
        # Only well-defined WITHOUT voxelization: voxelizaton_with_fusion() merges
        # points from different frames into one anchor, destroying the 1:1
        # Gaussian -> (frame, pixel) correspondence this relies on.
        frame_idx_list, dyn_flag_list, group_idx_list = [], [], []
        track_frame_idx = not self.cfg.voxelize
        if self.cfg.voxelize:
            for b_i in range(b):
                neural_pts, neural_feats = self.voxelizaton_with_fusion(
                    anchor_feats[b_i],
                    pts_all[b_i].permute(0, 3, 1, 2).contiguous(),
                    self.voxel_size,
                    conf=conf[b_i],
                )
                neural_feats_list.append(neural_feats)
                neural_pts_list.append(neural_pts)
        else:
            # (v, h, w) row-major frame index. Selected with the SAME conf mask, in the
            # SAME order as the Gaussians below, so entry i here IS Gaussian i.
            fidx_vhw = (
                torch.arange(v, device=pts_all.device).view(v, 1, 1).expand(v, h, w)
            )
            for b_i in range(b):
                neural_feats_list.append(
                    anchor_feats[b_i].permute(0, 2, 3, 1)[conf_valid_mask[b_i]]
                )
                neural_pts_list.append(pts_all[b_i][conf_valid_mask[b_i]])
                frame_idx_list.append(fidx_vhw[conf_valid_mask[b_i]])
                if dyn_mask is not None:
                    dyn_flag_list.append(
                        dyn_mask[b_i].to(pts_all.device).float()[conf_valid_mask[b_i]]
                    )
                if self._dyn_group_map is not None:
                    group_idx_list.append(self._dyn_group_map[b_i][conf_valid_mask[b_i]])

        max_voxels = max(f.shape[0] for f in neural_feats_list)
        neural_feats = self.pad_tensor_list(
            neural_feats_list, (max_voxels,), value=-1e10
        )

        neural_pts = self.pad_tensor_list(
            neural_pts_list, (max_voxels,), -1e4
        )  # -1 == invalid voxel

        # Pad the labels identically. Padded slots are inert: neural_feats is padded
        # with -1e10 -> sigmoid -> opacity 0, so they render nothing regardless. The
        # -1 frame index simply never matches a view.
        gaussian_frame_idx = (
            self.pad_tensor_list(frame_idx_list, (max_voxels,), -1)
            if track_frame_idx
            else None
        )
        gaussian_group_idx = (
            self.pad_tensor_list(group_idx_list, (max_voxels,), -1)
            if (track_frame_idx and group_idx_list)
            else None
        )
        gaussian_dyn_flag = (
            self.pad_tensor_list(dyn_flag_list, (max_voxels,), 0.0)
            if (track_frame_idx and dyn_flag_list)
            else None
        )

        depths = neural_pts[..., -1].unsqueeze(-1)
        densities = neural_feats[..., 0].sigmoid()

        assert len(densities.shape) == 2, "the shape of densities should be (B, N)"
        assert neural_pts.shape[1] > 1, "the number of voxels should be greater than 1"

        opacity = self.map_pdf_to_opacity(densities, global_step).squeeze(-1)
        if self.cfg.opacity_conf:
            shift = torch.quantile(depth_conf, self.cfg.conf_threshold)
            opacity = opacity * torch.sigmoid(depth_conf - shift)[
                conf_valid_mask
            ].unsqueeze(
                0
            )  # little bit hacky

        # Apply dynamic mask to suppress dynamic regions (no-op unless suppress_dynamic_gaussians=True).
        if dyn_mask is not None and self.cfg.enable_dynamic_detection:
            if self.cfg.suppress_dynamic_gaussians:
                print("Applying dynamic mask to Gaussian opacities...")
            opacity = self.apply_dynamic_mask_to_gaussians(
                opacity, pts_all, dyn_mask, conf_valid_mask
            )

        # GS Prune, but only works when bs = 1
        # if want to support bs > 1, need to random prune gaussians based on the rank of opacity like LongLRM
        # Note: we not prune gaussians here, but we will try it in the future
        if self.cfg.gs_prune and b == 1:
            opacity_threshold = self.cfg.opacity_threshold
            gaussian_usage = opacity > opacity_threshold  # (B, N)

            print(
                f"based on opacity threshold {opacity_threshold}, pruned {gaussian_usage.shape[1] - neural_pts.shape[1]} gaussians out of {gaussian_usage.shape[1]}"
            )

            if (gaussian_usage.sum() / gaussian_usage.numel()) > self.cfg.gs_keep_ratio:
                # rank by opacity
                num_keep = int(gaussian_usage.shape[1] * self.cfg.gs_keep_ratio)
                idx_sort = opacity.argsort(dim=1, descending=True)
                keep_idx = idx_sort[:, :num_keep]
                gaussian_usage = torch.zeros_like(gaussian_usage, dtype=torch.bool)
                gaussian_usage.scatter_(1, keep_idx, True)

            neural_pts = neural_pts[gaussian_usage].view(b, -1, 3).contiguous()
            depths = depths[gaussian_usage].view(b, -1, 1).contiguous()
            neural_feats = (
                neural_feats[gaussian_usage].view(b, -1, self.raw_gs_dim).contiguous()
            )
            opacity = opacity[gaussian_usage].view(b, -1).contiguous()
            # Keep the per-frame compositing labels aligned with the pruned Gaussians.
            if gaussian_frame_idx is not None:
                gaussian_frame_idx = (
                    gaussian_frame_idx[gaussian_usage].view(b, -1).contiguous()
                )
            if gaussian_dyn_flag is not None:
                gaussian_dyn_flag = (
                    gaussian_dyn_flag[gaussian_usage].view(b, -1).contiguous()
                )
            if gaussian_group_idx is not None:
                gaussian_group_idx = (
                    gaussian_group_idx[gaussian_usage].view(b, -1).contiguous()
                )

            print(
                f"finally pruned {gaussian_usage.shape[1] - neural_pts.shape[1]} gaussians out of {gaussian_usage.shape[1]}"
            )

        gaussians = self.gaussian_adapter.forward(
            neural_pts,
            depths,
            opacity,
            neural_feats[..., 1:].squeeze(2),
        )

        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                pts_all[..., -1].flatten(2, 3).unsqueeze(-1).unsqueeze(-1),
                "b v (h w) srf s -> b v h w srf s",
                h=h,
                w=w,
            )

        infos = {}
        infos["scene_scale"] = scene_scale
        infos["voxelize_ratio"] = densities.shape[1] / (h * w * v)

        # Add dynamic detection info if available
        if dyn_mask is not None:
            infos["dyn_mask"] = dyn_mask
            infos["dyn_map"] = dyn_map
            dyn_ratio = dyn_mask.float().mean().item()
            print(f"Dynamic detection: {dyn_ratio*100:.1f}% of pixels detected as dynamic")

        # Per-Gaussian labels for per-frame dynamic compositing (see construction above).
        # gaussian_frame_idx[b, n] = the view n was unprojected from; gaussian_dyn_flag[b, n]
        # = 1 if n sits on a moving object. The decoder uses these to render dynamic
        # Gaussians ONLY into their own frame. Both are None when voxelize=True (the
        # Gaussian->frame mapping does not survive voxel fusion).
        infos["gaussian_frame_idx"] = gaussian_frame_idx
        infos["gaussian_dyn_flag"] = gaussian_dyn_flag
        if self._dyn_group_motion is not None:
            gc, gp, gv = self._dyn_group_motion
            infos["dyn_group_centroid"] = gc      # [B,V,K,3]
            infos["dyn_group_pred"] = gp          # [B,V,K,3] leave-one-out prediction
            infos["dyn_group_valid"] = gv         # [B,V,K]
            infos["gaussian_group_idx"] = gaussian_group_idx

        # --- First-order MOTION MODEL for the dynamic content ------------------
        # Gaussian positions come from the FROZEN depth/pose heads, so a moving object
        # is reconstructed only where it was in its OWN source frame. To render it at
        # another timestamp it has to be MOVED. We estimate its motion as the 3D
        # centroid of the dynamic points per frame, then (leak-free) predict the
        # centroid at each target view j by a linear fit over the OTHER frames only —
        # frame j's own mask/depth is never used, so this stays valid under
        # leave-one-out. The decoder displaces dynamic Gaussians by
        # pred_centroid[j] - centroid[i].
        # Rigid (translation-only) and one motion for all dynamic content: a
        # deliberate first-order model, not a deformation field.
        if dyn_mask is not None and self.cfg.enable_dynamic_detection:
            dm = (dyn_mask > 0.5) & conf_valid_mask                     # [B,V,H,W]
            n_dyn = dm.flatten(2).sum(-1)                              # [B,V]
            ctr = (pts_all * dm.unsqueeze(-1)).flatten(2, 3).sum(2)    # [B,V,3]
            ctr = ctr / n_dyn.clamp_min(1).unsqueeze(-1).float()
            valid = n_dyn >= 32                                        # too few pts -> unusable
            infos["dyn_centroid"] = ctr
            infos["dyn_centroid_valid"] = valid
            infos["dyn_centroid_pred"] = predict_centroid_leave_one_out(ctr, valid)
        # ----------------------------------------------------------------------

        # Store per-frame Gaussian parameters for temporal consistency loss (Fix 3)
        # out shape: [B, V, raw_gs_dim+1, H, W] where raw_gs_dim = 1 + 7 + 3*d_sh
        # Channel layout: [0]=opacity, [1:4]=scales, [4:8]=rotations, [8:]=SH
        if self.cfg.use_temporal_attention or self.training:
            # Parse per-frame Gaussian parameters from the head output
            # anchor_feats has shape [B, V, raw_gs_dim, H, W]
            d_sh = self.gaussian_adapter.d_sh  # (sh_degree + 1)^2
            per_frame_opacity = anchor_feats[:, :, 0]  # [B, V, H, W]
            per_frame_scales = anchor_feats[:, :, 1:4].permute(0, 1, 3, 4, 2)  # [B, V, H, W, 3]
            per_frame_rotations = anchor_feats[:, :, 4:8].permute(0, 1, 3, 4, 2)  # [B, V, H, W, 4]
            per_frame_sh = anchor_feats[:, :, 8:].permute(0, 1, 3, 4, 2)  # [B, V, H, W, 3*d_sh]

            infos["per_frame_gaussians"] = {
                "opacity": per_frame_opacity.detach() if not self.training else per_frame_opacity,
                "scales": per_frame_scales.detach() if not self.training else per_frame_scales,
                "rotations": per_frame_rotations.detach() if not self.training else per_frame_rotations,
                "sh": per_frame_sh.detach() if not self.training else per_frame_sh,
            }

        print(
            f"scene scale: {scene_scale:.3f}, pixel-wise num: {h*w*v}, after voxelize: {neural_pts.shape[1]}, voxelize ratio: {infos['voxelize_ratio']:.3f}"
        )
        print(
            f"Gaussians attributes: \n"
            f"opacities: mean: {gaussians.opacities.mean()}, min: {gaussians.opacities.min()}, max: {gaussians.opacities.max()} \n"
            f"scales: mean: {gaussians.scales.mean()}, min: {gaussians.scales.min()}, max: {gaussians.scales.max()}"
        )

        print("B:", b, "V:", v, "H:", h, "W:", w)
        extrinsic_padding = (
            torch.tensor([0, 0, 0, 1], device=device, dtype=extrinsic.dtype)
            .view(1, 1, 1, 4)
            .repeat(b, v, 1, 1)
        )
        intrinsic = intrinsic.clone()  # Create a new tensor
        intrinsic = torch.stack(
            [intrinsic[:, :, 0] / w, intrinsic[:, :, 1] / h, intrinsic[:, :, 2]], dim=2
        )

        return EncoderOutput(
            gaussians=gaussians,
            pred_pose_enc_list=pred_pose_enc_list,
            pred_context_pose=dict(
                extrinsic=torch.cat([extrinsic, extrinsic_padding], dim=2).inverse(),
                intrinsic=intrinsic,
            ),
            depth_dict=dict(depth=depth_map, conf_valid_mask=conf_valid_mask),
            infos=infos,
            distill_infos=distill_infos,
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
