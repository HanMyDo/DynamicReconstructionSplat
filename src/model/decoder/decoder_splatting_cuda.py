from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import torchvision

from ..types import Gaussians
# from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput
from math import sqrt 
from gsplat import rasterization

from ...misc.utils import vis_depth_map

DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]

@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]
    
    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def rendering_fn(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
        gaussian_frame_idx: Tensor | None = None,
        gaussian_dyn_flag: Tensor | None = None,
        gaussian_only_view: Tensor | None = None,
        leave_one_out: bool = False,
        dyn_centroid: Tensor | None = None,
        dyn_centroid_pred: Tensor | None = None,
        dyn_centroid_valid: Tensor | None = None,
        dyn_group_centroid: Tensor | None = None,
        dyn_group_pred: Tensor | None = None,
        dyn_group_valid: Tensor | None = None,
        gaussian_group_idx: Tensor | None = None,
        gaussian_disp: Tensor | None = None,
        gaussian_disp_valid: Tensor | None = None,
        per_frame_compositing: bool = False,
    ) -> DecoderOutput:
        B, V, _, _  = intrinsics.shape
        H, W = image_shape
        rendered_imgs, rendered_depths, rendered_alphas = [], [], []
        xyzs, opacitys, rotations, scales, features = gaussians.means, gaussians.opacities, gaussians.rotations, gaussians.scales, gaussians.harmonics.permute(0, 1, 3, 2).contiguous()
        covariances = gaussians.covariances
        for i in range(B):
            xyz_i = xyzs[i].float()
            feature_i = features[i].float()
            covar_i = covariances[i].float()
            scale_i = scales[i].float()
            rotation_i = rotations[i].float()
            opacity_i = opacitys[i].squeeze().float()
            test_w2c_i = extrinsics[i].float().inverse() # (V, 4, 4)
            test_intr_i_normalized = intrinsics[i].float()
            # Denormalize the intrinsics into standred format
            test_intr_i = test_intr_i_normalized.clone()
            test_intr_i[:, 0] = test_intr_i_normalized[:, 0] * W
            test_intr_i[:, 1] = test_intr_i_normalized[:, 1] * H
            sh_degree = (int(sqrt(feature_i.shape[-2])) - 1)

            rendering_list = []
            rendering_depth_list = []
            rendering_alpha_list = []
            for j in range(V):
                # --- Per-frame dynamic compositing ------------------------------
                # Default (labels None) = original behaviour: every Gaussian renders
                # into every view, so a moving object appears at all V of its past
                # positions ("ghosting").
                # When enabled: a Gaussian on a moving object is rendered ONLY into the
                # view it was unprojected from. Static Gaussians still render into all
                # views (so the background keeps its multi-view fusion).
                #   gate = 1                       for static Gaussians
                #   gate = 1 if frame_idx == j     for dynamic Gaussians
                #   gate = 0                       for dynamic Gaussians of other frames
                # leave_one_out = drop view j's OWN Gaussians when rendering view j
                # (see (2) below) — the honest control against self-reprojection.
                opacity_ij = opacity_i
                if gaussian_frame_idx is not None:
                    fidx_i = gaussian_frame_idx[i].to(opacity_i.device)
                    own_frame = (fidx_i == j).float()          # 1 if Gaussian came from view j
                    gate = torch.ones_like(opacity_i)

                    # (1) Per-frame dynamic compositing (needs the dynamic flags):
                    #     dynamic Gaussians survive ONLY in their own frame.
                    if gaussian_dyn_flag is not None and per_frame_compositing:
                        dyn_i = gaussian_dyn_flag[i].to(opacity_i.device).float()
                        gate = gate * (1.0 - dyn_i * (1.0 - own_frame))

                    # (2) Leave-one-out: drop view j's OWN Gaussians entirely (static
                    #     AND dynamic), so view j must be reconstructed from the OTHER
                    #     frames. This is the honest control: without it, view j's
                    #     dynamic content is rendered from Gaussians unprojected FROM
                    #     view j (project->unproject->project), which is close to
                    #     self-reprojection and inflates dynamic PSNR for a trivial
                    #     reason. Under LOO, reconstructing a moving object requires
                    #     actually MODELLING its motion — which this architecture
                    #     cannot do — so a large LOO gap is the expected, reportable
                    #     result, not a bug.
                    # (2b) HYBRID pre-fused static sets. A fused voxel has no single
                    #      source frame, so LOO cannot drop view j's contribution by
                    #      frame index — the exclusion has to happen INSIDE the fusion
                    #      instead (voxelize_static_hybrid(exclude_frame=j)). The
                    #      encoder therefore emits one static set PER TARGET VIEW,
                    #      labelled with only_view = j. Such a Gaussian:
                    #        - renders ONLY into view j (its set was built for view j),
                    #        - is EXEMPT from the LOO drop below, because view j was
                    #          already excluded when the set was fused. Applying the
                    #          drop again would delete the entire static background.
                    #      only_view < 0 means "normal Gaussian", i.e. unchanged
                    #      behaviour for every existing run.
                    if gaussian_only_view is not None:
                        ov_i = gaussian_only_view[i].to(opacity_i.device)
                        is_pref = (ov_i >= 0)
                        gate = gate * torch.where(
                            is_pref, (ov_i == j).float(), torch.ones_like(opacity_i)
                        )
                        if leave_one_out:
                            # drop own-frame Gaussians ONLY for non-prefused ones
                            gate = gate * torch.where(
                                is_pref, torch.ones_like(opacity_i), 1.0 - own_frame
                            )
                    elif leave_one_out:
                        gate = gate * (1.0 - own_frame)

                    opacity_ij = opacity_i * gate
                # ----------------------------------------------------------------

                # --- (3) MOTION DISPLACEMENT of dynamic Gaussians ---------------
                # Positions come from the frozen depth/pose heads, so a dynamic
                # Gaussian sits where its object was in ITS OWN source frame i. To
                # render target view j we translate it by the object's estimated
                # motion between t_i and t_j:
                #     disp = pred_centroid[j] - centroid[i]
                # pred_centroid[j] is fitted from the OTHER frames only (see
                # predict_centroid_leave_one_out), so this never reads frame j and
                # stays valid under leave-one-out. Static Gaussians are untouched.
                xyz_ij = xyz_i
                if (gaussian_disp is not None and gaussian_frame_idx is not None
                        and gaussian_dyn_flag is not None):
                    # SCENE FLOW (takes precedence): per-Gaussian displacement toward
                    # target frame j, interpolated from the tracks' OBSERVED positions
                    # at j (direct correspondence — see dyn_motion.py "UPGRADE").
                    # gaussian_disp[i][:, j] is already zero where invalid/own-frame;
                    # the gates below only make that explicit.
                    fidx = gaussian_frame_idx[i].to(xyz_i.device).long().clamp_min(0)
                    dynf = gaussian_dyn_flag[i].to(xyz_i.device).float()
                    move = dynf * (1.0 - (fidx == j).float())
                    if gaussian_disp_valid is not None:
                        move = move * gaussian_disp_valid[i].to(xyz_i.device)[:, j].float()
                    xyz_ij = xyz_i + move.unsqueeze(-1) * gaussian_disp[i].to(xyz_i.device)[:, j].float()
                elif (dyn_group_centroid is not None and dyn_group_pred is not None
                        and gaussian_group_idx is not None and gaussian_frame_idx is not None
                        and gaussian_dyn_flag is not None):
                    # PIECEWISE-RIGID: each Gaussian follows the group (moving object)
                    # it belongs to, so a person and a box get different velocities.
                    fidx = gaussian_frame_idx[i].to(xyz_i.device).long().clamp_min(0)
                    gidx = gaussian_group_idx[i].to(xyz_i.device).long()
                    dynf = gaussian_dyn_flag[i].to(xyz_i.device).float()
                    has_g = (gidx >= 0).float()          # -1 = static / unassigned
                    gidx = gidx.clamp_min(0)
                    move = dynf * has_g * (1.0 - (fidx == j).float())
                    if dyn_group_valid is not None:
                        gv = dyn_group_valid[i].to(xyz_i.device).float()      # [V,K]
                        move = move * gv[fidx, gidx] * gv[j, gidx]
                    src_c = dyn_group_centroid[i].to(xyz_i.device)[fidx, gidx]   # [N,3]
                    tgt_c = dyn_group_pred[i].to(xyz_i.device)[j, gidx]          # [N,3]
                    xyz_ij = xyz_i + move.unsqueeze(-1) * (tgt_c - src_c)
                elif (dyn_centroid is not None and dyn_centroid_pred is not None
                        and gaussian_frame_idx is not None and gaussian_dyn_flag is not None):
                    fidx = gaussian_frame_idx[i].to(xyz_i.device).long().clamp_min(0)  # -1 padding -> 0
                    dynf = gaussian_dyn_flag[i].to(xyz_i.device).float()                # [N]
                    # A Gaussian rendered into its OWN source frame is already at the
                    # correct place for that timestamp — displacing it would MOVE the
                    # object off its own observation. Zero the displacement there.
                    # (Under leave_one_out these are gated out anyway, but this keeps
                    # --track_dynamic correct when used WITHOUT LOO.)
                    move = dynf * (1.0 - (fidx == j).float())                          # [N]
                    # A frame with too few dynamic points has a MEANINGLESS centroid
                    # (the mean of ~nothing), so displacing its Gaussians by
                    # pred[j] - garbage flings them across the scene and corrupts even
                    # static image regions. Only move Gaussians whose SOURCE frame had
                    # a usable centroid, and only toward a usable prediction.
                    if dyn_centroid_valid is not None:
                        okv = dyn_centroid_valid[i].to(xyz_i.device).float()           # [V]
                        move = move * okv[fidx]
                        if okv.sum() < 2:      # no motion estimate at all -> no displacement
                            move = move * 0.0
                    move = move.unsqueeze(-1)                                          # [N,1]
                    src_c = dyn_centroid[i].to(xyz_i.device)[fidx]                      # [N,3]
                    tgt_c = dyn_centroid_pred[i].to(xyz_i.device)[j].unsqueeze(0)       # [1,3]
                    xyz_ij = xyz_i + move * (tgt_c - src_c)
                # ----------------------------------------------------------------
                rendering, alpha, _ = rasterization(xyz_ij, rotation_i, scale_i, opacity_ij, feature_i,
                                                test_w2c_i[j:j+1], test_intr_i[j:j+1], W, H, sh_degree=sh_degree, 
                                                # near_plane=near[i].mean(), far_plane=far[i].mean(),
                                                render_mode="RGB+D", packed=False,
                                                near_plane=1e-10,
                                                backgrounds=self.background_color.unsqueeze(0).repeat(1, 1),
                                                radius_clip=0.1,
                                                covars=covar_i,
                                                rasterize_mode='classic') # (V, H, W, 3) 
                rendering_img, rendering_depth = torch.split(rendering, [3, 1], dim=-1)
                rendering_img = rendering_img.clamp(0.0, 1.0)
                rendering_list.append(rendering_img.permute(0, 3, 1, 2))
                rendering_depth_list.append(rendering_depth)
                rendering_alpha_list.append(alpha)
            rendered_depths.append(torch.cat(rendering_depth_list, dim=0).squeeze())
            rendered_imgs.append(torch.cat(rendering_list, dim=0))
            rendered_alphas.append(torch.cat(rendering_alpha_list, dim=0).squeeze())
        return DecoderOutput(torch.stack(rendered_imgs), torch.stack(rendered_depths), torch.stack(rendered_alphas), lod_rendering=None)

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
        gaussian_frame_idx: Tensor | None = None,
        gaussian_dyn_flag: Tensor | None = None,
        gaussian_only_view: Tensor | None = None,
        leave_one_out: bool = False,
        dyn_centroid: Tensor | None = None,
        dyn_centroid_pred: Tensor | None = None,
        dyn_centroid_valid: Tensor | None = None,
        dyn_group_centroid: Tensor | None = None,
        dyn_group_pred: Tensor | None = None,
        dyn_group_valid: Tensor | None = None,
        gaussian_group_idx: Tensor | None = None,
        gaussian_disp: Tensor | None = None,
        gaussian_disp_valid: Tensor | None = None,
        per_frame_compositing: bool = False,
    ) -> DecoderOutput:

        return self.rendering_fn(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode, cam_rot_delta, cam_trans_delta,
                                 gaussian_frame_idx=gaussian_frame_idx, gaussian_dyn_flag=gaussian_dyn_flag, gaussian_only_view=gaussian_only_view, leave_one_out=leave_one_out,
                                 dyn_centroid=dyn_centroid, dyn_centroid_pred=dyn_centroid_pred,
                                 dyn_centroid_valid=dyn_centroid_valid,
                                 dyn_group_centroid=dyn_group_centroid, dyn_group_pred=dyn_group_pred,
                                 dyn_group_valid=dyn_group_valid, gaussian_group_idx=gaussian_group_idx,
                                 gaussian_disp=gaussian_disp, gaussian_disp_valid=gaussian_disp_valid,
                                 per_frame_compositing=per_frame_compositing)

