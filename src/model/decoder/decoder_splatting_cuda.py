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
from src.model.encoder.dyn_motion import near_camera_reject, compensate_dyn_opacity
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
        dyn_opacity_comp: float = 0.0,
        dyn_unsupported: str = "drop",
        gaussian_track_dist: Tensor | None = None,
        dyn_far_static: float = 0.0,
        dyn_nearest_source: int = 0,
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
            # Which branch each off-frame dynamic Gaussian took, summed over targets:
            # [flow, rigid, static, dropped]. Accumulated as a tensor and read ONCE
            # after the loop -- a .item() per target would sync the GPU V times per
            # batch. Printed for the first batch only. Attempt 1 at this fallback
            # measured as noise and it took a whole GPU run to work out that the
            # branch had barely fired, which this makes visible immediately.
            _acc = torch.zeros(4, device=xyz_i.device) if gaussian_disp is not None else None
            # [survived, total] for OFF-FRAME DYNAMIC gaussians, so the effect of
            # --dyn_nearest_source is visible in the log rather than assumed. Three
            # separate changes in this file's history looked wired and silently did
            # nothing; a read-out is cheaper than finding that out from a GPU run.
            _keep_acc = torch.zeros(2, device=xyz_i.device)
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
                # Set here, not inside the block below, so section (3) can test it
                # unconditionally.
                fb_disp, fb_ok, far_ok = None, None, None
                if gaussian_frame_idx is not None:
                    fidx_i = gaussian_frame_idx[i].to(opacity_i.device)
                    own_frame = (fidx_i == j).float()          # 1 if Gaussian came from view j
                    gate = torch.ones_like(opacity_i)

                    # DISTANCE-GATED RESCUE (--dyn_far_static M). The radius gate
                    # rejects two populations that want OPPOSITE treatment:
                    #   - a few spacings away: a real mover whose correspondence
                    #     failed. KEEPING it renders the object at a stale position
                    #     = the ghosts/haze over the scene.
                    #   - METRES away (never near any track): almost certainly a
                    #     mask FALSE POSITIVE, i.e. static content. DROPPING it
                    #     deletes background in every view, and with --bg_color
                    #     1 1 1 the result is the white speckles in the renders.
                    # 'drop' treats both as the first case and 'static' treats both
                    # as the second; measured, each fixes one artefact and causes
                    # the other. This keeps only the FAR ones, which should give
                    # solid movers AND no speckles.
                    if dyn_far_static > 0 and gaussian_track_dist is not None:
                        far_ok = (gaussian_track_dist[i].to(opacity_i.device)
                                  > dyn_far_static).to(opacity_i.dtype)

                    # FALLBACK DISPLACEMENT (--dyn_unsupported rigid). Flow-gated
                    # compositing DELETES every off-frame dynamic Gaussian that no
                    # track supports. That is right when the own-frame copy survives
                    # to cover the object, and catastrophic under leave-one-out, where
                    # it does not: ~46% of the object is deleted and renders as
                    # background (WHITE, with --bg_color 1 1 1). Measured on
                    # synchronous2 LOO: dynamic 19.69 -> 11.39 dB, and worse than the
                    # baseline on PSNR, LPIPS *and* SSIM -- a complete-but-wrong frame
                    # beats a partial one on every metric.
                    # So DON'T delete: fall back to the piecewise-rigid group motion,
                    # which is a LEAVE-ONE-OUT prediction (dyn_group_pred, fitted from
                    # the other frames) and therefore stays valid under LOO. Flow where
                    # tracks support it, rigid where they don't, delete only where
                    # neither has an estimate.
                    if (dyn_unsupported == "rigid" and gaussian_disp is not None
                            and dyn_group_pred is not None
                            and dyn_group_centroid is not None
                            and gaussian_group_idx is not None):
                        # .long(): these index into the [V,K,...] group tensors, and
                        # gaussian_frame_idx carries -1 padding, so clamp first.
                        fidx_l = fidx_i.to(xyz_i.device).long().clamp_min(0)
                        gidx_f = gaussian_group_idx[i].to(xyz_i.device).long()
                        has_g = (gidx_f >= 0)              # -1 = static / unassigned
                        gidx_c = gidx_f.clamp_min(0)
                        src_f = dyn_group_centroid[i].to(xyz_i.device)[fidx_l, gidx_c]
                        tgt_f = dyn_group_pred[i].to(xyz_i.device)[j, gidx_c]
                        ok = has_g
                        if dyn_group_valid is not None:
                            # Both ends must have a usable centroid, exactly as the
                            # piecewise-rigid branch below requires.
                            gvf = dyn_group_valid[i].to(xyz_i.device).bool()
                            ok = ok & gvf[fidx_l, gidx_c] & gvf[j, gidx_c]
                        fb_ok = ok.to(xyz_i.dtype)
                        fb_disp = (tgt_f - src_f) * fb_ok.unsqueeze(-1)
                    # The opacity the surviving Gaussians render WITH. Only (1b)
                    # changes it; everywhere else it stays opacity_i, so every
                    # existing recipe renders bit-identically.
                    opac_eff = opacity_i

                    # (1) Per-frame dynamic compositing (needs the dynamic flags):
                    #     dynamic Gaussians survive ONLY in their own frame.
                    if gaussian_dyn_flag is not None and per_frame_compositing:
                        dyn_i = gaussian_dyn_flag[i].to(opacity_i.device).float()
                        # FLOW-GATED COMPOSITING. Plain pfd deletes EVERY off-frame
                        # dynamic Gaussian, which removes all ghosting but leaves the
                        # object with one frame's worth of density. Scene flow moves
                        # every one, but only ~54% have tracks near enough to be
                        # relocated correctly -- the rest keep their stale position and
                        # ghost exactly as before. Neither is right on its own, and run
                        # together they cancel (pfd zeroes what flow moves).
                        # The displacement layer already knows, per Gaussian and per
                        # target, whether tracks support the motion. Use it: KEEP a
                        # Gaussian that can be relocated (it will contribute at the
                        # right place), DROP one that cannot (it could only ghost).
                        # Coverage stops being a defect and becomes the relocate/drop
                        # split. With no flow, disp_valid is None and this is plain pfd.
                        keep = own_frame
                        if gaussian_disp_valid is not None:
                            dv = gaussian_disp_valid[i].to(opacity_i.device)[:, j].float()
                            keep = (own_frame + dv).clamp(max=1.0)
                        if fb_disp is not None:
                            # A Gaussian the fallback can move is no longer a "could
                            # only ghost" case, so it survives the gate too.
                            keep = (keep + fb_ok).clamp(max=1.0)
                        if far_ok is not None and dyn_unsupported != "static":
                            keep = (keep + far_ok).clamp(max=1.0)
                        if dyn_unsupported == "static":
                            # NOTHING is deleted for want of a motion estimate. A
                            # Gaussian with no estimate renders where it already is,
                            # exactly like a static one. Most of them ARE static:
                            # ~45% of "dynamic" Gaussians sit metres from any track,
                            # i.e. they are mask false positives, and no track-derived
                            # model (flow or rigid) can ever place them. Under
                            # leave-one-out, deleting them punches holes in the
                            # BACKGROUND, and a complete-but-wrong frame beats a
                            # partial one on PSNR, LPIPS and SSIM alike.
                            # A genuine mover caught here ghosts from one extra
                            # position, which is the price.
                            keep = torch.ones_like(keep)

                        # APPLIED LAST, deliberately: this RESTRICTS whatever the
                        # policy above decided to keep. Placed before the 'static'
                        # branch it was silently a no-op, because that branch sets
                        # keep = ones and overwrote it.
                        if dyn_nearest_source > 0:
                            # TEMPORAL SOURCE RESTRICTION. Every source frame carries its
                            # OWN monocular depth map, and those disagree: voxel fusion of
                            # the same surface across frames merges almost nothing once the
                            # camera has moved (measured ratio 0.694 at stride 8). So the
                            # V-1 copies of a mover do not stack into a surface -- they
                            # scatter in DEPTH into a cloud, which does not occlude the
                            # background behind it. That is the translucent, patchy person,
                            # and the stale ones among them are the ghosts.
                            # Averaging scattered copies cannot fix it; picking ONE can,
                            # because a single frame's reconstruction is internally
                            # consistent (one depth map). This is the render-time analogue
                            # of --ply_own_frame_only, which is what made the PLY exports
                            # sharp for exactly this reason.
                            # Ranked by |i - j| with j itself excluded (LOO drops it anyway),
                            # so window edges pick the nearest available frames rather than
                            # a fixed offset. Cost: fewer contributors, so the mover is
                            # sharper but thinner -- --dyn_opacity_comp carries that.
                            _ord = sorted((x for x in range(V) if x != j),
                                          key=lambda x: abs(x - j))[:dyn_nearest_source]
                            _allow = torch.zeros(V, dtype=torch.bool, device=opacity_i.device)
                            _allow[torch.tensor(_ord, device=opacity_i.device)] = True
                            near_ok = _allow[fidx_i.long().clamp_min(0)].to(opacity_i.dtype)
                            keep = keep * (own_frame + near_ok).clamp(max=1.0)
                        _offdyn = dyn_i * (1.0 - own_frame)
                        _keep_acc[0] += (_offdyn * keep).sum()
                        _keep_acc[1] += _offdyn.sum()
                        gate = gate * (1.0 - dyn_i * (1.0 - keep))

                        # (1b) OPACITY COMPENSATION for the contributors the gate
                        # removed. The pretrained head never chose these opacities in
                        # isolation: AnySplat renders every Gaussian into every view,
                        # so a surface is composited from ~V of them and each one only
                        # has to carry 1/V of the alpha. The gate above breaks that
                        # contract for dynamic Gaussians -- own-frame always survives,
                        # a relocated one survives and lands in the SAME place (so it
                        # still stacks), but a radius-rejected one is dropped outright.
                        # The survivors are then asked to cover a surface with a
                        # fraction of the alpha budget it was calibrated for, which is
                        # exactly the "under-covered / semi-transparent person".
                        #
                        # Match the ALPHA, not the opacity. V contributors at opacity o
                        # give 1-(1-o)^V; n survivors reproduce that at
                        #     o' = 1 - (1-o)^(V/n).
                        # n is estimated per target view from the survivor fraction
                        # among dynamic Gaussians -- a scalar, because which ones land
                        # on a given surface point is not knowable per Gaussian here.
                        # `dyn_opacity_comp` scales the exponent between 1 (off, the
                        # measured behaviour) and the full correction, so it can be
                        # swept rather than trusted.
                        opac_eff = compensate_dyn_opacity(
                            opac_eff, dyn_i, keep, V, dyn_opacity_comp)

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

                    opacity_ij = opac_eff * gate
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
                        # BINARISE for the move. Under --dyn_conf_opacity this tensor
                        # carries a confidence in [0,1] rather than a flag, and a
                        # fractional move would put the Gaussian PART of the way to
                        # where tracking says the object went -- a position nothing
                        # supports. Move it fully or not at all; the confidence acts
                        # on opacity (see the gate above), not on geometry.
                        flow_ok = (gaussian_disp_valid[i].to(xyz_i.device)[:, j] > 0).float()
                    else:
                        flow_ok = torch.ones_like(move)
                    disp_ij = gaussian_disp[i].to(xyz_i.device)[:, j].float()
                    if fb_disp is not None:
                        # Flow takes precedence where it is supported; the fallback
                        # fills in the rest. Binary choice, never a blend: a Gaussian
                        # part-way between two estimates sits where nothing supports it.
                        use_fb = (1.0 - flow_ok) * fb_ok
                        disp_ij = disp_ij * flow_ok.unsqueeze(-1) + fb_disp * use_fb.unsqueeze(-1)
                        move = move * (flow_ok + use_fb).clamp(max=1.0)
                    else:
                        move = move * flow_ok
                    if _acc is not None:
                        _off = dynf * (1.0 - (fidx == j).float())
                        _fbk = fb_ok if fb_disp is not None else torch.zeros_like(flow_ok)
                        _rest = _off * (1.0 - flow_ok) * (1.0 - _fbk)
                        _acc[0] += (_off * flow_ok).sum()
                        _acc[1] += (_off * (1.0 - flow_ok) * _fbk).sum()
                        if dyn_unsupported == "static":
                            _kept = _rest
                        elif far_ok is not None:
                            _kept = _rest * far_ok
                        else:
                            _kept = torch.zeros_like(_rest)
                        _acc[2] += _kept.sum()
                        _acc[3] += (_rest - _kept).sum()
                    xyz_ij = xyz_i + move.unsqueeze(-1) * disp_ij
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
                # (4) A relocation that lands in front of the whole original scene is
                # not a plausible motion, and gsplat is called with near_plane=1e-10
                # so nothing culls it -- it renders as a screen-filling splat. Drop
                # those. Only ever touches Gaussians that were actually displaced.
                if xyz_ij is not xyz_i:
                    _moved = (xyz_ij != xyz_i).any(dim=-1)
                    _bad = near_camera_reject(xyz_i, xyz_ij, test_w2c_i[j], _moved)
                    if bool(_bad.any()):
                        opacity_ij = opacity_ij * (~_bad).to(opacity_ij.dtype)
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
            if _acc is not None and i == 0:
                _t = _acc.sum().clamp_min(1.0)
                _f, _r, _st, _d = (_acc / _t * 100.0).tolist()
                _ka = _keep_acc.tolist()
                if _ka[1] > 0:
                    print(f"[DynKeep] off-frame dynamic gaussians surviving the gate: "
                          f"{100.0 * _ka[0] / _ka[1]:.1f}%  (nearest_source="
                          f"{dyn_nearest_source or 'off'})", flush=True)
                print(f"[DynUnsup/{dyn_unsupported}] of {int(_acc.sum().item())} off-frame "
                      f"dynamic (gaussian,target) pairs: flow {_f:.1f}%  rigid {_r:.1f}%  "
                      f"kept-static {_st:.1f}%  DROPPED {_d:.1f}%", flush=True)
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
        dyn_opacity_comp: float = 0.0,
        dyn_unsupported: str = "drop",
        gaussian_track_dist: Tensor | None = None,
        dyn_far_static: float = 0.0,
        dyn_nearest_source: int = 0,
    ) -> DecoderOutput:

        return self.rendering_fn(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode, cam_rot_delta, cam_trans_delta,
                                 gaussian_frame_idx=gaussian_frame_idx, gaussian_dyn_flag=gaussian_dyn_flag, gaussian_only_view=gaussian_only_view, leave_one_out=leave_one_out,
                                 dyn_centroid=dyn_centroid, dyn_centroid_pred=dyn_centroid_pred,
                                 dyn_centroid_valid=dyn_centroid_valid,
                                 dyn_group_centroid=dyn_group_centroid, dyn_group_pred=dyn_group_pred,
                                 dyn_group_valid=dyn_group_valid, gaussian_group_idx=gaussian_group_idx,
                                 gaussian_disp=gaussian_disp, gaussian_disp_valid=gaussian_disp_valid,
                                 dyn_unsupported=dyn_unsupported,
                                 gaussian_track_dist=gaussian_track_dist,
                                 dyn_far_static=dyn_far_static,
                                 dyn_nearest_source=dyn_nearest_source,
                                 per_frame_compositing=per_frame_compositing,
                                 dyn_opacity_comp=dyn_opacity_comp)

