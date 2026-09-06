"""Geometric dynamic-mask detection from flow residual.

WHY THIS EXISTS. VGGT4D detects motion from ATTENTION DISSIMILARITY between a
frame and its neighbours, which responds to how FAST a pixel moves. On a walking
person that finds the swinging arm and misses the torso and head, because they
barely displace between frames 0.2 s apart -- so the mask covers a limb, and
per-frame compositing cannot protect the rest of the person, who then ghosts as
many overlapping copies across every rendered view. Verified faithful to the
original demo, so this is the published method's behaviour, not a porting bug.

THE GEOMETRIC SIGNAL. Given depth and camera poses, every STATIC pixel's motion
between two frames is fully determined: unproject with depth, move by the
relative pose, reproject. Subtracting that prediction from measured optical flow
leaves a residual that is ~0 on static geometry and non-zero on anything moving
INDEPENDENTLY of the camera. A slow torso still moves differently from the wall
behind it, so the residual covers the whole object, not just its fastest parts.

Failure mode to respect: the prediction inherits depth error, so residual is also
large where depth is wrong (thin structures, edges, distant regions). The
threshold is therefore taken over the whole sequence rather than per frame, and
depth confidence gates the result where available.
"""
from typing import Optional

import torch
import torch.nn.functional as F

_RAFT = {}


def _raft(device):
    if str(device) not in _RAFT:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        m = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _RAFT[str(device)] = m
    return _RAFT[str(device)]


def _to44(m: torch.Tensor) -> torch.Tensor:
    """Accept a [3,4] world2cam -- what VGGT's pose head actually emits -- or a [4,4].

    `pose_encoding_to_extri_intri` returns [B, V, 3, 4]; inverting that directly
    raises "A must be batches of square matrices". `refine_dynamic_mask` already
    pads the bottom row, so this path has to as well.
    """
    m = m.float()
    if m.shape[-2] == 3:
        bottom = torch.zeros(1, 4, device=m.device, dtype=m.dtype)
        bottom[0, 3] = 1.0
        m = torch.cat([m, bottom], dim=0)
    return m


def raft_flow(model, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Dense flow a -> b as [H, W, 2], at ANY input size.

    torchvision's RAFT asserts H and W are divisible by 8. Dynamic detection runs
    at the original VGGT4D resolution of 518 on the long edge, and 518 % 8 == 6,
    so every call raises before producing anything -- while the eval-time tracker
    at 448 has always been fine. Replicate-pad up to the next multiple of 8, run,
    then crop the flow back, which is a no-op when the size already divides.
    """
    H, W = a.shape[-2:]
    ph, pw = (-H) % 8, (-W) % 8
    if ph or pw:
        a = F.pad(a, (0, pw, 0, ph), mode="replicate")
        b = F.pad(b, (0, pw, 0, ph), mode="replicate")
    flow = model(a, b)[-1][0]                       # [2, H+ph, W+pw]
    return flow[:, :H, :W].permute(1, 2, 0)         # [H, W, 2]


def _pixel_grid(H: int, W: int, device) -> torch.Tensor:
    """[H, W, 2] of (u, v) pixel centres."""
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                          torch.arange(W, device=device, dtype=torch.float32),
                          indexing="ij")
    return torch.stack([u, v], dim=-1)


def induced_flow(depth_i: torch.Tensor, K_i: torch.Tensor, K_j: torch.Tensor,
                 w2c_i: torch.Tensor, w2c_j: torch.Tensor) -> torch.Tensor:
    """Flow a STATIC scene would produce from frame i to frame j.

    depth_i [H,W]; K [3,3]; w2c [4,4] world-to-camera.
    -> [H,W,2] displacement in pixels. Points behind camera j are returned as 0.
    """
    H, W = depth_i.shape
    dev = depth_i.device
    w2c_i, w2c_j = _to44(w2c_i), _to44(w2c_j)
    uv = _pixel_grid(H, W, dev)                                   # [H,W,2]
    ones = torch.ones(H, W, 1, device=dev)
    pix = torch.cat([uv, ones], dim=-1).reshape(-1, 3, 1)         # [N,3,1]

    cam_i = (torch.inverse(K_i.float()) @ pix) * depth_i.reshape(-1, 1, 1)   # [N,3,1]
    cam_i_h = torch.cat([cam_i, torch.ones(cam_i.shape[0], 1, 1, device=dev)], dim=1)
    world = torch.inverse(w2c_i.float()) @ cam_i_h                # [N,4,1]
    cam_j = (w2c_j.float() @ world)[:, :3]                        # [N,3,1]
    z = cam_j[:, 2:3]
    proj = K_j.float() @ cam_j                                    # [N,3,1]
    uv_j = proj[:, :2, 0] / proj[:, 2:3, 0].clamp(min=1e-6)       # [N,2]
    flow = (uv_j - uv.reshape(-1, 2)).reshape(H, W, 2)
    behind = (z.reshape(H, W) <= 1e-6)
    return torch.where(behind.unsqueeze(-1), torch.zeros_like(flow), flow)


@torch.no_grad()
def flow_residual_map(images: torch.Tensor, depth: torch.Tensor,
                      extrinsic: torch.Tensor, intrinsic: torch.Tensor,
                      depth_conf: Optional[torch.Tensor] = None,
                      conf_quantile: float = 0.1) -> torch.Tensor:
    """Per-pixel evidence that a pixel moves independently of the camera.

    images [V,3,H,W] in [0,1]; depth [V,H,W]; extrinsic [V,3,4] or [V,4,4] world2cam;
    intrinsic [V,3,3] in PIXELS. -> [V,H,W] residual magnitude in pixels.

    Each frame is compared with its neighbours on both sides and the results
    combined by MINIMUM: a pixel counts as moving only if it disagrees with the
    static prediction in every comparison, which suppresses one-off flow errors
    and occlusion artefacts at a single boundary.
    """
    dev = images.device
    V, _, H, W = images.shape
    model = _raft(dev)
    imgs = images.float().clamp(0, 1) * 2.0 - 1.0            # RAFT wants [-1,1]

    res = torch.full((V, H, W), float("inf"), device=dev)
    for i in range(V):
        for j in (i - 1, i + 1):
            if not (0 <= j < V):
                continue
            measured = raft_flow(model, imgs[i:i + 1], imgs[j:j + 1])   # [H,W,2]
            pred = induced_flow(depth[i], intrinsic[i], intrinsic[j],
                                extrinsic[i], extrinsic[j])
            res[i] = torch.minimum(res[i], (measured - pred).norm(dim=-1))
    res = torch.where(torch.isinf(res), torch.zeros_like(res), res)

    if depth_conf is not None:
        # Where depth is unreliable the prediction is unreliable too, so a large
        # residual says nothing about motion. Zero those out rather than trusting them.
        thr = torch.quantile(depth_conf.flatten().float(), conf_quantile)
        res = res * (depth_conf > thr).float()
    return res
