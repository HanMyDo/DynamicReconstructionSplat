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

    images [V,3,H,W] in [0,1]; depth [V,H,W]; extrinsic [V,4,4] world2cam;
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
            measured = model(imgs[i:i + 1], imgs[j:j + 1])[-1][0].permute(1, 2, 0)  # [H,W,2]
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
