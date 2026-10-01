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


def _warp_mask(src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample `src` [H,W] at p + flow[p]. `flow` must map THIS frame -> src's frame.

    A gather, not a scatter: computing the backward flow and sampling is exact,
    whereas pushing pixels forward leaves holes wherever the source expands.
    NEAREST because a mask is labels -- bilinear would invent half-dynamic pixels
    along every silhouette, which is where precision is decided.
    """
    H, W = src.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=src.device, dtype=src.dtype),
                            torch.arange(W, device=src.device, dtype=src.dtype),
                            indexing="ij")
    x = xx + flow[..., 0]
    y = yy + flow[..., 1]
    grid = torch.stack([2.0 * x / max(W - 1, 1) - 1.0,
                        2.0 * y / max(H - 1, 1) - 1.0], dim=-1).unsqueeze(0)
    out = F.grid_sample(src[None, None], grid, mode="nearest",
                        padding_mode="zeros", align_corners=True)
    return out[0, 0]


@torch.no_grad()
def propagate_masks(masks: torch.Tensor, images: torch.Tensor,
                    max_carry: int = 5) -> torch.Tensor:
    """Carry a confident detection through frames where the detector loses it.

    WHY. The attention score is not stationary over a long sequence, and
    thresholding it fails either way round. Measured on Dynamic Replica 0cde48,
    300 frames, against ground truth: recall holds at 0.98 for 120 frames then
    collapses to 0.08 -- while the object KEEPS MOVING (GT travel 2.51 px/frame in
    frames 0-29 vs 2.73 in 180-209, near identical, recall 0.978 vs 0.119). One
    global threshold cannot adapt; per-chunk thresholds adapt to an arbitrary
    boundary instead of to content and were catastrophic either side of it (0.119
    -> 0.014). Both are the wrong tool for a drifting score.

    So stop re-deciding every frame independently. A region confidently detected
    at t is still the same object at t+1, and RAFT already tells us where it went.

    BOUNDED by max_carry: a region may be carried at most that many consecutive
    frames without being re-detected. Unbounded propagation would smear the object
    along its whole path -- here it travels 1087 px -- turning a recall fix into a
    precision disaster. The carry count travels WITH the pixels, so it measures
    frames-since-evidence for that piece of object, not for that screen position.

    Runs in both directions and unions: forward repairs a detection that fades,
    backward repairs one that starts late. Neither alone covers both.

    masks [V,H,W] in {0,1}; images [V,3,H,W] in [0,1]. -> [V,H,W] float {0,1}
    """
    V, _, H, W = images.shape
    if V < 2 or max_carry <= 0:
        return masks
    dev = images.device
    model = _raft(dev)
    imgs = images.float().clamp(0, 1) * 2.0 - 1.0            # RAFT wants [-1,1]
    det = (masks > 0.5).float()

    def sweep(order):
        out = det.clone()
        carry = torch.zeros(H, W, device=dev)                # frames since evidence
        prev_i = order[0]
        for t in order[1:]:
            # flow from THIS frame back to the previous one in sweep order, so the
            # warp is a gather (see _warp_mask).
            flow = raft_flow(model, imgs[t:t + 1], imgs[prev_i:prev_i + 1])
            warped = _warp_mask(out[prev_i], flow)
            carry = _warp_mask(carry, flow)
            d = det[t] > 0.5
            fresh = warped > 0.5
            carry = torch.where(d, torch.zeros_like(carry), carry + 1.0)
            keep = fresh & (carry <= max_carry) & (~d)
            out[t] = torch.where(d | keep, torch.ones_like(out[t]), torch.zeros_like(out[t]))
            prev_i = t
        return out

    fwd = sweep(list(range(V)))
    bwd = sweep(list(range(V - 1, -1, -1)))
    out = torch.maximum(fwd, bwd)
    before, after = float(det.mean()), float(out.mean())
    print(f"[MaskProp] dynamic pixels {100 * before:.1f}% -> {100 * after:.1f}% "
          f"(max_carry={max_carry}, both directions)", flush=True)
    return out


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
