"""Per-Gaussian displacement from a cached Track4World pairwise field.

This replaces `knn_flow_displacement` when `--dyn_t4w_cache` is given. The
difference is structural, not a tuning change: Track4World's field is DENSE per
pixel, so every Gaussian gets its own displacement and there is no trust radius,
no k-nearest-track interpolation and no "unsupported" population -- the machinery
measured as saturated on Oct 16, where the gate bought +1.67 dB of dynamic at the
cost of ssim/lpips on 2/2 sequences because sparse tracks cannot say what a
Gaussian far from any track should do.

WHY A Sim(3) FIT PER PAIR. Track4World estimates its own `metric_scale` for each
two-frame inference -- measured across one window's 240 pairs it ranges 2.2755 to
3.1872, a 40% spread -- and its world frame is its own. A 40% scale error on a 1 m
displacement is 40 cm of misplacement, so the raw field cannot be applied to our
Gaussians. Instead, for each pair we fit a similarity transform from Track4World's
source-frame points to OUR means at the same pixels, and apply only its scale and
rotation to the displacement. A displacement is a difference, so the translation
cancels and never enters.

⚠️ KNOWN ERROR FLOOR. On crowd, adjacent frames, the static background's
displacement median is 0.24 m where it should be ~0 (p90 1.7 m for the movers), and
at the 8-step gap this window needs it rises to 0.78 m. Our Gaussians sit at ~3 m
depth, so expect help where motion is large and harm on static content that the
mask fails to exclude.
"""
from typing import Optional, Tuple

import numpy as np
import torch

_CACHE = {}


def _load(path: str):
    if path not in _CACHE:
        z = np.load(path, allow_pickle=True)
        _CACHE[path] = {
            "disp": z["disp"],            # [T,T,G,G,3] f16, source-frame pixels
            "wsrc": z["wsrc"],            # [T,T,g,g,3] f16, T4W source points
            "valid": z["valid"],          # [T,T] bool
            "grid": int(z["grid"][0]),
            "sub": int(z["sub"][0]),
        }
        d = _CACHE[path]
        print(f"[DynT4W] loaded {path}: disp {d['disp'].shape} "
              f"grid {d['grid']} sub {d['sub']}, {int(d['valid'].sum())} pairs")
    return _CACHE[path]


def _umeyama_sR(src: torch.Tensor, dst: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """Similarity fit src->dst; returns (scale, R). Translation is not needed:
    we only ever transform DIFFERENCES, where it cancels."""
    sm, dm = src.mean(0), dst.mean(0)
    s0, d0 = src - sm, dst - dm
    C = (d0.T @ s0) / max(src.shape[0], 1)
    U, S, Vt = torch.linalg.svd(C.double())
    D = torch.eye(3, dtype=torch.float64, device=src.device)
    if torch.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = (U @ D @ Vt)
    var = (s0.double() ** 2).sum(1).mean()
    scale = float((S * torch.diag(D)).sum() / var.clamp_min(1e-12))
    return scale, R.to(src.dtype)


@torch.no_grad()
def t4w_displacement(
    cache_path: str,
    gauss_pts: torch.Tensor,      # [N,3] our means
    gauss_fidx: torch.Tensor,     # [N] source frame, -1 for fused voxels
    gauss_pix: torch.Tensor,      # [N] linear pixel index y*W+x, -1 if unknown
    num_views: int,
    gauss_dyn: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> (disp [N,num_views,3], valid [N,num_views] float 0/1)."""
    d = _load(cache_path)
    G, SUB = d["grid"], d["sub"]
    dev = gauss_pts.device
    N = gauss_pts.shape[0]
    disp = torch.zeros(N, num_views, 3, device=dev, dtype=gauss_pts.dtype)
    valid = torch.zeros(N, num_views, device=dev, dtype=gauss_pts.dtype)

    have_pix = gauss_pix >= 0
    T = d["disp"].shape[0]
    n_fit, n_pairs = 0, 0
    for i in range(min(T, num_views)):
        sel = (gauss_fidx == i) & have_pix
        if gauss_dyn is not None:
            sel = sel & gauss_dyn
        if not bool(sel.any()):
            continue
        idx = sel.nonzero(as_tuple=False).squeeze(1)
        pix = gauss_pix[idx].long()
        py, px = (pix // G), (pix % G)
        ours = gauss_pts[idx]

        # Our means laid back onto the source grid, subsampled to match `wsrc`.
        # Scattering rather than assuming raster order, because conf_valid_mask
        # filters pixels: index // G would silently scramble the field.
        g = G // SUB
        for j in range(min(T, num_views)):
            if j == i or not bool(d["valid"][i, j]):
                continue
            n_pairs += 1
            w = torch.from_numpy(d["wsrc"][i, j].astype(np.float32)).to(dev)  # [g,g,3]
            ours_map = torch.full((g, g, 3), float("nan"), device=dev, dtype=torch.float32)
            keep = (py % SUB == 0) & (px % SUB == 0)
            if bool(keep.any()):
                ours_map[(py[keep] // SUB).clamp(max=g - 1),
                         (px[keep] // SUB).clamp(max=g - 1)] = ours[keep].float()
            m = torch.isfinite(ours_map).all(-1) & torch.isfinite(w).all(-1)
            if int(m.sum()) < 50:
                continue                      # too few correspondences to trust a fit
            s, R = _umeyama_sR(w[m], ours_map[m])
            if not np.isfinite(s) or s <= 0:
                continue
            n_fit += 1
            dd = torch.from_numpy(d["disp"][i, j].astype(np.float32)).to(dev)  # [G,G,3]
            v = dd[py, px]                                   # [n_i,3] in T4W units
            disp[idx, j] = (s * (v @ R.T.float())).to(disp.dtype)
            valid[idx, j] = 1.0

    print(f"[DynT4W] {n_fit}/{n_pairs} pairs aligned; "
          f"{int((valid > 0).any(1).sum())}/{N} gaussians given a displacement",
          flush=True)
    return disp, valid
