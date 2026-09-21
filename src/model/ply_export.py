from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor

# Standard SH DC-to-color constant (same as INRIA 3DGS and gsplat)
C0 = 0.28209479177387814


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
    dyn_mask_flat: np.ndarray | None = None,
    dyn_opacity_scale: float = 0.05,
    max_scale_frac: float = 0.011,
):
    """max_scale_frac: drop Gaussians whose LARGEST axis exceeds this fraction of the
    scene's robust (p1-p99) diagonal. 0 disables.

    WHY A CEILING. Measured on balloon t15 (2.79M Gaussians, scene diagonal 1.80):
    50 Gaussians are over 0.1 world units across -- each spanning 6% of the whole
    scene -- and they alone carry 9.4% of all opacity-weighted splat area. The 998
    above 0.05 carry 20%. In a viewer they are huge, faint (median opacity 0.013 vs
    0.031 overall), round discs, which is the concentric-ring haze that made these
    files look broken. The default 0.011 (~0.02 world here) removes 1.9% of the
    Gaussians and 31% of the haze; user-confirmed as the best of 0.05/0.03/0.02.

    CEILING, NOT FLOOR. prune_ply.py only ever had --min_scale, and opacity pruning
    has far worse leverage for this defect: dropping everything below opacity 0.02
    deletes 34% of the file to remove 22% of the haze, against 1.9% for 31% here.
    The defect is SIZE, so the cut has to be on size.

    WHY THE EXPORTER AND NOT A POST-HOC TOOL. Upstream AnySplat ships this file
    unfiltered and does not need a filter: it runs voxelize=true, and fusion never
    produces these. We run voxelize=false (per-pixel Gaussians are what every
    dynamic mechanism here needs), so we inherited an exporter built for a
    representation we do not use.

    RELATIVE, not absolute, because the threshold has to mean the same thing in a
    scene of a different size; VGGT geometry is near-unit-scaled but not exactly.
    """
    if max_scale_frac > 0 and means.shape[0] > 0:
        _m = means.detach().cpu().float().numpy()
        _lo, _hi = np.percentile(_m, 1, axis=0), np.percentile(_m, 99, axis=0)
        _diag = float(np.linalg.norm(_hi - _lo))
        _limit = max_scale_frac * _diag
        _keep = scales.detach().max(dim=-1).values <= _limit
        _n_drop = int((~_keep).sum())
        if 0 < _n_drop < means.shape[0]:
            means, scales = means[_keep], scales[_keep]
            rotations, harmonics = rotations[_keep], harmonics[_keep]
            opacities = opacities[_keep]
            if dyn_mask_flat is not None and len(dyn_mask_flat) == len(_keep):
                dyn_mask_flat = dyn_mask_flat[_keep.detach().cpu().numpy()]
            print(f"[PLY export] dropped {_n_drop} oversized Gaussians "
                  f"(max axis > {_limit:.4f} = {max_scale_frac:.3f} x scene diagonal "
                  f"{_diag:.2f}), {means.shape[0]} remain")
        elif _n_drop:
            print(f"[PLY export] max_scale_frac {max_scale_frac} would drop EVERY "
                  f"Gaussian -- skipping the cut, inspect the scene scale")

    if shift_and_scale:
        # Shift the scene so that the median Gaussian is at the origin.
        means = means - means.median(dim=0).values

        # Rescale the scene so that most Gaussians are within range [-1, 1].
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since current model use SH_degree = 4,
    # which require large memory to store, we can only save the DC band to save memory.
    f_dc = harmonics[..., 0]  # [N, 3]: DC SH coefficient for each RGB channel
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    # Diagnostics: print f_dc statistics to help verify color encoding
    f_dc_np = f_dc.detach().cpu().float().numpy()
    rgb_from_dc = (f_dc_np * C0 + 0.5).clip(0, 1)
    print(f"[PLY export] f_dc range: [{f_dc_np.min():.3f}, {f_dc_np.max():.3f}], "
          f"mean={f_dc_np.mean():.3f}")
    print(f"[PLY export] Implied RGB from DC: min={rgb_from_dc.min(axis=0)}, "
          f"max={rgb_from_dc.max(axis=0)}, mean={rgb_from_dc.mean(axis=0)}")

    # Optionally suppress dynamic Gaussians for cleaner PLY presentation.
    # dyn_mask_flat: 1 = dynamic, 0 = static, same length as opacities.
    if dyn_mask_flat is not None and len(dyn_mask_flat) == len(opacities):
        dyn_weights = 1.0 - (1.0 - dyn_opacity_scale) * dyn_mask_flat
        opacities = opacities * torch.from_numpy(dyn_weights).to(opacities.device, opacities.dtype)
    elif dyn_mask_flat is not None:
        print(f"[PLY export] dyn_mask length {len(dyn_mask_flat)} != gaussians {len(opacities)}, skipping mask")

    # Opacity: PLY format expects pre-sigmoid logit; fix the original AnySplat bug.
    opacity_logit = opacities.clamp(1e-6, 1 - 1e-6).logit()

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0 if save_sh_dc_only else f_rest.shape[1])]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacity_logit[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    ]
    if save_sh_dc_only:
        # remove f_rest from attributes
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    # Also export a vertex-colored point cloud for easy inspection in any viewer.
    # Colors are computed from DC SH coefficients using the standard formula.
    _export_colored_pointcloud(means, rgb_from_dc, Path(str(path).replace(".ply", "_rgb.ply")))


def _export_colored_pointcloud(
    means: Float[Tensor, "gaussian 3"],
    rgb: np.ndarray,  # [N, 3] in [0, 1]
    path: Path,
):
    """Export a plain XYZ+RGB point cloud PLY for viewing in any PLY viewer."""
    xyz = means.detach().cpu().numpy()
    rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["red"] = rgb_uint8[:, 0]
    elements["green"] = rgb_uint8[:, 1]
    elements["blue"] = rgb_uint8[:, 2]

    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
