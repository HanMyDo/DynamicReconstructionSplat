"""Complete a partial dynamic mask into whole objects.

WHY THIS EXISTS. VGGT4D's detector responds to attention dissimilarity, which is
strongest where a surface moves fast and textured. On a person it fires on the
limbs and the head outline and misses the torso interior, so the mask arrives as
DISCONNECTED PARTS of one object. Every downstream mechanism then splits that
person in half: the masked parts get relocated or dropped, the unmasked parts
stay where they were and render from every frame in the window at once. The
result is a handled fragment surrounded by chaotic scatter -- worse than either
treating the whole person as dynamic or ignoring it entirely.

--mask_otsu_level 2 raised recall (dyn_frac 0.056 -> 0.159) but only lowered the
threshold; it cannot connect a torso the detector never scored. That is a shape
problem, and shape operations fix it:

  close      bridge the gaps between parts of one object (limb -> torso -> head)
  fill       fill interior holes, so a ring of edges becomes a solid body
  min_area   delete specks, which are the false positives that make the
             per-frame-compositing drop tear real background out of the scene
  dilate     a small safety margin, since a mask edge slightly inside the object
             leaves a rim of the object behind

Order matters: close before fill (fill cannot help until the outline is closed),
min_area after fill (so a legitimate object is never judged on its outline alone).
"""
from typing import Optional

import numpy as np


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def complete_mask(m: np.ndarray, close: int = 0, fill: bool = False,
                  min_area: int = 0, dilate: int = 0) -> np.ndarray:
    """m: [H,W] in {0,1} (any float dtype). -> same shape/dtype, completed.

    Every step is a no-op at its default, so the unprocessed mask is reproduced
    exactly when nothing is requested.
    """
    from scipy import ndimage as ndi

    b = m > 0.5
    if not b.any():
        return m
    if close > 0:
        b = ndi.binary_closing(b, structure=_disk(close))
    if fill:
        b = ndi.binary_fill_holes(b)
    if min_area > 0:
        lab, n = ndi.label(b)
        if n > 0:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0                                  # background
            keep = np.isin(lab, np.flatnonzero(sizes >= min_area))
            b = b & keep
    if dilate > 0:
        b = ndi.binary_dilation(b, structure=_disk(dilate))
    return b.astype(m.dtype)


def motion_gate(m: np.ndarray, residual: np.ndarray, mult: float = 3.0,
                min_pixels: int = 50, quantile: float = 0.75,
                floor: float = 0.5) -> np.ndarray:
    """Drop mask components that do not actually move. -> same shape/dtype as m.

    WHY. Attention over-fires on STATIC structure beside a moving object -- the desk
    edge and the chair next to a person -- because it responds to attention
    dissimilarity in that neighbourhood, not to motion. Raising the threshold to
    exclude them also drops the person's slow parts, which is the whole reason the
    published masks are arm-only. Recall and precision cannot both be bought from
    one attention threshold.

    Geometry separates them cleanly, because the two signals fail differently: flow
    residual (measured flow minus the flow the camera alone would produce) is ~0 on
    anything static REGARDLESS of how close it sits to a moving object.

    Deciding per COMPONENT rather than per pixel is the point. A pixel-wise
    intersection re-erodes the person wherever the residual is locally noisy --
    undoing the shape completion that made the mask cover a whole object. A whole
    component is kept or dropped together, so the person survives intact including
    the parts where the residual is weak, while the chair goes entirely.

    The bar is RELATIVE to the frame's own static regions (pixels outside the mask),
    which is where depth error puts a noise floor -- an absolute threshold would need
    retuning per scene. `floor` keeps a near-perfect static prediction from making the
    bar zero and admitting everything.
    """
    from scipy import ndimage as ndi

    b = m > 0.5
    if mult <= 0 or not b.any():
        return m
    bg = residual[~b]
    base = float(np.quantile(bg, quantile)) if bg.size else 0.0
    thr = mult * max(base, floor)

    lab, n = ndi.label(b)
    keep = np.zeros_like(b)
    for i in range(1, n + 1):
        sel = lab == i
        if sel.sum() < min_pixels:
            continue                      # too small to judge; specks go
        if float(np.quantile(residual[sel], quantile)) > thr:
            keep |= sel
    return keep.astype(m.dtype)


def motion_gate_masks(masks: np.ndarray, residual: np.ndarray, mult: float = 3.0,
                      min_pixels: int = 50, quantile: float = 0.75) -> np.ndarray:
    """[V,H,W] masks gated against [V,H,W] residual, each frame independently."""
    if mult <= 0:
        return masks
    return np.stack([motion_gate(masks[i], residual[i], mult, min_pixels, quantile)
                     for i in range(masks.shape[0])], axis=0)


def complete_masks(masks: np.ndarray, close: int = 0, fill: bool = False,
                   min_area: int = 0, dilate: int = 0) -> np.ndarray:
    """[V,H,W] -> [V,H,W], each frame completed independently."""
    if close <= 0 and not fill and min_area <= 0 and dilate <= 0:
        return masks
    return np.stack([complete_mask(masks[i], close, fill, min_area, dilate)
                     for i in range(masks.shape[0])], axis=0)


def largest_dyn_clusters(means: np.ndarray, dyn: np.ndarray, keep_frac: float,
                         link_frac: float = 0.004) -> np.ndarray:
    """Keep only dynamic Gaussians belonging to a SUBSTANTIAL 3D cluster. -> bool [N]

    WHY. `min_area` removes specks in the 2D mask, but a false positive that is a
    respectable blob in one frame -- a chair, an instrument on a bench -- survives
    it and is then carried all the way into the export. Measured on balloon at
    672x896: only 14.8% of the Gaussians flagged dynamic lie within 0.3 world units
    of their own densest spot, and the middle 80% of them span 0.84 of the scene
    diagonal. So most of "the moving object" is furniture scattered through the room.

    That is far more damaging in the PLY than in the render. `--ply_own_frame_only`
    keeps (static) OR (dynamic from frame j), so a misclassified BACKGROUND patch
    loses its other V-1 copies -- a hole -- and the survivor is then boosted by
    --dyn_opacity_comp into one of the most opaque splats in the file. Holes plus
    bright specks, scene-wide. The render does not do this: there the other copies
    are still drawn, so the same error is only a slight over-opacity.

    Grouping in 3D is what separates the two cases, because a real moving object is
    ONE connected body while the false positives are scattered islands -- a
    distinction that does not exist in the 2D mask, where they all look like blobs.

    keep_frac is relative to the LARGEST component. MEASURED on balloon at 672x896
    (36,335 dynamic gaussians, scene diagonal 1.79):

        none        14.8% within 0.3 world of the centroid, extent 0.84 of diagonal
        0.25        63.7%                                   extent 0.61
        0.50        95.5%                                   extent 0.24  <- one body

    0.5 keeps a single component and is what makes the object look like an object.
    0.25 keeps two, which is right when the scene HAS two genuine movers (balloon
    has a person and a balloon, far apart) and wrong when the second is furniture --
    the filter cannot tell those apart, so look before choosing.

    link_frac is the connection radius as a fraction of the scene diagonal. It
    barely matters (0.015 -> 0.002 moved the result by 7 points) because the
    components are already well separated; the default is small enough not to bridge
    a person to the floor they stand on.

    PRESENTATION, and a blunt one: it cannot tell a true mover from a false positive,
    only compact from scattered. A genuinely isolated small moving object would be
    dropped too. Prefer fixing the mask; use this when a figure is needed sooner.
    """
    out = np.ones(len(means), dtype=bool)
    if keep_frac <= 0:
        return out
    sel = np.flatnonzero(dyn)
    if sel.size < 10:
        return out
    pts = means[sel]
    lo, hi = np.percentile(means, 1, axis=0), np.percentile(means, 99, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    v = max(link_frac * diag, 1e-6)
    ijk = np.floor((pts - pts.min(0)) / v).astype(np.int64)

    # Flood fill over OCCUPIED voxels only, in a dict. A dense grid would allocate
    # the whole bounding volume -- and one far outlier makes that enormous -- while
    # the occupied set is at most one voxel per gaussian. No scipy either, which
    # matters because this also has to run wherever a PLY gets inspected.
    occ: dict = {}
    for idx, key in enumerate(map(tuple, ijk)):
        occ.setdefault(key, []).append(idx)
    label = {}
    sizes = []
    nbr = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for start in occ:
        if start in label:
            continue
        cid = len(sizes)
        label[start] = cid
        stack, total = [start], 0
        while stack:
            k = stack.pop()
            total += len(occ[k])
            for dx, dy, dz in nbr:
                nk = (k[0] + dx, k[1] + dy, k[2] + dz)
                if nk in occ and nk not in label:
                    label[nk] = cid
                    stack.append(nk)
        sizes.append(total)
    if len(sizes) <= 1:
        return out
    sizes = np.asarray(sizes)
    bar = keep_frac * sizes.max()
    keep_dyn = np.zeros(sel.size, dtype=bool)
    for key, idxs in occ.items():
        if sizes[label[key]] >= bar:
            keep_dyn[idxs] = True
    out[sel] = keep_dyn
    print(f"[ply] 3D cluster filter: {len(sizes)} dynamic components, kept "
          f"{int((sizes >= bar).sum())} (>= {keep_frac:g} x largest); "
          f"{int(keep_dyn.sum())}/{sel.size} dynamic gaussians survive "
          f"(link radius {v:.4f}, scene diag {diag:.2f})", flush=True)
    return out
