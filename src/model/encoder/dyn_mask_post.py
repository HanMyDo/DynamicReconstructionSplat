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
