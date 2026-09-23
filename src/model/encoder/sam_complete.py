"""Complete a motion SEED into whole objects with SAM 2.

WHY THIS EXISTS. The attention detector and the Otsu threshold trade recall
against precision along a single axis, and measured on balloon there is no
setting that gives both:

    otsu 1 + chunk 512 + global_post   arms and hands only, NO furniture   dyn 0.039
    otsu 2 + chunk 512 + global_post   whole person AND the chairs, desk    dyn 0.264

A chair's attention score sits BETWEEN the person's arm and the person's torso,
so no threshold separates them -- that is what a threshold does, compare
magnitudes. Morphology cannot bridge it either: closing, filling and dilating the
otsu-1 mask moved it 0.039 -> 0.044, about 13%, nowhere near arm -> person. It
can close a gap; it cannot invent a torso.

So the missing component is an OBJECT PRIOR, and the seed decides everything:

  - Prompt with otsu 1. Those seeds are precise and furniture-free, and they
    carry the MOTION evidence -- they sit on what actually moved. SAM grows
    "arm" into "person" and is never prompted on a chair, so no chair appears.
  - Do NOT prompt with otsu 2, and do NOT use SAM 3's text prompts. Both throw
    the motion criterion away: "person" segments a stationary person too, and an
    otsu-2 seed already contains the furniture we are trying to exclude.
    Completion applied to otsu 2 is measured -- it grew a chair patch into a
    whole chair.

SAM 2 rather than SAM 3 for the same reason plus cost: 162 MB against 3.45 GB
and ~3.4x faster, and SAM 3's only advantage is the open-vocabulary text
prompting that would defeat the design.

PER-FRAME, NOT VIDEO PROPAGATION. SAM 2's video predictor would add temporal
consistency, but it wants a directory of JPEGs and cross-frame object identity
management, and we already have a seed for EVERY frame from the detector -- the
thing propagation is normally needed for. Per-frame keeps the failure mode local
(one bad frame, not a drifting track) and the code an order of magnitude smaller.
Revisit if flicker turns out to matter.
"""
from typing import Optional

import numpy as np


_PREDICTOR = None


def _get_predictor(model_id: str, ckpt: Optional[str], device: str):
    """Lazily build the SAM 2 predictor. Imported here so the repo does not
    hard-depend on sam2 for every run that never asks for it."""
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if ckpt:
        from sam2.build_sam import build_sam2
        # A local checkpoint needs its matching config; sam2 resolves the config
        # by name from its own package, so pass the name rather than a path.
        cfg = model_id if model_id.endswith(".yaml") else f"{model_id}.yaml"
        _PREDICTOR = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=device))
    else:
        _PREDICTOR = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    print(f"[SAM] loaded {model_id}" + (f" from {ckpt}" if ckpt else " (hub)"),
          flush=True)
    return _PREDICTOR


def _seed_points(comp: np.ndarray, n: int) -> np.ndarray:
    """Interior points of one seed component, as [n, 2] (x, y).

    Eroded first so a prompt never lands on a boundary pixel, where SAM has to
    guess which side of the edge is meant. Falls back to the raw component when
    erosion empties it, which happens on the thin seeds otsu 1 produces.
    """
    from scipy import ndimage as ndi

    inner = ndi.binary_erosion(comp, iterations=2)
    if not inner.any():
        inner = comp
    ys, xs = np.nonzero(inner)
    if len(ys) <= n:
        pick = np.arange(len(ys))
    else:
        # Spread the points: the centroid plus the extremes of the principal
        # axis, so an elongated limb gets prompted along its length rather than
        # n times in the same spot.
        c = np.array([ys.mean(), xs.mean()])
        d = np.stack([ys, xs], 1).astype(np.float64) - c
        proj = d @ (np.linalg.svd(d, full_matrices=False)[2][0])
        order = np.argsort(proj)
        pick = np.unique(np.linspace(0, len(order) - 1, n).astype(int))
        pick = order[pick]
    return np.stack([xs[pick], ys[pick]], axis=1).astype(np.float32)


def sam_complete_masks(images: np.ndarray, seeds: np.ndarray,
                       model_id: str = "facebook/sam2-hiera-base-plus",
                       ckpt: Optional[str] = None, device: str = "cuda",
                       n_points: int = 3, min_seed_area: int = 40,
                       max_growth: float = 20.0) -> np.ndarray:
    """images [V,H,W,3] uint8; seeds [V,H,W] in {0,1} -> completed [V,H,W] float.

    max_growth is the safety rail that matters. Completion is the operation that
    turned a chair patch into a whole chair, so a returned mask more than
    `max_growth` times its seed's area is REJECTED and the seed kept. A runaway
    is caught by size before it reaches the mask, and the failure is local.

    The output is the UNION of SAM's masks with the original seed, so a frame
    where SAM returns nothing degrades to current behaviour rather than losing
    the object entirely.
    """
    from scipy import ndimage as ndi

    pred = _get_predictor(model_id, ckpt, device)
    V = images.shape[0]
    out = (seeds > 0.5).astype(np.float32).copy()
    n_comp = n_grown = n_reject = 0

    for v in range(V):
        seed = seeds[v] > 0.5
        if not seed.any():
            continue
        lab, n = ndi.label(seed)
        if n == 0:
            continue
        keep = [i for i in range(1, n + 1) if (lab == i).sum() >= min_seed_area]
        if not keep:
            continue
        pred.set_image(images[v])          # image encoder runs ONCE per frame
        for i in keep:
            comp = lab == i
            n_comp += 1
            pts = _seed_points(comp, n_points)
            try:
                masks, scores, _ = pred.predict(
                    point_coords=pts,
                    point_labels=np.ones(len(pts), dtype=np.int32),
                    multimask_output=True,
                )
            except Exception as e:                       # OOM, bad prompt, ...
                print(f"[SAM] frame {v} component {i} failed ({e})", flush=True)
                continue
            m = masks[int(np.argmax(scores))].astype(bool)
            if m.sum() > max_growth * comp.sum():
                n_reject += 1
                continue                                 # keep the seed alone
            out[v][m] = 1.0
            n_grown += 1

    print(f"[SAM] {n_comp} seed components over {V} frames: {n_grown} completed, "
          f"{n_reject} rejected for growing past {max_growth:g}x the seed | "
          f"dyn fraction {seeds.mean():.3f} -> {out.mean():.3f}", flush=True)
    return out
