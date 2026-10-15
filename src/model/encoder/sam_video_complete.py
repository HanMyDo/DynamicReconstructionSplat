"""Complete sparse motion seeds into full-sequence masks with SAM 2 VIDEO propagation.

WHY THIS EXISTS, AND WHY IT IS NOT sam_complete.py. The per-frame path was chosen
because "we already have a seed for EVERY frame from the detector -- the thing
propagation is normally needed for". That holds for the attention detector (measured:
0.0% empty frames on balloon). It does NOT hold for the geometric flow detector, which
is far more precise -- furniture-free, dyn_frac 0.023 against attention's 0.268 -- and
far sparser: 42.1% of frames carry no seed at all, in runs of up to 73 consecutive
frames. RAFT mask-warping (--mask_propagate) cannot bridge that: at 5 frames it closed
only the short gaps, leaving runs of 22-68, and no warp survives 70 frames of drift.

SAM 2's video predictor is built for exactly this. Its memory bank carries an object
across hundreds of frames from one prompt, so the sparse-but-clean flow detections
become anchors and the gaps fill themselves. That gives flow's precision with full
temporal coverage -- the combination neither detector reaches alone.

ANCHOR CHOICE. Seeds are ranked by area and the largest are used as prompts, because a
seed that covers a torso propagates reliably while one covering three pixels of a hand
does not. Propagation runs BACKWARD as well as forward from the first anchor, or every
frame before it would be left empty.

OBJECT IDENTITY. Connected components of an anchor seed become separate SAM object ids,
so a person and a thrown balloon are tracked as two objects rather than one blob that
has to stay connected. Their masks are unioned on output, which is all the pipeline
consumes.
"""
from typing import List, Optional, Tuple

import numpy as np


def _rank_anchor_frames(seeds: np.ndarray, max_anchors: int, min_area: int) -> List[int]:
    """Frames whose seed is big enough to prompt with, best first, spread over time.

    Taking simply the N largest tends to pick N neighbouring frames of the same moment,
    which propagates no better than one. So the sequence is split into max_anchors
    buckets and the best frame of each bucket is used, giving coverage along the video.
    """
    areas = seeds.reshape(len(seeds), -1).sum(1)
    ok = np.flatnonzero(areas >= min_area)
    if ok.size == 0:
        return []
    buckets = np.array_split(ok, min(max_anchors, ok.size))
    return [int(b[np.argmax(areas[b])]) for b in buckets if b.size]


def _components(mask: np.ndarray, min_area: int) -> List[np.ndarray]:
    """Split a seed into connected components, largest first."""
    try:
        import cv2
        n, lab = cv2.connectedComponents(mask.astype(np.uint8))
        out = [(lab == i) for i in range(1, n)]
    except Exception:
        out = [mask.astype(bool)]
    out = [c for c in out if c.sum() >= min_area]
    out.sort(key=lambda c: -c.sum())
    return out


def sam_video_complete(
    images: np.ndarray,
    seeds: np.ndarray,
    model_id: str = "facebook/sam2-hiera-base-plus",
    ckpt: Optional[str] = None,
    device: str = "cuda",
    max_anchors: int = 8,
    min_seed_area: int = 40,
    max_objects_per_anchor: int = 3,
) -> np.ndarray:
    """images [N,H,W,3] uint8, seeds [N,H,W] bool -> completed masks [N,H,W] bool.

    Falls back to returning the seeds unchanged if SAM 2's video predictor is not
    importable, so a missing optional dependency degrades rather than crashes a
    50-minute detection run.
    """
    import torch

    n_frames, H, W = seeds.shape
    anchors = _rank_anchor_frames(seeds, max_anchors, min_seed_area)
    if not anchors:
        print("[sam-video] no seed large enough to anchor on; returning seeds unchanged",
              flush=True)
        return seeds.astype(bool)

    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)
    except Exception as exc:                                  # pragma: no cover
        try:
            from sam2.build_sam import build_sam2_video_predictor
            predictor = build_sam2_video_predictor(model_id, ckpt, device=device)
        except Exception:
            print(f"[sam-video] video predictor unavailable ({exc}); seeds unchanged",
                  flush=True)
            return seeds.astype(bool)

    # SAM 2's video predictor MUST run under autocast: its memory attention stores the
    # bank in half precision while the projection weights stay float32, so without this
    # propagation dies with "mat1 and mat2 must have the same dtype, but got BFloat16
    # and Float". The official demo enters autocast globally for the same reason.
    if device.startswith("cuda"):
        _dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        _amp = torch.autocast("cuda", dtype=_dt)
    else:
        from contextlib import nullcontext
        _amp = nullcontext()

    # init_state historically wants a directory of JPEGs; newer builds accept frames
    # directly. Try the in-memory form first and fall back to a temporary directory,
    # because writing 439 JPEGs costs disk and a re-encode of every frame.
    # A WHOLE-SEQUENCE propagation holds every frame resized to 1024x1024 (~12 MB each)
    # plus a memory bank that grows with objects x frames, so past a few hundred frames
    # the offload flags are the difference between fitting and a CUDA OOM. They are
    # keyword-only and absent from older sam2 builds, hence the widening fallback: a
    # TypeError means this build does not know the kwarg, so drop it and retry.
    _init_kwargs = [
        dict(offload_video_to_cpu=True, offload_state_to_cpu=True),
        dict(offload_video_to_cpu=True),
        dict(),
    ]

    def _init(**fixed):
        for kw in _init_kwargs:
            try:
                return predictor.init_state(**fixed, **kw)
            except TypeError:
                continue
        return None

    state = None
    with _amp:
        try:
            state = _init(video_path=None, images=images)
        except Exception:
            state = None
        if state is None:
            import tempfile, os
            from PIL import Image
            tmp = tempfile.mkdtemp(prefix="sam2_frames_")
            for i, im in enumerate(images):
                Image.fromarray(im).save(os.path.join(tmp, f"{i:06d}.jpg"), quality=95)
            state = _init(video_path=tmp)

    obj = 0
    anchored: List[Tuple[int, int]] = []
    with _amp:
        for f in anchors:
            for comp in _components(seeds[f], min_seed_area)[:max_objects_per_anchor]:
                predictor.add_new_mask(state, frame_idx=f, obj_id=obj, mask=comp)
                anchored.append((f, obj))
                obj += 1
    print(f"[sam-video] {len(anchors)} anchor frames, {obj} objects "
          f"(seeds empty on {100.0 * (seeds.reshape(n_frames, -1).sum(1) == 0).mean():.1f}% "
          f"of frames)", flush=True)

    out = np.zeros((n_frames, H, W), dtype=bool)
    with torch.inference_mode(), _amp:
        for reverse in (False, True):                 # forward, then backward
            try:
                gen = predictor.propagate_in_video(state, reverse=reverse)
            except TypeError:                         # older builds: forward only
                if reverse:
                    break
                gen = predictor.propagate_in_video(state)
            for f_idx, _ids, logits in gen:
                m = (logits > 0).float().squeeze(1).any(0).cpu().numpy().astype(bool)
                if m.shape != (H, W):                 # predictor may work at its own res
                    from PIL import Image as _I
                    m = np.asarray(_I.fromarray(m.astype(np.uint8) * 255)
                                   .resize((W, H), _I.NEAREST)) > 127
                out[f_idx] |= m
    # Never lose a detection the propagation missed.
    out |= seeds.astype(bool)
    print(f"[sam-video] completed: empty frames "
          f"{100.0 * (out.reshape(n_frames, -1).sum(1) == 0).mean():.1f}%, "
          f"mean area {out.mean():.4f}", flush=True)
    return out
