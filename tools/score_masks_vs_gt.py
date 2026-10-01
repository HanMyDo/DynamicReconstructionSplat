"""Score predicted dynamic masks against ground truth.

WHY THIS EXISTS. On Bonn there is no dynamic-object ground truth, so every mask
judgement in this project has been made by looking at red overlays and arguing
about whether a chair counts. That is how a whole round of mask work (SAM,
normalisation, completion) was evaluated indirectly through PSNR, which is a poor
instrument for it: PSNR rewards COVERAGE, so a mask can score well while being
wrong about which pixels move.

Dynamic Replica ships binary foreground masks, so precision and recall become
measurable. Read the numbers as:

  recall    how much of the moving object the detector finds. LOW recall is the
            expensive failure -- an unmasked mover ghosts across all V frames.
  precision how much of what it flags actually moves. LOW precision costs little
            in the render (a wrongly masked static pixel just renders own-frame)
            but wrecks the PLY, where it becomes a hole plus a bright speck.

That asymmetry is measured, not assumed -- see [[mask-round-sep24-sam-negative]].
"""
import argparse, glob, os
import numpy as np
from PIL import Image


def load_bin(path, size=None):
    im = Image.open(path).convert("L")
    if size is not None and im.size != size:
        # NEAREST: a mask is labels, not intensities, and bilinear would invent
        # half-dynamic pixels along every silhouette -- exactly where precision
        # and recall are actually decided.
        im = im.resize(size, Image.NEAREST)
    return np.asarray(im) > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="directory of predicted mask PNGs")
    ap.add_argument("--gt", required=True, help="directory of GT mask PNGs")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N frames")
    args = ap.parse_args()

    P = sorted(glob.glob(os.path.join(args.pred, "*.png")))
    G = sorted(glob.glob(os.path.join(args.gt, "*.png")))
    if not P or not G:
        raise SystemExit(f"pred={len(P)} gt={len(G)} -- one of the directories is empty")
    if len(P) != len(G):
        print(f"WARNING: {len(P)} predicted vs {len(G)} GT masks; pairing by sorted "
              f"order over the first {min(len(P), len(G))}")
    n = min(len(P), len(G))
    if args.limit:
        n = min(n, args.limit)

    gt_size = Image.open(G[0]).size
    tp = fp = fn = 0
    ious, precs, recs = [], [], []
    for i in range(n):
        g = load_bin(G[i])
        p = load_bin(P[i], size=gt_size)
        t = int((p & g).sum()); f_p = int((p & ~g).sum()); f_n = int((~p & g).sum())
        tp += t; fp += f_p; fn += f_n
        u = t + f_p + f_n
        if u:
            ious.append(t / u)
            precs.append(t / max(t + f_p, 1))
            recs.append(t / max(t + f_n, 1))

    # Pixel-pooled over the whole sequence, which is what the render actually sees;
    # the per-frame mean is also printed because one catastrophic frame is invisible
    # in the pooled number.
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"frames scored        {n}")
    print(f"pooled precision     {prec:.3f}")
    print(f"pooled recall        {rec:.3f}")
    print(f"pooled F1            {f1:.3f}")
    print(f"pooled IoU           {tp / max(tp + fp + fn, 1):.3f}")
    print(f"per-frame IoU        mean {np.mean(ious):.3f}  p10 {np.percentile(ious, 10):.3f}"
          f"  p90 {np.percentile(ious, 90):.3f}")
    print(f"per-frame precision  mean {np.mean(precs):.3f}  p10 {np.percentile(precs, 10):.3f}")
    print(f"per-frame recall     mean {np.mean(recs):.3f}  p10 {np.percentile(recs, 10):.3f}")
    print(f"predicted dyn pixels {100 * (tp + fp) / max(tp + fp + fn + 1, 1):.1f}% of union; "
          f"GT dyn pixels {100 * (tp + fn) / max(tp + fp + fn + 1, 1):.1f}%")


if __name__ == "__main__":
    main()
