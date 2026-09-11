"""Per-frame PSNR between the panels of a comparison figure.

WHY. metrics.json reports a MEAN over hundreds of frames, in which a localized
catastrophe and a uniform mild gain are indistinguishable -- measured here: an
artefact that turned ~1/3 of frames near-black cost about 0.2 dB, and a
checkpoint that erased the moving person WON lpips by 0.042. A distribution
answers what a mean cannot: does B beat A on most frames, and what does B's
worst decile look like?

Costs nothing and needs no rerun: the comparison figures already contain
ground truth and both renders side by side, at full resolution.

Usage:
    python scan_frame_psnr.py cmp_balloon [--top 12]
"""
import argparse
import os

import numpy as np
from PIL import Image


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.dir) if n.endswith(".png"))
    if not names:
        raise SystemExit(f"no png in {args.dir}")

    pa, pb = [], []
    for n in names:
        im = np.asarray(Image.open(os.path.join(args.dir, n)).convert("RGB"), dtype=np.float32) / 255.0
        w = im.shape[1] // 3
        gt, a, b = im[:, :w], im[:, w:2 * w], im[:, 2 * w:3 * w]
        pa.append(psnr(gt, a))
        pb.append(psnr(gt, b))
    pa, pb = np.array(pa), np.array(pb)
    d = pb - pa

    print(f"{len(names)} frames                     {'run A':>10s} {'run B':>10s} {'B - A':>10s}")
    for lbl, q in (("mean", None), ("median", 50), ("worst decile (p10)", 10), ("worst frame", 0)):
        fa = pa.mean() if q is None else (pa.min() if q == 0 else np.percentile(pa, q))
        fb = pb.mean() if q is None else (pb.min() if q == 0 else np.percentile(pb, q))
        print(f"  {lbl:28s} {fa:10.2f} {fb:10.2f} {fb - fa:+10.2f}")
    print(f"\n  B beats A on {100.0 * (d > 0).mean():.0f}% of frames "
          f"({int((d > 0).sum())}/{len(d)})")
    print(f"  B is >1 dB WORSE on {100.0 * (d < -1).mean():.0f}% of frames "
          f"({int((d < -1).sum())})")

    for lbl, order in (("worst", np.argsort(d)[:args.top]),
                       ("BEST", np.argsort(-d)[:args.top])):
        print(f"\n{lbl} {len(order)} frames for B:")
        print(f"  {'frame':18s} {'A':>7s} {'B':>7s} {'B-A':>8s}")
        for i in order:
            print(f"  {names[i]:18s} {pa[i]:7.2f} {pb[i]:7.2f} {d[i]:+8.2f}")
    # ready to paste into a tar for figure selection
    best = [names[i] for i in np.argsort(-d)[:args.top]]
    print("\nbest frames, space separated (for tar/cp):")
    print(" ".join(best))


if __name__ == "__main__":
    main()
