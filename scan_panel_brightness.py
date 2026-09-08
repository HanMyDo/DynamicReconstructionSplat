"""Find frames where one panel of a comparison figure is washed out or veiled.

WHY. A semi-transparent layer over the render (the "brown"/"grey veil" artefacts)
shifts a panel's mean luminance while leaving structure visible underneath, so it
is obvious to the eye and nearly invisible to PSNR averaged over a sequence. This
locates the affected frames so a probe can be aimed at them instead of guessing an
index, and says whether the problem is a handful of windows or the whole run.

Panels are assumed left-to-right as make_comparison_figure writes them:
    ground truth | run A | run B

Usage:
    python scan_panel_brightness.py cmp_synchronous2 [--panels 3] [--top 12]
"""
import argparse
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--panels", type=int, default=3)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.dir) if n.endswith(".png"))
    if not names:
        raise SystemExit(f"no png in {args.dir}")

    rows = []
    for n in names:
        im = np.asarray(Image.open(os.path.join(args.dir, n)).convert("L"), dtype=np.float32)
        w = im.shape[1] // args.panels
        means = [float(im[:, i * w:(i + 1) * w].mean()) for i in range(args.panels)]
        rows.append((n, means))

    arr = np.array([m for _, m in rows])                      # [N, panels]
    print(f"{len(rows)} frames, mean luminance per panel (0-255):")
    for i in range(args.panels):
        lbl = ["ground truth", "run A", "run B"][i] if args.panels == 3 else f"panel {i}"
        print(f"  {lbl:14s} mean={arr[:, i].mean():6.2f}  min={arr[:, i].min():6.2f}  max={arr[:, i].max():6.2f}")

    # A veil darkens (or washes out) one run relative to BOTH the GT and the other
    # run, so rank by how far B sits from GT compared with how far A sits from GT.
    if args.panels == 3:
        d_a = arr[:, 1] - arr[:, 0]
        d_b = arr[:, 2] - arr[:, 0]
        excess = np.abs(d_b) - np.abs(d_a)
        print(f"\nB-vs-GT deviation minus A-vs-GT deviation: "
              f"mean={excess.mean():+.2f}, worse in {(excess > 0).mean() * 100:.0f}% of frames")
        order = np.argsort(-excess)[:args.top]
        print(f"\nworst {len(order)} frames for run B:")
        print(f"  {'frame':18s} {'GT':>7s} {'A':>7s} {'B':>7s} {'B-GT':>8s} {'A-GT':>8s}")
        for i in order:
            n = rows[i][0]
            print(f"  {n:18s} {arr[i,0]:7.2f} {arr[i,1]:7.2f} {arr[i,2]:7.2f} "
                  f"{d_b[i]:+8.2f} {d_a[i]:+8.2f}")


if __name__ == "__main__":
    main()
