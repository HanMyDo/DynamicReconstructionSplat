"""Make an existing splat PLY viewable, without re-running anything.

WHY. Our exports are structurally valid but hard to view: median opacity ~0.03
(only ~1% of gaussians above 0.5) and median scale ~0.0015 world units in a
scene ~1.7 units across, with a quarter of them below 1e-4 -- sub-pixel at any
sane zoom. gsplat renders this correctly because a million tiny gaussians
accumulate to alpha ~0.97 at the eval camera, but a viewer with its own splat
sizing and exposure shows a faint haze and looks "broken".

Two independent operations, both reversible because the input is untouched:

  PRUNE   drop gaussians that cannot contribute -- below min_opacity, or smaller
          than min_scale. This is lossless in appearance and shrinks the file,
          which is also what makes it practical to download.
  BOOST   add to the opacity logit / log-scale of what remains. This CHANGES the
          model and is presentation only: say so if a boosted file is shown, and
          never use one to make a quality claim.

Usage:
    python prune_ply.py in.ply out.ply --min_opacity 0.02 --min_scale 1e-4
    python prune_ply.py in.ply out.ply --min_opacity 0.05 --opacity_boost 1.5 --scale_boost 0.7
"""
import argparse

import numpy as np
from plyfile import PlyData, PlyElement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--min_opacity", type=float, default=0.02,
                    help="drop gaussians whose sigmoid(opacity) is below this")
    ap.add_argument("--min_scale", type=float, default=1e-4,
                    help="drop gaussians whose largest exp(scale) is below this (sub-pixel)")
    ap.add_argument("--opacity_boost", type=float, default=0.0,
                    help="PRESENTATION ONLY: added to the opacity logit of survivors")
    ap.add_argument("--scale_boost", type=float, default=0.0,
                    help="PRESENTATION ONLY: added to the log-scale of survivors")
    args = ap.parse_args()

    ply = PlyData.read(args.src)
    v = ply["vertex"]
    data = v.data.copy()
    n0 = len(data)

    o = data["opacity"].astype(np.float64)
    sig = 1.0 / (1.0 + np.exp(-o))
    S = np.stack([data[f"scale_{i}"].astype(np.float64) for i in range(3)], axis=1)
    big = np.exp(S).max(axis=1)

    keep = np.isfinite(o) & np.isfinite(S).all(axis=1)
    keep &= sig >= args.min_opacity
    keep &= big >= args.min_scale
    data = data[keep]

    if args.opacity_boost:
        data["opacity"] = (data["opacity"].astype(np.float64) + args.opacity_boost).astype(np.float32)
    if args.scale_boost:
        for i in range(3):
            data[f"scale_{i}"] = (data[f"scale_{i}"].astype(np.float64) + args.scale_boost).astype(np.float32)

    PlyData([PlyElement.describe(data, "vertex")]).write(args.dst)
    s2 = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))
    print(f"{args.src} -> {args.dst}")
    print(f"  kept {len(data)}/{n0} ({100.0 * len(data) / max(n0, 1):.1f}%)")
    print(f"  opacity median {np.median(sig):.4f} -> {np.median(s2):.4f}")
    if args.opacity_boost or args.scale_boost:
        print("  NOTE: boosted file -- presentation only, not for quality claims")


if __name__ == "__main__":
    main()
