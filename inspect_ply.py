"""Diagnose a Gaussian-splat PLY that a viewer renders wrongly or not at all.

WHY. A malformed splat PLY usually loads without complaint and then shows
nothing, or a white fog, or a few giant blobs -- the file is structurally valid
and the numbers inside are not. The failures worth separating:

  NaN/Inf         one bad value can make a viewer drop the whole cloud. Reached
                  here via scales.log() when a scale is 0, or means from a
                  degenerate depth.
  opacity ~ 0     stored as a LOGIT: -6 is invisible, 0 is half, +6 is solid.
                  A cloud of invisible Gaussians looks like an empty file.
  scale too large stored as LOG. exp(scale) is in world units, so a Gaussian
                  metres wide fills the view with one colour.
  scale too small below ~1e-4 world units nothing covers a pixel -> invisible.
  bbox blown out  a few Gaussians at 1e6 make a viewer frame the scene so far
                  out that the real content is a dot.

Usage:
    python inspect_ply.py path/to/gaussians.ply [more.ply ...]
"""
import sys

import numpy as np
from plyfile import PlyData


def report(path):
    ply = PlyData.read(path)
    v = ply["vertex"]
    n = len(v.data)
    names = set(v.data.dtype.names)
    print(f"\n=== {path}\n  {n} gaussians, {len(names)} attributes")

    xyz = np.stack([v[a] for a in ("x", "y", "z")], axis=1).astype(np.float64)
    bad_xyz = ~np.isfinite(xyz).all(axis=1)
    print(f"  position   NaN/Inf {int(bad_xyz.sum())}"
          f"   bbox {np.nanmin(xyz, 0).round(2)} .. {np.nanmax(xyz, 0).round(2)}")
    fin = xyz[np.isfinite(xyz).all(axis=1)]
    if len(fin):
        ext = fin.max(0) - fin.min(0)
        print(f"             extent {ext.round(2)}   p1..p99 "
              f"{np.percentile(fin, 1, axis=0).round(2)} .. {np.percentile(fin, 99, axis=0).round(2)}")

    if "opacity" in names:
        o = np.asarray(v["opacity"], dtype=np.float64)
        s = 1.0 / (1.0 + np.exp(-o))
        print(f"  opacity    logit  min {o.min():+.2f} med {np.median(o):+.2f} max {o.max():+.2f}"
              f"   NaN/Inf {int((~np.isfinite(o)).sum())}")
        print(f"             actual med {np.median(s):.3f}"
              f"   below 0.01: {100 * (s < 0.01).mean():.1f}%   above 0.5: {100 * (s > 0.5).mean():.1f}%")

    sc = [a for a in ("scale_0", "scale_1", "scale_2") if a in names]
    if sc:
        S = np.stack([np.asarray(v[a], dtype=np.float64) for a in sc], axis=1)
        E = np.exp(S)
        print(f"  scale      log    min {S.min():+.2f} med {np.median(S):+.2f} max {S.max():+.2f}"
              f"   NaN/Inf {int((~np.isfinite(S)).sum())}")
        print(f"             world  med {np.median(E):.5f}  p99 {np.percentile(E[np.isfinite(E)], 99):.5f}"
              f"  max {np.nanmax(E):.3f}")
        print(f"             over 0.5 world units: {100 * (E > 0.5).mean():.3f}%"
              f"   under 1e-4: {100 * (E < 1e-4).mean():.1f}%")

    fdc = [a for a in ("f_dc_0", "f_dc_1", "f_dc_2") if a in names]
    if fdc:
        F = np.stack([np.asarray(v[a], dtype=np.float64) for a in fdc], axis=1)
        rgb = np.clip(F * 0.28209479177387814 + 0.5, 0, 1)
        print(f"  colour     f_dc [{F.min():.2f}, {F.max():.2f}]"
              f"   implied rgb mean {rgb.mean(0).round(3)}")

    # verdict
    msgs = []
    if bad_xyz.any():
        msgs.append(f"{int(bad_xyz.sum())} gaussians have NaN/Inf positions -- many viewers drop the file")
    if "opacity" in names:
        s = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
        if (s < 0.01).mean() > 0.9:
            msgs.append("over 90% of gaussians are effectively transparent")
    if sc:
        E = np.exp(np.stack([np.asarray(v[a], dtype=np.float64) for a in sc], axis=1))
        if not np.isfinite(E).all():
            msgs.append("non-finite scales (scale 0 -> log -inf?)")
        if (E > 0.5).mean() > 0.001:
            msgs.append(f"{100 * (E > 0.5).mean():.2f}% of gaussians are over 0.5 world units wide")
        if (E < 1e-4).mean() > 0.5:
            msgs.append("over half the gaussians are smaller than 1e-4 world units -- may not cover a pixel")
    print("  VERDICT    " + ("ok, nothing obviously malformed" if not msgs else "; ".join(msgs)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        report(p)
