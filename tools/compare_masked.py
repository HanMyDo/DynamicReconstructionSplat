"""Score two methods overall AND split by dynamic/static, against the same masks.

WHY THE SPLIT MATTERS. An overall PSNR gap can come from two very different places:
the quality of the underlying reconstructor, or the handling of the moving object.
Only the second is what a dynamics mechanism controls. Splitting by mask separates
them, and on a frozen-reconstructor method it is the difference between "our
mechanism is worse" and "our reconstructor is older".

Using ONE mask set for BOTH methods is what makes psnr_dynamic comparable here --
the trap in [[mask-round-sep24-sam-negative]] was comparing dynamic PSNR across runs
with DIFFERENT masks, where the two numbers average over different pixels. The mask
is an evaluation region, not an input; neither method sees it.

Everything is resized DOWN to the smaller render, as compare_methods.py does, so the
comparison is not decided by rendering size.
"""
import argparse, os
from glob import glob
import cv2
import numpy as np


def load(src, panels=1):
    fs = sorted(sum((glob(os.path.join(src, e)) for e in ("*.png", "*.jpg")), []))
    out = []
    for f in fs:
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        if im.ndim == 3 and panels > 1:
            w = im.shape[1] // panels
            im = im[:, (panels - 1) * w: panels * w]
        out.append(im)
    return out


def psnr(a, b, m=None):
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    d = (a - b) ** 2
    if m is not None:
        if m.sum() < 16:
            return None
        d = d[m]
    mse = float(d.mean())
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--a", required=True); ap.add_argument("--a_label", default="A")
    ap.add_argument("--a_panels", type=int, default=1)
    ap.add_argument("--b", required=True); ap.add_argument("--b_label", default="B")
    ap.add_argument("--b_panels", type=int, default=1)
    ap.add_argument("--mask_dir", required=True)
    args = ap.parse_args()

    G = load(args.gt_dir); A = load(args.a, args.a_panels); B = load(args.b, args.b_panels)
    M = load(args.mask_dir)
    n = min(len(G), len(A), len(B), len(M))
    if n == 0:
        raise SystemExit(f"nothing to compare: gt={len(G)} a={len(A)} b={len(B)} mask={len(M)}")
    h = min(min(x.shape[0] for x in A), min(x.shape[0] for x in B))
    w = min(min(x.shape[1] for x in A), min(x.shape[1] for x in B))
    print(f"{args.a_label}: {len(A)} frames   {args.b_label}: {len(B)} frames   "
          f"scoring {n} at {w}x{h}")

    rows = {"all": [[], []], "dyn": [[], []], "stat": [[], []]}
    frac = []
    for i in range(n):
        g = cv2.resize(G[i], (w, h), interpolation=cv2.INTER_AREA)
        a = cv2.resize(A[i], (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(B[i], (w, h), interpolation=cv2.INTER_AREA)
        m = M[i]
        if m.ndim == 3:
            m = m[..., 0]
        # NEAREST: a mask is labels. Bilinear would invent half-dynamic pixels along
        # every silhouette, which is exactly where the two methods differ most.
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 127
        frac.append(float(m.mean()))
        m3 = np.repeat(m[..., None], g.shape[2], axis=2) if g.ndim == 3 else m
        for k, mm in (("all", None), ("dyn", m3), ("stat", ~m3)):
            for j, x in enumerate((a, b)):
                v = psnr(x, g, mm)
                if v is not None:
                    rows[k][j].append(v)

    print(f"\n{'':10s}{args.a_label:>12s}{args.b_label:>12s}{'A - B':>10s}")
    for k, name in (("all", "overall"), ("dyn", "dynamic"), ("stat", "static")):
        a_m = float(np.mean(rows[k][0])); b_m = float(np.mean(rows[k][1]))
        print(f"{name:10s}{a_m:12.2f}{b_m:12.2f}{a_m - b_m:+10.2f}")
    print(f"\nmean dynamic pixel fraction {np.mean(frac):.3f}  "
          f"(both methods scored against the SAME masks)")


if __name__ == "__main__":
    main()
