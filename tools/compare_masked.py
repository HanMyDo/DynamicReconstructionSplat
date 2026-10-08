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
    ap.add_argument("--lpips", action="store_true",
                    help="Also report LPIPS and SSIM on a CROP around the dynamic region. "
                         "WHY THIS MATTERS: PSNR punishes a MISPLACED object twice (wrong "
                         "where it is, wrong where it should be) and a MISSING one only once, "
                         "so a method that renders nothing where the mover is can OUTSCORE one "
                         "that renders it imperfectly. That is the failure mode this thesis is "
                         "about, so a PSNR-only verdict on dynamic content is not sufficient. "
                         "Follows eval_gaussian_head.py's convention: crop to the mask bbox "
                         "padded 8px, min 32px so VGG has support, because masking to black "
                         "creates edges the perceptual net reacts to, and masked SSIM is "
                         "unreliable from zero-padding bias.")
    ap.add_argument("--mask_erode", type=int, default=0,
                    help="Erode the dynamic mask by this many pixels before scoring. "
                         "WHY: our detector over-flags badly -- on synchronous2 at stride 12 "
                         "the mask covers 36%% of the frame while the moving people are maybe "
                         "5-10%%, so psnr_dynamic is dominated by STATIC content sitting inside "
                         "the mask. A method can then win the dynamic column by rendering walls "
                         "well while missing the people entirely. Eroding concentrates the "
                         "region on the mask's interior (the objects) and drops the halo. "
                         "Report the erosion radius with any number produced this way.")
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

    lp = None
    if args.lpips:
        import sys
        # Run as `python tools/compare_masked.py`, sys.path[0] is tools/, so the repo
        # root holding src/ is not importable. Add it.
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import torch
        from src.evaluation.metrics import compute_lpips, compute_ssim
        lp = {"lpips": [[], []], "ssim": [[], []]}

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
        if args.mask_erode > 0:
            k = 2 * args.mask_erode + 1
            m = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        frac.append(float(m.mean()))
        m3 = np.repeat(m[..., None], g.shape[2], axis=2) if g.ndim == 3 else m
        for k, mm in (("all", None), ("dyn", m3), ("stat", ~m3)):
            for j, x in enumerate((a, b)):
                v = psnr(x, g, mm)
                if v is not None:
                    rows[k][j].append(v)

        if lp is not None and m.any():
            ys, xs = np.where(m)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            pad = 8
            y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
            x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
            if (y1 - y0) >= 32 and (x1 - x0) >= 32:
                def t(im):
                    c = im[y0:y1, x0:x1]
                    if c.ndim == 2:
                        c = np.repeat(c[..., None], 3, axis=2)
                    c = c[..., :3][..., ::-1].copy()          # BGR (cv2) -> RGB
                    return torch.from_numpy(c).permute(2, 0, 1).float().div(255.).unsqueeze(0)
                gt_t = t(g)
                for j, x in enumerate((a, b)):
                    x_t = t(x)
                    lp["lpips"][j].append(compute_lpips(gt_t, x_t).mean().item())
                    lp["ssim"][j].append(compute_ssim(x_t, gt_t).mean().item())

    print(f"\n{'':10s}{args.a_label:>12s}{args.b_label:>12s}{'A - B':>10s}")
    for k, name in (("all", "overall"), ("dyn", "dynamic"), ("stat", "static")):
        a_m = float(np.mean(rows[k][0])); b_m = float(np.mean(rows[k][1]))
        print(f"{name:10s}{a_m:12.2f}{b_m:12.2f}{a_m - b_m:+10.2f}")
    if lp is not None and lp["lpips"][0]:
        print(f"\n{'(dyn crop)':10s}{args.a_label:>12s}{args.b_label:>12s}{'A - B':>10s}")
        for k, name, better in (("lpips", "LPIPS", "lower"), ("ssim", "SSIM", "higher")):
            a_m = float(np.mean(lp[k][0])); b_m = float(np.mean(lp[k][1]))
            print(f"{name:10s}{a_m:12.4f}{b_m:12.4f}{a_m - b_m:+10.4f}   ({better} is better)")
        print(f"           on {len(lp['lpips'][0])} frames with a crop >= 32px")

    print(f"\nmean dynamic pixel fraction {np.mean(frac):.3f}  "
          f"(both methods scored against the SAME masks"
          f"{', eroded %d px' % args.mask_erode if args.mask_erode else ''})")


if __name__ == "__main__":
    main()
