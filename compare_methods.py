"""Compare two reconstruction methods on the same frames, fairly.

WHY THIS EXISTS. Our PSNR comes from metrics.json at our eval resolution;
StreamSplat's comes from an mp4 at theirs. PSNR is strongly resolution-dependent,
so those two numbers cannot be put in one table -- the comparison would be decided
by rendering size rather than reconstruction quality. This scores both with the
SAME code at the SAME resolution against the SAME ground truth, and emits the
side-by-side figure at the same time.

Each side may be a video, a directory of predicted frames, or a directory of
GT|pred panels (as our eval writes) -- `--a_panels 2` takes the right half.

Everything is resized DOWN to the smaller of the two renders. Upsampling the
smaller one would invent detail it never produced and flatter it.

⚠️ A matched resolution does not make a matched PROTOCOL. Reconstructing a view
from frames 0.1 s apart is a different problem from reconstructing it from a 2 s
window; compare cells with the same temporal baseline, not the best cell of each.

Usage:
    python compare_methods.py --gt_dir .../rgb \
        --a ours_dir --a_panels 2 --a_label "ours" \
        --b render_video_0.mp4 --b_label "StreamSplat" \
        --out cmp_methods
"""
import argparse
import os
from glob import glob

import cv2
import numpy as np


def load_frames(src, panels=1, limit=None):
    """-> list of BGR arrays, from a video or a directory of images."""
    out = []
    if os.path.isdir(src):
        files = sorted(sum((glob(os.path.join(src, e)) for e in ("*.png", "*.jpg")), []))
        for f in files[:limit]:
            im = cv2.imread(f)
            if panels > 1:                      # GT|pred panels -> take the LAST
                w = im.shape[1] // panels
                im = im[:, (panels - 1) * w: panels * w]
            out.append(im)
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {src}")
        while True:
            ok, fr = cap.read()
            if not ok or (limit and len(out) >= limit):
                break
            out.append(fr)
        cap.release()
    if not out:
        raise SystemExit(f"no frames from {src}")
    return out


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))


def label(im, text):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (12 + 9 * len(text), 22), (0, 0, 0), -1)
    cv2.putText(im, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--a", required=True); ap.add_argument("--a_label", default="A")
    ap.add_argument("--a_panels", type=int, default=1)
    ap.add_argument("--a_offset", type=int, default=0)
    ap.add_argument("--b", required=True); ap.add_argument("--b_label", default="B")
    ap.add_argument("--b_panels", type=int, default=1)
    ap.add_argument("--b_offset", type=int, default=0)
    ap.add_argument("--out", default=None, help="write GT|A|B figures here")
    ap.add_argument("--fig_stride", type=int, default=20)
    args = ap.parse_args()

    gt_files = sorted(sum((glob(os.path.join(args.gt_dir, e)) for e in ("*.png", "*.jpg")), []))
    A = load_frames(args.a, args.a_panels)
    B = load_frames(args.b, args.b_panels)

    # common size = the smaller render; never upsample a method onto its rival
    hA, wA = A[0].shape[:2]
    hB, wB = B[0].shape[:2]
    H, W = min(hA, hB), min(wA, wB)
    print(f"{args.a_label}: {len(A)} frames at {wA}x{hA}")
    print(f"{args.b_label}: {len(B)} frames at {wB}x{hB}")
    print(f"scoring both at {W}x{H}\n")

    rs = lambda im: im if im.shape[:2] == (H, W) else cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    sa, sb, n = [], [], 0
    for k in range(min(len(A), len(B))):
        ga, gb = args.a_offset + k, args.b_offset + k
        if max(ga, gb) >= len(gt_files):
            break
        gt_a, gt_b = rs(cv2.imread(gt_files[ga])), rs(cv2.imread(gt_files[gb]))
        fa, fb = rs(A[k]), rs(B[k])
        sa.append(psnr(fa, gt_a))
        sb.append(psnr(fb, gt_b))
        if args.out and k % args.fig_stride == 0:
            strip = np.hstack([label(gt_a, "ground truth"),
                               label(fa, args.a_label), label(fb, args.b_label)])
            cv2.imwrite(os.path.join(args.out, f"f{k:05d}.png"), strip)
        n += 1

    sa, sb = np.array(sa), np.array(sb)
    print(f"{n} frames compared        {args.a_label:>14s} {args.b_label:>14s} {'A - B':>10s}")
    for lbl, f in (("mean", np.mean), ("median", np.median),
                   ("p10", lambda x: np.percentile(x, 10)), ("min", np.min)):
        va, vb = f(sa), f(sb)
        print(f"  {lbl:22s} {va:14.2f} {vb:14.2f} {va - vb:+10.2f}")
    print(f"\n  {args.a_label} beats {args.b_label} on {100 * (sa > sb).mean():.0f}% of frames")
    if args.out:
        print(f"  figures -> {args.out}")
    print("\nNOTE: a matched resolution is not a matched protocol -- compare only "
          "cells with the same temporal baseline.")


if __name__ == "__main__":
    main()
