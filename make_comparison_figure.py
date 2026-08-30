"""Build GT | baseline | ours comparison frames from two eval runs.

WHY THIS EXISTS. eval writes one image per (batch, view) as GT|pred panels, per run.
A "vanilla vs ours" figure therefore has to be assembled ACROSS two runs, matching
filenames (identical batch/view = identical GT), which is exactly what this does.

The comparison that reads visually is NOT flow-vs-no-flow (+0.71 dB, ~12% of pixels)
but VANILLA vs FULL MODEL (+2.70 psnr / +1.44 dynamic): fine-tuning sharpens the
scene and scene flow puts the moving object in the right place. Under a long window
the baseline shows the object at several past positions at once (ghosting) while the
motion-compensated model collapses those onto its position at t.

Usage:
    python make_comparison_figure.py --a runA/images --b runB/images --out fig/
    ffmpeg -framerate 10 -pattern_type glob -i 'fig/*.png' -c:v libx264 -pix_fmt yuv420p out.mp4
"""
import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw


def split_panels(im, n):
    w, h = im.size
    pw = w // n
    return [im.crop((i * pw, 0, (i + 1) * pw, h)) for i in range(n)]


def label(im, text, pad=6):
    d = ImageDraw.Draw(im)
    tw = d.textlength(text) + 2 * pad
    d.rectangle([0, 0, tw, 20], fill=(0, 0, 0))
    d.text((pad, 4), text, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="images/ dir of run A (the baseline)")
    ap.add_argument("--b", required=True, help="images/ dir of run B (ours)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", type=int, default=2,
                    help="panels per saved image: 2 for GT|pred, 3 with --image_error_map")
    ap.add_argument("--label_a", default="VGGT baseline")
    ap.add_argument("--label_b", default="ours (fine-tuned + scene flow)")
    ap.add_argument("--crop", default=None,
                    help="x,y,w,h in the ORIGINAL panel, to zoom on the moving object")
    ap.add_argument("--no_labels", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = sorted(set(os.listdir(args.a)) & set(os.listdir(args.b)))
    names = [n for n in names if n.endswith(".png")]
    if not names:
        raise SystemExit("no filenames common to both runs -- were they rendered with "
                         "the same --image_batch_start / --image_views?")
    box = tuple(int(v) for v in args.crop.split(",")) if args.crop else None

    for n in names:
        pa = split_panels(Image.open(os.path.join(args.a, n)).convert("RGB"), args.panels)
        pb = split_panels(Image.open(os.path.join(args.b, n)).convert("RGB"), args.panels)
        cells = [pa[0], pa[1], pb[1]]                      # GT (shared), baseline, ours
        if box:
            x, y, w, h = box
            cells = [c.crop((x, y, x + w, y + h)) for c in cells]
        if not args.no_labels:
            cells = [label(c.copy(), t) for c, t in
                     zip(cells, ["ground truth", args.label_a, args.label_b])]
        w, h = cells[0].size
        out = Image.new("RGB", (w * 3, h))
        for i, c in enumerate(cells):
            out.paste(c, (i * w, 0))
        out.save(os.path.join(args.out, n))
    print(f"wrote {len(names)} frames to {args.out}")
    print("video:  ffmpeg -framerate 10 -pattern_type glob -i '"
          f"{args.out}/*.png' -c:v libx264 -pix_fmt yuv420p comparison.mp4")


if __name__ == "__main__":
    main()
