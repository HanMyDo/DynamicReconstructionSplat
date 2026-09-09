"""Put mask overlays from several precompute runs side by side, per frame.

WHY. Mask changes are judged by eye, and flipping between folders loses exactly
what matters: whether the SAME frame is covered differently. This lays the runs
out in one strip per frame, labelled, so a difference in coverage or a stray blob
is visible without remembering what the other folder looked like.

Frames common to every run are used, so runs over different spans still compare.

Usage:
    python compare_overlays.py out_dir LABEL=path/to/overlays [LABEL=... ...] \
        [--stride 10] [--limit 40]
"""
import argparse
import os

from PIL import Image, ImageDraw


def label(im, text, pad=6):
    d = ImageDraw.Draw(im)
    w = d.textlength(text) + 2 * pad
    d.rectangle([0, 0, w, 20], fill=(0, 0, 0))
    d.text((pad, 4), text, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+", help="LABEL=/path/to/overlays")
    ap.add_argument("--stride", type=int, default=10, help="use every Nth common frame")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"expected LABEL=path, got {spec!r}")
        lbl, path = spec.split("=", 1)
        names = {n for n in os.listdir(path) if n.endswith(".png")}
        if not names:
            raise SystemExit(f"no overlays in {path}")
        runs.append((lbl, path, names))

    common = sorted(set.intersection(*[n for _, _, n in runs]))
    if not common:
        raise SystemExit("no frame names common to all runs")
    picked = common[::max(1, args.stride)][:args.limit]
    os.makedirs(args.out, exist_ok=True)

    for n in picked:
        tiles = []
        for lbl, path, _ in runs:
            im = Image.open(os.path.join(path, n)).convert("RGB")
            tiles.append(label(im.copy(), lbl))
        w, h = tiles[0].size
        strip = Image.new("RGB", (w * len(tiles), h))
        for i, t in enumerate(tiles):
            strip.paste(t.resize((w, h)) if t.size != (w, h) else t, (i * w, 0))
        strip.save(os.path.join(args.out, n))

    print(f"{len(common)} frames common to {len(runs)} runs; wrote {len(picked)} to {args.out}")
    print(f"tar czf {args.out}.tgz {args.out}/")


if __name__ == "__main__":
    main()
