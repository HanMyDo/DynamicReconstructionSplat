"""One PNG per sequence: GT | ours | NeoVerse | MoVieS, a row per time horizon.

Built from the paired directories `cmp_<tag>/<key>_<base>_s<S>/` that
score_baseline_table.sh already wrote, so the frames are exactly the ones the
reported numbers were computed on -- no re-rendering and no risk of showing a
different frame than was scored.

THREE THINGS THIS HAS TO GET RIGHT, each of which would otherwise be a silent
misrepresentation:

  - `ours/` holds GT|prediction PANELS (the eval writes two panels and
    compare_masked is told --a_panels 2), so only the RIGHT half is our render.
    Showing the whole panel would put GT where our output should be.
  - `<key>_neoverse_s<S>/movies/` holds NEOVERSE's frames: build_movies_comparison
    names its output subdirectory "movies" whichever baseline it paired. The
    baseline identity comes from the PARENT directory, not that name.
  - The three methods render at different sizes and aspect ratios -- ours 448x448,
    NeoVerse 448x336, MoVieS 518x392 -- so every cell is resized to one 4:3 box.
    For ours that UN-SQUASHES the 448-square grid back to the scene's true 4:3,
    which is what a reader should see.

USAGE
  python tools/make_baseline_figure.py --cmp cmp_bt --out figs_bt
"""
import argparse
import os
import pathlib
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CELL_W, CELL_H = 384, 288          # 4:3, the sequences' true aspect
PAD = 6
LABEL_H = 26
ROWLAB_W = 92


def put(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def cell(path, right_half=False):
    im = cv2.imread(str(path))
    if im is None:
        return np.full((CELL_H, CELL_W, 3), 40, np.uint8)
    if right_half:
        im = im[:, im.shape[1] // 2:]
    return cv2.resize(im, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmp", default="cmp_bt", help="directory of paired comparisons")
    ap.add_argument("--out", default="figs_bt", help="output directory for the PNGs")
    ap.add_argument("--strides", default=None,
                    help="comma-separated strides to include, e.g. '4' for the 2 s row "
                         "only (seconds = stride / 2). Default: all available.")
    ap.add_argument("--order", default="gt,ours,neoverse,movies",
                    help="column order, from gt,ours,neoverse,movies")
    ap.add_argument("--cmp_alt", default=None,
                    help="a SECOND pairing directory supplying one more column, for an arm "
                         "paired separately -- e.g. the plain AnySplat + vanilla VGGT "
                         "baseline in cmp_van alongside the adopted config in cmp_bt3. "
                         "Referenced as `alt` in --order.")
    ap.add_argument("--alt_label", default="alt", help="label for the --cmp_alt column")
    ap.add_argument("--ours_label", default="ours",
                    help="label for the `ours` column -- the pairing tool always writes "
                         "that directory name, whatever arm was paired into it.")
    ap.add_argument("--pick", type=float, default=0.5,
                    help="which paired frame to show, as a fraction through the "
                         "available ones (0.5 = middle)")
    args = ap.parse_args()

    root = Path(args.cmp)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # <key>_<base>_s<S>  -> which sequences, horizons and baselines exist
    rows = defaultdict(dict)
    pat = re.compile(r"^(.+)_(movies|neoverse)_s(\d+)$")
    for d in sorted(root.iterdir()):
        m = pat.match(d.name)
        if m and (d / "gt").is_dir():
            rows[m.group(1)].setdefault(int(m.group(3)), {})[m.group(2)] = d

    if not rows:
        raise SystemExit(f"no <key>_<base>_s<N> directories under {root}")

    for key, by_stride in sorted(rows.items()):
        strides = sorted(by_stride)
        if args.strides:
            want = {int(x) for x in args.strides.split(",")}
            strides = [S for S in strides if S in want]
            if not strides:
                print(f"  {key}: none of strides {sorted(want)} available "
                      f"({sorted(by_stride)}), skipped")
                continue
        panels = []
        for S in strides:
            srcs = by_stride[S]
            # Frames paired by BOTH baselines, so every cell in a row is the same
            # instant; a row built from different frames per method would be a lie.
            sets = []
            for b in ("neoverse", "movies"):
                if b in srcs:
                    sets.append({p.name for p in (srcs[b] / "gt").glob("*.png")})
            common = sorted(set.intersection(*sets)) if sets else []
            if not common:
                print(f"  {key} s{S}: no frames paired by both baselines, skipped")
                continue
            fname = common[min(int(len(common) * args.pick), len(common) - 1)]

            cells, labels = [], []
            base_for_gt = srcs.get("neoverse") or srcs.get("movies")
            for col in [c.strip() for c in args.order.split(",")]:
                if col == "gt":
                    cells.append(cell(base_for_gt / "gt" / fname))
                    labels.append("ground truth")
                elif col == "ours":
                    cells.append(cell(base_for_gt / "ours" / fname, True))
                    labels.append(args.ours_label)
                elif col == "alt":
                    # Same key/baseline/stride under the second pairing directory. Both
                    # pairings used --batch_stride 5, so the window starts and therefore
                    # the held-out frames coincide and `fname` means the same instant.
                    ad = None
                    if args.cmp_alt:
                        for b in ("neoverse", "movies"):
                            cand = pathlib.Path(args.cmp_alt) / f"{key}_{b}_s{S}" / "ours" / fname
                            if cand.is_file():
                                ad = cand
                                break
                    if ad is None:
                        print(f"  {key} s{S}: no alt frame {fname} under {args.cmp_alt}")
                        continue
                    cells.append(cell(ad, True))
                    labels.append(args.alt_label)
                elif col in srcs:
                    cells.append(cell(srcs[col] / "movies" / fname))
                    labels.append({"neoverse": "NeoVerse", "movies": "MoVieS"}[col])

            w = len(cells) * CELL_W + (len(cells) + 1) * PAD + ROWLAB_W
            row = np.full((CELL_H + 2 * PAD + LABEL_H, w, 3), 24, np.uint8)
            put(row, f"{S / 2:.0f}s", (8, CELL_H // 2 + LABEL_H), 0.8)
            put(row, f"frame {Path(fname).stem}", (8, CELL_H // 2 + LABEL_H + 26), 0.38,
                (170, 170, 170))
            for ci, (c, lab) in enumerate(zip(cells, labels)):
                x = ROWLAB_W + PAD + ci * (CELL_W + PAD)
                row[LABEL_H + PAD:LABEL_H + PAD + CELL_H, x:x + CELL_W] = c
                put(row, lab, (x + 4, LABEL_H - 7), 0.52)
            panels.append(row)

        if not panels:
            continue
        W = max(p.shape[1] for p in panels)
        canvas = np.full((sum(p.shape[0] for p in panels) + 34, W, 3), 24, np.uint8)
        put(canvas, f"{key}   --   leave-one-out, 16-frame window, held-out view",
            (10, 23), 0.6)
        y = 34
        for p in panels:
            canvas[y:y + p.shape[0], 0:p.shape[1]] = p
            y += p.shape[0]
        dst = out / f"{key}.png"
        cv2.imwrite(str(dst), canvas)
        print(f"{dst}  ({len(panels)} horizons, {canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
