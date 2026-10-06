"""Line up our renders with MoVieS's on the SAME held-out frames.

compare_methods.py pairs files by position in sorted order. MoVieS renders a sparse
subset (one held-out frame per window, e.g. 32, 45, 58, ...) while our eval renders
every frame, so feeding both directories straight in would compare our frame 0
against their frame 32 and average the result into something meaningless.

This reads the frame indices out of MoVieS's filenames (which is why they are named
by source frame) and copies the matching ground-truth frame and our render into
parallel directories, so sorted order IS frame order on both sides.

Our eval writes GT|pred panels, one per frame, with `--image_views 0` -- view 0 of
batch i is frame i -- so our render for frame i is the i-th file in images/.
"""
import argparse, glob, os, re, shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies_dir", required=True, help="dir of MoVieS pred_<idx>_<stem>.png")
    ap.add_argument("--ours_images", required=True, help="our eval's images/ (GT|pred panels)")
    ap.add_argument("--bonn_rgb", required=True, help="the sequence's rgb/ directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preds = sorted(glob.glob(os.path.join(args.movies_dir, "pred_*.png")))
    idxs = []
    for p in preds:
        m = re.search(r"pred_(\d{6})_", os.path.basename(p))
        if m:
            idxs.append((int(m.group(1)), p))
    if not idxs:
        raise SystemExit("no pred_<6 digits>_<stem>.png found -- is this an older run?")
    idxs.sort()

    ours = sorted(glob.glob(os.path.join(args.ours_images, "*.png")))
    rgb = sorted(glob.glob(os.path.join(args.bonn_rgb, "*.png")))
    for d in ("gt", "ours", "movies"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    n, skipped = 0, []
    for i, p in idxs:
        if i >= len(ours) or i >= len(rgb):
            skipped.append(i); continue
        shutil.copyfile(rgb[i],  os.path.join(args.out, "gt",     f"{i:06d}.png"))
        shutil.copyfile(ours[i], os.path.join(args.out, "ours",   f"{i:06d}.png"))
        shutil.copyfile(p,       os.path.join(args.out, "movies", f"{i:06d}.png"))
        n += 1
    print(f"paired {n} frames: {[i for i, _ in idxs[:8]]}{' ...' if len(idxs) > 8 else ''}")
    if skipped:
        print(f"SKIPPED {len(skipped)} frames past the end of ours/rgb: {skipped[:8]}")
    print(f"\nnow:\n  python compare_methods.py --gt_dir {args.out}/gt \\\n"
          f"    --a {args.out}/ours --a_panels 2 --a_label ours \\\n"
          f"    --b {args.out}/movies --b_label MoVieS \\\n"
          f"    --out cmp_movies")


if __name__ == "__main__":
    main()
