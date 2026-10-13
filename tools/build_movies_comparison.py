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
    ap.add_argument("--ours_offset", type=int, default=0,
                    help="source frame rendered by our k-th image = k + this. 0 for "
                         "--image_views 0 (view 0 of batch k IS frame k). For a MATCHED "
                         "leave-one-out comparison our eval must hold out the MIDDLE view, "
                         "--image_views 8 with --num_frames 16, because MoVieS holds out "
                         "frames//2 and interpolates from BOTH sides while view 0 would make "
                         "us extrapolate from one side. Then view 8 of batch k is frame "
                         "k + 8*stride, so pass --ours_offset $((8 * stride)).")
    ap.add_argument("--mask_dir", default=None,
                    help="optional <maskdir>/<seq>/masks, copied alongside so the "
                         "dynamic/static split can be scored. Using OUR masks for BOTH "
                         "methods is fair: the mask picks the evaluation region, it is "
                         "not an input, and MoVieS never sees it.")
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

    ours_paths = sorted(glob.glob(os.path.join(args.ours_images, "*.png")))
    # Map SOURCE FRAME -> our render by parsing the batch index out of the filename
    # (b{batch:04d}_v{view:02d}.png) rather than indexing positionally. Positional
    # indexing assumes our k-th image is window k, which is false whenever
    # --batch_stride > 1 (saved windows are 0, N, 2N, ...) or --image_batch_start > 0,
    # and it fails SILENTLY by comparing mismatched frames. With the index parsed,
    # --ours_offset is purely the view offset (8 * stride for --image_views 8).
    ours_by_frame, ours = {}, ours_paths
    for q in ours_paths:
        m = re.search(r"b(\d+)_v", os.path.basename(q))
        if m:
            ours_by_frame[int(m.group(1)) + args.ours_offset] = q
    if ours_by_frame:
        print(f"indexed {len(ours_by_frame)} of our renders by frame "
              f"({min(ours_by_frame)}..{max(ours_by_frame)})")
    else:
        print("WARNING: no b####_v## filenames -- falling back to positional indexing")
    rgb = sorted(glob.glob(os.path.join(args.bonn_rgb, "*.png")))
    masks = sorted(glob.glob(os.path.join(args.mask_dir, "*.png"))) if args.mask_dir else []
    for d in ("gt", "ours", "movies") + (("mask",) if masks else ()):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    n, skipped = 0, []
    for i, p in idxs:
        if ours_by_frame:
            src = ours_by_frame.get(i)
            if src is None or i >= len(rgb):
                skipped.append(i); continue
        else:
            j = i - args.ours_offset      # index INTO ours/ of the render of frame i
            if j < 0 or j >= len(ours) or i >= len(rgb):
                skipped.append(i); continue
            src = ours[j]
        shutil.copyfile(rgb[i], os.path.join(args.out, "gt",   f"{i:06d}.png"))
        shutil.copyfile(src,    os.path.join(args.out, "ours", f"{i:06d}.png"))
        shutil.copyfile(p,       os.path.join(args.out, "movies", f"{i:06d}.png"))
        if masks and i < len(masks):
            shutil.copyfile(masks[i], os.path.join(args.out, "mask", f"{i:06d}.png"))
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
