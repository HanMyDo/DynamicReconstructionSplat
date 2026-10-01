"""Lay a Dynamic Replica scene out the way the Bonn loader expects.

WHY THIS AND NOT A LOADER. VideoFrameDataset needs only <root>/<name>/rgb/*.png;
rgb.txt and groundtruth.txt are read only when BOTH exist, and eval uses predicted
poses regardless. So writing the frames into that shape makes every existing
script work unchanged -- mask precompute, probe, battery, PLY export -- with no
new dataset class and no new flags.

STEREO: take ONE side. Dynamic Replica stores 00001-left.jpg and 00001-right.jpg
in the same folder; feeding both would look to VGGT like a camera that teleports
sideways every frame, which is not a baseline it can reason about.

The GT foreground masks come along into a sibling gt_masks/ directory. They are
not used by the pipeline -- they are there so the predicted dynamic masks can be
scored against truth, which is the thing Bonn cannot give us.
"""
import argparse, glob, os, re, shutil
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/data/dynrep/real"),
                    help="directory holding the scene folders")
    ap.add_argument("--scene", required=True, help="e.g. ignacio_waving")
    ap.add_argument("--split", default="test", help="subfolder under the scene")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--out", default=os.path.expanduser("~/data/bonn/rgbd_bonn_dataset"),
                    help="root the eval reads with --data_dir")
    ap.add_argument("--name", default=None, help="output sequence name (default dynrep_<scene>)")
    args = ap.parse_args()

    # TWO LAYOUTS. The `real` split puts both stereo sides in one folder
    # (<scene>/<split>/frames_rect/00001-left.jpg) with masks/ alongside. The
    # synthetic splits give each side its own scene directory
    # (<scene>_left/images/<scene>_left-0000.png) and add depths, flow and
    # trajectories. Detect rather than ask, so the same command works for both.
    real_base = os.path.join(args.src, args.scene, args.split)
    syn_base = os.path.join(args.src, args.scene)
    if os.path.isdir(os.path.join(real_base, "frames_rect")):
        base = real_base
        frames = sorted(glob.glob(os.path.join(base, "frames_rect", f"*-{args.side}.*")))
        mask_dir = os.path.join(base, "masks")
        layout = "real"
    elif os.path.isdir(os.path.join(syn_base, "images")):
        base = syn_base
        frames = sorted(glob.glob(os.path.join(base, "images", "*.png")))
        mask_dir = os.path.join(base, "masks")
        layout = "synthetic"
    else:
        raise SystemExit(f"no frames_rect/ or images/ under {args.src}/{args.scene}")
    if not frames:
        raise SystemExit(f"no frames found for {args.scene}")

    # Pair images to masks by FRAME INDEX, not by filename: the synthetic split
    # writes images as <scene>-0000.png and masks as <scene>_0000.png, and ships
    # one more image than mask. Matching on the trailing digits survives both.
    def fidx(path):
        stem = os.path.basename(path).rsplit(".", 1)[0]
        if stem.endswith(".geometric"):
            stem = stem[: -len(".geometric")]
        m = re.search(r"(\d+)$", stem)
        return m.group(1).lstrip("0") or "0" if m else None

    masks_by_idx = {}
    for mp in glob.glob(os.path.join(mask_dir, "*.png")):
        k = fidx(mp)
        if k is not None:
            masks_by_idx[k] = mp

    name = args.name or f"dynrep_{args.scene}"
    rgb_dir = os.path.join(args.out, name, "rgb")
    msk_dir = os.path.join(args.out, name, "gt_masks")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    n_m, bad = 0, []
    for i, f in enumerate(frames):
        dst = os.path.join(rgb_dir, f"{i:06d}.png")
        # PNG because the completeness guards count rgb/*.png, and a mask set that
        # is one frame short makes eval fall back to LIVE detection for the rest --
        # a silently different protocol rather than an error.
        try:
            Image.open(f).convert("RGB").save(dst)
            # VERIFY. A truncated write here does not fail now, it fails an hour
            # later inside the mask job with "image file is truncated" and no clue
            # which frame. Decoding it back costs milliseconds and names the file.
            with Image.open(dst) as chk:
                chk.load()
        except Exception as e:
            bad.append((os.path.basename(f), type(e).__name__, str(e)[:60]))
            if os.path.exists(dst):
                os.remove(dst)
            continue
        src_m = masks_by_idx.get(fidx(f))
        if src_m and os.path.exists(src_m):
            shutil.copyfile(src_m, os.path.join(msk_dir, f"{i:06d}.png"))
            n_m += 1

    if bad:
        print(f"SKIPPED {len(bad)} unreadable source frames:")
        for b in bad[:10]:
            print(f"   {b[0]}  {b[1]}: {b[2]}")
        print("Frames are renumbered contiguously, so the sequence stays usable -- "
              "but it now has a time gap where those frames were.")

    n_ok = len(glob.glob(os.path.join(rgb_dir, "*.png")))
    print(f"{name} [{layout}]: {n_ok} frames written (of {len(frames)} found) -> {rgb_dir}")
    print(f"{'':>{len(name)}}  {n_m} GT masks -> {msk_dir}")
    if n_m and n_m != len(frames):
        print(f"WARNING: {len(frames) - n_m} frames have no GT mask")
    w, h = Image.open(frames[0]).size
    print(f"source resolution {w}x{h}; --image_size takes H W and both must divide by 14 "
          f"(392 518 is native AnySplat density at ~4:3)")


if __name__ == "__main__":
    main()
