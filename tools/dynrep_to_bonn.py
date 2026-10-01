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
import argparse, glob, os, shutil
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

    base = os.path.join(args.src, args.scene, args.split)
    frames = sorted(glob.glob(os.path.join(base, "frames_rect", f"*-{args.side}.*")))
    if not frames:
        raise SystemExit(f"no {args.side} frames under {base}/frames_rect")

    name = args.name or f"dynrep_{args.scene}"
    rgb_dir = os.path.join(args.out, name, "rgb")
    msk_dir = os.path.join(args.out, name, "gt_masks")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    n_m = 0
    for i, f in enumerate(frames):
        stem = os.path.basename(f).rsplit(".", 1)[0]          # 00001-left
        # PNG because the completeness guards count rgb/*.png, and a mask set that
        # is one frame short makes eval fall back to LIVE detection for the rest --
        # a silently different protocol rather than an error.
        Image.open(f).convert("RGB").save(os.path.join(rgb_dir, f"{i:06d}.png"))
        src_m = os.path.join(base, "masks", f"{stem}.png")
        if os.path.exists(src_m):
            shutil.copyfile(src_m, os.path.join(msk_dir, f"{i:06d}.png"))
            n_m += 1

    print(f"{name}: {len(frames)} frames -> {rgb_dir}")
    print(f"{'':>{len(name)}}  {n_m} GT masks -> {msk_dir}")
    if n_m and n_m != len(frames):
        print(f"WARNING: {len(frames) - n_m} frames have no GT mask")
    w, h = Image.open(frames[0]).size
    print(f"source resolution {w}x{h}; --image_size takes H W and both must divide by 14 "
          f"(392 518 is native AnySplat density at ~4:3)")


if __name__ == "__main__":
    main()
