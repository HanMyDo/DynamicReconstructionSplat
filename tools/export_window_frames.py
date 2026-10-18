"""Export one eval window as a %06d.png sequence Track4World can read.

WHY THIS EXISTS. Track4World's demo takes `--mp4_path` and loads it with
cv2.VideoCapture. Re-encoding our frames to H.264 would put compression
artifacts into a geometry/flow estimator, which is the one place we cannot
afford them. But cv2.VideoCapture also accepts a printf-style image sequence
("dir/%06d.png"), so symlinking a window's frames under sequential names feeds
it the ORIGINAL PNGs losslessly and costs no disk.

WHY THE SIDECAR MATTERS MORE THAN THE SYMLINKS. Track4World returns per-pixel
trajectories indexed by its own 0..T-1 frame order. To use them we have to know
which dataset frame each index was, because our Gaussians are indexed by source
frame and the eval window is strided (frame = start + k*stride). Getting that
mapping wrong is the same class of error as the --ours_offset bug that silently
produced a whole table of unmatched frames, so the mapping is written to disk
next to the symlinks rather than recomputed by whoever reads the output.

USAGE
  python tools/export_window_frames.py \
      --data_root ~/data/bonn/rgbd_bonn_dataset \
      --sequence rgbd_bonn_crowd --start 240 --num_frames 16 --stride 8 \
      --out /tmp/t4w_crowd_s8_240
  -> /tmp/t4w_crowd_s8_240/{000000.png,...}, frames.json, and the pattern on stdout
"""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True,
                    help="e.g. ~/data/bonn/rgbd_bonn_dataset or ~/data/tum")
    ap.add_argument("--sequence", required=True, help="e.g. rgbd_bonn_crowd")
    ap.add_argument("--start", type=int, default=0, help="first frame index of the window")
    ap.add_argument("--num_frames", type=int, default=16,
                    help="frames per window; 16 everywhere in this project")
    ap.add_argument("--stride", type=int, default=8,
                    help="temporal stride; the window spans (num_frames-1)*stride frames, "
                         "i.e. stride/2 seconds at 30 Hz")
    ap.add_argument("--out", required=True, help="directory to create")
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of symlink (needed if the reader cannot follow "
                         "links, or to move the window to another machine)")
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.data_root)) / args.sequence / "rgb"
    frames = sorted(list(root.glob("*.png")) + list(root.glob("*.jpg")))
    if not frames:
        raise SystemExit(f"no frames under {root}")

    idxs = [args.start + k * args.stride for k in range(args.num_frames)]
    if idxs[-1] >= len(frames):
        raise SystemExit(
            f"window runs past the end: needs frame {idxs[-1]} but {args.sequence} "
            f"has {len(frames)}. Max start for these settings is "
            f"{len(frames) - 1 - (args.num_frames - 1) * args.stride}.")

    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(idxs):
        src = frames[i].resolve()
        dst = out / f"{k:06d}{src.suffix}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            import shutil
            shutil.copyfile(src, dst)
        else:
            dst.symlink_to(src)

    suffix = frames[0].suffix
    meta = {
        "sequence": args.sequence,
        "data_root": str(root),
        "start": args.start,
        "num_frames": args.num_frames,
        "stride": args.stride,
        # THE mapping: track4world index k  ->  dataset frame index, and the file it
        # came from. The held-out frame of our eval protocol is index 8 of 16.
        "t4w_index_to_frame": idxs,
        "t4w_index_to_file": [frames[i].name for i in idxs],
        "held_out_t4w_index": args.num_frames // 2,
        "held_out_frame": idxs[args.num_frames // 2],
        "pattern": str(out / f"%06d{suffix}"),
    }
    (out / "frames.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {args.num_frames} frames to {out}")
    print(f"dataset frames {idxs[0]}..{idxs[-1]} (span {idxs[-1] - idxs[0]}, "
          f"{(idxs[-1] - idxs[0]) / 30.0:.2f} s at 30 Hz)")
    print(f"held out = t4w index {meta['held_out_t4w_index']} "
          f"= dataset frame {meta['held_out_frame']}")
    print(f"pattern: {meta['pattern']}")


if __name__ == "__main__":
    main()
