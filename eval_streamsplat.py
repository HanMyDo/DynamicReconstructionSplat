"""Per-frame metrics for a StreamSplat render video against the source frames.

WHY. splat_inference.py keeps only `results['pred_frames']` and writes them to an
mp4 -- the Gaussians are discarded, so there is nothing to feed into our own eval.
But the rendered frames survive in the video, and at their encoder setting the
compression sits around 40 dB while the signal being measured is around 19 dB, so
it contributes ~0.03 dB. Extracting from the video is therefore accurate enough
and avoids patching their code.

ALIGNMENT. With --frame_gap G they take every Gth frame as an anchor, reconstruct
from each consecutive PAIR, and render the G+1 frames spanning it; the shared
endpoint between consecutive pairs is skipped. So rendered frame k corresponds to
SOURCE frame k, and the intermediate frames were never given to the model.

⚠️ READ THE PROTOCOL BEFORE THE NUMBERS. At G=3 the two input frames are ~0.1 s
apart and the rendered frames sit at or beside them -- close to self-reprojection,
and a far easier task than reconstructing a target view from a 2 s window. These
numbers are only comparable to ours if ours is run at a matched baseline
(--num_frames 4 --frame_stride 1). Otherwise they compare different problems.

Usage:
    python eval_streamsplat.py --video render_video_0.mp4 \
        --gt_dir ~/data/bonn/rgbd_bonn_dataset/rgbd_bonn_balloon/rgb
"""
import argparse
import os
from glob import glob

import cv2
import numpy as np


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="render_video_N.mp4")
    ap.add_argument("--gt_dir", required=True, help="source frames (Bonn rgb/)")
    ap.add_argument("--offset", type=int, default=0,
                    help="source index of rendered frame 0 (0 for frame_gap runs)")
    ap.add_argument("--dump_dir", default=None, help="also write the rendered frames here")
    args = ap.parse_args()

    gt_files = sorted(sum((glob(os.path.join(args.gt_dir, e))
                           for e in ("*.png", "*.jpg")), []))
    if not gt_files:
        raise SystemExit(f"no frames in {args.gt_dir}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    scores, k = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gi = args.offset + k
        if gi >= len(gt_files):
            print(f"  ran out of source frames at rendered frame {k}")
            break
        gt = cv2.imread(gt_files[gi])
        # GT is resized to the render, never the reverse: upsampling the render
        # would invent detail it does not have and inflate the score.
        if gt.shape[:2] != frame.shape[:2]:
            gt = cv2.resize(gt, (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_AREA)
        scores.append(psnr(frame, gt))
        if args.dump_dir:
            cv2.imwrite(os.path.join(args.dump_dir, f"r{k:05d}.png"), frame)
        k += 1
    cap.release()

    s = np.array(scores)
    print(f"{len(s)} rendered frames vs {os.path.basename(args.gt_dir)}")
    print(f"  mean   {s.mean():6.2f} dB")
    print(f"  median {np.median(s):6.2f} dB")
    print(f"  p10    {np.percentile(s, 10):6.2f} dB")
    print(f"  min    {s.min():6.2f} dB   max {s.max():6.2f} dB")
    print("\nNOTE: comparable to ours ONLY at a matched temporal baseline; see the "
          "module docstring.")


if __name__ == "__main__":
    main()
