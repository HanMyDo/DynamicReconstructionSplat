"""Convert a window of a Bonn (or Dynamic Replica) sequence into MoVieS's .npz input.

MoVieS takes one npz per clip with:
    images    (F, 3, 294, 518) float32 in [0, 1]
    C2W       (F, 4, 4)        float32, camera-to-world
    fxfycxcy  (F, 4)           float32, intrinsics NORMALISED by (W, H, W, H)

Three conventions matter, all read off their shipped DAVIS clip:

1. C2W[0] IS THE IDENTITY. Poses are relative to the first frame, VGGT-style, so we
   pre-multiply by inv(C2W[0]). Absolute world coordinates would be out of
   distribution for a model trained on normalised cameras.
2. THE SCENE IS SCALE-NORMALISED. Their translations span 0.21 over 13 frames, which
   is camera motion as a FRACTION of scene depth, not metres. We divide translations
   by the clip's mean GT depth, so "1.0" means one average scene depth either way.
   Feeding raw metres would make the camera look stationary to the model.
3. INTRINSICS ARE NORMALISED, fx by W and fy by H separately (their fy/fx = 1.76 =
   518/294 is exactly the aspect ratio).

Poses come from the dataset's own ground truth, which is a DIFFERENCE from our method
worth stating: MoVieS receives externally supplied cameras, ours predicts its own.
"""
import argparse, glob, os
import numpy as np
from PIL import Image

# Bonn RGB-D Dynamic calibration (640x480). Override with --fx/--fy/--cx/--cy for
# another dataset; Dynamic Replica ships its own intrinsics per sequence.
BONN_K = dict(fx=542.822841, fy=542.576870, cx=315.593520, cy=237.756098, W=640, H=480)


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)


def load_tum_poses(path):
    """TUM format: timestamp tx ty tz qx qy qz qw. The pose is the CAMERA in the
    world frame, i.e. already camera-to-world."""
    ts, poses = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        v = [float(x) for x in line.split()]
        T = np.eye(4)
        T[:3, :3] = quat_to_R(*v[4:8])
        T[:3, 3] = v[1:4]
        ts.append(v[0]); poses.append(T)
    return np.array(ts), np.stack(poses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True, help="dir with rgb/, depth/, groundtruth.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0, help="index of the first frame")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--frames", type=int, default=13, help="MoVieS hardcodes 13")
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--height", type=int, default=294)
    for k, v in BONN_K.items():
        ap.add_argument(f"--{k}", type=float, default=v)
    args = ap.parse_args()

    rgb = sorted(glob.glob(os.path.join(args.seq_dir, "rgb", "*.png")))
    idx = [args.start + i * args.stride for i in range(args.frames)]
    if idx[-1] >= len(rgb):
        raise SystemExit(f"window runs past the sequence ({idx[-1]} >= {len(rgb)})")
    sel = [rgb[i] for i in idx]

    imgs = np.stack([
        np.asarray(Image.open(f).convert("RGB").resize((args.width, args.height), Image.BILINEAR),
                   dtype=np.float32).transpose(2, 0, 1) / 255.0
        for f in sel])                                              # (F, 3, H, W)

    # Poses: match each frame's timestamp (its filename) to the nearest GT pose.
    ts_all = np.array([float(os.path.basename(f)[:-4]) for f in sel])
    gt_ts, gt_T = load_tum_poses(os.path.join(args.seq_dir, "groundtruth.txt"))
    C2W = np.stack([gt_T[np.argmin(np.abs(gt_ts - t))] for t in ts_all])
    gap = max(abs(gt_ts[np.argmin(np.abs(gt_ts - t))] - t) for t in ts_all)
    C2W = np.linalg.inv(C2W[0])[None] @ C2W                         # frame 0 -> identity

    # Scale: metres -> fractions of scene depth, matching their normalisation.
    dep = sorted(glob.glob(os.path.join(args.seq_dir, "depth", "*.png")))
    if dep:
        d = np.asarray(Image.open(dep[min(idx[len(idx)//2], len(dep)-1)]), dtype=np.float32) / 5000.0
        mean_depth = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
    else:
        mean_depth = 1.0
        print("WARNING: no depth/ -- translations left in source units, scale may be off")
    C2W[:, :3, 3] /= max(mean_depth, 1e-6)

    fx = args.fx / args.W; fy = args.fy / args.H
    cx = args.cx / args.W; cy = args.cy / args.H
    fxfycxcy = np.tile(np.array([fx, fy, cx, cy], np.float32), (args.frames, 1))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, images=imgs.astype(np.float32),
             C2W=C2W.astype(np.float32), fxfycxcy=fxfycxcy)
    print(f"wrote {args.out}")
    print(f"  images   {imgs.shape} range [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"  C2W      {C2W.shape}  translation max {np.abs(C2W[:, :3, 3]).max():.4f} "
          f"(their DAVIS clip: 0.21)")
    print(f"  fxfycxcy {fxfycxcy[0]}  (their DAVIS clip: [2.509 4.422 0.506 0.506])")
    print(f"  mean scene depth {mean_depth:.3f} m; worst pose/frame timestamp gap {gap:.4f}s")
    print(f"  frames: {os.path.basename(sel[0])} .. {os.path.basename(sel[-1])}")


if __name__ == "__main__":
    main()
