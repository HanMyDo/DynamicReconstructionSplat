"""Verify a Track4World 3d_efep output is usable as a displacement field.

THE CLAIM BEING TESTED. If `trajectory_all_pointmap.npy` really holds
WORLD-centric positions, then across frames a STATIC point barely moves while a
moving one does. Camera motion is already accounted for, so the displacement
`traj[j,n] - traj[i,n]` is scene motion only -- which is exactly what our
Gaussians need and what RAFT tracks lifted through predicted depth only
approximate.

So the decisive statistic is the DISTRIBUTION of per-point displacement
magnitude. It should be strongly bimodal: a spike near zero (background) and a
tail (the mover). If instead everything moves by roughly the camera's own
translation, the array is camera-centric and needs the c2w applied first -- a
difference that would silently produce garbage displacements if assumed wrong.

It also checks the join: uv_index_map must cover frame 0 almost completely
(frame 0's grid seeds every pixel) and a pixel's point must round-trip.

USAGE
  python tools/check_track4world_output.py --dir ~/Track4World/results/cat/3d_efep_output
"""
import argparse
import os

import numpy as np


def pct(x, q):
    return float(np.percentile(x, q)) if x.size else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="a 3d_efep_output directory")
    ap.add_argument("--src", type=int, default=0, help="source frame i")
    ap.add_argument("--dst", type=int, default=-1,
                    help="target frame j (-1 = last)")
    args = ap.parse_args()

    d = os.path.expanduser(args.dir)
    traj = np.load(os.path.join(d, "trajectory_all_pointmap.npy"), mmap_mode="r")
    c2w = np.load(os.path.join(d, "c2w.npy"))
    uvp = os.path.join(d, "uv_index_map.npy")
    uv = np.load(uvp, mmap_mode="r") if os.path.exists(uvp) else None

    T, N, _ = traj.shape
    i = args.src
    j = (T - 1) if args.dst < 0 else args.dst
    print(f"trajectory  {traj.shape} {traj.dtype}")
    print(f"c2w         {c2w.shape}")
    print(f"uv_index_map {uv.shape if uv is not None else 'MISSING'}"
          f"{'' if uv is not None else '  <- patch not applied'}")
    print()

    # --- the join -----------------------------------------------------------
    if uv is not None:
        # The uv maps MUST be index-aligned with the trajectory. An earlier version
        # of the patch appended to a list, and the loop's five `continue` paths
        # silently dropped frames: 85 maps for 87 frames, misaligning everything
        # after the first skip. Refuse rather than report numbers off by a frame.
        if uv.shape[0] != T:
            raise SystemExit(
                f"uv_index_map has {uv.shape[0]} frames but the trajectory has {T}. "
                f"They are not index-aligned, so any lookup would be wrong. Re-apply "
                f"the patch with --repatch and re-run the demo.")
        empty = [t for t in range(T) if not bool((np.asarray(uv[t]) >= 0).any())]
        if empty:
            print(f"⚠️  {len(empty)} frame(s) have NO points at all "
                  f"(loop skipped them): {empty[:12]}{' ...' if len(empty) > 12 else ''}")
        for t in (0, T // 2, T - 1):
            m = np.asarray(uv[t])
            print(f"frame {t:3d}: {100.0 * (m >= 0).mean():5.1f}% of pixels have a "
                  f"point, max index {int(m.max())}")
        m0 = np.asarray(uv[i])
        ys, xs = np.nonzero(m0 >= 0)
        if ys.size:
            k = ys.size // 2
            n = int(m0[ys[k], xs[k]])
            print(f"\nspot check: frame {i} pixel (y={ys[k]}, x={xs[k]}) -> point {n}")
            print(f"  position at {i}: {np.asarray(traj[i, n])}")
            print(f"  position at {j}: {np.asarray(traj[j, n])}")
        print()

    # --- camera motion, for scale comparison --------------------------------
    cam = c2w[:, :3, 3]
    cam_step = float(np.linalg.norm(cam[j] - cam[i]))
    print(f"camera centre moved {cam_step:.4f} between frames {i} and {j}")

    # --- the decisive statistic --------------------------------------------
    a = np.asarray(traj[i], dtype=np.float64)
    b = np.asarray(traj[j], dtype=np.float64)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    disp = np.linalg.norm(b[ok] - a[ok], axis=1)
    print(f"{ok.sum()} of {N} points live at BOTH frames "
          f"({100.0 * ok.mean():.1f}%)")
    print(f"displacement |traj[{j}]-traj[{i}]|  (same units as the point cloud)")
    for q in (10, 25, 50, 75, 90, 99):
        print(f"   p{q:<3d} {pct(disp, q):8.4f}")
    print(f"   max  {disp.max():8.4f}" if disp.size else "   max      nan")

    # Bimodality is the signature of a world-centric field: most of the scene is
    # static. A median comparable to the camera's own motion means the positions
    # are camera-centric and c2w has NOT been applied.
    med = pct(disp, 50)
    print()
    if cam_step > 1e-6 and med > 0.5 * cam_step:
        print(f"⚠️  median displacement {med:.4f} is {med / cam_step:.2f}x the camera "
              f"motion {cam_step:.4f}")
        print("    => looks CAMERA-centric: apply c2w before differencing, or the")
        print("       displacements will contain the camera's own motion.")
    else:
        print(f"✅ median displacement {med:.4f} << camera motion {cam_step:.4f}")
        print("    => consistent with WORLD-centric positions: static scene stays put,")
        print("       so traj[j,n]-traj[i,n] is scene motion and can be used directly.")
    frac_moving = float((disp > max(10 * med, 1e-6)).mean())
    print(f"    {100 * frac_moving:.1f}% of points move more than 10x the median "
          f"(the mover, if the scene is mostly static)")


if __name__ == "__main__":
    main()
