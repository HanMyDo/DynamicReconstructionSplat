#!/usr/bin/env python3
"""Rank Bonn sequences by how well they DECOUPLE object motion from camera motion.

WHY: Bonn is monocular, so temporal distance and viewpoint change are coupled — holding
out a frame that is far in time is also far in viewpoint. That confound produced a flat
result (stride-8 LOO degraded static as much as dynamic). Sequences where the CAMERA
moves little but OBJECTS move a lot break the coupling: a held-out frame then differs
mainly because things MOVED, which is what we want to measure.

Reads each sequence's groundtruth.txt (TUM format: timestamp tx ty tz qx qy qz qw) and
reports camera translation/rotation speed. If precomputed masks exist, also reports the
dynamic-pixel fraction, and ranks by dynamic-fraction / camera-speed.

USAGE:
  python rank_sequences.py --data_dir /path/to/rgbd_bonn_dataset \
      [--mask_dir output_dyn_masks_precomputed_cs16_r518_st3_fs0]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def quat_to_R(q):
    """TUM quaternion (qx,qy,qz,qw) -> 3x3 rotation."""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_motion(gt_path):
    """-> (duration_s, trans_speed_m_per_s, rot_speed_deg_per_s, path_len_m, n_poses)."""
    ts, xyz, quats = [], [], []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            ts.append(float(p[0]))
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
            quats.append([float(p[4]), float(p[5]), float(p[6]), float(p[7])])
    if len(ts) < 2:
        return None
    ts = np.asarray(ts)
    xyz = np.asarray(xyz)
    duration = ts[-1] - ts[0]
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    path_len = float(steps.sum())
    trans_speed = path_len / duration if duration > 0 else 0.0

    # mean angular speed: geodesic angle between consecutive rotations / dt
    angs = []
    for i in range(len(quats) - 1):
        dR = quat_to_R(quats[i]).T @ quat_to_R(quats[i + 1])
        c = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
        dt = ts[i + 1] - ts[i]
        if dt > 0:
            angs.append(np.degrees(np.arccos(c)) / dt)
    rot_speed = float(np.mean(angs)) if angs else 0.0
    return duration, trans_speed, rot_speed, path_len, len(ts)


def dyn_fraction(mask_dir, seq):
    meta = Path(mask_dir) / seq / "meta.json"
    if not meta.exists():
        return None
    try:
        with open(meta) as f:
            return float(json.load(f).get("dyn_fraction_mean"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir containing rgbd_bonn_<seq>/ folders")
    ap.add_argument("--mask_dir", default=None, help="precomputed-mask parent (for dyn fraction)")
    args = ap.parse_args()

    root = Path(args.data_dir)
    rows = []
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        gt = seq_dir / "groundtruth.txt"
        if not gt.exists():
            continue
        m = camera_motion(gt)
        if m is None:
            continue
        dur, tspd, rspd, plen, n = m
        frac = dyn_fraction(args.mask_dir, seq_dir.name) if args.mask_dir else None
        # decoupling score: much dynamic content per unit of camera motion. Higher = better.
        score = (frac / tspd) if (frac is not None and tspd > 1e-6) else None
        rows.append((seq_dir.name, dur, tspd, rspd, plen, n, frac, score))

    if not rows:
        print(f"No sequences with groundtruth.txt found under {root}")
        return

    # Rank by CAMERA SPEED (available for every sequence) — dyn%/score are shown as
    # refinement where masks exist. Ranking by score would bury every sequence whose
    # masks we have not precomputed yet, which is most of them.
    rows.sort(key=lambda r: r[2])

    print(f"\n{len(rows)} sequences | ranked by camera translation speed "
          f"(lower = better decoupled); dyn%/score shown where masks exist")
    print("LOW cam speed + HIGH dyn fraction = held-out frames differ because things MOVED,\n"
          "not because the camera flew away. Those are the sequences worth evaluating on.\n")
    print(f"{'sequence':<42}{'dur_s':>7}{'cam_m/s':>9}{'cam_deg/s':>10}{'path_m':>8}"
          f"{'frames':>7}{'dyn%':>7}{'score':>8}")
    print("-" * 100)
    for name, dur, tspd, rspd, plen, n, frac, score in rows:
        fs = f"{frac*100:6.1f}" if frac is not None else "     -"
        ss = f"{score:7.2f}" if score is not None else "      -"
        print(f"{name:<42}{dur:7.1f}{tspd:9.3f}{rspd:10.2f}{plen:8.2f}{n:7d}{fs}{ss}")


if __name__ == "__main__":
    main()
