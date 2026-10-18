"""Patch Track4World's demo.py to save the pixel -> point-index lookup.

THE PROBLEM. `3d_efep` writes trajectory_all_pointmap.npy as [T, N, 3]: the world
position of every tracked POINT at every frame. That is exactly the displacement
field we want -- point n's motion from frame i to j is traj[j,n] - traj[i,n], with
no interpolation, no trust radius and no unsupported population. But our Gaussians
are indexed by (source frame, pixel), and nothing in the output says which point
is which pixel.

The information exists: `current_uv_map` holds the uv of every point at the
current frame and is maintained across the loop. It is simply never saved. Only
frame 0's block is recoverable without it, from uv_coords_init's raster order;
every point added later at a disocclusion is unrecoverable.

WHAT THIS ADDS. One [T, H, W] int32 array, uv_index_map.npy, where entry (t,y,x)
is the point index visible at that pixel in frame t, or -1. At 448x640 that is
1.1 MB per frame against the trajectory array's ~20 MB, so it is free.

WHY A PATCH SCRIPT AND NOT A MANUAL EDIT. Track4World lives outside our repo, so
an edit there is invisible to our history -- the same way an uncommitted edit to
submit_final_battery.sh sat on the cluster for eleven days. This script is the
record, is idempotent, and can be re-run after any `git pull` of theirs.

USAGE
  python tools/patch_track4world_uvmap.py --repo ~/Track4World          # apply
  python tools/patch_track4world_uvmap.py --repo ~/Track4World --check  # verify
"""
import argparse
import os
from pathlib import Path

MARK = "DynamicReconstructionSplat uv_index_map patch"

# Anchor 1: the init of the tracking uv map, immediately before the loop.
A1_OLD = """    # Initialize current tracking UV (on GPU)
    current_uv_map = uv_coords_init.clone().float()"""
A1_NEW = """    # Initialize current tracking UV (on GPU)
    current_uv_map = uv_coords_init.clone().float()
    # [{mark}] one [H,W] int32 pixel -> point-index map per frame, PRE-ALLOCATED
    # and written by t_start. The loop has five `continue` paths, so appending to a
    # list silently drops frames and misaligns every later index against
    # trajectory_all_pointmap -- measured on cat.mp4: 85 maps for 87 frames. A
    # skipped frame keeps -1, which is the truthful answer for it.
    _uv_index_maps = torch.full((NumFrames, H, W), -1, dtype=torch.int32)
    # Frame 0 is the init branch: it writes its positions and then `continue`s, so
    # the loop never reaches the recorder for it. Its uv map is known analytically
    # -- current_uv_map starts as uv_coords_init, the raster grid -- so fill it here
    # rather than leaving the anchor frame unusable.
    _uv_index_maps[0] = torch.arange(H * W, dtype=torch.int32).reshape(H, W)""".replace("{mark}", MARK)

# Anchor 2: the end of the per-frame loop body. The `torch.cat` is inside
# `if uv_rest.shape[0] > 0:`, and the dedent to `logger.info` is where the loop
# ends -- so inserting at 8-space indent here runs once per frame, after both the
# tracked-point update and any newly added points.
A2_OLD = """            current_uv_map = torch.cat([current_uv_map, uv_rest], dim=0)

    logger.info(f"Saved {NumFrames} frames to {save_dir}")"""
A2_NEW = """            current_uv_map = torch.cat([current_uv_map, uv_rest], dim=0)

        # [{mark}] Record which point is visible at each pixel of this frame.
        # current_uv_map[n] is point n's uv AT THIS FRAME, NaN where the track was
        # lost, so finite-and-in-bounds is the test for "this point is live here".
        # On a collision the LATER point wins: it was added more recently and is
        # the live track for that pixel.
        _idx = torch.full((H, W), -1, dtype=torch.int32)
        _uv = current_uv_map.detach().cpu().float()
        _ok = torch.isfinite(_uv).all(dim=1)
        _uvl = torch.zeros(_uv.shape, dtype=torch.long)
        _uvl[_ok] = _uv[_ok].round().long()
        _ok = _ok & (_uvl[:, 0] >= 0) & (_uvl[:, 0] < W) \\
                  & (_uvl[:, 1] >= 0) & (_uvl[:, 1] < H)
        _pid = torch.arange(_uv.shape[0], dtype=torch.int32)
        _idx[_uvl[_ok, 1], _uvl[_ok, 0]] = _pid[_ok]
        _uv_index_maps[t_start] = _idx

    logger.info(f"Saved {NumFrames} frames to {save_dir}")"""
A2_NEW = A2_NEW.replace("{mark}", MARK)

# Anchor 3: save it beside the arrays it indexes into. The c2w line appears TWICE
# in their file -- once in save_efep and once in the 2d/ff path -- so it is located
# relative to the dyn_mask save, which only save_efep has.
A3_ANCHOR_BEFORE = "trajectory_all_pointmap_dyn_mask.npy"
A3_OLD = """    np.save(f"{save_dir}/c2w.npy", camera_poses.cpu().numpy())"""
A3_NEW = """    np.save(f"{save_dir}/c2w.npy", camera_poses.cpu().numpy())
    # [{mark}] [T,H,W] int32: point index visible at (t,y,x), or -1. Index-aligned
    # with trajectory_all_pointmap by construction; an all -1 frame is one the loop
    # skipped via `continue`.
    np.save(f"{save_dir}/uv_index_map.npy", _uv_index_maps.numpy())
    _cov = (_uv_index_maps >= 0).flatten(1).any(1)
    logger.info(
        f"[{mark}] uv_index_map {tuple(_uv_index_maps.shape)}; "
        f"{int(_cov.sum())}/{_cov.numel()} frames populated"
    )""".replace("{mark}", MARK)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="~/Track4World", help="Track4World checkout")
    ap.add_argument("--check", action="store_true",
                    help="report whether the patch is applied; change nothing")
    ap.add_argument("--repatch", action="store_true",
                    help="restore demo.py.orig first, then apply -- for updating an "
                         "already-patched checkout to a newer version of this patch")
    args = ap.parse_args()

    demo = Path(os.path.expanduser(args.repo)) / "demo.py"
    if not demo.is_file():
        raise SystemExit(f"not found: {demo}")
    s = demo.read_text()

    if args.check:
        n = s.count(MARK)
        print(f"{demo}: {'PATCHED' if n else 'NOT patched'} ({n} marker(s))")
        return

    if MARK in s:
        if not args.repatch:
            print(f"{demo}: already patched, nothing to do (--repatch to update)")
            return
        backup = demo.with_suffix(".py.orig")
        if not backup.exists():
            raise SystemExit(f"--repatch needs {backup}, which is missing")
        s = backup.read_text()
        demo.write_text(s)
        print(f"restored {demo} from {backup}")

    for name, old in (("init", A1_OLD), ("loop", A2_OLD)):
        if s.count(old) != 1:
            raise SystemExit(
                f"anchor '{name}' matched {s.count(old)} times, expected 1 -- "
                f"their demo.py has changed; re-read it before forcing this.")
    # The save anchor is deliberately NOT unique on its own; find the occurrence
    # inside save_efep by starting the search after that function's own marker.
    if s.count(A3_ANCHOR_BEFORE) != 1:
        raise SystemExit(f"'{A3_ANCHOR_BEFORE}' matched {s.count(A3_ANCHOR_BEFORE)} "
                         f"times, expected 1 -- re-read their demo.py.")
    a3_at = s.find(A3_OLD, s.index(A3_ANCHOR_BEFORE))
    if a3_at < 0:
        raise SystemExit("no c2w save found after the dyn_mask save -- re-read "
                         "their demo.py.")

    backup = demo.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(s)
        print(f"backup -> {backup}")

    # Splice the save by POSITION, so the identical line in the 2d/ff path is
    # untouched. Done first, because the earlier edits shift later offsets.
    s = s[:a3_at] + A3_NEW + s[a3_at + len(A3_OLD):]
    s = s.replace(A1_OLD, A1_NEW, 1)
    s = s.replace(A2_OLD, A2_NEW, 1)
    demo.write_text(s)

    import ast
    ast.parse(s)                      # refuse to leave their file unparseable
    print(f"{demo}: patched (3 anchors), syntax OK")
    print("re-run the demo; expect a new uv_index_map.npy of shape [T,H,W] int32")


if __name__ == "__main__":
    main()
