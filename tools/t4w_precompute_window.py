"""Cache Track4World pairwise 3D displacement for one eval window.

WHAT IT PRODUCES. For every ordered pair (i, j) of the window's 16 frames, the dense
per-pixel displacement of frame i's content toward frame j, resampled onto OUR
448x448 Gaussian grid. Our Gaussians are one per pixel in raster order (the PLY
reports 3,211,264 for 16 frames = 16 x 448^2), so entry [i, j, y, x] is the
displacement to apply to the Gaussian of frame i at pixel (y, x) when rendering j.

WHY THE RAW FIELD IS NOT ENOUGH. `metric_scale` is estimated PER PAIR (measured
2.887 vs 2.994 on two pairs of the same window), and Track4World's world frame is
its own. So the displacement cannot be applied to our Gaussians directly: it has to
be mapped into our geometry's scale and rotation. That fit needs OUR positions,
which only exist inside the eval, so this tool also stores a SUBSAMPLED copy of
Track4World's `world_points[src]` per pair -- enough to fit a Sim(3) against our
means at eval time, at 1/16 the storage of the full map.

GEOMETRY NOTES
  - Track4World runs at 448x640 (640x480 fit to --image_size 640, floored to /64);
    ours is 448x448, the "448-square squash" of the same full frame. Both cover the
    same field of view, so a horizontal resize aligns them.
  - `flow_3d` is an ABSOLUTE position, not an offset (|flow_3d| p50 ~ scene depth),
    and sits closest to `world_points[src]`, so displacement = flow_3d - world_points[src].
  - The 3d_efep track path is NOT used: point survival on our strided windows is 0%
    at the gap from the first frame to the held-out middle.

USAGE
  python tools/t4w_precompute_window.py --frames /tmp/t4w_crowd_s8_240 \
      --out /data/hanmydo/t4w_cache/crowd_s8_240.npz --num_frames 16
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

GRID = 448          # our reconstruction grid; see --image_size in eval_gaussian_head
SUB = 4             # subsample factor for the stored world_points (Sim(3) fit only)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="dir from export_window_frames.py")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--num_frames", type=int, default=16,
                    help="window length; the export carries one extra trailing frame")
    ap.add_argument("--t4w_repo", default="~/Track4World")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--image_size", type=int, default=640)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--targets", default=None,
                    help="comma-separated target frames; default all (needed because "
                         "--eval_loo renders EVERY view leave-one-out)")
    args = ap.parse_args()

    repo = Path(os.path.expanduser(args.t4w_repo))
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import cv2
    import torch

    d = Path(os.path.expanduser(args.frames))
    files = sorted([p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg")])
    meta = json.loads((d / "frames.json").read_text()) if (d / "frames.json").exists() else {}
    imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files]
    h0, w0 = imgs[0].shape[:2]
    scale = min(args.image_size / h0, args.image_size / w0)
    H, W = int(h0 * scale) // 64 * 64, int(w0 * scale) // 64 * 64
    imgs = [cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR) for im in imgs]
    T = min(args.num_frames, len(imgs))
    print(f"{len(imgs)} frames at {H}x{W}; using the first {T} as the window")
    if meta:
        print(f"  {meta.get('sequence')} start={meta.get('start')} "
              f"stride={meta.get('stride')} held_out_idx={meta.get('held_out_t4w_index')}")

    targets = (list(range(T)) if args.targets is None
               else [int(x) for x in args.targets.split(",")])

    cfg = json.loads(Path(args.config or repo / "track4world/config/eval/v1.json").read_text())
    ckpt = Path(args.ckpt or repo / "checkpoints/track4world_da3.pth")
    from track4world.nets.model import Track4World
    print("building model ...")
    model = Track4World(**cfg["model"], seqlen=16, use_3d=True, use_model="depthanythingv3")
    model.load_pretrained_with_remap(torch.load(str(ckpt), map_location="cpu"))
    model.cuda()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.use_metric_scale = True
    print("model ready")

    disp = np.zeros((T, T, GRID, GRID, 3), dtype=np.float16)
    valid = np.zeros((T, T), dtype=bool)
    sg = GRID // SUB
    wsrc = np.zeros((T, T, sg, sg, 3), dtype=np.float16)
    mscale = np.full((T, T), np.nan, dtype=np.float32)

    def to_hw3(x):
        a = x.float().cpu().numpy()
        while a.ndim > 3:
            a = a[0]
        if a.shape[0] == 3 and a.shape[-1] != 3:
            a = np.transpose(a, (1, 2, 0))
        return a

    n_done = 0
    with torch.no_grad():
        for j in targets:
            for i in range(T):
                if i == j:
                    continue
                pair = np.stack([imgs[i], imgs[j]])
                t = torch.from_numpy(pair).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    res = model.infer_pair(t, iters=args.iters, sw=None,
                                           is_training=False, tracking3d=True,
                                           force_projection=True)
                geo, mot = res[0], res[1]
                f3 = to_hw3(mot["flow_3d"])                     # absolute target position
                wp = geo["world_points"].float().cpu().numpy()
                while wp.ndim > 4:
                    wp = wp[0]
                if wp.shape[-1] != 3 and wp.shape[1] == 3:
                    wp = np.transpose(wp, (0, 2, 3, 1))
                w0_ = wp[0]                                     # source frame, world
                dd = (f3 - w0_).astype(np.float32)              # [H,W,3] in T4W units
                # Resample onto OUR grid. Both are full-frame resizes of the same
                # 640x480, so this is a pure rescale -- no crop offset to account for.
                dd = cv2.resize(dd, (GRID, GRID), interpolation=cv2.INTER_LINEAR)
                ws = cv2.resize(w0_.astype(np.float32), (sg, sg),
                                interpolation=cv2.INTER_NEAREST)
                disp[i, j] = dd.astype(np.float16)
                wsrc[i, j] = ws.astype(np.float16)
                valid[i, j] = True
                try:
                    mscale[i, j] = float(geo["metric_scale"].float().cpu().numpy().reshape(-1)[0])
                except Exception:
                    pass
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  {n_done} pairs; last {i}->{j} "
                          f"median |disp| {np.linalg.norm(dd, axis=-1).mean():.3f}", flush=True)

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out, disp=disp, wsrc=wsrc, valid=valid, metric_scale=mscale,
        grid=np.array([GRID, GRID]), sub=np.array([SUB]),
        t4w_shape=np.array([H, W]),
        meta=np.array([json.dumps(meta)], dtype=object),
    )
    print(f"\n{n_done} pairs -> {out}")
    print(f"metric_scale over pairs: min {np.nanmin(mscale):.4f} "
          f"max {np.nanmax(mscale):.4f}  (per-pair, hence the Sim(3) fit at eval time)")


if __name__ == "__main__":
    main()
