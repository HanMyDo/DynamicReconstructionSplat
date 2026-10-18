"""Dense metric 3D scene flow between ARBITRARY frame pairs, via Track4World.

WHY NOT THE DEMO. `demo.py --mode 3d_efep` builds long-range tracks by CHAINING
consecutive flows, so point survival decays with the gap. Measured on our own
strided window (crowd, stride 8, so 0.27 s between consecutive window frames):
9.8% of points survive 1 step, 4.0% survive 4, and **0% survive 8** -- the gap
from the first frame to the held-out middle. There is simply no displacement
available for most of our Gaussians on that path.

WHAT THIS DOES INSTEAD. `model.infer_pair` returns `flow_3d` of shape
(T-1, 3, H, W): dense metric 3D scene flow for every pixel, t -> t+1. Feeding a
TWO-FRAME input [frame_i, frame_j] therefore yields the flow i -> j directly, with
no chaining, no survival decay and no index lookup -- a displacement for every
pixel by construction. That is exactly the per-Gaussian coverage the compositing
knobs failed to buy (see the Oct 16 saturation result), from a different direction.

WHAT TO CHECK IN THE OUTPUT. Scene flow excludes camera motion by definition, so
on a mostly-static scene the median |flow_3d| should be near zero with a heavy
tail where people move. A median comparable to the camera's own motion would mean
it is really optical flow lifted to 3D, and would need the pose removed first.

USAGE
  python tools/t4w_pair_flow.py --frames /tmp/t4w_crowd_s8_240 --pairs 0:8,4:8,15:8 \
      --out /tmp/t4w_crowd_s8_240/pairflow.npz
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_window(frames_dir, image_size=640):
    """Frames, resized exactly as demo.py does: fit image_size, floor to /64."""
    import cv2
    d = Path(os.path.expanduser(frames_dir))
    files = sorted([p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg")])
    if not files:
        raise SystemExit(f"no frames in {d}")
    imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files]
    h0, w0 = imgs[0].shape[:2]
    scale = min(image_size / h0, image_size / w0)
    H, W = int(h0 * scale) // 64 * 64, int(w0 * scale) // 64 * 64
    imgs = [cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR) for im in imgs]
    meta = {}
    mp = d / "frames.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    return np.stack(imgs), (H, W), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True,
                    help="a directory written by tools/export_window_frames.py")
    ap.add_argument("--pairs", default="0:8",
                    help="comma-separated src:dst indices into the exported window")
    ap.add_argument("--t4w_repo", default="~/Track4World")
    ap.add_argument("--ckpt", default=None, help="default: <repo>/checkpoints/track4world_da3.pth")
    ap.add_argument("--config", default=None, help="default: <repo>/track4world/config/eval/v1.json")
    ap.add_argument("--image_size", type=int, default=640)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--out", default=None, help="write an .npz of the flows")
    ap.add_argument("--inspect", action="store_true",
                    help="print every key and shape the model returns")
    args = ap.parse_args()

    repo = Path(os.path.expanduser(args.t4w_repo))
    sys.path.insert(0, str(repo))
    os.chdir(repo)                      # their imports assume the repo as cwd
    import torch

    imgs, (H, W), meta = load_window(args.frames, args.image_size)
    T = imgs.shape[0]
    print(f"{T} frames at {H}x{W} from {args.frames}")
    if meta:
        print(f"  {meta['sequence']} start={meta['start']} stride={meta['stride']}, "
              f"held out index {meta['held_out_t4w_index']} = frame {meta['held_out_frame']}")

    pairs = []
    for tok in args.pairs.split(","):
        i, j = tok.split(":")
        pairs.append((int(i), int(j)))
    for i, j in pairs:
        if not (0 <= i < T and 0 <= j < T):
            raise SystemExit(f"pair {i}:{j} outside the window (0..{T-1})")

    cfg = json.loads(Path(args.config or repo / "track4world/config/eval/v1.json").read_text())
    ckpt = Path(args.ckpt or repo / "checkpoints/track4world_da3.pth")

    from track4world.nets.model import Track4World
    print("building model ...")
    model = Track4World(**cfg["model"], seqlen=16, use_3d=True,
                        use_model="depthanythingv3")
    sd = torch.load(str(ckpt), map_location="cpu")
    model.load_pretrained_with_remap(sd)
    model.cuda()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.use_metric_scale = True       # metric flow, so magnitudes are metres
    print("model ready")

    out = {}
    with torch.no_grad():
        for i, j in pairs:
            pair = np.stack([imgs[i], imgs[j]])                   # [2,H,W,3]
            t = torch.from_numpy(pair).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                res = model.infer_pair(t, iters=args.iters, sw=None,
                                       is_training=False, tracking3d=True,
                                       force_projection=True)
            geo, motion = res[0], res[1]
            if args.inspect:
                for nm, dd in (("geometry res[0]", geo), ("motion res[1]", motion)):
                    print(f"\n--- {nm} ---")
                    for k, v in dd.items():
                        shp = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                        print(f"   {k:22s} {shp}")

            def hw3(x):
                # -> [H,W,3], whatever leading batch/time axes it arrived with.
                a = x.float().cpu().numpy()
                while a.ndim > 3:
                    a = a[0]
                if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
                    a = np.transpose(a, (1, 2, 0))
                return a

            f3 = hw3(motion["flow_3d"])
            cam = geo.get("camera_poses")
            cam_step = float("nan")
            if cam is not None:
                c = cam.float().cpu().numpy()
                while c.ndim > 3:
                    c = c[0]
                if c.shape[0] >= 2 and c.shape[-1] == 4:
                    cam_step = float(np.linalg.norm(c[1][:3, 3] - c[0][:3, 3]))

            # flow_3d is ABSOLUTE, not an offset: their flow_2d is documented as
            # "absolute optical flow" and the demo adds a pixel grid to convert
            # offsets into absolute coordinates. Measured here, |flow_3d| has a
            # median around 1 m and a max around 4 m on a room scene -- those are
            # POSITIONS. So the displacement is flow_3d minus the SOURCE position,
            # and which source (camera-frame `points` or `world_points`) has to
            # match flow_3d's own frame. Both are measured and the static floor
            # decides: scene flow excludes camera motion, so the correct pairing is
            # the one whose median sits near zero.
            print(f"\npair {i}->{j}   flow_3d {f3.shape}   camera step {cam_step:.4f}")
            print(f"   |flow_3d| p50 {np.percentile(np.linalg.norm(f3, axis=-1), 50):7.3f}"
                  f"  <- ~scene depth means these are POSITIONS, not offsets")
            best = None
            for src_key in ("points", "world_points"):
                if src_key not in geo:
                    continue
                sp = geo[src_key].float().cpu().numpy()
                while sp.ndim > 4:
                    sp = sp[0]
                s0 = sp[0] if sp.shape[0] <= max(i, j) else sp[i]
                if s0.shape[0] == 3 and s0.shape[-1] != 3:
                    s0 = np.transpose(s0, (1, 2, 0))
                if s0.shape[:2] != f3.shape[:2]:
                    print(f"   {src_key:14s} {s0.shape} vs flow {f3.shape}: skipped")
                    continue
                mag = np.linalg.norm(f3 - s0, axis=-1)
                med = float(np.median(mag)); p90 = float(np.percentile(mag, 90))
                print(f"   flow_3d - {src_key:14s} p50 {med:7.4f}  p90 {p90:7.4f}  "
                      f"max {mag.max():7.3f}  p90/p50 "
                      f"{p90 / med if med > 1e-9 else float(chr(110)+chr(97)+chr(110)):6.2f}")
                if best is None or med < best[1]:
                    best = (src_key, med, mag)
            if best is not None:
                print(f"   => smallest static floor: flow_3d - {best[0]} "
                      f"(median {best[1]:.4f} m vs camera step {cam_step:.4f} m)")
                out[f"dispmag_{i}_{j}"] = best[2].astype(np.float16)
            out[f"flow3d_{i}_{j}"] = f3.astype(np.float16)
            vc = motion.get("visconf_maps_e")
            if vc is not None:
                out[f"visconf_{i}_{j}"] = hw3(vc).astype(np.float16)

    if args.out:
        out["shape"] = np.array([H, W])
        out["pairs"] = np.array(pairs)
        np.savez_compressed(os.path.expanduser(args.out), **out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
