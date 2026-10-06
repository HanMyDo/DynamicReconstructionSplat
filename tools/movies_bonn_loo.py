"""Run MoVieS across a whole Bonn sequence under OUR leave-one-out protocol.

WHY NOT THEIR DEMO SCRIPT. infer_davis_nvs.py renders one clip in two modes that
were never meant to reproduce the input ("every timestep from camera 0", "the last
timestep from every camera"), so its output cannot be scored against ground truth.
Our eval holds a frame OUT of the window and renders it from the rest; to compare
fairly MoVieS has to do the same thing.

PROTOCOL, matched to ours as closely as their fixed 13-frame input allows:
  - slide a window of 13 frames at the given stride across the sequence
  - drop the MIDDLE frame from the input, leaving 12
  - render at that held-out frame's camera AND timestep
  - write pred/gt PNG pairs; scoring happens afterwards in compare_methods.py with
    OUR metric code, so PSNR/SSIM/LPIPS are computed identically for both methods

CAMERAS come from the dataset's ground truth, normalised the way their own
BaseDataset does it -- canonical only (inverse(C2W[0]) @ C2W), no scale change.
That is a real asymmetry worth stating: MoVieS is given cameras, ours predicts its
own, so a comparable score is achieved under a harder input condition.

Run from the MoVieS checkout so its `src` is importable:
    cd ~/MoVieS && python ~/DynamicReconstructionSplat/tools/movies_bonn_loo.py --seq_dir ...
"""
import sys; sys.path.append("extensions/vggt"); sys.path.append(".")
import argparse, glob, os
import numpy as np
import imageio.v2 as iio
import torch
from PIL import Image
from safetensors.torch import load_file
from src.options import opt_dict
from src.models import SplatRecon
from src.utils import *

BONN_K = dict(fx=542.822841, fy=542.576870, cx=315.593520, cy=237.756098, SRC_W=640, SRC_H=480)


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def load_tum_poses(path):
    ts, T = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        v = [float(x) for x in line.split()]
        M = np.eye(4); M[:3, :3] = quat_to_R(*v[4:8]); M[:3, 3] = v[1:4]
        ts.append(v[0]); T.append(M)
    return np.array(ts), np.stack(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4, help="frame spacing inside a window")
    ap.add_argument("--window_step", type=int, default=13, help="how far the window slides")
    ap.add_argument("--frames", type=int, default=13)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--max_windows", type=int, default=0, help="0 = whole sequence")
    args = ap.parse_args()

    rgb = sorted(glob.glob(os.path.join(args.seq_dir, "rgb", "*.png")))
    gt_ts, gt_T = load_tum_poses(os.path.join(args.seq_dir, "groundtruth.txt"))
    os.makedirs(args.out, exist_ok=True)

    opt = opt_dict["movies"]
    model = SplatRecon(opt, load_lpips=False)
    model.load_state_dict(load_file("resources/movies_ckpt.safetensors"), strict=True)
    model.eval().to("cuda")

    fx = BONN_K["fx"] / BONN_K["SRC_W"]; fy = BONN_K["fy"] / BONN_K["SRC_H"]
    cx = BONN_K["cx"] / BONN_K["SRC_W"]; cy = BONN_K["cy"] / BONN_K["SRC_H"]
    K1 = np.array([fx, fy, cx, cy], np.float32)

    span = (args.frames - 1) * args.stride
    starts = list(range(0, len(rgb) - span, args.window_step))
    if args.max_windows:
        starts = starts[:args.max_windows]
    print(f"{len(rgb)} frames -> {len(starts)} windows "
          f"({args.frames} frames, stride {args.stride}, step {args.window_step})")

    hold = args.frames // 2                     # middle frame is held out
    n = 0
    for w, s in enumerate(starts):
        idx = [s + i * args.stride for i in range(args.frames)]
        paths = [rgb[i] for i in idx]
        imgs = np.stack([
            np.asarray(Image.open(p).convert("RGB").resize((args.width, args.height),
                       Image.BILINEAR), np.float32).transpose(2, 0, 1) / 255.0
            for p in paths])
        tss = np.array([float(os.path.basename(p)[:-4]) for p in paths])
        C2W = np.stack([gt_T[np.argmin(np.abs(gt_ts - t))] for t in tss])
        C2W = np.linalg.inv(C2W[0])[None] @ C2W                 # canonical, their only normalisation

        keep = [i for i in range(args.frames) if i != hold]
        t_norm = np.linspace(0, 1, args.frames).astype(np.float32)

        to = lambda a: torch.from_numpy(np.asarray(a)).float().unsqueeze(0).to("cuda", torch.bfloat16)
        in_img = to(imgs[keep]); in_C2W = to(C2W[keep])
        in_K = to(np.tile(K1, (len(keep), 1))); in_t = to(t_norm[keep])
        out_t = to(t_norm[[hold]])

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bo, motions, motion_gs = model.backbone(
                in_img, in_C2W, in_K, in_t, out_t, frames_chunk_size=16)
            if motions is not None:   bo["offset"] = motions[:, 0, :, :3, ...]
            if motion_gs is not None: bo.update(motion_gs[0])
            r = model.gs_renderer.render(
                bo, in_C2W, in_K,
                to(C2W[[hold]]), to(K1[None]))

        pred = r["image"][0, 0].float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        gt = imgs[hold].transpose(1, 2, 0)
        # Name by the SOURCE frame, not the window counter. compare_methods.py pairs
        # files by sorted order, and the two methods hold out different frames at
        # different window strides -- positional matching would silently compare our
        # render of frame 130 against theirs of frame 91.
        stem = os.path.basename(paths[hold])[:-4]
        iio.imwrite(os.path.join(args.out, f"pred_{idx[hold]:06d}_{stem}.png"),
                    (pred * 255).astype(np.uint8))
        iio.imwrite(os.path.join(args.out, f"gt_{idx[hold]:06d}_{stem}.png"),
                    (gt * 255).astype(np.uint8))
        n += 1
        if w % 10 == 0:
            p = float(-10 * np.log10(((pred - gt) ** 2).mean() + 1e-12))
            print(f"  window {w:4d}/{len(starts)}  held-out frame {idx[hold]:5d}  psnr {p:.2f}",
                  flush=True)

    print(f"wrote {n} pred/gt pairs to {args.out}")
    print("held-out source frame indices:", [s + hold * args.stride for s in starts][:20],
          "..." if len(starts) > 20 else "")


if __name__ == "__main__":
    main()
