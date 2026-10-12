"""Run NeoVerse's reconstructor across a Bonn sequence under OUR leave-one-out protocol.

Mirrors tools/movies_bonn_loo.py so the same pairing and scoring tools work on the
output: a sliding 16-frame window, the MIDDLE frame held out, rendered at that
frame's camera and timestep, written as pred/gt pairs named by SOURCE frame index.

WHY NEOVERSE. It is the closest published competitor to our setting: feed-forward,
pose-free, monocular, general-domain, built on VGGT. (NoPo4D is not usable here --
it assumes a static camera rig and averages per-frame extrinsics across time, so a
moving camera collapses to a single averaged pose.)

WHAT WE COMPARE. The RECONSTRUCTION only. NeoVerse's full pipeline feeds the
reconstruction to a video diffusion model; comparing against that output would be
comparing against generation, which may hallucinate content never observed, and is
a different task. So we load `reconstructor.ckpt` alone (6 GB) and never touch the
75 GB Wan stack.

THREE PROTOCOL CHOICES, each a deviation worth stating:

1. `use_motion=True`. Their inference.py and app.py both pass FALSE, which gates
   `enable_motion` and `enable_dynamic_gs_attr` -- no velocities, no temporal
   attributes. That suits their use case (novel TRAJECTORY at OBSERVED timestamps)
   and would make a meaningless dynamic baseline. The released checkpoint does have
   the motion branch: with True it emits velocity_fwd/bwd and gs_fwd/bwd_attr.

2. The held-out frame is NOT passed in at all. `views` has an `is_target` flag that
   excludes a view from the reconstruction context, which looks like a built-in
   leave-one-out -- but the image would still be handed to the model, and whether
   its content is touched downstream is not obvious from the call site. We pass 15
   frames instead, so there is nothing to leak.

3. The held-out CAMERA is interpolated from its two neighbours (SLERP on rotation,
   midpoint on translation) in NeoVerse's own coordinate frame. A pose-free method
   predicts cameras only for the views it is given, so the held-out pose has to come
   from somewhere. The held-out frame is the MIDDLE of an evenly spaced window, so
   its neighbours bracket it symmetrically; on Bonn's slow-camera eval sequences
   this is accurate, and it avoids importing another method's poses into NeoVerse's
   frame and scale. ⚠️ This differs from the MoVieS runner, which is handed
   VGGT poses because MoVieS REQUIRES poses as input. State the difference.

Run (cluster, `neoverse` env, needs a GPU so use srun/sbatch):
  python tools/neoverse_bonn_loo.py --seq_dir ~/data/bonn/rgbd_bonn_dataset/rgbd_bonn_synchronous2 \
      --out ~/NeoVerse/out/loo_sync2_s12 --stride 12 --window_step 5
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


def slerp_pose(c2w_a, c2w_b, t=0.5):
    """Interpolate two 4x4 camera-to-world matrices: SLERP rotation, lerp translation.

    Nearest-neighbour or linear-matrix interpolation of a rotation does not produce a
    rotation; SLERP does. scipy ships it and is already a NeoVerse dependency.
    """
    from scipy.spatial.transform import Rotation, Slerp

    rots = Rotation.from_matrix(np.stack([c2w_a[:3, :3], c2w_b[:3, :3]]))
    r = Slerp([0.0, 1.0], rots)([t])[0]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.as_matrix()
    out[:3, 3] = (1.0 - t) * c2w_a[:3, 3] + t * c2w_b[:3, 3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True, help="Bonn sequence dir (contains rgb/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neoverse_dir", default=str(Path.home() / "NeoVerse"))
    ap.add_argument("--ckpt", default="models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--frames", type=int, default=16, help="window size INCLUDING the held-out frame")
    ap.add_argument("--stride", type=int, default=4, help="frame spacing inside a window")
    ap.add_argument("--window_step", type=int, default=5, help="how far the window slides")
    ap.add_argument("--max_windows", type=int, default=0, help="0 = whole sequence")
    # 336x448 is EXACTLY 4:3 like Bonn (640x480) and both are multiples of 14 (the
    # ViT patch), so no aspect distortion and no patch padding.
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--no_motion", action="store_true",
                    help="reproduce their shipped use_motion=False, as a control")
    args = ap.parse_args()

    sys.path.insert(0, args.neoverse_dir)
    from diffsynth.models.model_manager import ModelManager
    from diffsynth.utils.auxiliary import homo_matrix_inverse

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    ckpt = Path(args.neoverse_dir) / args.ckpt
    mm = ModelManager()
    mm.load_model(str(ckpt), device=dev, torch_dtype=torch.bfloat16)
    recon = mm.fetch_model("reconstructor")
    if recon is None:
        raise SystemExit("fetch_model('reconstructor') returned None")
    print(f"reconstructor: {type(recon).__name__}  motion={getattr(recon,'enable_motion',None)}")

    rgb = sorted((Path(args.seq_dir) / "rgb").glob("*.png"))
    span = (args.frames - 1) * args.stride
    starts = list(range(0, len(rgb) - span, args.window_step))
    if args.max_windows:
        starts = starts[: args.max_windows]
    hold = args.frames // 2                     # middle frame is held out
    use_motion = not args.no_motion
    print(f"{len(rgb)} frames -> {len(starts)} windows "
          f"({args.frames} frames, stride {args.stride}, step {args.window_step}, "
          f"hold={hold}, use_motion={use_motion})")

    n = 0
    for w, s in enumerate(starts):
        idx = [s + i * args.stride for i in range(args.frames)]
        keep = [i for i in range(args.frames) if i != hold]

        imgs = [Image.open(str(rgb[idx[i]])).convert("RGB")
                .resize((args.width, args.height), Image.BILINEAR) for i in keep]
        views = {
            "img": torch.stack([TF.to_tensor(im)[None] for im in imgs], dim=1).to(dev),
            "is_target": torch.zeros((1, len(keep)), dtype=torch.bool, device=dev),
            "is_static": torch.zeros((1, len(keep)), dtype=torch.bool, device=dev),
            # WITHIN-WINDOW positions, so the held-out timestamp is exactly `hold`
            # and the integer grid stays uniform with one gap at the held-out frame.
            "timestamp": torch.tensor(keep, dtype=torch.int64, device=dev).unsqueeze(0),
        }

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = recon(views, is_inference=True, use_motion=use_motion)

        c2w = preds["rendered_extrinsics"][0].float().cpu().numpy()   # [15,4,4]
        K = preds["rendered_intrinsics"][0].float()                    # [15,3,3]

        # Neighbours of the held-out frame within `keep`: positions hold-1 and hold.
        a, b = keep.index(hold - 1), keep.index(hold + 1)
        tgt_c2w = slerp_pose(c2w[a], c2w[b], 0.5)
        tgt_w2c = homo_matrix_inverse(
            torch.from_numpy(tgt_c2w).float().unsqueeze(0).to(dev))
        tgt_K = K[a].unsqueeze(0).to(dev)        # same physical camera throughout
        tgt_ts = torch.tensor([hold], dtype=torch.int64, device=dev)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            rgb_out, _, _ = recon.gs_renderer.rasterizer.forward(
                preds["splats"],
                render_viewmats=[tgt_w2c],
                render_Ks=[tgt_K],
                render_timestamps=[tgt_ts],
                sh_degree=0, width=args.width, height=args.height,
            )

        pred = rgb_out[0, 0].float().clamp(0, 1).cpu().numpy()
        gt = np.asarray(Image.open(str(rgb[idx[hold]])).convert("RGB")
                        .resize((args.width, args.height), Image.BILINEAR)) / 255.0

        # Name by the SOURCE frame, not the window counter: compare_methods.py pairs
        # files by sorted order, and different strides hold out different frames, so
        # positional matching would silently compare mismatched frames.
        stem = rgb[idx[hold]].stem
        Image.fromarray((pred * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"pred_{idx[hold]:06d}_{stem}.png"))
        Image.fromarray((gt * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"gt_{idx[hold]:06d}_{stem}.png"))
        n += 1
        if w % 10 == 0:
            p = float(-10 * np.log10(((pred - gt) ** 2).mean() + 1e-12))
            print(f"  window {w:4d}/{len(starts)}  held-out frame {idx[hold]:5d}  psnr {p:.2f}",
                  flush=True)

    print(f"wrote {n} pred/gt pairs to {args.out}")
    print("held-out source frame indices:",
          [s + hold * args.stride for s in starts][:20], "..." if len(starts) > 20 else "")


if __name__ == "__main__":
    main()
