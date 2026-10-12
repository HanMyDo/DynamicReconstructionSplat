"""Smoke test for the NeoVerse reconstructor as a BASELINE, before writing the LOO runner.

WHY A SMOKE TEST. NeoVerse ships an 81 GB checkpoint set, of which only the 6 GB
`reconstructor.ckpt` is relevant to us: we compare reconstructions, never their
generated video (a diffusion model may hallucinate content it never observed,
which is a different task). So this loads the reconstructor WITHOUT the Wan
diffusion stack and answers the three things the runner needs:

  1. does `ModelManager` load the reconstructor alone, and which class is it?
     (the DA3 adapter builds "pseudo Gaussians" and ignores motion entirely, so
     only the native WorldMirror reconstructor is a valid dynamic baseline)
  2. what does a forward pass return, with `use_motion=True`?
     Their own inference.py and app.py pass use_motion=FALSE, which gates both
     `enable_motion` and `enable_dynamic_gs_attr` -- i.e. no velocities and no
     temporal attributes. That is fine for their use case (novel trajectory at
     OBSERVED timestamps) and useless as a dynamic baseline, so we must pass True.
     The only forced override is `if imgs.shape[1] == 1`, so multi-frame is fine.
  3. what signature does the rasteriser take? The LOO protocol renders at a
     held-out frame's camera AND timestamp, so we need to pass both explicitly.

Run (on the cluster, in the `neoverse` env):
  python tools/neoverse_smoke.py --seq_dir ~/data/bonn/rgbd_bonn_dataset/rgbd_bonn_synchronous2
"""
import argparse
import inspect
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def describe(name, obj, indent=2):
    pad = " " * indent
    if torch.is_tensor(obj):
        print(f"{pad}{name}: Tensor {tuple(obj.shape)} {obj.dtype}")
    elif isinstance(obj, dict):
        print(f"{pad}{name}: dict({len(obj)})")
        for k, v in obj.items():
            describe(str(k), v, indent + 2)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__}({len(obj)})")
        for i, v in enumerate(obj[:4]):
            describe(f"[{i}]", v, indent + 2)
    else:
        print(f"{pad}{name}: {type(obj).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neoverse_dir", default=str(Path.home() / "NeoVerse"))
    ap.add_argument("--ckpt", default="models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--seq_dir", required=True, help="Bonn sequence dir (contains rgb/)")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    # 336x448 is EXACTLY 4:3 like Bonn (640x480) and both are multiples of 14
    # (the ViT patch), so no aspect distortion and no patch padding.
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--no_motion", action="store_true",
                    help="reproduce their shipped setting (use_motion=False) for comparison")
    args = ap.parse_args()

    sys.path.insert(0, args.neoverse_dir)
    from diffsynth.models.model_manager import ModelManager

    ckpt = Path(args.neoverse_dir) / args.ckpt
    print(f"loading {ckpt} ({ckpt.stat().st_size / 1e9:.2f} GB)")
    mm = ModelManager()
    mm.load_model(str(ckpt), device="cuda", torch_dtype=torch.bfloat16)
    recon = mm.fetch_model("reconstructor")
    if recon is None:
        raise SystemExit("fetch_model('reconstructor') returned None -- the manager did "
                         "not recognise this checkpoint under that name")
    print(f"reconstructor class: {type(recon).__name__}")
    print(f"  enable_motion         = {getattr(recon, 'enable_motion', 'n/a')}")
    print(f"  enable_dynamic_gs_attr= {getattr(recon, 'enable_dynamic_gs_attr', 'n/a')}")

    rgb = sorted((Path(args.seq_dir) / "rgb").glob("*.png"))
    idx = [args.start + i * args.stride for i in range(args.frames)]
    if idx[-1] >= len(rgb):
        raise SystemExit(f"window runs past the sequence ({idx[-1]} >= {len(rgb)})")
    print(f"{len(rgb)} frames available; using {idx[0]}..{idx[-1]} stride {args.stride}")

    imgs = [Image.open(p).convert("RGB").resize((args.width, args.height), Image.BILINEAR)
            for p in (rgb[i] for i in idx)]
    dev = "cuda"
    views = {
        "img": torch.stack([TF.to_tensor(im)[None] for im in imgs], dim=1).to(dev),
        "is_target": torch.zeros((1, len(imgs)), dtype=torch.bool, device=dev),
        "is_static": torch.zeros((1, len(imgs)), dtype=torch.bool, device=dev),
        # integer frame index, exactly as inference.py builds it
        "timestamp": torch.arange(0, len(imgs), dtype=torch.int64, device=dev).unsqueeze(0),
    }
    print(f"views['img'] = {tuple(views['img'].shape)}")

    use_motion = not args.no_motion
    print(f"\nforward(use_motion={use_motion}) ...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = recon(views, is_inference=True, use_motion=use_motion)

    print("\npredictions:")
    for k, v in preds.items():
        describe(k, v)

    print("\nrasterizer.forward signature:")
    print("  ", inspect.signature(recon.gs_renderer.rasterizer.forward))

    torch.cuda.synchronize()
    print(f"\npeak GPU: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
