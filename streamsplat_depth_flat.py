"""Depth-Anything-V2 depths for a FLAT directory of frames, for StreamSplat.

WHY. StreamSplat's inference needs a depth map per frame, and its own
`preprocess_depth_davis.py` hard-codes the DAVIS layout (ImageSets/2017/*.txt,
JPEGImages/Full-Resolution/<set>/*.jpg). Bonn sequences are a flat rgb/ folder,
so nothing in that script can be pointed at them.

This reproduces its pipeline EXACTLY -- same resize order (shorter side to 518,
then to a multiple of 14), same wrapper, same interpolation back to the frame
size, and the same per-frame min-max normalisation to uint8. That last one
matters: the depths are RELATIVE per frame, not metric, and StreamSplat is
trained to consume them that way. Deviating would silently change its input
distribution and make any comparison meaningless.

Run it from inside a StreamSplat checkout (it imports model.depth_wrapper) with
the StreamSplat conda environment active:

    cp streamsplat_depth_flat.py ../StreamSplat/
    cd ../StreamSplat
    python streamsplat_depth_flat.py \
        --input_dir  ~/data/bonn/rgbd_bonn_dataset/rgbd_bonn_balloon/rgb \
        --output_dir ~/data/streamsplat/balloon_depth
"""
import argparse
import os
import os.path as osp
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model.depth_wrapper import DepthAnythingWrapper


def resize_to_multiple_of_14(image: Image.Image) -> Image.Image:
    w, h = image.size
    return TF.resize(image, size=(round(h / 14) * 14, round(w / 14) * 14),
                     interpolation=tf.InterpolationMode.BILINEAR)


def resize_to_shorter_side(image: Image.Image, target_size: int = 518) -> Image.Image:
    w, h = image.size
    if w <= h:
        new_w, new_h = target_size, int(h * (target_size / w))
    else:
        new_h, new_w = target_size, int(w * (target_size / h))
    return TF.resize(image, size=(new_h, new_w),
                     interpolation=tf.InterpolationMode.BILINEAR)


class FlatFrames(Dataset):
    def __init__(self, input_dir: str, output_dir: str):
        self.files = sorted(sum((glob(osp.join(input_dir, e))
                                 for e in ("*.png", "*.jpg", "*.jpeg")), []))
        if not self.files:
            raise SystemExit(f"no frames found in {input_dir}")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.to_tensor = tf.Compose([tf.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        frame = Image.open(path).convert("RGB")
        # the DAVIS script feeds the SAME image down both paths -- the depth branch
        # is the resized copy, the frame branch keeps the original size so the
        # prediction can be interpolated back onto it.
        depth_in = resize_to_multiple_of_14(resize_to_shorter_side(frame))
        stem = osp.splitext(osp.basename(path))[0]
        return {
            "frames": self.to_tensor(frame),
            "depths": self.to_tensor(depth_in),
            "out": osp.join(self.output_dir, stem + "_pred.png"),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="flat directory of frames")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", default="vitl", help="matches Options.depth_model_name")
    args = ap.parse_args()

    ds = FlatFrames(args.input_dir, args.output_dir)
    # batch_size 1: frames keep their native size, so a batch cannot be collated
    # unless every frame is identical in size. Bonn is, but one odd frame would
    # crash a long run late, and depth inference is fast enough that it is moot.
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingWrapper(args.model_name).to(device).eval()
    print(f"{len(ds)} frames -> {args.output_dir}")

    with torch.no_grad():
        for n, batch in enumerate(dl):
            frames = batch["frames"].to(device)
            depths = batch["depths"].to(device)
            pred = model(depths).detach()                      # [B, H, W]
            pred = F.interpolate(pred[:, None], size=frames.shape[-2:],
                                 mode="bilinear", align_corners=True)
            for i in range(pred.shape[0]):
                d = pred[i].cpu().numpy()
                if d.ndim == 3:
                    d = d[0]
                d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(batch["out"][i], d)
            if (n + 1) % 50 == 0:
                print(f"  {n + 1}/{len(ds)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
