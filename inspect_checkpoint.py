"""Print what a checkpoint actually contains. Usage: python inspect_checkpoint.py <ckpt> [filter]

Written to answer: did the temporal attention block LEARN anything, or did the optimiser
drive its contribution to zero? `output_scale` is initialised to 0 and gated through
tanh(), so output_scale ~ 0 means the block is a learned identity -- loading its weights
or not then makes no difference to the output, which is exactly what we observe
(identical metrics with 126 vs 134 restored tensors).
"""
import sys
import torch

ckpt_path = sys.argv[1]
filt = sys.argv[2] if len(sys.argv) > 2 else "temporal"

c = torch.load(ckpt_path, map_location="cpu")
sd = c["model_state_dict"]
cfg = c.get("config", {}) or {}

print(f"checkpoint: {ckpt_path}")
print(f"saved_prefixes: {c.get('saved_prefixes')}")
print(f"epoch: {c.get('epoch')}  tensors: {len(sd)}")
for f in ("use_temporal_attention", "temporal_spatial_downsample", "temporal_num_heads",
          "use_vggt4d", "anchor_weight", "depth_consis_weight", "unfreeze_depth_head",
          "frame_stride", "num_frames"):
    if f in cfg:
        print(f"  cfg.{f} = {cfg[f]}")

print(f"\n--- tensors matching '{filt}' ---")
hits = 0
for k, v in sd.items():
    if filt not in k:
        continue
    hits += 1
    if v.numel() <= 8:
        print(f"  {k:70s} {tuple(v.shape)}  values={v.flatten().tolist()}")
    else:
        print(f"  {k:70s} {tuple(v.shape)}  |w| mean={v.abs().mean():.6f}  std={v.std():.6f}")
if hits == 0:
    print("  (none)")

# THE decisive number
for k, v in sd.items():
    if k.endswith("output_scale"):
        s = float(v.flatten()[0])
        import math
        print(f"\noutput_scale = {s:.6f}  ->  tanh = {math.tanh(s):.6f}")
        print("   |tanh| < 0.01  =>  the block is a LEARNED IDENTITY (contributes ~nothing)")
        print("   |tanh| > 0.1   =>  the block genuinely contributes")
