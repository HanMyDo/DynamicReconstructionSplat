"""Correctness tests for TemporalAttentionBlock.

THE PROPERTY THAT MATTERS IS TEST 1: identity at initialisation.

`output_scale` is initialised to 0 specifically so the block starts as a no-op and can
only help. The original implementation did not satisfy that: it added the residual in
DOWNSAMPLED space and returned upsample(x_down), discarding the full-resolution input,
so the block returned upsample(avgpool(x)) -- a low-pass filter applied unconditionally.
That cost -0.54 psnr / -0.59 static / -0.28 dyn on 3/3 sequences and produced a
scene-independent LPIPS penalty (+0.0260/+0.0274/+0.0264), which is what a fixed blur
looks like. The measurement was of the bug, not of temporal attention.

Run:  python tests/test_temporal_attention.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model.encoder.heads.vggt_dpt_gs_head import TemporalAttentionBlock


def main() -> int:
    torch.manual_seed(0)
    B, S, C, H, W = 2, 6, 32, 28, 28
    fails = []

    for ds in (1, 2, 4):
        blk = TemporalAttentionBlock(dim=C, num_heads=4, spatial_downsample=ds).eval()
        x = torch.randn(B * S, C, H, W)
        with torch.no_grad():
            y = blk(x, B, S)

        # 1. IDENTITY AT INIT: output_scale starts at 0, so the block must be a no-op.
        #    A pooled/upsampled output fails this at ds > 1 while passing at ds == 1,
        #    which is exactly how the original bug hid.
        same = torch.allclose(y, x, atol=1e-6)
        err = (y - x).abs().max().item()
        print(f"ds={ds}: identity at init = {same}   max|y-x| = {err:.3e}")
        if not same:
            fails.append(f"ds={ds}: NOT identity at init (max err {err:.3e}) -- "
                         "the block alters features before it has learned anything")

        # 2. HIGH FREQUENCY PRESERVED: a checkerboard survives an identity block.
        hf = torch.zeros(B * S, C, H, W)
        hf[:, :, ::2, ::2] = 1.0
        with torch.no_grad():
            y_hf = blk(hf, B, S)
        kept = (y_hf - hf).abs().max().item()
        print(f"      high-freq preserved = {kept < 1e-6}  (max err {kept:.3e})")
        if kept >= 1e-6:
            fails.append(f"ds={ds}: high-frequency detail destroyed (max err {kept:.3e})")

        # 3. Shape is unchanged.
        if y.shape != x.shape:
            fails.append(f"ds={ds}: shape changed {tuple(x.shape)} -> {tuple(y.shape)}")

    # 4. ONCE TRAINED (scale != 0) the block must actually DO something, otherwise it is
    #    inert and any null result would be meaningless.
    blk = TemporalAttentionBlock(dim=C, num_heads=4, spatial_downsample=2).eval()
    with torch.no_grad():
        blk.output_scale.fill_(1.0)
        x = torch.randn(B * S, C, H, W)
        y = blk(x, B, S)
    moved = (y - x).abs().max().item()
    print(f"\nwith output_scale=1: block changes output = {moved > 1e-4}  (max {moved:.3e})")
    if moved <= 1e-4:
        fails.append("block is INERT even at output_scale=1 -- attention never reaches the output")

    # 5. It must be a TEMPORAL mechanism: shuffling frames must change the result.
    #    (This is also the criterion separating temporal from multi-view methods.)
    with torch.no_grad():
        xb = x.view(B, S, C, H, W)
        y_a = blk(xb.reshape(B * S, C, H, W), B, S).view(B, S, C, H, W)
        perm = torch.tensor([2, 0, 1, 5, 3, 4])
        y_b = blk(xb[:, perm].reshape(B * S, C, H, W), B, S).view(B, S, C, H, W)[:, perm.argsort()]
    order_matters = not torch.allclose(y_a, y_b, atol=1e-5)
    print(f"frame order matters (i.e. genuinely temporal) = {order_matters}")
    if not order_matters:
        fails.append("frame order does NOT change the output -- the block is not temporal")

    print()
    if fails:
        for f in fails:
            print("FAIL:", f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
