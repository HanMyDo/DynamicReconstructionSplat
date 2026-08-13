"""Correctness tests for hybrid static-fusion (EncoderAnySplat.voxelize_static_hybrid).

Run on the cluster (needs torch + torch_scatter):
    python tests/test_hybrid_fusion.py

The property that matters most is test 3. Fusing static points across frames means
leave-one-out can no longer drop view j's own Gaussians by frame index, so if the
fusion still SAW view j, that view's own depth estimate leaks into its own render
(project->unproject->project = the self-reprojection shortcut) and static PSNR
inflates for a trivial reason. Static is where our entire measured gain sits
(+1.51 dB), so a leak here would silently invalidate the headline result.
Test 3 corrupts the excluded frame beyond recognition and demands the output be
bit-identical -- that is the only way to prove no information path exists.
"""
import sys
import types
from pathlib import Path

# Repo root on sys.path: `python tests/test_hybrid_fusion.py` puts tests/ there,
# not the root, so `import src...` fails. Do this BEFORE importing src.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model.encoder.anysplat import EncoderAnySplat

# The method never touches `self`, so a dummy instance exercises the real code.
fuse = lambda *a, **k: EncoderAnySplat.voxelize_static_hybrid(types.SimpleNamespace(), *a, **k)


def main() -> int:
    torch.manual_seed(0)
    V, C, H, W, vs = 4, 8, 6, 6, 0.05

    # Every frame observes the SAME static surface (+ tiny per-frame noise), which
    # is the situation fusion is supposed to collapse into one Gaussian per voxel.
    base = torch.rand(3, H, W)
    pts = torch.stack([base + 0.001 * torch.randn(3, H, W) for _ in range(V)])
    feat = torch.randn(V, C, H, W)
    conf = torch.rand(V, H, W)
    static = torch.ones(V, H, W, dtype=torch.bool)

    fails = []

    # 1. The whole point: V redundant copies -> ~1 Gaussian per voxel.
    p_all, f_all = fuse(feat, pts, vs, conf, static)
    ratio = p_all.shape[0] / (V * H * W)
    print(f"1. redundancy removed : {V*H*W} pts -> {p_all.shape[0]} voxels (ratio {ratio:.3f})")
    if ratio >= 0.9:
        fails.append("fusion did not merge redundant copies (voxel_size too small?)")

    # 2. exclude_frame actually changes the fused set.
    p_ex, f_ex = fuse(feat, pts, vs, conf, static, exclude_frame=1)
    print(f"2. exclude_frame      : {p_ex.shape[0]} voxels, differs={not torch.equal(p_all, p_ex)}")

    # 3. LOO EXACTNESS -- the honesty test. Corrupt the excluded frame completely;
    #    the output must not move at all. Any change = an information path from
    #    view j into view j's own render = self-reprojection.
    pts_c, feat_c, conf_c = pts.clone(), feat.clone(), conf.clone()
    pts_c[1] += 5.0
    feat_c[1] = torch.randn(C, H, W)
    conf_c[1] = torch.rand(H, W)
    p_c, f_c = fuse(feat_c, pts_c, vs, conf_c, static, exclude_frame=1)
    exact = torch.equal(p_ex, p_c) and torch.equal(f_ex, f_c)
    print(f"3. LOO EXACT          : corrupting excluded frame changes nothing = {exact}")
    if not exact:
        fails.append("LOO LEAK: excluded frame still influences the fused output")

    # 4. Dynamic pixels must stay OUT of the fusion (they keep per-frame identity).
    half = static.clone()
    half[:, :, :3] = False
    p_half, _ = fuse(feat, pts, vs, conf, half)
    print(f"4. dynamic excluded   : {p_half.shape[0]} < {p_all.shape[0]} = {p_half.shape[0] < p_all.shape[0]}")
    if p_half.shape[0] >= p_all.shape[0]:
        fails.append("masking out dynamic pixels did not reduce the fused set")

    # 5. Fusion is a convex combination -> fused points cannot leave the input range.
    inside = bool(p_all.min() >= pts.min() - 1e-4 and p_all.max() <= pts.max() + 1e-4)
    print(f"5. convex combination : fused points within input range = {inside}")
    if not inside:
        fails.append("fused points outside input range (weights are not a partition of unity)")

    # 6. Degenerate input must not crash (a fully-dynamic window is legal).
    p_empty, f_empty = fuse(feat, pts, vs, conf, torch.zeros_like(static))
    print(f"6. empty static set   : returns {tuple(p_empty.shape)}, {tuple(f_empty.shape)} without crashing")

    print()
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
