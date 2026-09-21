"""Correctness tests for `compensate_dyn_opacity` (dyn_motion.py), the opacity
fix-up that per-frame compositing needs.

THE PROPERTY THAT MATTERS IS TEST 3: after compensation, the alpha composed by
the SURVIVING dynamic Gaussians must equal the alpha the FULL set would have
composed. That is the whole claim -- the head sized each Gaussian to carry a
share of a V-fold stack, the gate removes part of the stack, and this restores
the total. Everything else is a guard against the fix doing damage elsewhere:
static Gaussians untouched, strength 0 bit-identical to the measured behaviour,
opacity never leaving [0, 1], and no boost when the gate removed nothing.

Loads dyn_motion.py directly (importlib) so the test does not drag in the full
encoder package (gsplat, torch_scatter, ...).

Run:  python tests/test_dyn_opacity_comp.py
"""
import importlib.util
import sys
from pathlib import Path

import torch

_DM_PATH = Path(__file__).resolve().parents[1] / "src/model/encoder/dyn_motion.py"
_spec = importlib.util.spec_from_file_location("dyn_motion", _DM_PATH)
dyn_motion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dyn_motion)
compensate = dyn_motion.compensate_dyn_opacity


def composed_alpha(o, n):
    """Alpha from n independent contributors of opacity o, over-composited."""
    return 1.0 - (1.0 - o) ** n


def main():
    fails = []
    torch.manual_seed(0)

    V = 16
    N = 400
    # Half the Gaussians dynamic, opacities spanning the range the head produces
    # (including the low tail, which is where under-coverage actually bites).
    opacity = torch.rand(N) * 0.5 + 0.01
    dyn = torch.zeros(N)
    dyn[: N // 2] = 1.0

    # ---- 1. strength 0 is EXACTLY off ----------------------------------
    keep_half = torch.zeros(N)
    keep_half[: N // 4] = 1.0                       # half the dynamic ones survive
    out = compensate(opacity, dyn, keep_half, V, strength=0.0)
    t1 = torch.equal(out, opacity)
    print(f"[1] strength=0 is bit-identical: {'PASS' if t1 else 'FAIL'}")
    if not t1:
        fails.append(1)

    # ---- 2. static Gaussians are never touched -------------------------
    out = compensate(opacity, dyn, keep_half, V, strength=1.0)
    stat = dyn <= 0.5
    t2a = torch.equal(out[stat], opacity[stat])
    t2b = bool((out[dyn > 0.5] >= opacity[dyn > 0.5]).all())   # dynamic only ever rises
    t2 = t2a and t2b
    print(f"[2] static untouched, dynamic never lowered: {'PASS' if t2 else 'FAIL'} "
          f"(static identical={t2a}, dynamic monotone={t2b})")
    if not t2:
        fails.append(2)

    # ---- 3. THE PROPERTY: surviving contributors compose to the full alpha ----
    # Survivor fraction 0.5 of V=16 -> 8 contributors instead of 16. Compensated,
    # those 8 must compose to what 16 uncompensated ones would have.
    o = opacity[dyn > 0.5]
    n_keep = float(keep_half[dyn > 0.5].mean()) * V          # = 8.0
    o_comp = out[dyn > 0.5]
    want = composed_alpha(o, V)                              # what the head assumed
    got = composed_alpha(o_comp, n_keep)                     # what now renders
    err = (got - want).abs().max().item()
    t3 = err < 1e-5
    print(f"[3] compensated survivors match full-stack alpha: {'PASS' if t3 else 'FAIL'} "
          f"(n_keep={n_keep:.1f}/{V}, max |alpha error| = {err:.2e}, "
          f"uncompensated would be off by "
          f"{(composed_alpha(o, n_keep) - want).abs().max().item():.3f})")
    if not t3:
        fails.append(3)

    # ---- 4. nothing removed -> nothing added ---------------------------
    # Every dynamic Gaussian survives: n_keep == V, exponent 1, no change. This is
    # the case that matters for --per_frame_dynamic on a window where flow could
    # relocate everything; a fix that still boosts there would over-cover.
    keep_all = torch.ones(N)
    out_all = compensate(opacity, dyn, keep_all, V, strength=1.0)
    t4 = torch.allclose(out_all, opacity, atol=1e-6)
    print(f"[4] full survival is a no-op: {'PASS' if t4 else 'FAIL'} "
          f"(max delta {float((out_all - opacity).abs().max()):.2e})")
    if not t4:
        fails.append(4)

    # ---- 5. strength interpolates monotonically, stays in [0, 1] -------
    prev = opacity[dyn > 0.5]
    mono = True
    for st in (0.25, 0.5, 0.75, 1.0):
        cur = compensate(opacity, dyn, keep_half, V, strength=st)[dyn > 0.5]
        mono = mono and bool((cur >= prev - 1e-7).all())
        prev = cur
    t5 = mono and bool(((prev >= 0.0) & (prev <= 1.0)).all())
    print(f"[5] strength monotone and opacity stays in [0,1]: {'PASS' if t5 else 'FAIL'} "
          f"(max compensated opacity {float(prev.max()):.4f})")
    if not t5:
        fails.append(5)

    # ---- 6. degenerate: no dynamic Gaussians, and none surviving -------
    # Both reachable per target view -- a window with no moving object, and a view
    # where the gate dropped everything. Neither may produce NaN or touch anything.
    t6a = torch.equal(compensate(opacity, torch.zeros(N), keep_half, V, 1.0), opacity)
    out_none = compensate(opacity, dyn, torch.zeros(N), V, 1.0)
    t6b = bool(torch.isfinite(out_none).all())
    t6c = torch.equal(out_none[stat], opacity[stat])
    t6 = t6a and t6b and t6c
    print(f"[6] no-dynamic and none-surviving are safe: {'PASS' if t6 else 'FAIL'} "
          f"(no-dyn identical={t6a}, finite={t6b}, static intact={t6c})")
    if not t6:
        fails.append(6)

    # ---- 7. the low-opacity tail is helped MOST ------------------------
    # The mechanism's whole point: a Gaussian that was carrying 1/V of the alpha is
    # hurt more by losing contributors than one that was already near-opaque. The
    # RATIO of compensated to original opacity must therefore fall as opacity rises.
    lo = opacity[dyn > 0.5].min()
    hi = opacity[dyn > 0.5].max()
    probe = torch.tensor([lo, hi])
    pc = compensate(probe, torch.ones(2), torch.tensor([1.0, 0.0]), V, 1.0)
    t7 = (pc[0] / probe[0]) > (pc[1] / probe[1])
    print(f"[7] low-opacity tail gains most: {'PASS' if t7 else 'FAIL'} "
          f"(x{pc[0] / probe[0]:.2f} at o={probe[0]:.3f} vs "
          f"x{pc[1] / probe[1]:.2f} at o={probe[1]:.3f})")
    if not t7:
        fails.append(7)

    print(f"\n{'ALL TESTS PASS' if not fails else f'FAILED: tests {fails}'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
