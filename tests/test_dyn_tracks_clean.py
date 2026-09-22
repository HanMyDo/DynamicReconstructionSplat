"""Correctness tests for the Tier-3 track-scaffold cleanup in dyn_motion.py:
smooth_tracks_temporal (per-frame depth error entering the displacement twice)
and drop_static_tracks (mask false positives seeding tracks on static background).

THE PROPERTIES THAT MATTER ARE TESTS 2 AND 5: a constant-velocity trajectory with
a single bad depth sample must come back closer to the truth than it went in, and
a track that does not move must be marked unusable while the moving ones survive.
Everything else guards against the cleanup doing damage: both are exact no-ops at
their defaults, smoothing must not flatten real motion, and neither may resurrect
a frame the tracker already marked unusable.

Loads dyn_motion.py directly (importlib) so the test does not drag in the full
encoder package (gsplat, torch_scatter, ...).

Run:  python tests/test_dyn_tracks_clean.py
"""
import importlib.util
import sys
from pathlib import Path

import torch

_DM_PATH = Path(__file__).resolve().parents[1] / "src/model/encoder/dyn_motion.py"
_spec = importlib.util.spec_from_file_location("dyn_motion", _DM_PATH)
dyn_motion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dyn_motion)
smooth = dyn_motion.smooth_tracks_temporal
drop_static = dyn_motion.drop_static_tracks


def main():
    fails = []
    torch.manual_seed(0)
    V, Nt = 16, 40

    # A clean constant-velocity scaffold: every track moves 0.01 world units/frame
    # along x, which is the regime the displacement layer is built for.
    t = torch.arange(V, dtype=torch.float32).view(V, 1, 1)
    base = torch.rand(1, Nt, 3)
    vel = torch.tensor([0.01, 0.0, 0.0]).view(1, 1, 3)
    clean = base + t * vel
    ok = torch.ones(V, Nt, dtype=torch.bool)

    # ---- 1. width < 3 is EXACTLY off --------------------------------------
    t1 = all(torch.equal(smooth(clean, ok, w), clean) for w in (0, 1, 2))
    print(f"[1] width<3 is bit-identical: {'PASS' if t1 else 'FAIL'}")
    if not t1:
        fails.append(1)

    # ---- 2. THE PROPERTY: a depth outlier is removed ----------------------
    # One frame of one track samples the background past a silhouette and jumps
    # half a world unit. That is what puts a relocated copy in the wrong place.
    bad = clean.clone()
    bad[8, 0] += torch.tensor([0.0, 0.0, 0.5])
    err_in = (bad - clean).norm(dim=-1).max().item()
    out = smooth(bad, ok, 3)
    err_out = (out - clean).norm(dim=-1).max().item()
    t2 = err_out < 0.1 * err_in
    print(f"[2] single depth outlier removed: {'PASS' if t2 else 'FAIL'} "
          f"(max error {err_in:.3f} -> {err_out:.3f})")
    if not t2:
        fails.append(2)

    # ---- 3. real motion is NOT flattened ----------------------------------
    # A smoother that also shrank genuine displacement would trade one error for
    # another, and the metric would not tell them apart.
    sm = smooth(clean, ok, 3)
    span_in = (clean[-1] - clean[0]).norm(dim=-1).mean().item()
    span_out = (sm[-1] - sm[0]).norm(dim=-1).mean().item()
    node_err = (sm - clean).abs().max().item()
    t3 = abs(span_out - span_in) / span_in < 1e-6 and node_err < 1e-5
    print(f"[3] constant velocity preserved EXACTLY: {'PASS' if t3 else 'FAIL'} "
          f"(span {span_in:.4f} -> {span_out:.4f}, max node error {node_err:.2e})")
    if not t3:
        fails.append(3)

    # ---- 4. unusable frames neither vote nor get rewritten ----------------
    ok2 = ok.clone()
    ok2[5, :] = False
    junk = clean.clone()
    junk[5] += 10.0                      # garbage, but already marked unusable
    out = smooth(junk, ok2, 3)
    t4a = torch.allclose(out[5], junk[5])                 # left alone
    t4b = (out[4] - clean[4]).norm(dim=-1).max().item() < 1e-4   # did not pollute neighbours
    t4 = t4a and t4b
    print(f"[4] unusable frames excluded from the window: {'PASS' if t4 else 'FAIL'} "
          f"(untouched={t4a}, neighbour clean={t4b})")
    if not t4:
        fails.append(4)

    # ---- 5. THE PROPERTY: static tracks dropped, movers kept --------------
    # Half the tracks are mask false positives sitting on static background.
    mixed = clean.clone()
    static_idx = torch.arange(0, Nt, 2)
    mixed[:, static_idx] = base[:, static_idx].expand(V, -1, 3).clone()
    ok3 = drop_static(mixed, torch.ones(V, Nt, dtype=torch.bool), 0.25)
    dropped = ~ok3.any(dim=0)
    t5a = bool(dropped[static_idx].all())
    t5b = bool((~dropped[torch.arange(1, Nt, 2)]).all())
    t5 = t5a and t5b
    print(f"[5] static dropped, movers kept: {'PASS' if t5 else 'FAIL'} "
          f"(static dropped={t5a}, movers kept={t5b}, "
          f"{int(dropped.sum())}/{Nt} dropped)")
    if not t5:
        fails.append(5)

    # ---- 6. frac 0 is EXACTLY off, and a uniform scaffold is untouched ----
    okin = torch.ones(V, Nt, dtype=torch.bool)
    t6a = torch.equal(drop_static(mixed, okin, 0.0), okin)
    # every track moving the same amount -> median == travel -> nothing below it
    t6b = torch.equal(drop_static(clean, okin, 0.25), okin)
    t6 = t6a and t6b
    print(f"[6] frac=0 off, uniform scaffold untouched: {'PASS' if t6 else 'FAIL'} "
          f"(off={t6a}, uniform={t6b})")
    if not t6:
        fails.append(6)

    # ---- 7. an already-unusable frame is never resurrected ----------------
    ok4 = torch.ones(V, Nt, dtype=torch.bool)
    ok4[3, 7] = False
    out4 = drop_static(clean, ok4, 0.25)
    t7 = not bool(out4[3, 7])
    print(f"[7] drop_static only ever removes: {'PASS' if t7 else 'FAIL'}")
    if not t7:
        fails.append(7)

    # ---- 8. a track with <2 usable frames is not judged -------------------
    # It cannot have a measurable travel, so dropping it would be a guess.
    ok5 = torch.ones(V, Nt, dtype=torch.bool)
    ok5[:, 3] = False
    ok5[0, 3] = True
    out5 = drop_static(clean, ok5, 0.25)
    t8 = bool(out5[0, 3])
    print(f"[8] unjudgeable track left alone: {'PASS' if t8 else 'FAIL'}")
    if not t8:
        fails.append(8)

    # ---- 9. KNOWN LIMITATION, asserted so it cannot change silently -------
    # Odd extension builds the pad from the endpoint itself, so an outlier ON
    # frame 0 or V-1 survives. Accepted deliberately: the alternative (truncating
    # the window) shrinks EVERY displacement by 1/(V-1), and a systematic bias on
    # all frames is worse than a missed outlier on two of them.
    edge = clean.clone()
    edge[0, 0] += torch.tensor([0.0, 0.0, 0.5])
    e_err = (smooth(edge, ok, 3) - clean).norm(dim=-1).max().item()
    t9 = e_err > 0.4
    print(f"[9] boundary outlier survives (documented trade): "
          f"{'as expected' if t9 else 'CHANGED -- re-read the docstring'} "
          f"(residual {e_err:.3f})")
    if not t9:
        fails.append(9)

    print(f"\n{'ALL TESTS PASS' if not fails else f'FAILED: tests {fails}'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
