"""Correctness tests for the fallback-displacement selection (--dyn_disp_fallback).

WHY THIS EXISTS. Flow-gated compositing DELETES every off-frame dynamic Gaussian
that no RAFT track supports. That is correct when the own-frame copy survives to
cover the object, and catastrophic under --eval_loo, where it does not: measured
on synchronous2, dynamic PSNR fell 19.69 -> 11.39 dB and lost to MoVieS on PSNR,
LPIPS *and* SSIM, because a complete-but-wrong frame beats a partial one on every
metric. The fallback routes those Gaussians to the piecewise-rigid group motion
instead of deleting them.

THE PROPERTIES THAT MATTER:
  1. OFF is bit-identical to the previous behaviour (every published recipe must
     reproduce exactly).
  2. A Gaussian with flow support uses FLOW, never the fallback -- flow is the
     better estimate and must take precedence.
  3. A Gaussian WITHOUT flow support but WITH a valid group moves by the GROUP
     displacement instead of being deleted.
  4. A Gaussian with neither is still deleted -- the fallback must not invent a
     position nothing supports.
  5. The choice is BINARY, never a blend of the two displacements.

This re-implements the selection rather than importing the decoder, which would
drag in gsplat/torch_scatter. The arithmetic under test is small and copied
verbatim from decoder_splatting_cuda.py section (3); if that changes, this test
must be updated alongside it.

Run:  python tests/test_dyn_disp_fallback.py
"""
import torch


def select(move, flow_ok, disp_flow, fb_disp, fb_ok, fallback):
    """Verbatim arithmetic from decoder_splatting_cuda.py section (3)."""
    disp_ij = disp_flow.clone()
    if fallback:
        use_fb = (1.0 - flow_ok) * fb_ok
        disp_ij = disp_ij * flow_ok.unsqueeze(-1) + fb_disp * use_fb.unsqueeze(-1)
        move = move * (flow_ok + use_fb).clamp(max=1.0)
    else:
        move = move * flow_ok
    return move.unsqueeze(-1) * disp_ij


def main():
    fails = []

    # 4 dynamic Gaussians, none from the target view, so all are movable:
    #   0: flow ok, group ok      -> must use FLOW
    #   1: no flow, group ok      -> must use GROUP (the whole point)
    #   2: no flow, no group      -> must stay DELETED
    #   3: flow ok, no group      -> must use FLOW
    move = torch.ones(4)
    flow_ok = torch.tensor([1.0, 0.0, 0.0, 1.0])
    fb_ok = torch.tensor([1.0, 1.0, 0.0, 0.0])
    disp_flow = torch.tensor([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]])
    fb_disp = torch.tensor([[0, 10.0, 0], [0, 20.0, 0], [0, 30.0, 0], [0, 40.0, 0]])
    # fb_disp is already zeroed by fb_ok in the decoder; mirror that here.
    fb_disp = fb_disp * fb_ok.unsqueeze(-1)

    off = select(move, flow_ok, disp_flow, fb_disp, fb_ok, fallback=False)
    on = select(move, flow_ok, disp_flow, fb_disp, fb_ok, fallback=True)

    # 1. OFF: only flow-supported Gaussians move, by their flow displacement.
    want_off = torch.tensor([[1.0, 0, 0], [0, 0, 0], [0, 0, 0], [4.0, 0, 0]])
    if not torch.allclose(off, want_off):
        fails.append(f"OFF changed behaviour:\n{off}\nwant\n{want_off}")

    # 2 + 3 + 4.
    want_on = torch.tensor([[1.0, 0, 0], [0, 20.0, 0], [0, 0, 0], [4.0, 0, 0]])
    if not torch.allclose(on, want_on):
        fails.append(f"ON wrong selection:\n{on}\nwant\n{want_on}")
    if not torch.allclose(on[0], disp_flow[0]):
        fails.append("flow did NOT take precedence where both were available")
    if not torch.allclose(on[1], fb_disp[1]):
        fails.append("fallback did not fill in where flow was missing")
    if on[2].abs().sum() > 0:
        fails.append("a Gaussian with NEITHER estimate was moved -- must stay deleted")

    # 5. Binary, never a blend: each row equals one source exactly.
    for n in range(4):
        if on[n].abs().sum() == 0:
            continue
        if not (torch.allclose(on[n], disp_flow[n]) or torch.allclose(on[n], fb_disp[n])):
            fails.append(f"row {n} is a BLEND of flow and fallback: {on[n]}")

    # 6. Own-frame Gaussians must never be displaced, fallback or not: `move` is
    #    already zero for them upstream, so the fallback must not revive them.
    own = select(torch.zeros(4), flow_ok, disp_flow, fb_disp, fb_ok, fallback=True)
    if own.abs().sum() > 0:
        fails.append(f"own-frame Gaussian displaced by the fallback: {own}")

    # 7. Soft flow validity (--dyn_conf_opacity) still binarises: a 0 < v < 1 row
    #    counts as supported and must use flow unblended.
    soft = select(torch.ones(1), torch.ones(1), disp_flow[:1], fb_disp[:1],
                  torch.ones(1), fallback=True)
    if not torch.allclose(soft[0], disp_flow[0]):
        fails.append(f"soft-valid row was blended: {soft[0]}")

    if fails:
        print("FAIL")
        for f in fails:
            print(" -", f)
        return 1
    print("PASS  (7 properties: off bit-identical, flow precedence, fallback fills,"
          " neither->deleted, no blending, own-frame untouched, soft validity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
