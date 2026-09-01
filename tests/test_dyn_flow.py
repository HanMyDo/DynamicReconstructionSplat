"""Correctness tests for the track-correspondence scene-flow motion model
(dyn_motion.py "UPGRADE": collect_dyn_tracks + knn_flow_displacement).

THE PROPERTY THAT MATTERS IS TEST 3: a rigidly translating cluster of dynamic
Gaussians must be displaced by EXACTLY the object's observed motion (direct
correspondence), with static Gaussians, own-frame slots and out-of-trust-radius
Gaussians untouched. The piecewise-rigid predecessor failed on articulated /
multi-object motion because it extrapolated a constant-velocity fit; this
mechanism has no motion model to mis-fit — if these invariants hold, its errors
can only come from the tracker or the depth lift, not from the interpolation.

Loads dyn_motion.py directly (importlib) so the test does not drag in the full
encoder package (torch_scatter, gsplat, ...).

Run:  python tests/test_dyn_flow.py
"""
import importlib.util
import sys
from pathlib import Path

import torch

_DM_PATH = Path(__file__).resolve().parents[1] / "src/model/encoder/dyn_motion.py"
_spec = importlib.util.spec_from_file_location("dyn_motion", _DM_PATH)
dyn_motion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dyn_motion)


class MockTrackHead:
    """Stands in for VGGT's TrackHead. Ground-truth 2D motion: every tracked
    pixel moves by `delta` px per frame. The mock decodes each input position's
    ORIGINAL frame id from the image content (channel 0 is filled with
    frame_id / 100), so it also verifies that _run_track_head's frame
    permutation is applied consistently to tokens/images and inverted on the
    outputs. Queries are taken in the frame at input position 0 (like the real
    tracker samples fmaps[:, 0])."""

    def __init__(self, delta):
        self.delta = torch.as_tensor(delta, dtype=torch.float32)

    def __call__(self, toks, images, patch_start_idx, query_points, iters=None):
        S = images.shape[1]
        fids = (images[0, :, 0, 0, 0] * 100).round()            # [S] original frame ids
        q = query_points[0]                                      # [Nq, 2] at frame fids[0]
        coords = torch.stack(
            [q + (fids[p] - fids[0]) * self.delta for p in range(S)], dim=0
        ).unsqueeze(0)                                           # [1, S, Nq, 2]
        vis = torch.ones(1, S, q.shape[0])
        return [coords], vis, None


def make_scene(V=3, H=32, W=32, scale=0.01, delta=(2.0, 1.0), block=6, base=(4, 5)):
    """Flat world plane: pts[f, y, x] = (x*scale, y*scale, 0) — so image motion of
    delta px/frame lifts to world motion of delta*scale/frame. The dynamic block
    sits at `base` in frame 0 and translates by delta each frame."""
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    plane = torch.stack([xs * scale, ys * scale, torch.zeros_like(xs, dtype=torch.float32)], -1)
    pts_all = plane.unsqueeze(0).expand(V, H, W, 3).clone().unsqueeze(0)  # [1,V,H,W,3]
    dyn = torch.zeros(1, V, H, W)
    for f in range(V):
        x0 = int(base[0] + f * delta[0])
        y0 = int(base[1] + f * delta[1])
        dyn[0, f, y0:y0 + block, x0:x0 + block] = 1.0
    image = torch.zeros(1, V, 3, H, W)
    for f in range(V):
        image[0, f, 0] = f / 100.0                               # frame id for the mock
    return pts_all, dyn, image


def main() -> int:
    torch.manual_seed(0)
    fails = []
    V, H, W = 3, 32, 32
    scale, delta = 0.01, (2.0, 1.0)
    pts_all, dyn, image = make_scene(V, H, W, scale, delta)
    conf = torch.ones(1, V, H, W, dtype=torch.bool)

    # 1. NEAREST LIFT: fractional pixels snap to the nearest cell; out-of-bounds flagged.
    tracks = torch.tensor([[[3.4, 7.6], [-2.0, 5.0]]]).expand(V, 2, 2)
    lifted, ok = dyn_motion._lift_tracks_nearest(pts_all[0], tracks)
    want = torch.tensor([3 * scale, 8 * scale, 0.0])
    t1 = torch.allclose(lifted[0, 0], want, atol=1e-6) and bool(ok[0, 0]) and not bool(ok[0, 1])
    print(f"[1] nearest lift + bounds: {'PASS' if t1 else 'FAIL'} "
          f"(got {lifted[0, 0].tolist()}, want {want.tolist()}; ok={ok[0].tolist()})")
    if not t1:
        fails.append(1)

    # 2. PERMUTED TRACKER CALL: querying from frame qf must give tracks in ORIGINAL
    #    frame order, i.e. row f == q + (f - qf) * delta, for every qf.
    head = MockTrackHead(delta)
    toks = [torch.zeros(1, V, 4, 8)]
    q = torch.tensor([[[10.0, 12.0], [20.0, 9.0]]])
    t2 = True
    for qf in range(V):
        tr, vis = dyn_motion._run_track_head(head, toks, image, 5, q, qf)
        for f in range(V):
            want = q[0] + (f - qf) * torch.tensor(delta)
            if not torch.allclose(tr[f], want, atol=1e-5):
                t2 = False
    print(f"[2] frame-permuted tracker query: {'PASS' if t2 else 'FAIL'}")
    if not t2:
        fails.append(2)

    # 3. END-TO-END PHASE A+B: the dynamic block's Gaussians must move by exactly
    #    (j - i) * delta * scale; static Gaussians and own-frame slots by zero.
    out = dyn_motion.collect_dyn_tracks(
        head, toks, image, 5, pts_all, dyn, conf, n_query=256, query_all_frames=True)
    t3 = out is not None and out[0] is not None
    if t3:
        traj, ok = out[0]
        gpts = pts_all[0][conf[0]]
        gfidx = torch.arange(V).view(V, 1, 1).expand(V, H, W)[conf[0]]
        gdyn = dyn[0][conf[0]] > 0.5
        disp, valid = dyn_motion.knn_flow_displacement(
            traj, ok, gpts, gfidx, gdyn, V, k=4, gate_mult=3.0)
        for i in range(V):
            for j in range(V):
                sel = gdyn & (gfidx == i)
                if j == i:
                    if disp[sel][:, j].abs().max() > 1e-6:
                        t3 = False
                else:
                    want = torch.tensor([(j - i) * delta[0] * scale,
                                         (j - i) * delta[1] * scale, 0.0])
                    got = disp[sel][:, j]
                    if not bool((valid[sel][:, j] > 0).all()):
                        t3 = False
                    if (got - want).abs().max() > 1e-4:
                        t3 = False
        if disp[~gdyn].abs().max() > 1e-6 or valid[~gdyn].abs().max() > 1e-6:
            t3 = False
    print(f"[3] exact recovery of block translation (all i->j, static untouched): "
          f"{'PASS' if t3 else 'FAIL'}")
    if not t3:
        fails.append(3)

    # 4. TRUST-RADIUS GATE: a dynamic-flagged Gaussian far from every track gets
    #    NO displacement (moving it by a far object's flow scatters it into the
    #    background — the measured failure of the rigid modes).
    traj, ok = out[0]
    gpts2 = torch.cat([gpts, torch.tensor([[10.0, 10.0, 0.0]])], 0)
    gfidx2 = torch.cat([gfidx, torch.tensor([0])])
    gdyn2 = torch.cat([gdyn, torch.tensor([True])])
    disp2, valid2 = dyn_motion.knn_flow_displacement(
        traj, ok, gpts2, gfidx2, gdyn2, V, k=4, gate_mult=3.0)
    t4 = disp2[-1].abs().max() < 1e-6 and valid2[-1].abs().max() < 1e-6
    print(f"[4] far-from-scaffold Gaussian is gated (no motion): {'PASS' if t4 else 'FAIL'}")
    if not t4:
        fails.append(4)

    # 5. OCCLUSION AT THE TARGET: tracks invisible at frame j contribute nothing;
    #    if NO neighbour is visible at j, the Gaussian is invalid there (renders at
    #    its source position) instead of moving by garbage.
    ok_occ = ok.clone()
    ok_occ[2] = False
    disp3, valid3 = dyn_motion.knn_flow_displacement(
        traj, ok_occ, gpts, gfidx, gdyn, V, k=4, gate_mult=3.0)
    sel0 = gdyn & (gfidx == 0)
    t5 = (valid3[sel0][:, 2].abs().max() < 1e-6
          and disp3[sel0][:, 2].abs().max() < 1e-6
          and (valid3[sel0][:, 1] > 0).all())
    print(f"[5] target-frame occlusion -> invalid, not garbage: {'PASS' if t5 else 'FAIL'}")
    if not t5:
        fails.append(5)

    # 6. STRICT MODE on CONSTANT-VELOCITY motion: the leave-one-out fit reconstructs
    #    the held-out frame exactly, so strict must match the observed-j result. This
    #    is what makes the strict/non-strict comparison interpretable: any difference
    #    it reports on real data is unpredictable motion, not a broken control.
    disp_s, valid_s = dyn_motion.knn_flow_displacement(
        traj, ok, gpts, gfidx, gdyn, V, k=4, gate_mult=3.0, strict=True)
    t6 = torch.allclose(disp_s, disp, atol=1e-4) and torch.allclose(valid_s, valid)
    print(f"[6] strict == observed on constant-velocity motion: {'PASS' if t6 else 'FAIL'} "
          f"(max diff {(disp_s - disp).abs().max().item():.2e})")
    if not t6:
        fails.append(6)

    # 7. STRICT MODE NEVER READS FRAME j. Corrupting ONLY frame j's track positions
    #    must leave the strict displacement toward j unchanged, while the non-strict
    #    one must move. Without this, a "strict" run could silently still be leaking.
    traj_bad = traj.clone()
    traj_bad[2] += 5.0                                   # frame 2 tracks -> garbage
    d_bad_s, _ = dyn_motion.knn_flow_displacement(
        traj_bad, ok, gpts, gfidx, gdyn, V, k=4, gate_mult=3.0, strict=True)
    d_bad_o, _ = dyn_motion.knn_flow_displacement(
        traj_bad, ok, gpts, gfidx, gdyn, V, k=4, gate_mult=3.0, strict=False)
    sel01 = gdyn & (gfidx != 2)
    unchanged = torch.allclose(d_bad_s[sel01][:, 2], disp_s[sel01][:, 2], atol=1e-4)
    leaked = not torch.allclose(d_bad_o[sel01][:, 2], disp[sel01][:, 2], atol=1e-4)
    t7 = unchanged and leaked
    print(f"[7] strict ignores corrupted frame-j tracks (and non-strict does not): "
          f"{'PASS' if t7 else 'FAIL'} (strict unchanged={unchanged}, observed moved={leaked})")
    if not t7:
        fails.append(7)

    # 8. predict_tracks_loo recovers a linear trajectory exactly at every frame.
    lin = torch.stack([torch.tensor([1.0, 2.0, 3.0]) * f for f in range(V)], 0).unsqueeze(1)
    pred, pok = dyn_motion.predict_tracks_loo(lin, torch.ones(V, 1, dtype=torch.bool))
    t8 = torch.allclose(pred, lin, atol=1e-4) and bool(pok.all())
    print(f"[8] LOO fit reproduces a linear trajectory: {'PASS' if t8 else 'FAIL'} "
          f"(max diff {(pred - lin).abs().max().item():.2e})")
    if not t8:
        fails.append(8)

    # 9. LOCALLY WEIGHTED strict fit. (a) bandwidth=0 must reproduce the uniform fit
    #    BIT-FOR-BIT, or the measured +0.55 dB strict result silently stops being the
    #    configuration the code runs. (b) weighted least squares is exact for linear
    #    data at ANY bandwidth. (c) on ACCELERATING motion a local fit must beat the
    #    global one at predicting the held-out frame -- the reason the knob exists.
    okl = torch.ones(V, 1, dtype=torch.bool)
    p_uni, _ = dyn_motion.predict_tracks_loo(lin, okl)
    p_bw0, _ = dyn_motion.predict_tracks_loo(lin, okl, bandwidth=0.0)
    same0 = torch.equal(p_uni, p_bw0)
    p_loc, _ = dyn_motion.predict_tracks_loo(lin, okl, bandwidth=1.5)
    exact = torch.allclose(p_loc, lin, atol=1e-4)

    Vq = 6
    tq = torch.arange(Vq, dtype=torch.float32)
    quad = torch.stack([tq ** 2, 0.5 * tq ** 2, torch.zeros(Vq)], -1).unsqueeze(1)  # [V,1,3]
    okq = torch.ones(Vq, 1, dtype=torch.bool)
    e_glob = (dyn_motion.predict_tracks_loo(quad, okq)[0] - quad).norm(dim=-1).mean()
    e_loc = (dyn_motion.predict_tracks_loo(quad, okq, bandwidth=1.5)[0] - quad).norm(dim=-1).mean()
    better = e_loc < e_glob
    t9 = same0 and exact and better
    print(f"[9] locally weighted strict fit: {'PASS' if t9 else 'FAIL'} "
          f"(bw=0 identical={same0}, exact on linear={exact}, "
          f"accelerating err {e_glob:.3f} global -> {e_loc:.3f} local)")
    if not t9:
        fails.append(9)

    print(f"\n{'ALL TESTS PASS' if not fails else f'FAILED: tests {fails}'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
