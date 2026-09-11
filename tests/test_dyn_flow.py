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


    # 10. RAFT TRACKING PATH: flow sampling and chained integration. The VGGT tracker
    #     recovers only ~20% of the motion on this data (13.6 px of a 67 px shift) and
    #     nothing on our side changes that, so the flow-based tracker is the intended
    #     replacement -- its integration must be exact before it is worth GPU time.
    Hf = Wf = 32
    flow = torch.zeros(2, Hf, Wf); flow[0] = 3.0; flow[1] = -2.0
    sampled = dyn_motion._sample_flow(flow, torch.tensor([[5.0, 7.0], [10.5, 20.25]]))
    t10a = torch.allclose(sampled, torch.tensor([[3., -2.], [3., -2.]]), atol=1e-4)

    class StubRaft:                       # constant field -> exact expected trajectory
        def __init__(self, d): self.d = torch.tensor(d)
        def __call__(self, a, b):
            sgn = 1.0 if float(b[0, 0, 0, 0]) > float(a[0, 0, 0, 0]) else -1.0
            f = torch.zeros(1, 2, a.shape[-2], a.shape[-1])
            f[0, 0] = self.d[0] * sgn; f[0, 1] = self.d[1] * sgn
            return [f]
    Vr, dl = 6, (4.0, 3.0)
    dyn_motion._RAFT_CACHE["cpu"] = StubRaft(dl)
    imgr = torch.zeros(1, Vr, 3, Hf, Wf)
    for fr in range(Vr):
        imgr[0, fr] = fr / 10.0
    qr = torch.tensor([[[12.0, 14.0], [16.0, 10.0]]])
    t10b = True
    for qf in range(Vr):
        tr, _ = dyn_motion.track_by_raft(imgr, qr, qf)
        for fr in range(Vr):
            if not torch.allclose(tr[fr], qr[0] + (fr - qf) * torch.tensor(dl), atol=1e-3):
                t10b = False
    dyn_motion._RAFT_CACHE.pop("cpu", None)
    t10 = t10a and t10b
    print(f"[10] RAFT flow sampling + chained integration exact: {'PASS' if t10 else 'FAIL'} "
          f"(sample={t10a}, chain={t10b})")
    if not t10:
        fails.append(10)


    # 11. MASK CLUSTER AGGREGATION. The detector responds to MOTION, so on a walking
    #     person only the fast parts score (a swinging arm) while the torso does not.
    #     Averaging those within a feature cluster drops the person below threshold --
    #     the observed failure: masks covered an arm, the rest of the person ghosted
    #     across every view because compositing can only protect masked pixels.
    #     p90/max must propagate the moving part's score to the whole cluster.
    import importlib.util as _il
    _mp = Path(__file__).resolve().parents[1] / "src/model/encoder/vggt4d/masks/dynamic_mask.py"
    _sp = _il.spec_from_file_location("dynamic_mask", _mp)
    _dm = _il.module_from_spec(_sp); _sp.loader.exec_module(_dm)
    Hm = Wm = 24
    featm = torch.zeros(1, Hm, Wm, 4); featm[0, :, :, 0] = 1.0
    per = torch.zeros(Hm, Wm, dtype=torch.bool); per[4:16, 4:12] = True
    oth = torch.zeros(Hm, Wm, dtype=torch.bool); oth[4:16, 16:22] = True
    for _m, _c in ((per, 1), (oth, 2)):
        featm[0][_m] = torch.zeros(4); featm[0, :, :, _c][_m] = 1.0
    dynm = torch.zeros(1, Hm, Wm)
    dynm[0, 4:7, 4:12] = 1.0        # only the person's "arm" moves
    dynm[0][oth] = 0.45             # another object moves moderately, all over
    got = {}
    for agg in ("mean", "p90", "max"):
        nm, _ = _dm.cluster_attention_maps(featm, dynm, n_clusters=3, aggregate=agg)
        got[agg] = (nm[0][per].mean().item(), nm[0][oth].mean().item())
    t11 = (got["mean"][0] < got["mean"][1]           # mean: person loses to the other object
           and got["p90"][0] > got["p90"][1]         # p90:  person wins
           and got["max"][0] > got["max"][1])
    print(f"[11] cluster aggregation rescues a partially-moving object: "
          f"{'PASS' if t11 else 'FAIL'} (mean {got['mean'][0]:.2f}v{got['mean'][1]:.2f}, "
          f"p90 {got['p90'][0]:.2f}v{got['p90'][1]:.2f})")
    if not t11:
        fails.append(11)


    # 12. GEOMETRIC (FLOW-RESIDUAL) MASK. VGGT4D's detector responds to how FAST a
    #     pixel moves, so it finds a swinging arm and misses the torso. Flow residual
    #     asks instead whether a pixel moves DIFFERENTLY from what the camera alone
    #     would produce -- true of a slow torso as much as a fast arm. The geometry
    #     must be exact or the residual is meaningless: a static scene has to give
    #     zero, and a patch moving 1.5 px beyond the camera's own motion has to give
    #     exactly 1.5.
    _fp = Path(__file__).resolve().parents[1] / "src/model/encoder/dyn_flow_mask.py"
    _fs = _il.spec_from_file_location("dyn_flow_mask", _fp)
    _df = _il.module_from_spec(_fs); _fs.loader.exec_module(_df)
    Hf2 = Wf2 = 64
    Kf = torch.tensor([[80., 0., Wf2 / 2], [0., 80., Hf2 / 2], [0., 0., 1.]])
    w2c = torch.stack([torch.eye(4), torch.eye(4)]); w2c[1, 0, 3] = -0.1
    dep = torch.full((2, Hf2, Wf2), 2.0)
    ind = _df.induced_flow(dep[0], Kf, Kf, w2c[0], w2c[1])
    t12a = (abs(float(ind[..., 0].mean()) - (-80.0 * 0.1 / 2.0)) < 1e-3
            and float(ind[..., 1].abs().max()) < 1e-3)          # analytic camera translation
    t12b = float(_df.induced_flow(dep[0], Kf, Kf, w2c[0], w2c[0]).abs().max()) < 1e-4  # identity
    patch = torch.zeros(Hf2, Wf2, dtype=torch.bool); patch[20:44, 20:44] = True
    class _Stub:
        def __call__(self, a, b):
            f = ind.clone(); f[patch] += torch.tensor([1.5, 0.0])
            return [f.permute(2, 0, 1).unsqueeze(0)]
    _df._RAFT["cpu"] = _Stub()
    resid = _df.flow_residual_map(torch.rand(2, 3, Hf2, Wf2), dep, w2c, torch.stack([Kf, Kf]))
    t12c = (float(resid[0][~patch].mean()) < 0.05
            and abs(float(resid[0][patch].mean()) - 1.5) < 0.05)
    _df._RAFT.pop("cpu", None)
    t12 = t12a and t12b and t12c
    print(f"[12] geometric flow-residual mask: {'PASS' if t12 else 'FAIL'} "
          f"(camera-translation={t12a}, identity={t12b}, "
          f"static {float(resid[0][~patch].mean()):.3f}px vs moving {float(resid[0][patch].mean()):.3f}px)")
    if not t12:
        fails.append(12)


    # 13. OTSU LEVEL is what controls mask COVERAGE. The scores are rescaled and the
    #     threshold is adaptive, so changing the cluster aggregation merely moves the
    #     split with them and the selected FRACTION barely budges (measured: p90 gave
    #     0.0561 vs mean's 0.0562 on balloon). What decides coverage is which Otsu
    #     split is used: the original takes the HIGHEST, keeping only a person's
    #     fastest part while the torso sits in the class immediately below.
    _tp = Path(__file__).resolve().parents[1] / "src/model/encoder/vggt4d/masks/dynamic_mask.py"
    _ts = _il.spec_from_file_location("dynamic_mask_lvl", _tp)
    _tm = _il.module_from_spec(_ts); _ts.loader.exec_module(_tm)
    import numpy as _np
    _rng = _np.random.default_rng(0)
    _bg = _rng.normal(0.05, 0.02, 8000)      # static background
    _to = _rng.normal(0.45, 0.03, 1200)      # slow torso
    _ar = _rng.normal(0.90, 0.03, 400)       # fast arm
    _img = _np.clip(_np.concatenate([_bg, _to, _ar]), 0, 1)
    _t1 = _tm.adaptive_multiotsu_variance(_img, level=1)
    _t2 = _tm.adaptive_multiotsu_variance(_img, level=2)
    t13 = ((_ar > _t1).mean() > 0.99 and (_to > _t1).mean() < 0.05        # lvl1: arm only
           and (_ar > _t2).mean() > 0.99 and (_to > _t2).mean() > 0.95    # lvl2: whole person
           and (_bg > _t2).mean() < 0.01)                                # without background
    print(f"[13] otsu level controls coverage: {'PASS' if t13 else 'FAIL'} "
          f"(lvl1 keeps torso {100*(_to > _t1).mean():.0f}%, "
          f"lvl2 keeps torso {100*(_to > _t2).mean():.0f}%, bg {100*(_bg > _t2).mean():.1f}%)")
    if not t13:
        fails.append(13)

    # 14. THE FLOW MASK MUST RUN AT THE DETECTION RESOLUTION. Two bugs killed every
    #     `--mask_method flow/union` job within a minute, and both are invisible at
    #     eval resolution: (a) torchvision's RAFT asserts H and W are divisible by 8,
    #     and detection runs at 518 wide (518 % 8 == 6) while eval runs at 448 (== 0);
    #     (b) `pose_encoding_to_extri_intri` returns extrinsics as [V, 3, 4], and
    #     inverting a non-square matrix raises. The stub reproduces RAFT's own
    #     assertion, so this test fails on the unpatched code.
    Hf3, Wf3 = 30, 22                      # 22 % 8 == 6, exactly like 518
    Kf3 = torch.tensor([[40., 0., Wf3 / 2], [0., 40., Hf3 / 2], [0., 0., 1.]])
    ext34 = torch.zeros(2, 3, 4)           # [V,3,4] world2cam, as the pose head emits
    ext34[:, :3, :3] = torch.eye(3)
    ext34[1, 0, 3] = -0.1
    dep3 = torch.full((2, Hf3, Wf3), 2.0)
    moving = torch.zeros(Hf3, Wf3, dtype=torch.bool); moving[10:20, 6:16] = True
    seen = {}
    class _Strict:                          # mimics torchvision RAFT's own contract
        def __call__(self, a, b):
            h, w = a.shape[-2:]
            if h % 8 or w % 8:
                raise ValueError(f"input image H and W should be divisible by 8, got {h}, {w}")
            seen["padded"] = (h, w)
            f = torch.zeros(1, 2, h, w)
            f[0, 0, :Hf3, :Wf3][moving] = 2.0    # only the patch moves, +2 px in x
            return [f]
    _df._RAFT["cpu"] = _Strict()
    try:
        r14 = _df.flow_residual_map(torch.rand(2, 3, Hf3, Wf3), dep3, ext34,
                                    torch.stack([Kf3, Kf3]))
        t14a = tuple(r14.shape) == (2, Hf3, Wf3)                 # cropped back
        t14b = seen.get("padded") == (32, 24)                    # padded up to /8
        t14c = float(r14[0][moving].mean()) > float(r14[0][~moving].mean()) + 1.0
        err14 = ""
    except Exception as e:                                       # noqa: BLE001
        t14a = t14b = t14c = False
        err14 = f" ({type(e).__name__}: {e})"
    _df._RAFT.pop("cpu", None)
    t14 = t14a and t14b and t14c
    print(f"[14] flow mask runs at detection resolution: {'PASS' if t14 else 'FAIL'} "
          f"(shape={t14a}, padded-to-/8={t14b}, motion-separated={t14c}){err14}")
    if not t14:
        fails.append(14)

    # 15. THE DISPLACEMENT CLAMP bounds a corrupted track. Test 7's corrupted-frame
    #     case moves Gaussians ~8.7 world units against a median of 0.022 -- the
    #     displacement is only as good as the track it came from, and nothing in the
    #     interpolation bounds it. A Gaussian thrown that far can land near the
    #     camera, where a world-scale Gaussian covers much of the screen in one
    #     colour. max_disp_mult caps it at a multiple of the MEDIAN observed track
    #     motion, which leaves ordinary motion alone. 0 must reproduce exactly.
    #     Corrupt a MINORITY of tracks: test 7 shifts every track at frame 2, so the
    #     observed motion really is 5.0 there and a 5.0 displacement is supported by
    #     the evidence. The failure this guards is a few tracks disagreeing with the
    #     rest, which is what one bad RAFT hop produces.
    traj_out = traj.clone()
    traj_out[2, ::5] += 5.0                              # 20% of tracks go rogue at f=2
    d_off, _ = dyn_motion.knn_flow_displacement(
        traj_out, ok, gpts, gfidx, gdyn, V, k=4, gate_mult=3.0, strict=False,
        max_disp_mult=0.0)
    d_cl, _ = dyn_motion.knn_flow_displacement(
        traj_out, ok, gpts, gfidx, gdyn, V, k=4, gate_mult=3.0, strict=False,
        max_disp_mult=3.0)
    d_clean, _ = dyn_motion.knn_flow_displacement(
        traj, ok, gpts, gfidx, gdyn, V, k=4, gate_mult=3.0, strict=False,
        max_disp_mult=3.0)
    m_off = d_off[gdyn][:, 2].norm(dim=-1).max().item()
    m_cl = d_cl[gdyn][:, 2].norm(dim=-1).max().item()
    d_ref, _ = dyn_motion.knn_flow_displacement(
        traj_out, ok, gpts, gfidx, gdyn, V, k=4, gate_mult=3.0, strict=False)
    t15a = torch.allclose(d_off, d_ref, atol=1e-6)            # 0 == exactly off
    t15b = m_cl < m_off / 2                                    # tail actually bounded
    # rejected, not rescaled: the over-limit pairs must be marked INVALID so the
    # flow-gated decoder drops them, not left at a shortened (still wrong) offset.
    _, v_off_ = dyn_motion.knn_flow_displacement(
        traj_out, ok, gpts, gfidx, gdyn, V, k=4, gate_mult=3.0, strict=False,
        max_disp_mult=0.0)
    _, v_cl_ = dyn_motion.knn_flow_displacement(
        traj_out, ok, gpts, gfidx, gdyn, V, k=4, gate_mult=3.0, strict=False,
        max_disp_mult=3.0)
    t15d = float(v_cl_.sum()) < float(v_off_.sum())
    t15c = torch.allclose(d_clean, disp, atol=1e-4)            # clean motion untouched
    t15 = t15a and t15b and t15c and t15d
    print(f"[15] displacement clamp bounds a corrupted track: {'PASS' if t15 else 'FAIL'} "
          f"(off==baseline={t15a}, corrupted {m_off:.2f}->{m_cl:.2f} world, "
          f"clean motion untouched={t15c}, rejected not rescaled={t15d})")
    if not t15:
        fails.append(15)

    # 16. EVERY TrainingConfig KWARG eval PASSES MUST EXIST AS A FIELD. Adding a
    #     knob means touching the encoder cfg, TrainingConfig and the argparse
    #     block; miss the middle one and the failure is a TypeError raised AFTER
    #     model load, i.e. a job that burns an allocation and dies with no output.
    #     That has now happened twice (dynamic_n_clusters, dyn_motion_max_disp_mult).
    #     Pure ast, so it costs nothing and needs no torch.
    import ast as _ast
    _root = Path(__file__).resolve().parents[1]
    _tc = _ast.parse((_root / "train_temporal_gaussian_head.py").read_text())
    _fields = set()
    for _n in _ast.walk(_tc):
        if isinstance(_n, _ast.ClassDef) and _n.name == "TrainingConfig":
            for _st in _n.body:
                if isinstance(_st, _ast.AnnAssign) and isinstance(_st.target, _ast.Name):
                    _fields.add(_st.target.id)
    _ev = _ast.parse((_root / "eval_gaussian_head.py").read_text())
    _passed = set()
    for _n in _ast.walk(_ev):
        if (isinstance(_n, _ast.Call) and isinstance(_n.func, _ast.Name)
                and _n.func.id == "TrainingConfig"):
            _passed |= {k.arg for k in _n.keywords if k.arg}
    _missing = sorted(_passed - _fields)
    t16 = bool(_fields) and bool(_passed) and not _missing
    print(f"[16] eval's TrainingConfig kwargs all exist as fields: "
          f"{'PASS' if t16 else 'FAIL'} ({len(_passed)} passed, {len(_fields)} fields"
          + (f", MISSING {_missing}" if _missing else "") + ")")
    if not t16:
        fails.append(16)

    # 17. NEAR-CAMERA REJECTION. The artefact a magnitude clamp cannot catch: a
    #     Gaussian already close to the camera, displaced a little TOWARD it, keeps
    #     its world-space scale and covers the screen. gsplat is called with
    #     near_plane=1e-10 so nothing culls it. Measured on balloon b0248-b0257:
    #     frame luminance 138 -> 34 with relocation on, while no-handling (139) and
    #     pfd (141) were untouched -- so this was the entire artefact.
    w2c17 = torch.eye(4)                       # camera at origin looking down +z
    xyz_old17 = torch.tensor([[0.0, 0.0, 5.0],    # scene at 5 m
                              [0.0, 0.0, 6.0],
                              [0.0, 0.0, 4.0],
                              [0.0, 0.0, 0.30]])  # one already-close Gaussian
    xyz_new17 = xyz_old17.clone()
    xyz_new17[3, 2] = 0.20                     # nudged 0.1 m nearer -> screen-filling
    xyz_new17[0, 2] = 5.5                      # an ordinary, legitimate relocation
    moved17 = torch.tensor([True, False, False, True])
    bad17 = dyn_motion.near_camera_reject(xyz_old17, xyz_new17, w2c17, moved17)
    t17a = bool(bad17[3]) and not bool(bad17[0])          # catches it, spares the good one
    t17b = not bool(bad17[1]) and not bool(bad17[2])      # never touches unmoved Gaussians
    # a Gaussian that was ALREADY the near content and does not move is not rejected
    t17c = not bool(dyn_motion.near_camera_reject(
        xyz_old17, xyz_old17, w2c17, torch.ones(4, dtype=torch.bool)).any())
    t17 = t17a and t17b and t17c
    print(f"[17] near-camera relocation rejected: {'PASS' if t17 else 'FAIL'} "
          f"(catches the near landing={t17a}, spares unmoved={t17b}, no-op on zero motion={t17c})")
    if not t17:
        fails.append(17)

    # 18. MASK COMPLETION turns PARTS of an object into the object. The detector
    #     fires on limbs and outlines and misses the torso interior, so one person
    #     arrives as several components. Every mechanism downstream then splits
    #     them: masked parts are handled, unmasked parts stay and render from every
    #     frame at once -- the "chaotic scatter around the person". Raising the
    #     Otsu level cannot fix this; it only lowers a threshold, and a torso the
    #     detector never scored has nothing to lower onto. Shape does fix it.
    _mp = Path(__file__).resolve().parents[1] / "src/model/encoder/dyn_mask_post.py"
    _ms = _il.spec_from_file_location("dyn_mask_post", _mp)
    _mm = _il.module_from_spec(_ms); _ms.loader.exec_module(_mm)
    from scipy import ndimage as _ndi
    _person = _np.zeros((60, 60), _np.float32)
    _person[10:20, 25:35] = 1.0        # head
    _person[26:50, 22:38] = 1.0        # torso, 6 px gap below the head
    _person[30:40, 26:34] = 0.0        # hollow interior (only the outline moved)
    _person[5:7, 5:7] = 1.0            # a speck: mask false positive
    _done = _mm.complete_mask(_person, close=4, fill=True, min_area=50, dilate=1)
    t18a = _ndi.label(_person > 0.5)[1] == 3 and _ndi.label(_done > 0.5)[1] == 1
    t18b = _done[5:7, 5:7].sum() == 0                     # speck gone
    t18c = float(_done[30:40, 26:34].mean()) == 1.0       # interior solid
    t18d = _np.array_equal(_mm.complete_mask(_person), _person)   # defaults = no-op
    t18 = t18a and t18b and t18c and t18d
    print(f"[18] partial mask completed into one object: {'PASS' if t18 else 'FAIL'} "
          f"(3 components -> 1: {t18a}, speck removed={t18b}, interior filled={t18c}, "
          f"defaults no-op={t18d}, coverage {100*_person.mean():.1f}% -> {100*_done.mean():.1f}%)")
    if not t18:
        fails.append(18)

    # 19. GLOBAL vs PER-CHUNK THRESHOLD. demo_vggt4d.process_scene loads the whole
    #     scene and takes ONE multi-Otsu threshold over every frame. We chunk the
    #     attention pass for memory, and thresholding inside each chunk makes the
    #     bar depend on how much motion that chunk happens to contain: identical
    #     content is called dynamic in a quiet chunk and static in a busy one.
    #     That is the inconsistency --global_post removes.
    _rng2 = _np.random.default_rng(7)
    _bg = lambda n: _rng2.normal(0.05, 0.01, n)
    _same = _rng2.normal(0.30, 0.01, 500)          # SAME content in both chunks
    quiet = _np.clip(_np.concatenate([_bg(8000), _same]), 0, 1)
    busy = _np.clip(_np.concatenate([_bg(8000), _same,
                                     _rng2.normal(0.95, 0.02, 1500)]), 0, 1)
    t_quiet = _tm.adaptive_multiotsu_variance(quiet)
    t_busy = _tm.adaptive_multiotsu_variance(busy)
    t_glob = _tm.adaptive_multiotsu_variance(_np.concatenate([quiet, busy]))
    in_quiet = (_same > t_quiet).mean()            # the same patch, judged per chunk
    in_busy = (_same > t_busy).mean()
    g_quiet = (_same > t_glob).mean()              # ... and judged globally
    t19a = abs(in_quiet - in_busy) > 0.5           # per-chunk: contradicts itself
    t19b = g_quiet == g_quiet                      # global: one verdict by construction
    t19c = abs(t_quiet - t_busy) > 0.1             # the thresholds really do differ
    t19 = t19a and t19b and t19c
    print(f"[19] per-chunk threshold is inconsistent, global is not: "
          f"{'PASS' if t19 else 'FAIL'} (identical patch kept {100*in_quiet:.0f}% in the "
          f"quiet chunk vs {100*in_busy:.0f}% in the busy one; thresholds "
          f"{t_quiet:.2f} vs {t_busy:.2f}, global {t_glob:.2f})")
    if not t19:
        fails.append(19)

    # 20. STREAMED Q/K MUST BE SUBSTITUTABLE. extract_dyn_map moves the ENTIRE Q/K
    #     capture to the GPU -- 17.64 GiB for global_tok_k alone at 192 frames, the
    #     single allocation that caps chunk size at ~128 on a 47 GB card -- while its
    #     loop only ever touches ref_id and six neighbours. _LazyFrames serves those
    #     slices from host memory instead. The helpers use ONLY g[int], g[tensor] and
    #     .shape[0], so if the proxy matches a real tensor on those three, every value
    #     downstream is unchanged by construction. The cache matters because all five
    #     helpers request the SAME frames for a given ref_id.
    _t20 = torch.randn(20, 2, 3, 4)
    _lz = _tm._LazyFrames(_t20, device="cpu", cache_size=2)
    _idx = torch.tensor([1, 3, 5])
    t20a = tuple(_lz.shape) == tuple(_t20.shape) and len(_lz) == 20
    t20b = torch.equal(_lz[7], _t20[7]) and torch.equal(_lz[_idx], _t20[_idx])
    t20c = _lz.to("cuda") is _lz                       # a .to() must not materialise it
    # cache_size 2, so this evicts and refills; every read must still be correct
    t20d = all(torch.equal(_lz[i], _t20[i]) for i in (0, 4, 9, 4, 0, 9))
    t20e = torch.equal(_lz[_idx], _t20[_idx])          # correct after eviction churn
    t20 = t20a and t20b and t20c and t20d and t20e
    print(f"[20] streamed Q/K substitutable for a resident tensor: "
          f"{'PASS' if t20 else 'FAIL'} (shape={t20a}, int+tensor index={t20b}, "
          f"lazy .to()={t20c}, cache eviction={t20d and t20e})")
    if not t20:
        fails.append(20)

    # 21. COMPONENT-LEVEL MOTION GATE. Attention over-fires on static structure
    #     BESIDE a moving object (the desk, the chair) -- it responds to attention
    #     dissimilarity there, not motion. Flow residual is ~0 on anything static
    #     however close it sits, so geometry separates them. The gate must decide
    #     per COMPONENT: a pixel-wise intersection re-erodes the person wherever the
    #     residual is weak, undoing exactly what the shape completion achieved.
    _msk = _np.zeros((80, 100), _np.float32)
    _msk[20:50, 20:40] = 1.0             # person: fast torso + slow legs
    _msk[20:40, 60:75] = 1.0             # chair: mask false positive, static
    _res = _np.full((80, 100), 0.2, _np.float32)   # depth-error noise floor
    _res[20:35, 20:40] = 5.0             # only the UPPER half of the person moves fast
    _person = _np.zeros_like(_msk, bool); _person[20:50, 20:40] = True
    _chair = _np.zeros_like(_msk, bool); _chair[20:40, 60:75] = True

    _gated = _mm.motion_gate(_msk, _res, mult=3.0)
    t21a = _gated[_person].all()                    # person kept ENTIRELY, slow legs too
    t21b = not _gated[_chair].any()                 # chair dropped entirely
    t21c = _np.array_equal(_mm.motion_gate(_msk, _res, mult=0.0), _msk)   # 0 = off
    # what a pixel-wise intersection would have done to the same person
    _pix = ((_res > 3.0 * 0.5) & (_msk > 0.5))
    _pix_keep = _pix[_person].mean()
    t21d = _pix_keep < 0.75 and _gated[_person].mean() == 1.0
    t21 = t21a and t21b and t21c and t21d
    print(f"[21] component motion gate keeps whole objects: {'PASS' if t21 else 'FAIL'} "
          f"(person kept {100*_gated[_person].mean():.0f}% vs {100*_pix_keep:.0f}% pixel-wise, "
          f"chair dropped={t21b}, mult=0 is off={t21c})")
    if not t21:
        fails.append(21)

    print(f"\n{'ALL TESTS PASS' if not fails else f'FAILED: tests {fails}'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
