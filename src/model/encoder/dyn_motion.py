"""Per-group motion model for dynamic Gaussians, driven by VGGT4D's point tracker.

WHY THIS EXISTS
Gaussian positions come from the FROZEN depth/pose heads, so a dynamic Gaussian sits
where its object was in ITS OWN source frame. Rendering another timestamp therefore
shows the object at the wrong place ("ghosting"). To fix it we must MOVE those
Gaussians — a geometry change, unreachable by fine-tuning the appearance head.

The first attempt used ONE 3D centroid for all dynamic content, i.e. a single rigid
translation. That failed on scenes with several independently-moving things (a person
AND a box get averaged into one bogus velocity), and a wrong displacement scatters
Gaussians into background regions, hurting even static pixels.

This module upgrades that to PIECEWISE-rigid motion:
  1. `TrackHead` follows a set of query points across the window. The tracker only
     accepts queries in frame 0 (base_track_predictor samples fmaps[:, 0]), which is
     fine: ONE call gives each point's position in EVERY frame, so we get genuine
     correspondence over time for the cost of one head call.
  2. Each tracked point is lifted to 3D by sampling the frozen world-point map at its
     tracked pixel — so trajectories live in the same world frame as the Gaussians.
  3. Tracks are clustered into K groups, giving one coherent motion per moving object.
  4. Per group and per frame we take the centroid, then predict the centroid at each
     target frame j by a LEAVE-ONE-OUT linear (constant-velocity) fit that EXCLUDES
     frame j — so nothing is read off the held-out frame.
  5. Every dynamic pixel is assigned to its nearest group, so each Gaussian inherits
     the motion of the object it actually belongs to.

K=1 reduces to the previous single-centroid behaviour.

--------------------------------------------------------------------------------
UPGRADE (Aug 2026): TRACK-CORRESPONDENCE SCENE FLOW (collect_dyn_tracks +
knn_flow_displacement). The piecewise-rigid model above measured ~null
(dynamic -0.06..-0.28 dB vs no handling) for identifiable reasons:
  (a) it EXTRAPOLATES: a constant-velocity fit over the other frames predicts the
      centroid at frame j, even though the tracker OBSERVED every point at j;
  (b) it is RIGID per cluster: one translation cannot represent a walking person;
  (c) queries come only from frame 0, so objects appearing later never move.
The scene-flow mode fixes all three:
  1. Displacement of a Gaussian toward target frame j is interpolated from the
     tracked points' OBSERVED positions at j (direct correspondence, no motion
     model fitted at all).
  2. Interpolation is per-Gaussian: K nearest tracks in 3D (source frame),
     inverse-distance weighted -> a non-rigid deformation field (MoSca /
     Shape-of-Motion style scaffold, feed-forward).
  3. Queries are drawn from EVERY frame's dynamic pixels (the tracker only takes
     frame-0 queries, so frames are permuted per call to put each query frame
     first).

PROTOCOL NOTE (thesis): using the track position AT frame j reads frame j's
pixels for GEOMETRY. This is the standard monocular dynamic-NVS protocol
("motion fitted on the full video, appearance held out") — under leave-one-out
the Gaussians (appearance + source geometry) still come exclusively from the
other frames. The strict no-look-at-j variant remains available as the
piecewise-rigid mode above; report both.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _lift_tracks_to_world(pts_all_b: torch.Tensor, tracks_b: torch.Tensor) -> torch.Tensor:
    """Sample the world-point map at tracked pixel positions.

    pts_all_b: [V, H, W, 3] world points (frozen geometry).
    tracks_b:  [V, Nq, 2] tracked pixel coords (x, y) in each frame.
    -> [V, Nq, 3] world position of every track in every frame.
    """
    V, H, W, _ = pts_all_b.shape
    grid = tracks_b.clone().float()
    grid[..., 0] = (grid[..., 0] / max(W - 1, 1)) * 2.0 - 1.0   # x -> [-1, 1]
    grid[..., 1] = (grid[..., 1] / max(H - 1, 1)) * 2.0 - 1.0   # y -> [-1, 1]
    maps = pts_all_b.permute(0, 3, 1, 2).float()                # [V, 3, H, W]
    out = F.grid_sample(maps, grid.unsqueeze(1), mode="bilinear",
                        align_corners=True, padding_mode="border")  # [V,3,1,Nq]
    return out[:, :, 0, :].permute(0, 2, 1).contiguous()        # [V, Nq, 3]


def _predict_loo_linear(traj: torch.Tensor, ok: torch.Tensor, min_pts: int = 3) -> torch.Tensor:
    """Leave-one-out constant-velocity prediction.

    traj: [V, K, 3] value per frame per group. ok: [V, K] bool usable.
    For each target frame j, fit a line over the OTHER usable frames and evaluate at j,
    so the prediction at j never uses frame j. -> [V, K, 3]
    """
    V, K, _ = traj.shape
    idx = torch.arange(V, device=traj.device, dtype=traj.dtype)
    pred = traj.clone()
    for k in range(K):
        for j in range(V):
            m = ok[:, k].clone()
            m[j] = False                       # exclude the target frame (no leakage)
            n = int(m.sum())
            if n == 0:
                continue                       # keep own value -> zero displacement
            t = idx[m]
            c = traj[m, k]                     # [n, 3]
            if n < min_pts:
                pred[j, k] = c.mean(0)
                continue
            tm = t.mean()
            cm = c.mean(0)
            dt = t - tm
            den = (dt * dt).sum()
            if den < 1e-8:
                pred[j, k] = cm
                continue
            slope = (dt.unsqueeze(-1) * (c - cm)).sum(0) / den
            pred[j, k] = cm + slope * (idx[j] - tm)
    return pred


@torch.no_grad()
def compute_dyn_group_motion(
    track_head,
    aggregated_tokens_list,
    image: torch.Tensor,
    patch_start_idx: int,
    pts_all: torch.Tensor,
    dyn_mask: torch.Tensor,
    conf_valid_mask: torch.Tensor,
    n_query: int = 384,
    n_groups: int = 4,
    min_track_pts: int = 8,
    gate_radius_mult: float = 2.5,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Estimate piecewise-rigid motion of the dynamic content.

    pts_all:         [B, V, H, W, 3] world points   dyn_mask/conf_valid_mask: [B, V, H, W]
    Returns (centroid, pred, valid, group_map) or None if tracking is not possible:
        centroid  [B, V, K, 3]  per-group per-frame 3D centroid
        pred      [B, V, K, 3]  leave-one-out predicted centroid at each target frame
        valid     [B, V, K]     bool, group had enough visible tracks in that frame
        group_map [B, V, H, W]  long, group id per pixel (-1 where not dynamic)
    """
    if track_head is None:
        return None
    B, V, H, W, _ = pts_all.shape
    dev = pts_all.device
    dyn = (dyn_mask > 0.5) & conf_valid_mask

    centroid = torch.zeros(B, V, n_groups, 3, device=dev, dtype=pts_all.dtype)
    valid = torch.zeros(B, V, n_groups, device=dev, dtype=torch.bool)
    group_map = torch.full((B, V, H, W), -1, device=dev, dtype=torch.long)
    pred = torch.zeros_like(centroid)
    any_ok = False

    for b in range(B):
        # --- query points: dynamic pixels of FRAME 0 (the tracker's required frame) ---
        ys, xs = torch.nonzero(dyn[b, 0], as_tuple=True)
        if ys.numel() < min_track_pts:
            continue                                    # nothing to track in frame 0
        if ys.numel() > n_query:                        # uniform subsample
            pick = torch.randperm(ys.numel(), device=dev)[:n_query]
            ys, xs = ys[pick], xs[pick]
        q = torch.stack([xs.float(), ys.float()], dim=-1).unsqueeze(0)   # [1, Nq, 2] (x, y)

        try:
            # Tokens are [B,S,P,C]; slice to THIS batch item so it matches images[b:b+1]
            # (a full-batch token list with a 1-image tensor silently mismatches for B>1).
            # .float(): the aggregator runs in bf16/fp16 but the heads expect fp32 tokens
            # (same treatment as camera_head/depth_head elsewhere in this repo).
            toks_b = [t[b:b + 1].float() for t in aggregated_tokens_list]
            with torch.amp.autocast("cuda", enabled=False):
                coord_preds, vis, _ = track_head(
                    toks_b, images=image[b:b + 1].float(),
                    patch_start_idx=patch_start_idx, query_points=q,
                )
        except Exception as e:                          # tracker unavailable/OOM -> skip
            print(f"[DynMotion] tracking failed ({e}); no motion model this batch")
            continue

        tracks = coord_preds[-1][0].float()             # [V, Nq, 2]
        if vis is not None:
            v0 = vis[0]
            while v0.dim() > 2:
                v0 = v0.squeeze(-1)
            vis_ok = v0 > 0.5                            # [V, Nq]
        else:
            vis_ok = torch.ones(tracks.shape[:2], device=dev, dtype=torch.bool)

        traj = _lift_tracks_to_world(pts_all[b], tracks)  # [V, Nq, 3]

        # --- group the tracks so each moving object gets its own motion -------------
        # Cluster on frame-0 3D position: spatially separated objects (person vs box)
        # separate cleanly, and it needs no motion estimate to bootstrap.
        p0 = traj[0]                                     # [Nq, 3]
        K = max(1, min(n_groups, p0.shape[0]))
        try:
            from sklearn.cluster import KMeans
            lab = torch.as_tensor(
                KMeans(n_clusters=K, n_init=4, random_state=0)
                .fit_predict(p0.detach().cpu().numpy()),
                device=dev, dtype=torch.long)
        except Exception:
            lab = torch.zeros(p0.shape[0], device=dev, dtype=torch.long)
            K = 1

        # --- per-group, per-frame centroid over VISIBLE tracks ---------------------
        radius = torch.zeros(V, n_groups, device=dev, dtype=pts_all.dtype)
        for k in range(K):
            sel = lab == k
            if int(sel.sum()) == 0:
                continue
            m = vis_ok[:, sel]                           # [V, nk]
            cnt = m.sum(1)                               # [V]
            c = (traj[:, sel] * m.unsqueeze(-1)).sum(1) / cnt.clamp_min(1).unsqueeze(-1)
            centroid[b, :, k] = c
            valid[b, :, k] = cnt >= min_track_pts
            # spatial extent of the group's tracks = how far its motion can be trusted
            d = (traj[:, sel] - c.unsqueeze(1)).norm(dim=-1)          # [V, nk]
            radius[:, k] = ((d * m).sum(1) / cnt.clamp_min(1)).clamp_min(1e-3)

        pred[b] = _predict_loo_linear(centroid[b], valid[b])

        # --- assign every dynamic pixel to its nearest group ----------------------
        # Uses the group's centroid IN THAT FRAME, so a pixel follows the object it
        # actually sits on rather than a global average.
        for s in range(V):
            sel = dyn[b, s]
            if not bool(sel.any()):
                continue
            p = pts_all[b, s][sel]                       # [n, 3]
            cs = centroid[b, s, :K]                      # [K, 3]
            okk = valid[b, s, :K]
            if not bool(okk.any()):
                continue
            d = torch.cdist(p.unsqueeze(0).float(), cs.unsqueeze(0).float())[0]  # [n, K]
            d = d.masked_fill(~okk.unsqueeze(0), float("inf"))
            nearest = d.argmin(-1)                                   # [n]
            dmin = d.gather(1, nearest.unsqueeze(1))[:, 0]
            # Only adopt a group's motion if the pixel plausibly belongs to it.
            # Otherwise leave it as -1 -> the decoder does not move it. Pixels of
            # objects that appear after frame 0 (never tracked) end up here, which is
            # the safe behaviour: no motion estimate, no displacement.
            keep = dmin < (gate_radius_mult * radius[s, :K].gather(0, nearest))
            lbl = torch.where(keep, nearest, torch.full_like(nearest, -1))
            group_map[b, s][sel] = lbl
        any_ok = True

    if not any_ok:
        return None

    # --- DIAGNOSTIC: did the mechanism actually DO anything? ----------------------
    # Without this, "tracking ~= baseline" is ambiguous: it could mean the
    # displacement does not help, OR that the validity/distance gates suppressed
    # nearly all displacement so the mechanism never really acted. Report the share
    # of dynamic pixels that received a motion, and how far they would move.
    n_dyn_px = int(dyn.sum())
    n_assigned = int((group_map >= 0).sum())
    mags = []
    for b in range(B):
        for k in range(n_groups):
            ok = valid[b, :, k]
            if int(ok.sum()) < 2:
                continue
            for i in range(V):
                if not bool(ok[i]):
                    continue
                d = (pred[b, :, k] - centroid[b, i, k]).norm(dim=-1)   # over targets j
                mags.append(d[ok])
    if mags:
        m = torch.cat(mags)
        print(f"[DynMotion] assigned {n_assigned}/{n_dyn_px} dynamic px "
              f"({100.0 * n_assigned / max(n_dyn_px, 1):.1f}%) | displacement "
              f"median={m.median().item():.4f} mean={m.mean().item():.4f} "
              f"p90={m.quantile(0.9).item():.4f} (world units)")
    else:
        print(f"[DynMotion] assigned {n_assigned}/{n_dyn_px} dynamic px — NO usable motion")
    # -----------------------------------------------------------------------------
    return centroid, pred, valid, group_map


# =============================================================================
# TRACK-CORRESPONDENCE SCENE FLOW (see module docstring, "UPGRADE")
# =============================================================================

def _lift_tracks_nearest(pts_all_b: torch.Tensor, tracks_b: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample the world-point map at tracked pixels with NEAREST sampling.

    Bilinear sampling (used by the centroid path) blends object and background
    depth at occlusion boundaries, producing phantom mid-air 3D points exactly
    where tracks sit (object silhouettes). Nearest avoids that.

    pts_all_b: [V, H, W, 3]   tracks_b: [V, Nq, 2] pixel coords (x, y).
    -> ([V, Nq, 3] world positions, [V, Nq] bool in-image-bounds)
    """
    V, H, W, _ = pts_all_b.shape
    x = tracks_b[..., 0].round().long()
    y = tracks_b[..., 1].round().long()
    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x = x.clamp(0, W - 1)
    y = y.clamp(0, H - 1)
    flat = pts_all_b.reshape(V, H * W, 3)
    idx = (y * W + x).unsqueeze(-1).expand(-1, -1, 3)           # [V, Nq, 3]
    return flat.gather(1, idx), in_bounds


def _run_track_head(track_head, toks_b: list, image_b1: torch.Tensor,
                    patch_start_idx: int, q: torch.Tensor, query_frame: int
                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the tracker with `query_frame` moved to position 0, outputs restored
    to the ORIGINAL frame order.

    The tracker samples query features from fmaps[:, 0] only; frames are
    otherwise treated symmetrically, so a permutation of the (already
    aggregated) token sequence dim + images is valid and lets any frame serve
    as the query frame.

    toks_b: list of [1, S, P, C]; image_b1: [1, S, 3, H, W]; q: [1, Nq, 2].
    -> (tracks [S, Nq, 2], vis [S, Nq] or None) in original frame order.
    """
    S = image_b1.shape[1]
    dev = image_b1.device
    order = torch.arange(S, device=dev)
    if query_frame != 0:
        order = torch.cat([order[query_frame:query_frame + 1],
                           order[:query_frame], order[query_frame + 1:]])
    toks_p = [t.index_select(1, order) for t in toks_b]
    img_p = image_b1.index_select(1, order)
    with torch.amp.autocast("cuda", enabled=False):
        coord_preds, vis, _ = track_head(
            toks_p, images=img_p, patch_start_idx=patch_start_idx, query_points=q,
        )
    inv = torch.empty_like(order)
    inv[order] = torch.arange(S, device=dev)
    tracks = coord_preds[-1][0].float().index_select(0, inv)     # [S, Nq, 2]
    if vis is not None:
        v0 = vis[0]
        while v0.dim() > 2:
            v0 = v0.squeeze(-1)
        vis = v0.index_select(0, inv)                            # [S, Nq]
    return tracks, vis


@torch.no_grad()
def collect_dyn_tracks(
    track_head,
    aggregated_tokens_list,
    image: torch.Tensor,
    patch_start_idx: int,
    pts_all: torch.Tensor,
    dyn_mask: torch.Tensor,
    conf_valid_mask: torch.Tensor,
    n_query: int = 1024,
    query_all_frames: bool = True,
    min_track_pts: int = 8,
) -> Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
    """Phase A of the scene-flow motion model: build a track scaffold.

    Runs the tracker with query points on the DYNAMIC pixels of each query frame
    (all frames by default — objects that appear mid-window get tracked from the
    frame they appear in) and lifts every track to world space through the
    frozen point map.

    Must run while the aggregated tokens still exist. Total query budget
    `n_query` is split across query frames.

    pts_all [B,V,H,W,3];  dyn_mask/conf_valid_mask [B,V,H,W].
    -> per batch item: (traj [V,Nt,3], ok [V,Nt] bool) or None. None overall if
       nothing tracked.
    """
    if track_head is None:
        return None
    B, V, H, W, _ = pts_all.shape
    dev = pts_all.device
    dyn = (dyn_mask.to(dev) > 0.5) & conf_valid_mask

    query_frames = list(range(V)) if query_all_frames else [0]
    quota = max(n_query // len(query_frames), 64)

    out: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
    any_ok = False
    for b in range(B):
        # fp32 tokens per batch item (aggregator runs bf16; heads expect fp32 —
        # same treatment as compute_dyn_group_motion above).
        toks_b = [t[b:b + 1].float() for t in aggregated_tokens_list]
        img_b = image[b:b + 1].float()
        traj_parts, ok_parts = [], []
        for qf in query_frames:
            ys, xs = torch.nonzero(dyn[b, qf], as_tuple=True)
            if ys.numel() < min_track_pts:
                continue
            if ys.numel() > quota:
                pick = torch.randperm(ys.numel(), device=dev)[:quota]
                ys, xs = ys[pick], xs[pick]
            q = torch.stack([xs.float(), ys.float()], dim=-1).unsqueeze(0)  # [1,Nq,2] (x,y)
            try:
                tracks, vis = _run_track_head(
                    track_head, toks_b, img_b, patch_start_idx, q, qf)
            except Exception as e:          # tracker unavailable/OOM -> skip this frame
                print(f"[DynFlow] tracking from frame {qf} failed ({e})")
                continue
            # DIAGNOSTIC: how far do the tracks actually travel, in 2D and in 3D?
            # These two numbers localise a lost-motion bug immediately. Compare the 2D
            # figure against the dynamic mask's own centroid displacement over the same
            # span (measurable straight from the mask PNGs): if the tracks barely move
            # while the mask does, the TRACKER is failing; if the tracks move but the 3D
            # trajectory does not, the DEPTH LIFT is flattening it.
            _d2d = (tracks - tracks[qf:qf + 1]).norm(dim=-1).max(0).values   # [Nq]
            traj, in_b = _lift_tracks_nearest(pts_all[b], tracks)   # [V,Nq,3], [V,Nq]
            _d3d = (traj - traj[qf:qf + 1]).norm(dim=-1).max(0).values       # [Nq]
            print(f"[DynTracks] qf={qf} n={tracks.shape[1]} | 2D travel median="
                  f"{_d2d.median().item():.1f}px p90={_d2d.quantile(0.9).item():.1f}px "
                  f"| 3D travel median={_d3d.median().item():.4f} "
                  f"p90={_d3d.quantile(0.9).item():.4f} world", flush=True)
            ok = in_b
            if vis is not None:
                ok = ok & (vis > 0.5)
            # A track is only a correspondence if its own query position is usable.
            ok = ok & ok[qf:qf + 1].expand_as(ok)
            traj_parts.append(traj)
            ok_parts.append(ok)
        if traj_parts:
            out.append((torch.cat(traj_parts, dim=1), torch.cat(ok_parts, dim=1)))
            any_ok = True
        else:
            out.append(None)
    return out if any_ok else None


@torch.no_grad()
def predict_tracks_loo(traj: torch.Tensor, ok: torch.Tensor, min_pts: int = 2,
                       bandwidth: float = 0.0
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """LEAVE-ONE-OUT prediction of every track's position at every frame.

    THE HONEST CONTROL for the scene-flow mode. Using a track's OBSERVED position
    at target frame j reads frame j's pixels, which is standard practice for
    monocular dynamic NVS but is information the leave-one-out protocol otherwise
    withholds. Here each track's position at j is instead fitted (constant
    velocity) over the OTHER frames only, so nothing about j is read. Comparing
    the two isolates what the non-rigid per-Gaussian interpolation contributes
    from what OBSERVING j contributes.

    `bandwidth` > 0 makes it a LOCALLY weighted fit: support frames are weighted
    exp(-0.5*((f-j)/bandwidth)^2), so the velocity is estimated from the frames
    NEAREST the target instead of the whole window. Measured motivation: strict
    prediction recovered +0.55 dB dynamic where observing frame j gave +1.34, and
    a single global constant-velocity fit spans ~1.3 s at stride 8 -- far longer
    than a walking person stays linear. bandwidth=0 keeps the uniform global fit
    (the measured configuration), so existing results reproduce exactly.

    Vectorised over tracks (the per-group version above loops, which is fine for
    K~4 groups but not for ~1k tracks x V frames x 920 batches).

    traj [V,Nt,3], ok [V,Nt] -> (pred [V,Nt,3], pred_ok [V,Nt])
    """
    V, Nt, _ = traj.shape
    dev = traj.device
    t = torch.arange(V, device=dev, dtype=torch.float32)
    c = traj.float()
    okf = ok.float()
    pred = torch.zeros_like(c)
    pred_ok = torch.zeros(V, Nt, device=dev, dtype=torch.bool)
    for j in range(V):
        m = okf.clone()
        m[j] = 0.0                                   # exclude the target frame
        n = m.sum(0)                                 # [Nt] usable frames (unweighted)
        if bandwidth > 0:
            kern = torch.exp(-0.5 * ((t - t[j]) / bandwidth) ** 2)
            m = m * kern.unsqueeze(-1)               # locally weighted least squares
        s = m.sum(0)                                 # [Nt] weight mass
        w = m.unsqueeze(-1)
        cm = (c * w).sum(0) / s.clamp_min(1e-8).unsqueeze(-1)     # [Nt,3] weighted mean pos
        tm = (t.unsqueeze(-1) * m).sum(0) / s.clamp_min(1e-8)     # [Nt]   weighted mean time
        dt = t.view(V, 1) - tm.view(1, Nt)                        # [V,Nt]
        den = (m * dt * dt).sum(0)                                # [Nt]
        num = (w * dt.unsqueeze(-1) * (c - cm.unsqueeze(0))).sum(0)  # [Nt,3]
        slope = num / den.clamp_min(1e-8).unsqueeze(-1)
        # 2 points already determine a velocity, and velocity IS the model here. Falling
        # back to the mean sooner would predict a moving object as stationary, i.e. it
        # would understate the strict mode and make the control look worse than it is.
        fit = (n >= min_pts) & (den > 1e-8)
        pred[j] = torch.where(fit.unsqueeze(-1), cm + slope * (t[j] - tm).unsqueeze(-1), cm)
        pred_ok[j] = n >= 2
    return pred, pred_ok


@torch.no_grad()
def knn_flow_displacement(
    traj: torch.Tensor,
    ok: torch.Tensor,
    gauss_pts: torch.Tensor,
    gauss_fidx: torch.Tensor,
    gauss_dyn: torch.Tensor,
    num_views: int,
    k: int = 8,
    gate_mult: float = 3.0,
    min_frame_tracks: int = 4,
    strict: bool = False,
    pred_bandwidth: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Phase B: per-Gaussian displacement toward every target frame, by direct
    track correspondence (NOT extrapolation).

    For a dynamic Gaussian of source frame i and target frame j:
        disp = sum_k w_k * (traj[j, k] - traj[i, k]) / sum_k w_k
    over its K nearest tracks in 3D at frame i (inverse-distance weights),
    restricted to tracks visible at BOTH i and j. A Gaussian farther from its
    nearest track than gate_mult x (median track NN spacing) gets NO motion —
    moving by a far-away object's flow scatters Gaussians into the background,
    the measured failure of the rigid modes.

    `strict` swaps the OBSERVED target-frame track positions for leave-one-out
    predictions (see predict_tracks_loo), so frame j is never read. The source
    frame i is always the observed position — i is not the held-out frame.

    traj [V,Nt,3], ok [V,Nt];  gauss_pts [N,3], gauss_fidx [N], gauss_dyn [N] bool.
    -> (disp [N, num_views, 3], valid [N, num_views] float 0/1). disp is 0 where
       invalid or own-frame.
    """
    N = gauss_pts.shape[0]
    dev = gauss_pts.device
    disp = torch.zeros(N, num_views, 3, device=dev, dtype=gauss_pts.dtype)
    valid = torch.zeros(N, num_views, device=dev, dtype=gauss_pts.dtype)
    traj_t, ok_t = (predict_tracks_loo(traj, ok, bandwidth=pred_bandwidth)
                    if strict else (traj, ok))

    for i in range(num_views):
        sel = gauss_dyn & (gauss_fidx == i)
        n_i = int(sel.sum())
        if n_i == 0:
            continue
        mi = ok[i]                                   # tracks usable at source frame i
        M = int(mi.sum())
        if M < min_frame_tracks:
            continue
        P = traj[i, mi].float()                      # [M, 3] track positions at i
        G = gauss_pts[sel].float()                   # [n_i, 3]

        # Adaptive trust radius from the scaffold's own density at frame i.
        d_tt = torch.cdist(P, P)
        d_tt.fill_diagonal_(float("inf"))
        spacing = d_tt.min(dim=1).values.median()
        gate_r = gate_mult * spacing.clamp_min(1e-4)

        kk = min(k, M)
        d_gt = torch.cdist(G, P)                     # [n_i, M]
        dist, idx = d_gt.topk(kk, dim=1, largest=False)
        w0 = (dist <= gate_r).float() / (dist + 1e-6)          # [n_i, kk]

        tr_i = traj[:, mi]                           # [V, M, 3] observed
        tr_t = traj_t[:, mi]                         # [V, M, 3] target lookup (LOO if strict)
        ok_t_i = ok_t[:, mi]                         # [V, M]
        nb_src = tr_i[i].float()[idx]                # [n_i, kk, 3]
        for j in range(num_views):
            if j == i:
                continue                             # own frame: already in place
            w = w0 * ok_t_i[j].float()[idx]          # drop neighbours unusable at j
            wsum = w.sum(dim=1, keepdim=True)        # [n_i, 1]
            good = wsum[:, 0] > 0
            if not bool(good.any()):
                continue
            nb_disp = tr_t[j].float()[idx] - nb_src  # [n_i, kk, 3] flow i->j
            d_ij = (w.unsqueeze(-1) * nb_disp).sum(dim=1) / wsum.clamp_min(1e-8)
            d_ij = d_ij * good.unsqueeze(-1).float()
            row = torch.zeros(N, 3, device=dev, dtype=disp.dtype)
            row[sel] = d_ij.to(disp.dtype)
            disp[:, j] += row
            vrow = torch.zeros(N, device=dev, dtype=valid.dtype)
            vrow[sel] = good.to(valid.dtype)
            valid[:, j] += vrow

    # --- DIAGNOSTIC: did the mechanism actually DO anything? --------------------
    n_dyn = int(gauss_dyn.sum())
    if n_dyn > 0:
        v_off = valid.clone()
        own = gauss_fidx.clamp_min(0).long().unsqueeze(1)      # [N,1]
        v_off.scatter_(1, own, 0.0)                            # ignore own-frame slots
        cover = v_off[gauss_dyn].sum() / max(n_dyn * max(num_views - 1, 1), 1)
        mags = disp[gauss_dyn].norm(dim=-1)
        mags = mags[v_off[gauss_dyn] > 0]
        if mags.numel() > 0:
            print(f"[DynFlow{'/strict' if strict else ''}] moved {100.0 * float(cover):.1f}% of dynamic "
                  f"(gaussian, target) pairs | displacement "
                  f"median={mags.median().item():.4f} mean={mags.mean().item():.4f} "
                  f"p90={mags.quantile(0.9).item():.4f} (world units)")
        else:
            print("[DynFlow] NO usable displacement (gates removed everything)")
    return disp, valid
