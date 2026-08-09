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
"""
from typing import Optional, Tuple

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
