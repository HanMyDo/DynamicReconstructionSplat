"""
Temporal Consistency Loss for Dynamic Gaussian Splatting (Fix 3).

This loss enforces temporal consistency in Gaussian predictions:
- For STATIC regions: Gaussian parameters should be consistent across adjacent frames
- For DYNAMIC regions: No consistency penalty (allow change)

The loss uses the dynamic mask from VGGT4D to distinguish static from dynamic regions.
"""

from dataclasses import dataclass
from typing import Optional

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn.functional as F

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossTemporalConsistencyCfg:
    """Configuration for temporal consistency loss."""
    weight: float = 0.1
    # Which Gaussian parameters to apply consistency loss to
    use_opacity: bool = True
    use_scales: bool = True
    use_rotations: bool = True
    use_sh: bool = True  # Spherical harmonics (appearance)
    # Dynamic mask handling
    static_only: bool = True  # Only apply to static regions (requires dyn_mask)
    dynamic_weight: float = 0.0  # Weight for dynamic regions (0 = no penalty)
    # Loss type
    loss_type: str = "l1"  # "l1" or "l2"


@dataclass
class LossTemporalConsistencyCfgWrapper:
    temporal_consistency: LossTemporalConsistencyCfg


class LossTemporalConsistency(Loss[LossTemporalConsistencyCfg, LossTemporalConsistencyCfgWrapper]):
    """
    Temporal consistency loss for Gaussian parameters.

    Requires the encoder to provide per-frame Gaussian parameters in infos['per_frame_gaussians']:
    {
        'opacity': [B, V, H, W] or [B, V, N],
        'scales': [B, V, H, W, 3] or [B, V, N, 3],
        'rotations': [B, V, H, W, 4] or [B, V, N, 4],
        'sh': [B, V, H, W, C] or [B, V, N, C],  # Spherical harmonics
    }

    Also optionally uses infos['dyn_mask'] to weight the loss.
    """

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
        infos: Optional[dict] = None,
    ) -> Float[Tensor, ""]:
        """
        Compute temporal consistency loss.

        Note: This loss requires additional `infos` argument passed from the training loop.
        """
        if infos is None:
            return torch.tensor(0.0, device=gaussians.means.device)

        per_frame = infos.get('per_frame_gaussians', None)
        if per_frame is None:
            # No per-frame data available, skip this loss
            return torch.tensor(0.0, device=gaussians.means.device)

        device = gaussians.means.device
        total_loss = torch.tensor(0.0, device=device)
        num_components = 0

        # Get dynamic mask if available
        dyn_mask = infos.get('dyn_mask', None)  # [B, V, H, W]

        # Compute consistency loss for each Gaussian parameter type
        if self.cfg.use_opacity and 'opacity' in per_frame:
            loss = self._compute_temporal_loss(
                per_frame['opacity'], dyn_mask, device
            )
            total_loss = total_loss + loss
            num_components += 1

        if self.cfg.use_scales and 'scales' in per_frame:
            loss = self._compute_temporal_loss(
                per_frame['scales'], dyn_mask, device
            )
            total_loss = total_loss + loss
            num_components += 1

        if self.cfg.use_rotations and 'rotations' in per_frame:
            # For rotations, use quaternion distance or simple L1/L2
            loss = self._compute_temporal_loss(
                per_frame['rotations'], dyn_mask, device
            )
            total_loss = total_loss + loss
            num_components += 1

        if self.cfg.use_sh and 'sh' in per_frame:
            loss = self._compute_temporal_loss(
                per_frame['sh'], dyn_mask, device
            )
            total_loss = total_loss + loss
            num_components += 1

        if num_components > 0:
            total_loss = total_loss / num_components

        return self.cfg.weight * total_loss

    def _compute_temporal_loss(
        self,
        params: Tensor,
        dyn_mask: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        """
        Compute temporal consistency loss for a single parameter type.

        Args:
            params: Per-frame parameters [B, V, ...] where V is number of frames
            dyn_mask: Dynamic mask [B, V, H, W] (1=dynamic, 0=static)
            device: Target device

        Returns:
            Scalar loss tensor
        """
        if params.shape[1] < 2:
            # Need at least 2 frames for temporal loss
            return torch.tensor(0.0, device=device)

        # Compute difference between adjacent frames
        # params_t: frames 0..V-2, params_t1: frames 1..V-1
        params_t = params[:, :-1]  # [B, V-1, ...]
        params_t1 = params[:, 1:]  # [B, V-1, ...]

        # Compute per-element difference
        if self.cfg.loss_type == "l1":
            diff = (params_t - params_t1).abs()
        else:  # l2
            diff = (params_t - params_t1) ** 2

        # Apply weighting based on dynamic mask
        if dyn_mask is not None and self.cfg.static_only:
            # static_mask: 1 for static, 0 for dynamic
            static_mask = 1.0 - dyn_mask[:, 1:].float()  # Align with diff (V-1 frames)

            # Handle shape mismatch (dyn_mask is [B, V, H, W], params might be different)
            if diff.dim() > static_mask.dim():
                # Expand static_mask to match diff dimensions
                for _ in range(diff.dim() - static_mask.dim()):
                    static_mask = static_mask.unsqueeze(-1)

            # Interpolate static_mask if spatial dimensions don't match
            if static_mask.shape[2:4] != diff.shape[2:4] and diff.dim() >= 4:
                static_mask = F.interpolate(
                    static_mask.flatten(0, 1),
                    size=diff.shape[2:4],
                    mode='nearest'
                ).view(*static_mask.shape[:2], *diff.shape[2:4], *static_mask.shape[4:])

            # Weight the difference
            static_weight = 1.0
            dynamic_weight = self.cfg.dynamic_weight
            weights = static_mask * static_weight + (1 - static_mask) * dynamic_weight
            diff = diff * weights

            # Compute mean only over valid (weighted) regions
            if weights.sum() > 0:
                loss = diff.sum() / (weights.sum() + 1e-8)
            else:
                loss = diff.mean()
        else:
            # No dynamic mask, apply uniformly
            loss = diff.mean()

        return loss


class LossTemporalRenderingConsistency(Loss[LossTemporalConsistencyCfg, LossTemporalConsistencyCfgWrapper]):
    """
    Alternative temporal consistency loss using cross-frame rendering.

    Renders frame t using Gaussians predicted from frame t-1 and supervises
    against frame t's ground truth. This is more expensive but provides
    stronger supervision signal.

    Note: This requires access to the decoder during loss computation,
    which needs special handling in the training loop.
    """

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
        infos: Optional[dict] = None,
    ) -> Float[Tensor, ""]:
        # This loss requires special handling in the training loop
        # to render with cross-frame Gaussians. For now, return 0.
        # The actual implementation would be in the training script.
        return torch.tensor(0.0, device=gaussians.means.device)
