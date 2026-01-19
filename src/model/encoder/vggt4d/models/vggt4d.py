import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from src.model.encoder.vggt4d.models.aggregator import AggregatorFor4D
from src.model.encoder.vggt.heads.camera_head import CameraHead
from src.model.encoder.vggt.heads.dpt_head import DPTHead
from src.model.encoder.vggt.heads.track_head import TrackHead


class VGGTFor4D(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        self.aggregator = AggregatorFor4D(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(
            dim_in=2 * embed_dim, patch_size=patch_size)

    def forward(self, images: torch.Tensor, dyn_masks: torch.Tensor = None, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT4D model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            dyn_masks (torch.Tensor, optional): Masks for the images, in range [0, 1].
                Shape: [S, H, W] or [B, S, H, W], where S is the sequence length, H is the height, W is the width.
                Default: None
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            tuple: (predictions, qk_dict, enc_feat, aggregated_tokens_list)
                - predictions (dict): Contains pose_enc, depth, depth_conf, world_points, etc.
                - qk_dict (dict): Q/K tensors for dynamic mask extraction
                - enc_feat (torch.Tensor): Encoder features (patch tokens)
                - aggregated_tokens_list (list): Token outputs from each layer
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if dyn_masks is not None and len(dyn_masks.shape) == 3:
            dyn_masks = dyn_masks.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx, qk_dict, enc_feat = self.aggregator(
            images, dyn_masks)

        predictions = {}

        with torch.amp.autocast("cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                # pose encoding of the last iteration
                predictions["pose_enc"] = pose_enc_list[-1]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            # track of the last iteration
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        return predictions, qk_dict, enc_feat, aggregated_tokens_list
