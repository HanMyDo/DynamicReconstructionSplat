# VGGT4D - Visual Geometry Grounded Transformer for 4D Scenes
# Extended from VGGT to handle dynamic scenes with motion-aware attention

from src.model.encoder.vggt4d.models.vggt4d import VGGTFor4D
from src.model.encoder.vggt4d.models.aggregator import AggregatorFor4D
from src.model.encoder.vggt4d.utils import organize_qk_dict
from src.model.encoder.vggt4d.masks import (
    extract_dyn_map,
    batch_extract_dyn_map,
    cluster_attention_maps,
    adaptive_multiotsu_variance,
)

__all__ = [
    "VGGTFor4D",
    "AggregatorFor4D",
    "organize_qk_dict",
    "extract_dyn_map",
    "batch_extract_dyn_map",
    "cluster_attention_maps",
    "adaptive_multiotsu_variance",
]
