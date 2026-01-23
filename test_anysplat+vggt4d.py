"""
Test script for AnySplat with VGGT4D integration and dynamic mask detection.
This script builds the model from config (not HuggingFace) to use our modifications.
"""
from pathlib import Path
import torch
import os
import sys
import json
import numpy as np

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "DynamicReconstructionSplat"))

from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.model.ply_export import export_ply
from src.utils.image import process_image

# Import config classes
from src.model.encoder.anysplat import EncoderAnySplatCfg, OpacityMappingCfg
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from src.model.encoder.visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg


def create_encoder_config(use_vggt4d=True, enable_dynamic_detection=True):
    """Create encoder config with VGGT4D settings."""
    return EncoderAnySplatCfg(
        name="anysplat",
        anchor_feat_dim=83,
        voxel_size=0.001,
        n_offsets=2,
        d_feature=32,
        add_view=False,
        num_monocular_samples=32,
        backbone=None,
        visualizer=EncoderVisualizerEpipolarCfg(
            num_samples=8,
            min_resolution=256,
            export_ply=False,
        ),
        gaussian_adapter=GaussianAdapterCfg(
            gaussian_scale_min=0.5,
            gaussian_scale_max=15.0,
            sh_degree=4,
        ),
        apply_bounds_shim=True,
        opacity_mapping=OpacityMappingCfg(
            initial=0.0,
            final=0.0,
            warm_up=1,
        ),
        gaussians_per_pixel=1,
        num_surfaces=1,
        gs_params_head_type="dpt_gs",
        pose_free=True,
        pred_head_type="depth",
        # VGGT4D settings
        use_vggt4d=use_vggt4d,
        vggt4d_weights_path=None,  # Will initialize from VGGT-1B
        enable_dynamic_detection=enable_dynamic_detection,
        dynamic_mask_threshold=None,  # Use adaptive Otsu threshold
        dynamic_n_clusters=64,
        suppress_dynamic_gaussians=True,
    )


def create_decoder_config():
    """Create decoder config."""
    return DecoderSplattingCUDACfg(name="splatting_cuda", background_color=[0.0, 0.0, 0.0], make_scale_invariant=False)


def main():
    print("=" * 60)
    print("Testing AnySplat with VGGT4D Integration")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create configs
    print("\nCreating model configs...")
    encoder_cfg = create_encoder_config(
        use_vggt4d=True,
        enable_dynamic_detection=True
    )
    decoder_cfg = create_decoder_config()

    # Build model
    print("\nBuilding AnySplat with VGGT4D...")
    print("  - use_vggt4d: True")
    print("  - enable_dynamic_detection: True")
    model = AnySplat(encoder_cfg, decoder_cfg)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Load test images
    image_folder = "DynamicReconstructionSplat/examples/vrnerf/rgbd_bonn_crowd3/rgb"
    if not os.path.exists(image_folder):
        # Try alternative path (when running from inside the repo)
        image_folder = "examples/vrnerf/rgbd_bonn_crowd3/rgb"

    if not os.path.exists(image_folder):
        print(f"\nERROR: Image folder not found at {image_folder}")
        print("Please provide test images or adjust the path.")
        return

    # Sample evenly spaced frames from the sequence
    all_images = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    n_sample = 8  # Number of frames to sample
    step = max(1, len(all_images) // n_sample)
    image_names = all_images[::step][:n_sample]

    if len(image_names) < 2:
        print(f"\nERROR: Need at least 2 images, found {len(image_names)}")
        return

    print(f"\nLoading {len(image_names)} images from {image_folder}...")
    for name in image_names:
        print(f"  - {os.path.basename(name)}")

    images = [process_image(img) for img in image_names]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device)
    b, v, _, h, w = images.shape
    print(f"Image tensor shape: {images.shape}")

    # Run Inference
    print("\n" + "=" * 60)
    print("Running inference with VGGT4D...")
    print("=" * 60)

    with torch.no_grad():
        gaussians, pred_context_pose = model.inference((images + 1) * 0.5)

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Gaussians means shape: {gaussians.means.shape}")
    print(f"Gaussians count: {gaussians.means.shape[1]}")
    print(f"Opacities - mean: {gaussians.opacities.mean():.4f}, "
          f"min: {gaussians.opacities.min():.4f}, max: {gaussians.opacities.max():.4f}")
    print(f"Scales - mean: {gaussians.scales.mean():.4f}, "
          f"min: {gaussians.scales.min():.4f}, max: {gaussians.scales.max():.4f}")

    # Check dynamic detection results
    if hasattr(model.encoder, 'dyn_mask') and model.encoder.dyn_mask is not None:
        dyn_mask = model.encoder.dyn_mask
        dyn_ratio = dyn_mask.float().mean().item()
        print(f"\nDynamic Detection Results:")
        print(f"  - Mask shape: {dyn_mask.shape}")
        print(f"  - Dynamic pixels: {dyn_ratio*100:.1f}%")
    else:
        print("\nDynamic detection: No mask generated (may be disabled or no dynamic content)")

    # Save results
    output_dir = "output_anysplat+vggt4d_test"
    os.makedirs(output_dir, exist_ok=True)

    pred_all_extrinsic = pred_context_pose['extrinsic']
    pred_all_intrinsic = pred_context_pose['intrinsic']

    print(f"\nSaving results to {output_dir}/...")

    # Save videos
    save_interpolated_video(
        pred_all_extrinsic, pred_all_intrinsic,
        b, h, w, gaussians, output_dir, model.decoder
    )

    # Save PLY
    plyfile = os.path.join(output_dir, "gaussians.ply")
    export_ply(
        gaussians.means[0],
        gaussians.scales[0],
        gaussians.rotations[0],
        gaussians.harmonics[0],
        gaussians.opacities[0],
        Path(plyfile),
        save_sh_dc_only=True,
    )

    # Save camera poses
    poses_file = os.path.join(output_dir, "camera_poses.json")
    poses_data = {
        "extrinsics": pred_all_extrinsic[0].cpu().numpy().tolist(),
        "intrinsics": pred_all_intrinsic[0].cpu().numpy().tolist(),
        "num_views": v,
        "image_names": [os.path.basename(name) for name in image_names],
        "config": {
            "use_vggt4d": True,
            "enable_dynamic_detection": True,
        }
    }
    with open(poses_file, 'w') as f:
        json.dump(poses_data, f, indent=2)

    # Save dynamic mask if available
    if hasattr(model.encoder, 'dyn_mask') and model.encoder.dyn_mask is not None:
        np.save(
            os.path.join(output_dir, "dynamic_mask.npy"),
            model.encoder.dyn_mask.cpu().numpy()
        )
        print(f"  - Dynamic mask: dynamic_mask.npy")

    if hasattr(model.encoder, 'dyn_map') and model.encoder.dyn_map is not None:
        np.save(
            os.path.join(output_dir, "dynamic_map.npy"),
            model.encoder.dyn_map.cpu().numpy()
        )
        print(f"  - Dynamic map: dynamic_map.npy")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    print(f"Output files:")
    print(f"  - RGB video: {output_dir}/rgb.mp4")
    print(f"  - Depth video: {output_dir}/depth.mp4")
    print(f"  - 3D Gaussians: {output_dir}/gaussians.ply")
    print(f"  - Camera poses: {output_dir}/camera_poses.json")


if __name__ == "__main__":
    main()
