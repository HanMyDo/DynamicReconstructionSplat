import torch
from einops import rearrange


def organize_qk_dict(qk_dict: dict, n_img: int) -> dict:
    """Reorganize Q/K tensors from aggregator output for dynamic mask extraction.

    Args:
        qk_dict: Dictionary containing 'global_q', 'global_k', 'frame_q', 'frame_k'
                 from the VGGT4D aggregator output
        n_img: Number of images in the sequence

    Returns:
        Dictionary with reorganized Q/K tensors including:
        - global_cam_q/k: Camera tokens [n_img, n_layer, n_head, 1, c]
        - global_reg_q/k: Register tokens [n_img, n_layer, n_head, 4, c]
        - global_tok_q/k: Patch tokens [n_img, n_layer, n_head, n_tok, c]
        - frame_cam_q/k, frame_reg_q/k, frame_tok_q/k: Same for frame attention
    """
    global_q = qk_dict["global_q"]
    global_k = qk_dict["global_k"]
    frame_q = qk_dict["frame_q"]
    frame_k = qk_dict["frame_k"]

    n_tok = global_q.shape[-2] // n_img

    patch_start_idx = 5

    global_q = rearrange(
        global_q, "n_layer 1 1 n_head (n_img n_tok) c -> n_img n_layer n_head n_tok c", n_img=n_img, n_tok=n_tok)
    global_k = rearrange(
        global_k, "n_layer 1 1 n_head (n_img n_tok) c -> n_img n_layer n_head n_tok c", n_img=n_img, n_tok=n_tok)

    global_cam_q = global_q[..., 0:1, :]
    global_cam_k = global_k[..., 0:1, :]
    global_reg_q = global_q[..., 1:patch_start_idx, :]
    global_reg_k = global_k[..., 1:patch_start_idx, :]
    global_tok_q = global_q[..., patch_start_idx:, :]
    global_tok_k = global_k[..., patch_start_idx:, :]

    frame_q = rearrange(
        frame_q, "n_layer 1 n_img n_head n_tok c -> n_img n_layer n_head n_tok c", n_img=n_img, n_tok=n_tok)
    frame_k = rearrange(
        frame_k, "n_layer 1 n_img n_head n_tok c -> n_img n_layer n_head n_tok c", n_img=n_img, n_tok=n_tok)

    frame_cam_q = frame_q[..., 0:1, :]
    frame_cam_k = frame_k[..., 0:1, :]
    frame_reg_q = frame_q[..., 1:patch_start_idx, :]
    frame_reg_k = frame_k[..., 1:patch_start_idx, :]
    frame_tok_q = frame_q[..., patch_start_idx:, :]
    frame_tok_k = frame_k[..., patch_start_idx:, :]

    return {
        "global_cam_q": global_cam_q,
        "global_cam_k": global_cam_k,
        "global_reg_q": global_reg_q,
        "global_reg_k": global_reg_k,
        "global_tok_q": global_tok_q,
        "global_tok_k": global_tok_k,
        "frame_cam_q": frame_cam_q,
        "frame_cam_k": frame_cam_k,
        "frame_reg_q": frame_reg_q,
        "frame_reg_k": frame_reg_k,
        "frame_tok_q": frame_tok_q,
        "frame_tok_k": frame_tok_k,

        "global_q": global_tok_q,
        "global_k": global_tok_k,
        "frame_q": frame_tok_q,
        "frame_k": frame_tok_k,
    }
