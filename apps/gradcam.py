from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image


def _find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Conv2d:
    """Return the last nn.Conv2d layer in the model."""
    last_conv: Optional[torch.nn.Conv2d] = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError("No Conv2d layer found in the model for Grad-CAM.")

    return last_conv


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_index: int | None = None,
    target_layer: torch.nn.Module | None = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap as a [H, W] numpy array in [0, 1]."""
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor.")
    if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
        raise ValueError("input_tensor must have shape [1, 3, H, W].")
    if input_tensor.shape[1] != 3:
        raise ValueError("input_tensor must have 3 channels (RGB).")

    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    if target_layer is None:
        target_layer = _find_last_conv_layer(model)
    if not isinstance(target_layer, torch.nn.Conv2d):
        raise TypeError("target_layer must be an nn.Conv2d layer.")

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        # Save the feature maps from the target layer.
        activations["value"] = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        # Save the gradients of the target layer output.
        gradients["value"] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    if hasattr(target_layer, "register_full_backward_hook"):
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
    else:
        backward_handle = target_layer.register_backward_hook(backward_hook)

    try:
        model.zero_grad()
        logits = model(input_tensor)

        # Support binary classifiers with shape [1, 1] and optional class_index.
        if logits.dim() == 2 and logits.shape[0] == 1:
            if class_index is None:
                target_logit = logits[0, 0]
            else:
                if not (0 <= class_index < logits.shape[1]):
                    raise ValueError(
                        f"class_index {class_index} out of range for logits shape {logits.shape}."
                    )
                target_logit = logits[0, class_index]
        elif logits.dim() == 1 and logits.numel() == 1:
            if class_index not in (None, 0):
                raise ValueError(
                    f"class_index {class_index} invalid for single-logit output."
                )
            target_logit = logits[0]
        else:
            raise ValueError(
                f"Unsupported logits shape {tuple(logits.shape)}; expected [1, 1]."
            )

        # Backprop from the melanoma logit to the target conv layer.
        target_logit.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError(
                "Failed to capture activations/gradients; check the target layer."
            )

        act = activations["value"]  # [1, C, H, W]
        grad = gradients["value"]  # [1, C, H, W]

        # Global-average-pool gradients over spatial dims to get channel weights.
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = torch.relu(cam)

        # Resize to match input size if needed.
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        if cam.shape[2] != input_h or cam.shape[3] != input_w:
            cam = F.interpolate(
                cam, size=(input_h, input_w), mode="bilinear", align_corners=False
            )

        # Normalize to [0, 1].
        cam_min = cam.min()
        cam_max = cam.max()
        if (cam_max - cam_min) > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        cam_np = cam.squeeze().detach().cpu().numpy()
        return cam_np
    finally:
        forward_handle.remove()
        backward_handle.remove()


def overlay_heatmap_on_image(
    image_pil: Image.Image, cam_np: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Blend a Grad-CAM heatmap with the original image."""
    if not isinstance(image_pil, Image.Image):
        raise TypeError("image_pil must be a PIL.Image.Image.")
    if not isinstance(cam_np, np.ndarray) or cam_np.ndim != 2:
        raise ValueError("cam_np must be a 2D numpy array.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    image_rgb = image_pil.convert("RGB")
    cam_clipped = np.clip(cam_np, 0.0, 1.0)

    # Apply a matplotlib colormap to create a color heatmap.
    heatmap_rgba = cm.get_cmap("magma")(cam_clipped)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    heatmap = Image.fromarray(heatmap_rgb)
    if heatmap.size != image_rgb.size:
        heatmap = heatmap.resize(image_rgb.size, resample=Image.BILINEAR)

    overlay = Image.blend(image_rgb, heatmap, alpha=alpha)
    return overlay
