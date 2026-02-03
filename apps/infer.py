import os

import timm
import torch
from PIL import Image
from torchvision import transforms

from apps.gradcam import generate_gradcam, overlay_heatmap_on_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Cache the loaded model so repeated calls stay fast.
_MODEL: torch.nn.Module | None = None
_DEVICE: torch.device | None = None


def _build_eval_transform() -> transforms.Compose:
    """Match the validation preprocessing used during training."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


_EVAL_TRANSFORM = _build_eval_transform()


def load_model(
    model_path: str = "models/classifier_v1.pt",
    model_name: str = "efficientnet_b0",
) -> torch.nn.Module:
    """Load the EfficientNet-B0 classifier and put it in eval mode."""
    # Stop early with a clear message if the weights are missing.
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Train the model or copy the weights to that location."
        )

    # Use GPU if it is available; otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the same architecture used for training (single logit output).
    model = timm.create_model(model_name, pretrained=False, num_classes=1)

    # Load either a full checkpoint dict or a raw state dict.
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Cache for reuse in predict_risk.
    global _MODEL, _DEVICE
    _MODEL = model
    _DEVICE = device
    return model


def predict_risk(image_pil: Image.Image) -> float:
    """Return melanoma risk probability for a single PIL image."""
    if not isinstance(image_pil, Image.Image):
        raise TypeError("predict_risk expects a PIL.Image.Image input.")

    # Lazily load the model the first time this function is called.
    model = _MODEL if _MODEL is not None else load_model()
    device = _DEVICE if _DEVICE is not None else next(model.parameters()).device

    # Apply validation transforms and add a batch dimension.
    image_rgb = image_pil.convert("RGB")
    image_tensor = _EVAL_TRANSFORM(image_rgb).unsqueeze(0).to(device)

    # Run inference and map logits to a probability.
    with torch.no_grad():
        logits = model(image_tensor).view(-1)
        prob = torch.sigmoid(logits)[0].item()

    return float(prob)


def predict_risk_with_cam(
    image_pil: Image.Image, alpha: float = 0.45
) -> tuple[float, Image.Image]:
    """Return melanoma risk probability and a Grad-CAM overlay image."""
    if not isinstance(image_pil, Image.Image):
        raise TypeError("predict_risk_with_cam expects a PIL.Image.Image input.")

    # Lazily load the model the first time this function is called.
    model = _MODEL if _MODEL is not None else load_model()
    device = _DEVICE if _DEVICE is not None else next(model.parameters()).device

    # Apply validation transforms and add a batch dimension.
    image_rgb = image_pil.convert("RGB")
    image_tensor = _EVAL_TRANSFORM(image_rgb).unsqueeze(0).to(device)

    # Compute Grad-CAM with gradients enabled even in eval mode.
    with torch.set_grad_enabled(True):
        cam_np = generate_gradcam(model, image_tensor)

    # Run a clean forward pass for the risk score.
    with torch.no_grad():
        logits = model(image_tensor).view(-1)
        risk = torch.sigmoid(logits)[0].item()

    # Overlay heatmap on the original image (resize to 224x224 for alignment).
    overlay_base = image_rgb.resize((224, 224), resample=Image.BILINEAR)
    overlay = overlay_heatmap_on_image(overlay_base, cam_np, alpha=alpha)

    return float(risk), overlay
