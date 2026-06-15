import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# Gain maps cached by (path, mtime); decoding the reference is slow.
_GAIN_CACHE: Dict[Tuple[str, float], Optional[np.ndarray]] = {}

# Clamp so a near-black reference pixel can't blow up the image.
_GAIN_MIN = 0.25
_GAIN_MAX = 4.0


# Falloff is low-frequency: compute the gain on a small copy (upscaled at apply time)
# so the blur kernel stays tiny.
_GAIN_WORK_SIZE = 256


def _compute_gain(reference: ImageBuffer) -> np.ndarray:
    """Per-channel gain = mean(blur) / blur, on a downsampled copy."""
    ref = reference.astype(np.float32)
    h, w = ref.shape[:2]
    scale = min(1.0, _GAIN_WORK_SIZE / max(h, w))
    if scale < 1.0:
        ref = cv2.resize(ref, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    sigma = max(ref.shape[:2]) / 16.0
    blur = cv2.GaussianBlur(ref, (0, 0), sigmaX=sigma, sigmaY=sigma)
    eps = 1e-4
    blur = np.clip(blur, eps, None)
    means = blur.reshape(-1, blur.shape[2]).mean(axis=0)
    gain = means[None, None, :] / blur
    return np.clip(gain, _GAIN_MIN, _GAIN_MAX).astype(np.float32)


def load_reference_gain(path: str) -> Optional[np.ndarray]:
    """Per-channel gain map for the reference, decoded like a negative (no WB, linear)."""
    if not path or not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None

    key = (path, mtime)
    if key in _GAIN_CACHE:
        return _GAIN_CACHE[key]

    gain: Optional[np.ndarray] = None
    try:
        from negpy.services.rendering.preview_manager import PreviewManager

        reference, _, _ = PreviewManager().load_linear_preview(path, use_camera_wb=False, full_resolution=False)
        gain = _compute_gain(reference)
    except Exception:
        logger.exception("Flat-field: failed to load reference %s", path)
        gain = None

    _GAIN_CACHE[key] = gain
    return gain


def flatfield_token(config: FlatFieldConfig) -> str:
    """Identity of the active correction, folded into the render source hash. Empty when inactive."""
    if not config.enabled or not config.reference_path:
        return ""
    try:
        mtime = os.path.getmtime(config.reference_path)
    except OSError:
        return ""
    return f"|ff:{config.reference_path}:{mtime}"


def apply_flatfield(image: ImageBuffer, config: FlatFieldConfig) -> ImageBuffer:
    """Multiply the linear source by the reference gain map. No-op when inactive or unreadable."""
    if not config.enabled or not config.reference_path:
        return image
    gain = load_reference_gain(config.reference_path)
    if gain is None:
        return image
    if gain.shape[:2] != image.shape[:2]:
        gain = cv2.resize(gain, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (image * gain).astype(np.float32)
