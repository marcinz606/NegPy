import hashlib
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.flatfield.models import FlatFieldConfig

# Clamp so a near-black reference pixel can't blow up the image.
_GAIN_MIN = 0.25
_GAIN_MAX = 4.0

# Falloff is low-frequency: compute the gain on a small copy (upscaled at apply time)
# so the blur kernel stays tiny.
_GAIN_WORK_SIZE = 256

# Resolved gains keyed by profile id: (gain map, content token). A cached ``None``
# marks a known-missing profile so a broken reference doesn't re-hit the store every
# render. Populated lazily through the injected provider — the desktop app wires it
# to the on-disk profile store (services/assets/flatfield.py) at startup; tests may
# seed this map directly.
GainEntry = Tuple[np.ndarray, str]
_GAIN_CACHE: Dict[str, Optional[GainEntry]] = {}
_gain_provider: Optional[Callable[[str], Optional[GainEntry]]] = None


def set_gain_provider(provider: Optional[Callable[[str], Optional[GainEntry]]]) -> None:
    """Inject the ``profile_id -> (gain, token)`` resolver and drop any cached gains."""
    global _gain_provider
    _gain_provider = provider
    _GAIN_CACHE.clear()


def invalidate_gain(profile_id: Optional[str] = None) -> None:
    """Drop a cached gain (all when None) after a profile is re-baked or deleted."""
    if profile_id is None:
        _GAIN_CACHE.clear()
    else:
        _GAIN_CACHE.pop(profile_id, None)


def _resolve(profile_id: str) -> Optional[GainEntry]:
    if not profile_id:
        return None
    if profile_id in _GAIN_CACHE:
        return _GAIN_CACHE[profile_id]
    entry = _gain_provider(profile_id) if _gain_provider is not None else None
    _GAIN_CACHE[profile_id] = entry
    return entry


def compute_gain(reference: ImageBuffer) -> np.ndarray:
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


def gain_token(gain: np.ndarray) -> str:
    """Stable content id for a baked gain map, folded into the render source hash."""
    return hashlib.blake2b(np.ascontiguousarray(gain, dtype=np.float32).tobytes(), digest_size=8).hexdigest()


def flatfield_token(config: FlatFieldConfig) -> str:
    """Identity of the active correction, folded into the render source hash. Empty when inactive."""
    if not config.apply or not config.profile_id:
        return ""
    entry = _resolve(config.profile_id)
    if entry is None:
        return ""
    return f"|ff:{config.profile_id}:{entry[1]}"


def apply_flatfield(image: ImageBuffer, config: FlatFieldConfig) -> ImageBuffer:
    """Multiply the linear source by the reference gain map. No-op when inactive or unresolved."""
    if not config.apply or not config.profile_id:
        return image
    entry = _resolve(config.profile_id)
    if entry is None:
        return image
    gain = entry[0]
    if gain.shape[:2] != image.shape[:2]:
        gain = cv2.resize(gain, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (image * gain).astype(np.float32)
