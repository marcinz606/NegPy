"""
Pure heuristics for auto-detecting the film process mode (C41 / B&W / E-6)
from a raw linear scan, before any inversion or normalization.
"""

from typing import Optional

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.normalization import get_analysis_crop
from negpy.features.process.models import ProcessMode

# Tuned against real sample scans; see tests/test_process_detect.py.
_ANALYSIS_BUFFER = 0.12  # centre-crop ratio: drops film rebate / borders
_MAX_ANALYSIS_DIM = 256  # downsample longest edge to this for speed
_BW_CORR_THRESHOLD = 0.99  # min channel correlation above this -> monochrome
_C41_ORANGE_THRESHOLD = 2.0  # red-over-blue cast above this -> orange mask (C41)


def _downsample(img: ImageBuffer, max_dim: int) -> ImageBuffer:
    """Strided downsample so analysis stays cheap on full-res previews."""
    longest = max(img.shape[0], img.shape[1])
    if longest <= max_dim:
        return img
    step = int(np.ceil(longest / max_dim))
    return img[::step, ::step]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flattened channels."""
    a = a.ravel() - float(a.mean())
    b = b.ravel() - float(b.mean())
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b))) + 1e-12
    return float(np.sum(a * b) / denom)


def detect_process_mode(raw: Optional[ImageBuffer]) -> ProcessMode:
    """
    Classify a raw linear scan as C41, B&W or E-6.
    Falls back to C41 (the default) on ambiguous or invalid input.
    """
    if raw is None or raw.ndim != 3 or raw.shape[2] < 3:
        return ProcessMode.C41

    img = get_analysis_crop(raw[:, :, :3].astype(np.float32), _ANALYSIS_BUFFER)
    img = _downsample(img, _MAX_ANALYSIS_DIM)
    img = np.clip(np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if img.size == 0:
        return ProcessMode.C41

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # B&W: channels stay near-perfectly correlated even with a colour tint;
    # real colour (C41/E-6) has varied hues and lower correlation.
    min_corr = min(_corr(r, g), _corr(g, b), _corr(r, b))
    if min_corr > _BW_CORR_THRESHOLD:
        return ProcessMode.BW

    # C41 vs E-6: orange-mask red-over-blue ratio (global mean + brightest pixels).
    mean_ratio = (float(np.mean(r)) + 1e-6) / (float(np.mean(b)) + 1e-6)
    base_ratio = (float(np.percentile(r, 98)) + 1e-6) / (float(np.percentile(b, 98)) + 1e-6)
    orange_score = 0.5 * (mean_ratio + base_ratio)

    if orange_score > _C41_ORANGE_THRESHOLD:
        return ProcessMode.C41
    return ProcessMode.E6
