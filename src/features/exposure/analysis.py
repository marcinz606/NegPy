import numpy as np
from typing import Tuple
from src.core.types import ImageBuffer
from src.features.exposure.normalization import (
    measure_log_negative_bounds,
    LogNegativeBounds,
)


def prepare_exposure_analysis(
    img: ImageBuffer,
) -> Tuple[ImageBuffer, LogNegativeBounds]:
    """
    Helper to normalize raw negative image for sensitometric analysis.
    Extracts bounds and normalizes the subject area (center crop) to density-like space.
    """
    epsilon = 1e-6

    # 1. Measure Log Bounds
    bounds = measure_log_negative_bounds(img)
    floors, ceils = bounds.floors, bounds.ceils

    # 2. Extract Subject Area (Center 60%)
    h, w = img.shape[:2]
    mh, mw = int(h * 0.20), int(w * 0.20)
    subject_linear = img[mh : h - mh, mw : w - mw]

    # 3. Normalize subject in log domain
    subject_log = np.log10(np.clip(subject_linear, epsilon, 1.0))
    norm_subject_log = np.zeros_like(subject_log)
    for ch in range(3):
        f, c = floors[ch], ceils[ch]
        norm_subject_log[:, :, ch] = (subject_log[:, :, ch] - f) / (max(c - f, epsilon))

    return norm_subject_log, bounds


def analyze_sensitometry(norm_subject_log: ImageBuffer) -> Tuple[float, float]:
    """
    Calculates Sensitometric Metrics for the Red Channel (Structure).

    Returns:
        Tuple[float, float]: (measured_dr, midpoint_r)
        - measured_dr: Dynamic Range (P99 - P1)
        - midpoint_r: Midpoint of the range (P99 + P1) / 2
    """
    p1_r = float(np.percentile(norm_subject_log[:, :, 0], 1.0))
    p99_r = float(np.percentile(norm_subject_log[:, :, 0], 99.0))
    measured_dr = max(p99_r - p1_r, 0.1)  # Avoid division by zero
    midpoint_r = (p99_r + p1_r) / 2.0
    return measured_dr, midpoint_r
