import numpy as np
from typing import Tuple
from src.core.types import ImageBuffer
from src.core.constants import PIPELINE_CONSTANTS
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


def solve_photometric_exposure(
    norm_subject_log: ImageBuffer, bounds: LogNegativeBounds
) -> Tuple[float, float, float, float, float]:
    """
    Analyzes a raw negative to determine the optimal Photometric Exposure settings using a Sensitometric Solver.
    Returns UI Slider units: (cyan, magenta, yellow, density, grade)
    """
    # --- 1. Auto-Filtration (CMY): "Base Neutralization" ---
    base_r = float(np.percentile(norm_subject_log[:, :, 0], 0.1))
    base_g = float(np.percentile(norm_subject_log[:, :, 1], 0.1))
    base_b = float(np.percentile(norm_subject_log[:, :, 2], 0.1))

    cyan_offset = 0.0
    magenta_offset = base_r - base_g
    yellow_offset = base_r - base_b

    # --- 2. Sensitometry Analysis ---
    measured_dr, midpoint_r = analyze_sensitometry(norm_subject_log)

    # --- 3. Auto-Grade (Contrast) ---
    physical_slope = PIPELINE_CONSTANTS["target_paper_range"] / measured_dr
    physical_slope = float(np.clip(physical_slope, 1.0, 4.0))

    auto_grade_ui = (physical_slope - 1.0) / PIPELINE_CONSTANTS["grade_multiplier"]

    # --- 4. Auto-Density (Exposure) ---
    exposure_shift = 1.0 - midpoint_r
    auto_density_ui = (exposure_shift - 0.1) / PIPELINE_CONSTANTS["density_multiplier"]

    return (
        cyan_offset,
        magenta_offset,
        yellow_offset,
        auto_density_ui,
        auto_grade_ui,
    )
