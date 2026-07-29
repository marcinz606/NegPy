"""The chart must solve the same per-channel Cast Removal curves the engine renders.

Regression guard: the chart used to pass the raw slider strength (not the
confidence-weighted effective strength), never passed the neutral axis (silently
falling back to the shadow-tie branch) and dropped the curvatures — so the plotted
per-channel curves diverged from the render exactly when Cast Removal was active.
"""

from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import (
    cast_solve_inputs,
    curve_params_from_metrics,
    normalized_shadow_refs,
    per_channel_curve_params,
)
from negpy.features.exposure.processor import NormalizationProcessor, PhotometricProcessor

_H, _W = 600, 400


def _curved_negative() -> np.ndarray:
    """Curved neutral axis + green content block (see test_midtone_neutral)."""
    E = np.linspace(0.0, 1.0, _H, dtype=np.float32)
    gamma, curv, mask = (0.66, 0.71, 0.68), (0.0, 0.30, 0.12), (0.0, -0.12, -0.22)
    log = np.empty((_H, _W, 3), np.float32)
    for ch in range(3):
        log[:, :, ch] = (-0.2 + mask[ch] - gamma[ch] * E - curv[ch] * E * E)[:, None]
    gx = slice(int(0.82 * _W), _W)
    log[:, gx, 1], log[:, gx, 0], log[:, gx, 2] = -0.22, -0.50, -0.62
    return (10.0**log).astype(np.float32)


def _metrics_and_config():
    cfg = WorkspaceConfig()
    process = replace(cfg.process, analysis_buffer=0.0)
    exposure = replace(cfg.exposure, cast_removal_strength=0.8)
    ctx = PipelineContext(scale_factor=1.0, original_size=(_H, _W), process_mode="C41")
    norm = NormalizationProcessor(process).process(_curved_negative(), ctx)
    PhotometricProcessor(exposure).process(norm, ctx)
    return ctx.metrics, exposure


def test_chart_wiring_matches_render():
    metrics, config = _metrics_and_config()
    # The shared resolver the chart and the step wedge both call — so this pins all of them.
    slopes, pivots, curvs = curve_params_from_metrics(config, "C41", metrics)

    np.testing.assert_allclose(slopes, metrics["print_slopes"], atol=1e-12)
    assert max(abs(c) for c in curvs) > 1e-6  # curved fixture engages the quadratic

    # Anti-vacuity: the old wiring (raw slider strength, no neutral axis) diverges.
    bounds = metrics["final_bounds"]
    old_slopes, _, old_curvs = per_channel_curve_params(
        config.grade,
        config.density,
        config.auto_normalize_contrast,
        config.cast_removal_strength,
        metrics.get("norm_density_range"),
        normalized_shadow_refs(bounds, metrics.get("shadow_log_refs")),
        metrics.get("textural_range"),
        anchor=metrics.get("metered_anchor"),
        grade_trims=(config.grade_trim_red, config.grade_trim_green, config.grade_trim_blue),
    )
    assert max(abs(c) for c in old_curvs) == 0.0  # shadow-tie branch: no curvature
    assert max(abs(a - b) for a, b in zip(old_slopes, slopes)) > 1e-6


def test_confidence_derates_chart_strength():
    metrics, config = _metrics_and_config()
    confidence = metrics["neutral_axis_refs"][3]
    assert 0.0 < confidence < 1.0  # content block makes the grey set imperfect
    strength, _, _ = cast_solve_inputs(
        metrics["final_bounds"],
        metrics.get("shadow_log_refs"),
        metrics.get("neutral_axis_refs"),
        config.cast_removal_strength,
    )
    assert abs(strength - confidence * config.cast_removal_strength) < 1e-12
