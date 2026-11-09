"""Survival-ratio estimation for Fade Restoration (features/process/fade.py), reading
Cast Removal's two-point neutral-detection algorithm rather than blind percentile spans.
A measure step, a build step, and the fail-closed reasons in between.
"""

import numpy as np

from fade_crossover_scene import estimate_fade
from negpy.features.exposure.normalization import prefilter_log_grid
from negpy.features.process.fade import (
    RATIO_BOUNDS,
    SPREAD_FLOOR,
    fade_ratios_from_neutral_axis,
)


def _ratios(image: np.ndarray) -> tuple[float, float, str]:
    """Estimate's (green, blue, reason) on the meter grid of a linear capture."""
    est = estimate_fade(prefilter_log_grid(image, None, 0.0))
    return est.green, est.blue, est.reason


def _synthetic_neutral_slide(
    ratio_g: float, ratio_b: float, span: float = 3.0, size: int = 200, seed: int = 0, ratio_r: float = 1.0
) -> np.ndarray:
    """A synthetic (size, size, 3) linear capture of a faded gray card. Unfaded, every pixel
    is neutral in measured density (a shared per-pixel factor t on all three channels): film
    is balanced so a gray subject reads gray, side absorptions included. The fade is then
    F = diag(ratio_r, ratio_r·ratio_g, ratio_r·ratio_b) on density, built here from scratch. `span` covers E-6's full
    transfer window when ratio_r == 1; ratio_r < 1 shrinks the actual span below it."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.03, 0.97, (size, size)).astype(np.float64)
    fade = np.diag([ratio_r, ratio_r * ratio_g, ratio_r * ratio_b])
    clean_log = np.stack([-t * span] * 3, axis=-1)
    measured_log = clean_log @ fade.T
    return np.power(10.0, measured_log).astype(np.float32)


def test_recovers_known_ratios_through_the_real_detector():
    """End-to-end: a synthetic slide with known survival ratios and
    run through the actual production detector (measure_neutral_axis_from_log) via
    neutral_axis_from_grid, recovers the true ratios via fade_ratios_from_neutral_axis."""
    ratio_g_true, ratio_b_true = 0.6, 0.85
    image = _synthetic_neutral_slide(ratio_g_true, ratio_b_true)

    ratio_g, ratio_b, reason = _ratios(image)
    assert reason == ""
    assert abs(ratio_g - ratio_g_true) < 1e-2
    assert abs(ratio_b - ratio_b_true) < 1e-2


def test_recovers_a_mild_fade_without_overshooting():
    """A mild fade is where a biased estimate hurts most: an overshoot over-corrects past
    the uncorrected render."""
    ratio_g_true, ratio_b_true = 0.85, 0.9
    image = _synthetic_neutral_slide(ratio_g_true, ratio_b_true)

    ratio_g, ratio_b, reason = _ratios(image)
    assert reason == ""
    assert abs(ratio_g - ratio_g_true) < 1e-2
    assert abs(ratio_b - ratio_b_true) < 1e-2


def test_recovers_known_ratios_on_a_severely_faded_frame():
    """The regression test for the measured-bounds fix: against E-6's fixed 0..3 window,
    a frame whose own density span is compressed by a real fade_ratio_r (here 0.1, so the
    frame's actual span is a tenth of what the fixed window expects) found no content in
    the detector's midtone/shadow luma bands at all and failed closed to (1.0, 1.0) --
    exactly the slides fade_ratio_r exists to help. Bounds measured from the frame itself
    put the detector's bands back on real content, recovering the true ratios regardless."""
    ratio_g_true, ratio_b_true = 0.6, 0.85
    image = _synthetic_neutral_slide(ratio_g_true, ratio_b_true, ratio_r=0.1)

    ratio_g, ratio_b, reason = _ratios(image)
    assert reason == ""
    assert abs(ratio_g - ratio_g_true) < 1e-2
    assert abs(ratio_b - ratio_b_true) < 1e-2


def test_documented_wrong_behavior_on_unequal_unfaded_spreads():
    """A legitimately unequal-spread scene with no fade at all still reads as non-unity.
    Documented wrong behavior -- a monochromatic slide defeats any estimator built on this
    assumption -- not a bug for a future pass to "fix"."""
    image = _synthetic_neutral_slide(1.6, 1.0)  # a genuinely more colorful green channel
    ratio_g, ratio_b, reason = _ratios(image)
    assert reason == ""
    assert ratio_g != 1.0


def test_fails_closed_when_spreads_agree():
    ratio_g, ratio_b, reason = fade_ratios_from_neutral_axis(((1.0, 1.01, 0.99), (0.0, 0.0, 0.0), None, 1.0))
    assert (ratio_g, ratio_b) == (1.0, 1.0)
    assert reason


def test_fails_closed_below_spread_floor():
    tiny = SPREAD_FLOOR / 2
    ratio_g, ratio_b, reason = fade_ratios_from_neutral_axis(((tiny, tiny, tiny), (0.0, 0.0, 0.0), None, 1.0))
    assert (ratio_g, ratio_b) == (1.0, 1.0)
    assert reason


def test_clamps_and_reports_an_out_of_bounds_ratio():
    lo, hi = RATIO_BOUNDS
    ratio_g, ratio_b, reason = fade_ratios_from_neutral_axis(((1.0, 10.0, 1.0), (0.0, 0.0, 0.0), None, 1.0))
    assert ratio_g == hi
    assert ratio_b == 1.0
    assert reason.startswith("Green read 10.0, outside 0.2 to 5, so it is set to 5.0.")
    assert "Red Survival" in reason
    assert "Blue" not in reason


def test_a_ratio_below_the_range_does_not_point_at_red_survival():
    _g, ratio_b, reason = fade_ratios_from_neutral_axis(((1.0, 1.0, 0.1), (0.0, 0.0, 0.0), None, 1.0))
    assert ratio_b == RATIO_BOUNDS[0]
    assert reason == "Blue read 0.1, outside 0.2 to 5, so it is set to 0.2."


def test_no_neutral_axis_found_fails_closed():
    ratio_g, ratio_b, reason = fade_ratios_from_neutral_axis(None)
    assert (ratio_g, ratio_b) == (1.0, 1.0)
    assert reason


def test_estimate_recovers_a_pure_fade_and_sets_red():
    """On a pure fade Estimate reads the ratios and puts the best-kept layer at 1.0."""
    image = _synthetic_neutral_slide(1.4, 1.2, ratio_r=0.7)
    est = estimate_fade(prefilter_log_grid(image, None, 0.0))
    assert est.reason == ""
    assert abs(est.green - 1.4) < 2e-2 and abs(est.blue - 1.2) < 2e-2
    assert abs(est.red - 1.0 / est.green) < 1e-9


def test_estimate_ignores_a_lifted_shadow():
    """A flat, lifted shadow in one channel does not move the midtone densities Estimate reads."""
    refs = ((-1.0, -1.2, -1.1), (-1.3, -1.5, -1.25), None, 0.9)
    g, b, _ = fade_ratios_from_neutral_axis(refs)
    assert abs(g - 1.2) < 1e-9 and abs(b - 1.1) < 1e-9


def test_estimate_meters_the_render_grid_before_the_fade():
    """A rotated, cropped slide with an E-6 crosstalk matrix: Estimate's grid is the render's
    meter grid with the fade taken out."""
    from dataclasses import replace

    from negpy.desktop.workers.chart_fit import ChartSource, fade_estimate_grid
    from negpy.features.process.models import ProcessMode
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
    from negpy.services.rendering.image_processor import ImageProcessor
    from tests.test_fade_fixtures import clean_slide, forward_fade_matrix

    ratio_g, ratio_b = 0.7, 0.85
    faded = 10.0 ** (np.log10(clean_slide(256, 3)) @ forward_fade_matrix(ratio_g, ratio_b).T)
    image = np.ascontiguousarray(faded[:, :192], dtype=np.float32)
    cfg = DEFAULT_WORKSPACE_CONFIG
    cfg = replace(
        cfg,
        process=replace(
            cfg.process,
            process_mode=ProcessMode.E6,
            crosstalk_matrix=(1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0),
            crosstalk_strength=1.0,
            crosstalk_process=ProcessMode.E6,
            fade_ratio_g=ratio_g,
            fade_ratio_b=ratio_b,
        ),
        geometry=replace(cfg.geometry, rotation=1, crop_rect=(0.1, 0.15, 0.85, 0.9), crop_from_auto=False),
    )
    grid = fade_estimate_grid(ChartSource(image, cfg, "fade-grid", None, None, (), 1))
    whole = fade_estimate_grid(
        ChartSource(image, replace(cfg, geometry=replace(cfg.geometry, crop_rect=None)), "fade-grid", None, None, (), 1)
    )
    assert grid is not None and whole is not None and grid.shape != whole.shape

    _, metrics = ImageProcessor(use_gpu=False).run_pipeline(
        image, cfg, "fade-grid", render_size_ref=256.0, prefer_gpu=False, cache_stages=False, wants_fade_grid=True
    )
    # The grid holds no fade, whatever the render's own.
    np.testing.assert_allclose(metrics["transfer_grid"], grid, atol=1e-5)


def test_estimate_reads_the_drawn_region_on_an_uncropped_slide():
    """Two halves faded apart: a region drawn on either half recovers that half's own fade."""
    from dataclasses import replace

    from negpy.desktop.workers.chart_fit import ChartSource, fade_estimate_grid
    from negpy.features.process.models import ProcessMode
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
    from tests.test_fade_fixtures import clean_slide, forward_fade_matrix

    def faded(ratio_g, ratio_b):
        return 10.0 ** (np.log10(clean_slide(256, 3)) @ forward_fade_matrix(ratio_g, ratio_b).T)

    image = np.ascontiguousarray(np.concatenate([faded(0.7, 0.85), faded(0.95, 0.5)], axis=1), dtype=np.float32)
    cfg = DEFAULT_WORKSPACE_CONFIG
    for rect, (ratio_g, ratio_b) in (((0.0, 0.0, 0.5, 1.0), (0.7, 0.85)), ((0.5, 0.0, 1.0, 1.0), (0.95, 0.5))):
        process = replace(cfg.process, process_mode=ProcessMode.E6, analysis_rect=rect)
        grid = fade_estimate_grid(ChartSource(image, replace(cfg, process=process), "fade-region", None, None, (), 1))
        est = estimate_fade(grid)
        assert est.reason == ""
        assert abs(est.green - ratio_g) < 0.03 and abs(est.blue - ratio_b) < 0.03


def test_the_estimate_grid_renders_again_only_when_more_than_the_mix_changes(monkeypatch):
    """A Mix drag estimates on every tick; the grid it reads is one render."""
    from dataclasses import replace

    from negpy.desktop.workers import chart_fit
    from negpy.features.process.models import ProcessMode
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
    from tests.test_fade_fixtures import clean_slide

    renders = []
    monkeypatch.setattr(chart_fit, "_grid_memo", None)
    monkeypatch.setattr(chart_fit, "_render_estimate_grid", lambda source, config: renders.append(config) or np.zeros((4, 4, 3)))
    image = np.ascontiguousarray(clean_slide(64, 3), dtype=np.float32)
    cfg = replace(DEFAULT_WORKSPACE_CONFIG, process=replace(DEFAULT_WORKSPACE_CONFIG.process, process_mode=ProcessMode.E6))

    def grid(**process):
        return chart_fit.fade_estimate_grid(
            chart_fit.ChartSource(image, replace(cfg, process=replace(cfg.process, **process)), "h", None, None, (), 1)
        )

    grid(fade_neutrals_mix=0.1)
    grid(fade_neutrals_mix=0.6, fade_ratio_b=0.8, fade_white_r=-0.1, fade_white_space="w", fade_crossover_table=(0.0,) * 24)
    assert len(renders) == 1
    grid(fade_neutrals_mix=0.6, crosstalk_strength=0.5)
    assert len(renders) == 2


def test_what_estimate_stores_leaves_the_estimate_grid_alone(monkeypatch):
    """The grid cache ignores the stored white, table and Mix, so they must not change the grid."""
    from dataclasses import replace

    from negpy.desktop.workers import chart_fit
    from negpy.features.process.models import ProcessMode
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
    from tests.test_fade_fixtures import clean_slide

    image = np.ascontiguousarray(clean_slide(64, 3), dtype=np.float32)
    cfg = replace(DEFAULT_WORKSPACE_CONFIG, process=replace(DEFAULT_WORKSPACE_CONFIG.process, process_mode=ProcessMode.E6))
    stored = replace(
        cfg,
        process=replace(
            cfg.process,
            fade_ratio_g=0.8,
            fade_white_anchor=True,
            fade_white_r=-0.2,
            fade_white_g=-0.1,
            fade_white_b=-0.3,
            fade_crossover=0.5,
            fade_crossover_table=(0.02,) * 24,
            fade_neutrals_mix=0.7,
            local_floors=(-1.9, -1.7, -1.5),
            local_ceils=(-0.1, -0.2, -0.3),
        ),
    )
    grids = []
    for c in (cfg, stored):
        monkeypatch.setattr(chart_fit, "_grid_memo", None)
        grids.append(chart_fit.fade_estimate_grid(chart_fit.ChartSource(image, c, "h", None, None, (), 1)))
    np.testing.assert_array_equal(grids[0], grids[1])


def _record_axis(monkeypatch, confidences):
    """Stub the neutral-axis meter: each call returns the next confidence and logs its inputs."""
    from negpy.features.process import fade

    calls = []
    axis = ((-1.7, -2.0, -1.0), (-2.5, -2.9, -1.4), None)

    def fake_axis(grid, bounds, roi, buffer, exclude_subjects=True, sky_sector=True):
        calls.append(((bounds.floors, bounds.ceils), exclude_subjects))
        conf = confidences[len(calls) - 1]
        return None if conf is None else (*axis, conf)

    monkeypatch.setattr(fade, "measure_neutral_axis_from_log", fake_axis)
    return calls


def _gray_grid():
    rng = np.random.default_rng(0)
    t = np.linspace(-0.1, -2.6, 64, dtype=np.float32)[None, :].repeat(64, axis=0)
    return np.stack([t, t * 1.2, t * 0.7], axis=-1) + rng.normal(0, 0.02, (64, 64, 3)).astype(np.float32)


def test_the_first_reading_takes_a_low_confidence_axis_with_the_sectors_excluded(monkeypatch):
    from negpy.features.process import fade

    calls = _record_axis(monkeypatch, [0.3])
    grid = _gray_grid()
    refs, reason = fade.neutral_axis_from_grid(grid)
    assert reason == "" and refs[3] == 0.3
    assert len(calls) == 1 and calls[0][1] is True
    first = fade._measured_slide_bounds(grid, fade.COLOR_EXTREME_PCT)
    assert calls[0][0] == (first.floors, first.ceils)


def test_a_failed_first_reading_falls_back_to_the_robust_bounds(monkeypatch):
    from negpy.features.process import fade

    calls = _record_axis(monkeypatch, [0.1, 0.3, 0.5])
    grid = _gray_grid()
    refs, reason = fade.neutral_axis_from_grid(grid)
    assert reason == "" and refs[3] == 0.5
    robust = fade._measured_slide_bounds(grid, fade.ROBUST_EXTREME_PCT)
    assert calls[1:] == [((robust.floors, robust.ceils), True), ((robust.floors, robust.ceils), False)]


def test_no_reading_over_its_gate_refuses(monkeypatch):
    from negpy.features.process import fade

    _record_axis(monkeypatch, [0.19, None, 0.39])
    refs, reason = fade.neutral_axis_from_grid(_gray_grid())
    assert refs is None and reason


def test_scanner_zeros_do_not_set_the_frame_bounds():
    from negpy.features.process.fade import _measured_slide_bounds

    grid = _gray_grid()
    zeros = grid.copy()
    zeros[0, :3, 1] = -6.0  # decoded zeros in green
    with_zeros = _measured_slide_bounds(zeros)
    without = _measured_slide_bounds(grid[1:])
    np.testing.assert_allclose(with_zeros.floors, without.floors, atol=0.02)
