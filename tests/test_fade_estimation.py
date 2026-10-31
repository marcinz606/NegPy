"""Survival-ratio estimation for Fade Restoration (features/process/fade.py), reading
Cast Removal's two-point neutral-detection algorithm rather than blind percentile spans.
A measure step, a build step, and the fail-closed reasons in between.
"""

import numpy as np

from negpy.features.process.fade import (
    RATIO_BOUNDS,
    SPREAD_FLOOR,
    estimate_fade_ratios,
    fade_ratios_from_neutral_axis,
)
from negpy.features.process.models import ProcessMode


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
    measure_neutral_axis_ratios, recovers the true ratios via fade_ratios_from_neutral_axis."""
    ratio_g_true, ratio_b_true = 0.6, 0.85
    image = _synthetic_neutral_slide(ratio_g_true, ratio_b_true)

    ratio_g, ratio_b, reason = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0)
    assert reason == ""
    assert abs(ratio_g - ratio_g_true) < 1e-2
    assert abs(ratio_b - ratio_b_true) < 1e-2


def test_recovers_a_mild_fade_without_overshooting():
    """A mild fade is where a biased estimate hurts most: an overshoot over-corrects past
    the uncorrected render."""
    ratio_g_true, ratio_b_true = 0.85, 0.9
    image = _synthetic_neutral_slide(ratio_g_true, ratio_b_true)

    ratio_g, ratio_b, reason = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0)
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

    ratio_g, ratio_b, reason = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0)
    assert reason == ""
    assert abs(ratio_g - ratio_g_true) < 1e-2
    assert abs(ratio_b - ratio_b_true) < 1e-2


def test_documented_wrong_behavior_on_unequal_unfaded_spreads():
    """A legitimately unequal-spread scene with no fade at all still reads as non-unity.
    Documented wrong behavior -- a monochromatic slide defeats any estimator built on this
    assumption -- not a bug for a future pass to "fix"."""
    image = _synthetic_neutral_slide(1.6, 1.0)  # a genuinely more colorful green channel
    ratio_g, ratio_b, reason = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0)
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
    assert reason.startswith("Implied green 10.0 is outside 0.2–5, set to 5.0")
    assert "Red Survival" in reason
    assert "blue" not in reason


def test_a_ratio_below_the_range_does_not_point_at_red_survival():
    _g, ratio_b, reason = fade_ratios_from_neutral_axis(((1.0, 1.0, 0.1), (0.0, 0.0, 0.0), None, 1.0))
    assert ratio_b == RATIO_BOUNDS[0]
    assert reason == "Implied blue 0.1 is outside 0.2–5, set to 0.2"


def test_no_neutral_axis_found_fails_closed():
    ratio_g, ratio_b, reason = fade_ratios_from_neutral_axis(None)
    assert (ratio_g, ratio_b) == (1.0, 1.0)
    assert reason


def test_fails_closed_for_non_transparency():
    image = _synthetic_neutral_slide(0.7, 1.4)
    ratio_g, ratio_b, reason = estimate_fade_ratios(image, ProcessMode.C41, None, 0.0)
    assert (ratio_g, ratio_b) == (1.0, 1.0)
    assert reason


def test_both_methods_recover_a_pure_fade_and_midtone_sets_red():
    """On a pure fade both methods read the same ratios; Midtone also puts the best-kept layer at 1.0."""
    from negpy.features.process.fade import EstimateMethod, estimate_fade

    image = _synthetic_neutral_slide(1.4, 1.2, ratio_r=0.7)
    estimates = estimate_fade(image, ProcessMode.E6, None, 0.0)
    midtone, spread = estimates[EstimateMethod.MIDTONE], estimates[EstimateMethod.SPREAD]
    for est in (midtone, spread):
        assert est.reason == ""
        assert abs(est.green - 1.4) < 2e-2 and abs(est.blue - 1.2) < 2e-2
    assert spread.red is None
    assert abs(midtone.red - 1.0 / midtone.green) < 1e-9


def test_midtone_ignores_a_lifted_shadow_that_misleads_spread():
    """A flat, lifted shadow in one channel compresses its spread; the midtone density does not move."""
    from negpy.features.process.fade import EstimateMethod, fade_ratios_from_neutral_axis

    refs = ((-1.0, -1.2, -1.1), (-1.3, -1.5, -1.25), None, 0.9)
    g_mid, b_mid, _ = fade_ratios_from_neutral_axis(refs, EstimateMethod.MIDTONE)
    g_spread, b_spread, _ = fade_ratios_from_neutral_axis(refs, EstimateMethod.SPREAD)
    assert abs(g_mid - 1.2) < 1e-9 and abs(b_mid - 1.1) < 1e-9
    assert abs(b_spread - 0.5) < 1e-9 and g_spread == 1.0


def test_estimate_fails_closed_for_both_methods_off_a_slide():
    from negpy.features.process.fade import estimate_fade

    image = _synthetic_neutral_slide(0.7, 1.4)
    for est in estimate_fade(image, ProcessMode.C41, None, 0.0).values():
        assert (est.red, est.green, est.blue) == (None, 1.0, 1.0) and est.reason


def test_estimate_reads_a_camera_buffer_in_working_space():
    """The render applies the camera matrix before the fade, so Estimate must too."""
    from negpy.features.process.capture_color import camera_to_working_matrix
    from negpy.features.process.fade import EstimateMethod, estimate_fade
    from negpy.features.process.models import ProcessMode
    from tests.test_fade_fixtures import clean_slide, forward_fade_matrix

    cam_xyz = [[0.6722, -0.0635, -0.0963], [-0.4287, 1.2460, 0.2028], [-0.0908, 0.2162, 0.5668], [0, 0, 0]]
    matrix = camera_to_working_matrix(cam_xyz, None)
    assert matrix is not None
    ratio_g, ratio_b = 0.75, 0.62
    working = 10.0 ** (np.log10(clean_slide(256, 0)) @ forward_fade_matrix(ratio_g, ratio_b).T)
    camera = np.clip(working @ np.linalg.inv(matrix).T, 1e-6, 1.0).astype(np.float32)

    bare = estimate_fade(camera, ProcessMode.E6, None, 0.1)[EstimateMethod.MIDTONE]
    mapped = estimate_fade(camera, ProcessMode.E6, None, 0.1, matrix)[EstimateMethod.MIDTONE]
    assert abs(mapped.green - ratio_g) < 0.03 and abs(mapped.blue - ratio_b) < 0.03
    assert abs(bare.blue - ratio_b) > 2 * abs(mapped.blue - ratio_b)
