"""The step wedge is a Stouffer T2115 printed through the frame's current settings.

The instrument's whole claim is that it shows where the paper's scale runs out: a hard
grade must visibly crush the toe steps together and block the shoulder steps, and the
usable span must narrow accordingly.
"""

import numpy as np

from negpy.features.exposure.analysis import (
    WEDGE_SEPARATION,
    WEDGE_STEP_DENSITY,
    WEDGE_STEPS,
    wedge_step_density,
    wedge_usable_span,
    wedge_vals,
)
from negpy.features.exposure.logic import compute_pivot, grade_to_slope, print_curve, print_curve_output
from negpy.features.exposure.models import ExposureConfig


def _printed(config) -> np.ndarray:
    slope = grade_to_slope(config.grade, None)
    pivot = compute_pivot(slope, config.density, d_min=0.0)
    return print_curve_output(print_curve(config, slope, pivot), wedge_vals())


def test_the_wedge_is_twentyone_steps_clear_first():
    vals = wedge_vals()
    assert len(vals) == WEDGE_STEPS
    assert vals[0] == 1.0  # clear step: film base, prints paper black
    assert vals[-1] == 0.0  # densest step: prints paper white
    steps = np.diff(vals)
    assert np.all(steps < 0)
    np.testing.assert_allclose(steps, steps[0])  # uniform


def test_a_three_density_scan_gives_the_t2115_increment():
    # 21 steps 0.15 D apart span 3.0 D over 20 intervals — pins the constant to the
    # physical wedge it claims to be.
    assert wedge_step_density(3.0) == 0.15
    assert wedge_step_density(1.5) == 0.075
    assert wedge_step_density(None) == WEDGE_STEP_DENSITY
    assert wedge_step_density(0.0) == WEDGE_STEP_DENSITY


def test_the_printed_wedge_runs_paper_black_to_paper_white():
    enc = _printed(ExposureConfig())
    assert len(enc) == WEDGE_STEPS
    assert np.all(np.diff(enc) >= 0)  # monotone: denser step prints lighter
    assert enc[0] < 0.15
    assert enc[-1] > 0.85


def test_a_hard_grade_crushes_the_toe_and_blocks_the_shoulder():
    hard = _printed(ExposureConfig(grade=50.0))
    normal = _printed(ExposureConfig(grade=115.0))

    # At the hardest grade the first two steps collapse into paper black...
    assert hard[1] - hard[0] < WEDGE_SEPARATION
    # ...while at the default grade the same neighbours still separate.
    assert normal[1] - normal[0] > WEDGE_SEPARATION

    hard_span = wedge_usable_span(hard)
    normal_span = wedge_usable_span(normal)
    assert hard_span is not None and normal_span is not None
    assert normal_span == (0, WEDGE_STEPS - 1)  # the whole scale is usable
    # The hard grade loses steps at both ends, not just one.
    assert hard_span[0] > normal_span[0]
    assert hard_span[1] < normal_span[1]


def test_a_flat_scale_has_no_usable_span():
    assert wedge_usable_span(np.zeros(WEDGE_STEPS)) is None
    assert wedge_usable_span(np.full(WEDGE_STEPS, 0.5)) is None


def test_the_wedge_and_the_chart_share_one_evaluator(qapp):
    """Fails the moment charts.py grows a second print-curve evaluator."""
    from negpy.desktop.view.widgets.charts import PhotometricCurveWidget

    config = ExposureConfig()
    w = PhotometricCurveWidget()
    w.update_curve(config)

    plt_x = np.array([p[0] for p in w._curve_pts], dtype=np.float64)
    plotted = np.array([p[1] for p in w._curve_pts], dtype=np.float64)

    slope = grade_to_slope(config.grade, None)
    pivot = compute_pivot(slope, config.density, d_min=0.0)
    shared = print_curve_output(print_curve(config, slope, pivot), 1.0 - plt_x)
    np.testing.assert_allclose(plotted, shared, atol=1e-12)
