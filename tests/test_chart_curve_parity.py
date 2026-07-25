"""The H&D chart must draw the same knees the engine renders with.

Regression guard for the grade-coupled toe/shoulder: the render path bases its
knees on grade_coupled_shape, and the chart used to pass the raw slider values,
diverging at hard grades.
"""

import numpy as np

from negpy.features.exposure.logic import (
    CharacteristicCurve,
    compute_pivot,
    grade_coupled_shape,
    grade_to_slope,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.kernel.image.logic import working_oetf_encode


def _reference_points(config, toe, shoulder):
    slope = grade_to_slope(config.grade, None)
    d_min = EXPOSURE_CONSTANTS["d_min"]
    pivot = compute_pivot(slope, config.density, d_min=d_min)
    curve = CharacteristicCurve(
        contrast=slope,
        pivot=pivot,
        d_min=d_min,
        toe=toe,
        toe_width=config.toe_width,
        shoulder=shoulder,
        shoulder_width=config.shoulder_width,
    )
    return slope, pivot, curve


def test_chart_uses_grade_coupled_knees(qapp):
    from negpy.desktop.view.widgets.charts import PhotometricCurveWidget

    # paper_dmin/paper_black pinned: reference math below doesn't model d_min/bpc.
    config = ExposureConfig(grade=50.0, paper_dmin=True, paper_black=True)  # hardest grade: coupling is maximal
    w = PhotometricCurveWidget()
    w.update_curve(config)

    plt_x = np.array([p[0] for p in w._curve_pts], dtype=np.float64)
    plotted = np.array([p[1] for p in w._curve_pts], dtype=np.float64)
    x_log_exp = 1.0 - plt_x

    slope, _, _ = _reference_points(config, 0.0, 0.0)
    toe_eff, shoulder_eff = grade_coupled_shape(slope, config.toe, config.shoulder)
    assert toe_eff > config.toe and shoulder_eff > config.shoulder  # coupling active at R50

    def _expected(toe, shoulder):
        _, _, curve = _reference_points(config, toe, shoulder)
        d = curve(x_log_exp.astype(np.float32))
        t = np.power(10.0, -np.asarray(d))
        return np.asarray(working_oetf_encode(t.astype(np.float32))).reshape(-1)

    np.testing.assert_allclose(plotted, _expected(toe_eff, shoulder_eff), atol=1e-4)
    # The old wiring (raw slider values) must NOT match — otherwise this test is vacuous.
    assert np.max(np.abs(plotted - _expected(config.toe, config.shoulder))) > 1e-3


def test_chart_plots_cast_curvature(qapp):
    from negpy.desktop.view.widgets.charts import PhotometricCurveWidget

    config = ExposureConfig(paper_dmin=True, paper_black=True)
    slopes, pivots = (4.2, 4.0, 3.8), (0.30, 0.32, 0.34)
    curvs = (0.5, 0.0, -0.4)
    w = PhotometricCurveWidget()
    w.update_curve(config, slope=slopes[1], pivot=pivots[1], slopes=slopes, pivots=pivots, curvatures=curvs)

    assert w._channel_curves, "curvature spread alone must diverge the channel traces"
    toe_eff, shoulder_eff = grade_coupled_shape(slopes[1], config.toe, config.shoulder)
    d_min = EXPOSURE_CONSTANTS["d_min"]

    def _expected(ch, curvature):
        curve = CharacteristicCurve(
            contrast=slopes[ch],
            pivot=pivots[ch],
            d_min=d_min,
            toe=toe_eff,
            toe_width=config.toe_width,
            shoulder=shoulder_eff,
            shoulder_width=config.shoulder_width,
            curvature=curvature,
        )
        plt_x = np.array([p[0] for p in w._channel_curves[ch][1]], dtype=np.float64)
        d = curve((1.0 - plt_x).astype(np.float32))
        t = np.power(10.0, -np.asarray(d))
        return np.asarray(working_oetf_encode(t.astype(np.float32))).reshape(-1)

    for ch in range(3):
        plotted = np.array([p[1] for p in w._channel_curves[ch][1]], dtype=np.float64)
        np.testing.assert_allclose(plotted, _expected(ch, curvs[ch]), atol=1e-4)
    # Anti-vacuity: dropping the curvature must NOT reproduce the red trace.
    plotted_r = np.array([p[1] for p in w._channel_curves[0][1]], dtype=np.float64)
    assert np.max(np.abs(plotted_r - _expected(0, 0.0))) > 1e-3


def test_curvature_spread_alone_triggers_divergence(qapp):
    from negpy.desktop.view.widgets.charts import PhotometricCurveWidget

    config = ExposureConfig()
    w = PhotometricCurveWidget()
    # Identical slopes/pivots, zero trims: only the curvature differs.
    w.update_curve(config, slope=4.0, pivot=0.3, slopes=(4.0, 4.0, 4.0), pivots=(0.3, 0.3, 0.3), curvatures=(0.3, 0.0, 0.0))
    assert w._channel_curves
    w.update_curve(config, slope=4.0, pivot=0.3, slopes=(4.0, 4.0, 4.0), pivots=(0.3, 0.3, 0.3), curvatures=(0.0, 0.0, 0.0))
    assert not w._channel_curves
