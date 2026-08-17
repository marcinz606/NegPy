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
from negpy.features.process.models import ProcessMode


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


def test_chart_honours_the_papers_own_dmax(qapp):
    """The render builds its constants from effective_constants(paper), so a paper that
    raises d_max above the 2.3 default must move the plotted curve too. The chart used to
    omit the profile and drew the default-paper curve for every paper."""
    from negpy.desktop.view.widgets.charts import PhotometricCurveWidget
    from negpy.features.exposure.logic import print_curve, print_curve_output
    from negpy.features.exposure.papers import effective_paper_profile

    config = ExposureConfig(paper_profile="kodak_endura")
    paper = effective_paper_profile(config.paper_profile, ProcessMode.C41)
    assert paper.d_max > EXPOSURE_CONSTANTS["d_max"]  # else this test proves nothing

    w = PhotometricCurveWidget()
    w.update_curve(config, process_mode=ProcessMode.C41)
    plt_x = np.array([p[0] for p in w._curve_pts], dtype=np.float64)
    plotted = np.array([p[1] for p in w._curve_pts], dtype=np.float64)

    slope = grade_to_slope(config.grade, None)
    d_min = paper.d_min if config.paper_dmin else 0.0
    pivot = compute_pivot(slope, config.density, d_min=d_min, paper=paper)
    curve = print_curve(config, slope, pivot, ProcessMode.C41)
    np.testing.assert_allclose(plotted, print_curve_output(curve, 1.0 - plt_x), atol=1e-9)

    # Anti-vacuity: the old paper-blind curve must differ.
    toe_eff, shoulder_eff = grade_coupled_shape(slope, config.toe, config.shoulder)
    blind = CharacteristicCurve(
        contrast=slope,
        pivot=pivot,
        d_min=d_min,
        toe=toe_eff,
        toe_width=config.toe_width,
        shoulder=shoulder_eff,
        shoulder_width=config.shoulder_width,
        bpc=not config.paper_black,
    )
    assert np.max(np.abs(plotted - print_curve_output(blind, 1.0 - plt_x))) > 1e-3


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


def _band_gap(widget) -> float:
    """Largest vertical gap between the base curve and the mask band's far edge."""
    base = dict(widget._curve_pts)
    return max(abs(base[x] - y) for x, y in widget._mask_pts)


def test_mask_band_is_absent_without_a_mask(qapp):
    from negpy.desktop.view.widgets.charts import PhotometricCurveWidget

    w = PhotometricCurveWidget()
    w.update_curve(ExposureConfig(), mask_centre=0.5)
    assert w._mask_pts == []


def test_mask_band_needs_the_metered_centre(qapp):
    """Without the centre from the render there is nothing to rotate about, so the band
    stays hidden rather than guessing one."""
    from negpy.desktop.view.widgets.charts import PhotometricCurveWidget

    w = PhotometricCurveWidget()
    w.update_curve(ExposureConfig(contrast_mask=0.4), mask_centre=None)
    assert w._mask_pts == []


def test_mask_band_holds_the_centre_and_opens_at_the_ends(qapp):
    """A flat area at the centre val prints where it always did; the band widens away
    from it, which is what the (1-g) remap does."""
    from negpy.desktop.view.widgets.charts import PhotometricCurveWidget

    centre = 0.5
    w = PhotometricCurveWidget()
    w.update_curve(ExposureConfig(contrast_mask=0.4), mask_centre=centre)
    assert w._mask_pts

    base = dict(w._curve_pts)
    # plt_x = 1 - val, so the centre val sits at plt_x = 1 - centre.
    at_centre = min(w._mask_pts, key=lambda p: abs(p[0] - (1.0 - centre)))
    assert abs(base[at_centre[0]] - at_centre[1]) < 5e-3

    ends = [p for p in w._mask_pts if p[0] < 0.15 or p[0] > 0.85]
    assert max(abs(base[x] - y) for x, y in ends) > 2e-2


def test_mask_band_reverses_with_the_gamma_sign(qapp):
    """Reduction pulls the ends toward the centre tone, increase pushes them away."""
    from negpy.desktop.view.widgets.charts import PhotometricCurveWidget

    reduce_w, increase_w = PhotometricCurveWidget(), PhotometricCurveWidget()
    reduce_w.update_curve(ExposureConfig(contrast_mask=0.4), mask_centre=0.5)
    increase_w.update_curve(ExposureConfig(contrast_mask=-0.4), mask_centre=0.5)

    base = dict(reduce_w._curve_pts)
    # The thin end of the negative (plt_x high) prints light; reduction darkens it back
    # toward the centre, increase drives it further.
    x = max(p[0] for p in reduce_w._mask_pts if p[0] < 0.95)
    red = next(y for px, y in reduce_w._mask_pts if px == x)
    inc = next(y for px, y in increase_w._mask_pts if px == x)
    assert red < base[x] < inc or inc < base[x] < red


def test_mask_band_is_off_for_the_flat_master(qapp):
    from negpy.desktop.view.widgets.charts import PhotometricCurveWidget

    w = PhotometricCurveWidget()
    w.update_curve(ExposureConfig(contrast_mask=0.4), slope=0.65, pivot=0.10, flat=True, mask_centre=0.5)
    assert w._mask_pts == []
