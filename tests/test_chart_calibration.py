import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.process import chart as ch
from negpy.kernel.image.logic import lab_to_rgb_working, working_oetf_encode

_CELL = 20
_GAP = 6


_PITCH = _CELL + _GAP
_C0 = _GAP + _CELL / 2
_SPAN_Y = 3 * _PITCH


def _corners_px() -> np.ndarray:
    return np.array([[_C0, _C0], [_C0 + 5 * _PITCH, _C0], [_C0 + 5 * _PITCH, _C0 + _SPAN_Y], [_C0, _C0 + _SPAN_Y]])


def _chart_image(rgb: np.ndarray, chart: ch.ChartSpec = ch.COLORCHECKER_24, orientation: int = 0) -> np.ndarray:
    """The card as linear working-RGB patches on black, sitting in the frame in ``orientation``."""
    img = np.zeros((_GAP + 4 * _PITCH, _GAP + 6 * _PITCH, 3), dtype=np.float32)
    half = _CELL // 2
    for patch, (x, y) in zip(ch.reading_order(chart, orientation), ch.patch_centers(_corners_px(), chart, orientation)):
        img[round(y) - half : round(y) + half, round(x) - half : round(x) + half] = rgb[patch]
    return img


def _reference_rgb(chart: ch.ChartSpec) -> np.ndarray:
    return np.clip(lab_to_rgb_working(ch.reference_lab(chart).astype(np.float32)), 1e-4, 1.0)


@pytest.mark.parametrize(
    "lab1, lab2, expected",
    [
        # Sharma, Wu & Dalal (2005) test pairs 1, 7, 17 and 25.
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
        ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ],
)
def test_delta_e2000_matches_the_published_pairs(lab1, lab2, expected) -> None:
    assert ch.delta_e2000(np.array([lab1]), np.array([lab2]))[0] == pytest.approx(expected, abs=1e-4)


def test_color_only_delta_e_leaves_lightness_out() -> None:
    ref = np.array([[50.0, 20.0, -10.0], [50.0, 20.0, -10.0]])
    lab = np.array([[70.0, 20.0, -10.0], [80.0, 26.0, -4.0]])
    assert ch.delta_e_color(ref, lab)[0] == 0.0
    assert ch.delta_e_color(ref, lab)[1] == pytest.approx(ch.delta_e2000(ref[1:], np.array([[50.0, 26.0, -4.0]]))[0])


def test_reference_lab_keeps_a_neutral_neutral() -> None:
    spec = ch.ChartSpec("gray", ("g",) * 24, ((50.0, 0.0, 0.0),) * 24, (0,))
    assert np.abs(ch.reference_lab(spec)[0] - (50.0, 0.0, 0.0)).max() < 0.05


def test_charts_are_complete() -> None:
    for chart in ch.CHARTS:
        assert len(chart.patch_names) == len(chart.lab_d50) == chart.patch_count
        assert chart.row_centers[0] == 0.0 and chart.row_centers[-1] == 1.0 and np.all(np.diff(chart.row_centers) > 0)
        assert all(abs(chart.lab_d50[i][1]) < 2.5 and abs(chart.lab_d50[i][2]) < 3.0 for i in chart.exposure_patches)


def test_patch_centers_span_the_corners_row_major() -> None:
    corners = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 3.0], [0.0, 3.0]])
    centers = ch.patch_centers(corners, ch.COLORCHECKER_24)
    assert centers.shape == (24, 2)
    assert np.allclose(centers[0], (0, 0)) and np.allclose(centers[5], (5, 0)) and np.allclose(centers[23], (5, 3))
    assert np.allclose(centers[7], (1, 1))


@pytest.mark.parametrize("perm", [(0, 1, 2, 3), (2, 0, 3, 1), (3, 2, 1, 0)])
def test_patch_centers_take_the_corners_in_any_order_and_any_way_up(perm) -> None:
    sideways = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 5.0], [0.0, 5.0]])[list(perm)]
    centers = {tuple(np.round(c, 6)) for c in ch.patch_centers(sideways, ch.COLORCHECKER_24)}
    assert centers == {(float(x), float(y)) for x in range(4) for y in range(6)}


def test_a_sideways_chart_with_crossed_dots_reads_its_colors() -> None:
    chart = ch.COLORCHECKER_24
    img = _chart_image(_reference_rgb(chart))
    width = img.shape[1]
    corners = np.array([(y, width - 1 - x) for x, y in _corners_px()])[[0, 2, 1, 3]]
    enc = working_oetf_encode(np.ascontiguousarray(np.rot90(img)))
    lab = ch.measure_lab(enc, corners, chart, ch.detect_orientation(enc, corners, chart))
    assert ch.delta_e2000(ch.reference_lab(chart), lab).mean() < 1.0


def test_uneven_rows_move_with_the_card() -> None:
    corners = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 10.0], [0.0, 10.0]])
    rows = lambda o: sorted({round(float(y), 6) for _, y in ch.patch_centers(corners, ch.DKC_PRO, o)})  # noqa: E731
    assert rows(0) == rows(1) == [0.0, 6.2, 10.0]
    assert rows(2) == rows(3) == [0.0, 3.8, 10.0]
    # The sample box takes the tighter of the column pitch and the nearest two rows.
    assert ch.sample_radius(corners, ch.DKC_PRO) == pytest.approx(0.5 * 0.4 * 3.8)
    narrow = np.array([[0.0, 0.0], [12.0, 0.0], [12.0, 10.0], [0.0, 10.0]])
    assert ch.sample_radius(narrow, ch.DKC_PRO) == pytest.approx(0.5 * 0.4 * 2.4)


@pytest.mark.parametrize("chart", ch.CHARTS, ids=lambda c: c.name)
@pytest.mark.parametrize("which", ch.ORIENTATIONS)
def test_orientation_is_found_from_the_colors(chart: ch.ChartSpec, which: int) -> None:
    img = working_oetf_encode(_chart_image(_reference_rgb(chart), chart, which))
    assert ch.detect_orientation(img, _corners_px(), chart) == which
    lab = ch.measure_lab(img, _corners_px(), chart, which)
    assert ch.delta_e2000(ch.reference_lab(chart), lab).mean() < 1.0


def test_fit_undoes_a_known_density_mix() -> None:
    chart = ch.SPYDERCHECKR_24
    density = -np.log10(_reference_rgb(chart))
    mix = np.array([[0.9, 0.15, -0.05], [-0.1, 0.85, 0.25], [0.05, 0.2, 0.75]])

    def render(matrix: np.ndarray) -> np.ndarray:
        m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        m = m / m.sum(axis=1, keepdims=True)
        out = 10.0 ** -(density @ mix.T @ m.T)
        return working_oetf_encode(_chart_image(np.clip(out, 0.0, 1.0).astype(np.float32), chart, 1))

    fit = ch.fit_chart_matrix(render, _corners_px(), chart)
    assert fit.orientation == 1
    assert fit.before.mean() > 4.0
    assert fit.after.mean() < 0.4 * fit.before.mean()
    assert fit.after_color.mean() < 0.4 * fit.before_color.mean()
    assert np.all(fit.after_color <= fit.after + 1e-9)
    assert np.allclose(np.reshape(fit.matrix, (3, 3)).diagonal(), 1.0)


def test_fit_stops_when_cancelled() -> None:
    chart = ch.SPYDERCHECKR_24
    img = working_oetf_encode(_chart_image(_reference_rgb(chart), chart))
    with pytest.raises(ch.FitCancelled):
        ch.fit_chart_matrix(lambda _m: img, _corners_px(), chart, cancelled=lambda: True)


def test_fit_config_meters_the_frame_on_its_own() -> None:
    from dataclasses import replace

    from negpy.desktop.workers.chart_fit import chart_fit_config

    base = WorkspaceConfig()
    base = replace(
        base,
        process=replace(base.process, use_luma_average=True, use_cast_average=True, lock_bounds=True, crosstalk_strength=0.3),
        exposure=replace(base.exposure, paper_profile="kodak_2383"),
    )
    cfg = chart_fit_config(base, np.eye(3) * 2.0)
    p = cfg.process
    assert p.crosstalk_strength == 1.0 and p.crosstalk_matrix[0] == 2.0
    assert not (p.use_luma_average or p.use_color_average or p.use_cast_average or p.lock_bounds)
    # A stock's dyes are a print look; fitted through them they would land in the rig's matrix.
    assert cfg.exposure.paper_profile == WorkspaceConfig().exposure.paper_profile


def test_worker_renders_through_the_real_pipeline() -> None:
    from negpy.desktop.workers.chart_fit import ChartFitWorker, ChartSource

    rng = np.random.default_rng(0)
    buf = rng.uniform(0.05, 0.9, (90, 120, 3)).astype(np.float32)
    worker = ChartFitWorker(ChartSource(buf, WorkspaceConfig(), "t", None, None, (), 1))
    out = worker.render(np.eye(3))
    assert out.shape == (90, 120, 3) and np.isfinite(out).all()


def test_dialog_saves_a_measured_profile(monkeypatch, tmp_path) -> None:
    from negpy.desktop.view.widgets.chart_calibration_dialog import ChartCalibrationDialog
    from negpy.desktop.workers.chart_fit import ChartSource
    from negpy.kernel.system.config import APP_CONFIG
    from negpy.services.assets.crosstalk import CrosstalkProfiles

    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    buf = np.full((60, 90, 3), 0.3, dtype=np.float32)
    dlg = ChartCalibrationDialog(ChartSource(buf, WorkspaceConfig(), "t", None, None, (), 1), "Adobe RGB")
    dlg._worker.wait()
    dlg._on_previewed(np.full((60, 90, 3), 0.5, dtype=np.float32))
    dlg.name_edit.setText("Test Rig")
    assert dlg.measure_btn.isEnabled()
    saved = []
    dlg.profile_saved.connect(saved.append)
    m = np.eye(3).ravel()
    m[1] = -0.1
    after = np.round(np.linspace(0.5, 7.4, 24), 1)
    dlg._on_fitted(ch.ChartFit(tuple(m), np.full(24, 9.0), after + 3.0, 1, np.full(24, 8.0), after))
    assert np.array_equal(dlg.view._scores, after[ch.reading_order(ch.CHARTS[0], 1)])
    assert "over the 24 patches" in dlg.result_label.text()
    assert "6 within 2, 10 within 5, 8 beyond" in dlg.result_label.text()
    assert saved == ["Test Rig"]
    assert CrosstalkProfiles.get_type("Test Rig") == "measured"
    assert CrosstalkProfiles.get_matrix("Test Rig")[1] == pytest.approx(-0.1)
    assert dlg.cancel_btn.text() == "Close"
    dlg.done(0)


def test_dialog_draws_the_boxes_the_way_the_card_sits() -> None:
    from negpy.desktop.view.widgets.chart_calibration_dialog import ChartCalibrationDialog
    from negpy.desktop.workers.chart_fit import ChartSource

    chart = ch.DKC_PRO
    enc = working_oetf_encode(_chart_image(_reference_rgb(chart), chart, 2))
    dlg = ChartCalibrationDialog(ChartSource(enc, WorkspaceConfig(), "t", None, None, (), 1), "Adobe RGB")
    dlg._worker.wait()
    h, w = enc.shape[:2]
    dlg.view._corners = _corners_px() / np.array([w, h])
    dlg._on_previewed(enc)
    dlg.chart_choice.setCurrentIndex(ch.CHARTS.index(chart))
    assert (dlg.view._chart, dlg.view._orientation) == (chart, 2)
    dlg.done(0)
