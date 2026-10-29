"""The Analysis waveform: print luma per image column, binned the same way on both engines,
and the chart view that shows it."""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest

from negpy.desktop.view.styles.color_vision import palette_for
from negpy.desktop.session import AppState
from negpy.features.exposure.analysis import (
    WAVEFORM_BINS,
    WAVEFORM_COLOR_SCALE,
    WAVEFORM_COLS,
    density_parade,
    scan_density_axis,
    scan_full_scale_columns,
    scan_density_of_val,
    waveform,
    waveform_color,
    waveform_parade,
)
from negpy.infrastructure.gpu.device import GPUDevice


def _require_gpu(metrics: dict) -> None:
    """Skip when run_pipeline fell back to the CPU: the scan keys exist on both engines, so a
    fallback would compare the CPU with itself. histogram_raw comes from the GPU only."""
    if "histogram_raw" not in metrics:
        pytest.skip("GPU render fell back to the CPU")


def test_a_left_to_right_ramp_climbs_column_by_column() -> None:
    ramp = np.linspace(0.0, 0.999, 512, dtype=np.float32)
    img = np.repeat(np.repeat(ramp[None, :, None], 8, axis=0), 3, axis=2)

    counts = waveform(img)

    assert counts.shape == (WAVEFORM_BINS, WAVEFORM_COLS)
    assert counts.sum() == 8 * 512
    peak_bin = counts.argmax(axis=0)
    assert peak_bin[0] == 0 and peak_bin[-1] == WAVEFORM_BINS - 1
    assert np.all(np.diff(peak_bin) >= 0)


def test_every_column_gets_its_share_of_a_narrow_frame() -> None:
    img = np.full((10, 64, 3), 0.5, dtype=np.float32)

    counts = waveform(img)

    # 64 pixels spread over 256 columns: every fourth column holds one pixel column.
    assert counts.sum(axis=0)[::4].tolist() == [10.0] * 64
    assert waveform(np.zeros((4, 4))) is None


def test_each_cell_sums_the_color_of_its_pixels() -> None:
    img = np.zeros((10, 512, 3), dtype=np.float32)
    img[:, :256] = (0.8, 0.3, 0.2)
    img[:, 256:] = (0.2, 0.3, 0.8)

    counts = waveform(img)
    sums = waveform_color(img)

    assert sums.shape == (3, WAVEFORM_BINS, WAVEFORM_COLS)
    mean = sums.sum(axis=1) / counts.sum(axis=0) / WAVEFORM_COLOR_SCALE
    np.testing.assert_allclose(mean[:, 0], (0.8, 0.3, 0.2), atol=1e-3)
    np.testing.assert_allclose(mean[:, -1], (0.2, 0.3, 0.8), atol=1e-3)
    assert waveform_color(np.zeros((4, 4))) is None


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestWaveformParity(unittest.TestCase):
    def test_wgsl_bins_match_the_cpu(self):
        from negpy.domain.models import WorkspaceConfig
        from negpy.services.rendering.image_processor import ImageProcessor

        rng = np.random.default_rng(3)
        grad = np.linspace(0.05, 0.9, 300, dtype=np.float32)
        img = np.repeat(grad[None, :, None], 200, axis=0).repeat(3, axis=2)
        img = np.ascontiguousarray(img + rng.uniform(0, 0.02, img.shape).astype(np.float32))
        proc = ImageProcessor()
        cfg = WorkspaceConfig()

        cpu_out, _ = proc.run_pipeline(img, cfg, "wave|cpu", render_size_ref=300, prefer_gpu=False, readback_metrics=True)
        _gpu_out, gpu_metrics = proc.run_pipeline(img, cfg, "wave|gpu", render_size_ref=300, prefer_gpu=True, readback_metrics=True)

        gpu_wave = gpu_metrics.get("waveform")
        if gpu_wave is None:
            self.skipTest("render fell back to the CPU engine")
        # A CPU render carries no in-shader waveform; the worker bins the float output.
        cpu_wave = waveform(cpu_out)
        self.assertEqual(cpu_wave.sum(), gpu_wave.sum())
        a = cpu_wave / cpu_wave.sum()
        b = gpu_wave / gpu_wave.sum()
        self.assertLess(float(np.abs(a - b).sum()), 0.02)

    def test_wgsl_color_sums_match_the_cpu(self):
        from negpy.domain.models import WorkspaceConfig
        from negpy.services.rendering.image_processor import ImageProcessor

        grad = np.linspace(0.05, 0.9, 300, dtype=np.float32)
        img = np.stack([grad, grad[::-1], np.full_like(grad, 0.4)], axis=-1)
        img = np.ascontiguousarray(np.repeat(img[None], 200, axis=0))
        proc = ImageProcessor()
        cfg = WorkspaceConfig()

        cpu_out, _ = proc.run_pipeline(img, cfg, "wavec|cpu", render_size_ref=300, prefer_gpu=False, readback_metrics=True)
        _gpu_out, gpu_metrics = proc.run_pipeline(img, cfg, "wavec|gpu", render_size_ref=300, prefer_gpu=True, readback_metrics=True)

        gpu_sums = gpu_metrics.get("waveform_color")
        if gpu_sums is None:
            self.skipTest("render fell back to the CPU engine")
        cpu_sums = waveform_color(cpu_out)
        # Per-column channel means: robust to a pixel landing one tone bin apart.
        cpu_mean = cpu_sums.sum(axis=1) / cpu_sums.sum()
        gpu_mean = gpu_sums.sum(axis=1) / gpu_sums.sum()
        self.assertLess(float(np.abs(cpu_mean - gpu_mean).sum()), 0.02)


def test_waveform_widget_paints_and_marks_clipped_columns(qapp) -> None:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget()
    widget.resize(256, 64)
    counts = np.zeros((WAVEFORM_BINS, WAVEFORM_COLS))
    counts[0, :64] = 5.0  # left quarter at paper black
    counts[64, 64:] = 5.0
    widget.set_data(counts)

    img = widget.grab().toImage()

    shadow, highlight = palette_for("standard").pair
    assert QColor(img.pixel(20, 61)).name() == QColor(shadow).name()
    assert QColor(img.pixel(200, 61)).name() != QColor(shadow).name()

    widget.set_clip_colors(*palette_for("tritan").pair)
    assert QColor(widget.grab().toImage().pixel(20, 61)).name() == QColor(palette_for("tritan").pair[0]).name()

    widget.set_log_scale(True)
    assert widget._images is None
    widget.set_data(None)
    widget.grab()


def test_the_view_choice_switches_the_chart_and_persists() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.view_btn.currentData.return_value = "waveform"

    RightPanel._on_analysis_view(panel, 1)

    panel.chart_stack.setCurrentIndex.assert_called_once_with(1)
    panel.controller.session.repo.save_global_setting.assert_called_once_with("analysis_view", "waveform")


def test_a_peek_clears_the_waveform() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.controller.state = AppState(negative_peek=True)

    RightPanel._update_histograms(panel, {"waveform": np.ones((WAVEFORM_BINS, WAVEFORM_COLS))})

    panel.waveform_widget.set_data.assert_called_once_with(None, None)


def test_waveform_widget_draws_each_cell_in_its_color(qapp) -> None:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget()
    widget.resize(WAVEFORM_COLS, WAVEFORM_BINS)
    counts = np.zeros((WAVEFORM_BINS, WAVEFORM_COLS))
    counts[64, :] = 4.0
    color = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    color[0, 64, :128] = 4.0 * 0.9 * WAVEFORM_COLOR_SCALE  # left half red
    color[:, 64, 128:] = 4.0 * 0.5 * WAVEFORM_COLOR_SCALE  # right half neutral
    widget.set_data(counts, color)

    img = widget.grab().toImage()
    y = WAVEFORM_BINS - 1 - 64

    red = QColor(img.pixel(60, y))
    gray = QColor(img.pixel(200, y))
    assert red.red() > 200 and red.green() < 60 and red.blue() < 60
    assert gray.red() > 200 and gray.red() == gray.green() == gray.blue()


def test_the_parade_bins_each_channel_on_its_own() -> None:
    img = np.zeros((10, 512, 3), dtype=np.float32)
    img[:, :256] = (0.8, 0.3, 0.2)
    img[:, 256:] = (0.2, 0.3, 0.8)

    parade = waveform_parade(img)

    assert parade.shape == (3, WAVEFORM_BINS, WAVEFORM_COLS)
    assert parade[0].sum() == parade[1].sum() == parade[2].sum() == 10 * 512
    peak = parade.argmax(axis=1)
    assert peak[:, 0].tolist() == [int(0.8 * WAVEFORM_BINS), int(0.3 * WAVEFORM_BINS), int(0.2 * WAVEFORM_BINS)]
    assert peak[:, -1].tolist() == [int(0.2 * WAVEFORM_BINS), int(0.3 * WAVEFORM_BINS), int(0.8 * WAVEFORM_BINS)]
    assert waveform_parade(np.zeros((4, 4))) is None


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
def test_wgsl_parade_matches_the_cpu() -> None:
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    grad = np.linspace(0.05, 0.9, 300, dtype=np.float32)
    img = np.stack([grad, grad[::-1], np.full_like(grad, 0.4)], axis=-1)
    img = np.ascontiguousarray(np.repeat(img[None], 200, axis=0))
    proc = ImageProcessor()
    cfg = WorkspaceConfig()

    cpu_out, _ = proc.run_pipeline(img, cfg, "parade|cpu", render_size_ref=300, prefer_gpu=False, readback_metrics=True)
    _gpu_out, gpu_metrics = proc.run_pipeline(img, cfg, "parade|gpu", render_size_ref=300, prefer_gpu=True, readback_metrics=True)

    gpu = gpu_metrics.get("waveform_parade")
    if gpu is None:
        return
    cpu = waveform_parade(cpu_out)
    assert cpu.sum() == gpu.sum()
    for c in range(3):
        assert float(np.abs(cpu[c] / cpu[c].sum() - gpu[c] / gpu[c].sum()).sum()) < 0.02


def test_the_parade_draws_three_panels_in_channel_colors(qapp) -> None:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.styles.theme import THEME
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    widget.resize(3 * 100, WAVEFORM_BINS)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[:, 64, :] = 4.0
    counts[0, -1, :32] = 1.0  # red reaches paper white on the left
    widget.set_data(counts)

    img = widget.grab().toImage()
    y = WAVEFORM_BINS - 1 - 64

    for panel, hex_ in enumerate((THEME.channel_red_text, THEME.channel_green_text, THEME.channel_blue_text)):
        got, want = QColor(img.pixel(panel * 101 + 50, y)), QColor(hex_)
        assert max(abs(got.red() - want.red()), abs(got.green() - want.green()), abs(got.blue() - want.blue())) <= 1
    highlight = QColor(palette_for("standard").pair[1]).name()
    assert QColor(img.pixel(5, 2)).name() == highlight
    assert QColor(img.pixel(106, 2)).name() != highlight
    widget.set_data(np.ones((WAVEFORM_BINS, WAVEFORM_COLS)))  # a luma waveform is not a parade
    assert widget._counts is None


def test_a_flat_peek_clears_the_parade() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.controller.state = AppState(flat_peek=True)
    parade = np.ones((3, WAVEFORM_BINS, WAVEFORM_COLS))

    RightPanel._update_histograms(panel, {"waveform_parade": parade, "density_parade": parade})

    panel._push_scan_domain.assert_called_once_with(False)
    panel.parade_widget.set_data.assert_called_once_with(None)


def test_peek_negative_shows_the_scan_on_the_parade() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.controller.state = AppState(negative_peek=True)
    scan = np.ones((3, WAVEFORM_BINS, WAVEFORM_COLS))

    RightPanel._update_histograms(panel, {"waveform_parade": np.zeros_like(scan), "density_parade": scan})

    panel._push_scan_domain.assert_called_once_with(True)
    panel.parade_widget.set_data.assert_called_once_with(scan)
    panel.waveform_widget.set_data.assert_called_once_with(None, None)


def test_the_print_parade_returns_after_the_peek() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.controller.state = AppState()
    printed = np.ones((3, WAVEFORM_BINS, WAVEFORM_COLS))

    RightPanel._update_histograms(panel, {"waveform_parade": printed, "density_parade": np.zeros_like(printed)})

    panel._push_scan_domain.assert_called_once_with(False)
    panel.parade_widget.set_data.assert_called_once_with(printed)


def test_the_scan_parade_bins_each_channel_scan_density_dense_at_the_bottom() -> None:
    log = np.zeros((10, 512, 3), dtype=np.float32)
    log[:, :256] = (-0.3, -1.0, -2.5)  # base-like red, dense blue
    log[:, 256:] = (-2.5, -1.0, -0.3)

    parade = density_parade(log)

    assert parade.shape == (3, WAVEFORM_BINS, WAVEFORM_COLS)
    assert parade[0].sum() == parade[1].sum() == parade[2].sum() == 10 * 512
    peak = parade.argmax(axis=1)
    want = [int(np.floor(scan_density_axis(d) * WAVEFORM_BINS)) for d in (0.3, 1.0, 2.5)]
    assert peak[:, 0].tolist() == want
    assert peak[:, -1].tolist() == want[::-1]
    # Denser is lower: D 2.5 sits below D 0.3.
    assert peak[0, -1] < peak[0, 0]
    assert density_parade(None) is None and density_parade(np.zeros((4, 4))) is None


def test_the_scan_parade_reads_the_crop_and_piles_overflow_at_the_axis_ends() -> None:
    log = np.full((20, 100, 3), -1.0, dtype=np.float32)
    log[:, :50, 0] = 0.2  # past sensor full scale: top edge bin
    log[:, 50:, 0] = -5.0  # denser than the axis reaches: bottom edge bin

    parade = density_parade(log, roi=(5, 15, 0, 100))

    assert parade[0].sum() == 10 * 100
    assert parade[0, -1, : WAVEFORM_COLS // 2].sum() == 10 * 50
    assert parade[0, 0, WAVEFORM_COLS // 2 :].sum() == 10 * 50
    assert parade[1, 0].sum() == parade[1, -1].sum() == 0


def test_a_val_reads_back_to_scan_density_through_its_bounds() -> None:
    from negpy.features.exposure.normalization import LogNegativeBounds

    bounds = LogNegativeBounds(floors=(-2.0, -2.2, -2.6), ceils=(-0.4, -0.7, -1.1))

    np.testing.assert_allclose(scan_density_of_val((0.0, 0.0, 0.0), bounds), (2.0, 2.2, 2.6))
    np.testing.assert_allclose(scan_density_of_val((1.0, 1.0, 1.0), bounds), (0.4, 0.7, 1.1))


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
def test_wgsl_scan_parade_matches_the_cpu() -> None:
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    grad = np.linspace(0.05, 0.9, 300, dtype=np.float32)
    img = np.stack([grad, grad[::-1], np.full_like(grad, 0.4)], axis=-1)
    img = np.ascontiguousarray(np.repeat(img[None], 200, axis=0))
    proc = ImageProcessor()
    cfg = WorkspaceConfig()

    _cpu_out, cpu_metrics = proc.run_pipeline(img, cfg, "dparade|cpu", render_size_ref=300, prefer_gpu=False, readback_metrics=True)
    _gpu_out, gpu_metrics = proc.run_pipeline(img, cfg, "dparade|gpu", render_size_ref=300, prefer_gpu=True, readback_metrics=True)

    _require_gpu(gpu_metrics)
    gpu = gpu_metrics["density_parade"]
    cpu = cpu_metrics["density_parade"]
    assert cpu.sum() == gpu.sum()
    for c in range(3):
        assert float(np.abs(cpu[c] / cpu[c].sum() - gpu[c] / gpu[c].sum()).sum()) < 0.02


def _row_lit(img, y: int) -> bool:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.styles.theme import THEME

    bg = QColor(THEME.bg_input).name()
    return any(QColor(img.pixel(x, y)).name() != bg for x in range(60, img.width() - 4))


def _row_across(img, y: int) -> bool:
    """A drawn line across a third of the row, over the empty chart and its separators."""
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.styles.theme import THEME

    lit = QColor(THEME.bg_input).lightness() + 20
    return sum(QColor(img.pixel(x, y)).lightness() > lit for x in range(img.width())) > img.width() // 3


def test_the_scan_parade_trades_zone_guides_for_density_guides_and_marks(qapp) -> None:
    from negpy.desktop.view.widgets.charts import WaveformWidget
    from negpy.features.exposure.analysis import encoded_of_zone

    widget = WaveformWidget(parade=True)
    widget.resize(300, 200)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[:, -1, 0] = 1.0  # one cell at the top edge, clear of every guide row
    widget.set_data(counts)
    zone_v = int(200 - encoded_of_zone(5) * 200)
    d2 = int(200 - float(scan_density_axis(2.0)) * 200)

    printed = widget.grab().toImage()
    assert _row_across(printed, zone_v) and not _row_across(printed, d2)

    widget.set_marker((0.5, 0.5, 0.5, 0.5))
    widget.set_scan_domain(True, np.array([[0.2, 0.7]] * 3))
    assert widget._marker is None
    scan = widget.grab().toImage()
    assert _row_across(scan, d2) and not _row_across(scan, zone_v)
    assert _row_across(scan, int(200 - 0.2 * 200)) and _row_across(scan, int(200 - 0.7 * 200))


def test_peek_negative_marks_each_channels_stretch_ends() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel
    from negpy.features.exposure.normalization import LogNegativeBounds

    panel = MagicMock()
    panel.controller.state = AppState(negative_peek=True)
    panel.scan_base_btn.isChecked.return_value = False
    panel._scan_marks_and_caption = lambda: RightPanel._scan_marks_and_caption(panel)
    panel._push_scan_domain = lambda negative: RightPanel._push_scan_domain(panel, negative)
    bounds = LogNegativeBounds(floors=(-2.0, -2.2, -2.6), ceils=(-0.4, -0.7, -1.1))

    RightPanel._update_histograms(panel, {"density_parade": np.ones((3, WAVEFORM_BINS, WAVEFORM_COLS)), "final_bounds": bounds})

    enabled, marks, caption = panel.parade_widget.set_scan_domain.call_args[0]
    assert panel.parade_widget.set_scan_domain.call_args[1]["base"] is None
    assert enabled and caption == "Scan density"
    np.testing.assert_allclose(marks, scan_density_axis(np.array([[2.0, 0.4], [2.2, 0.7], [2.6, 1.1]])))


def test_peek_negative_rings_the_scan_density_of_the_hovered_pixel() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel
    from negpy.features.exposure.densitometer import DensitometerReading
    from negpy.features.exposure.normalization import LogNegativeBounds

    panel = MagicMock()
    panel.controller.state = AppState(negative_peek=True)
    panel._scan_bounds = LogNegativeBounds(floors=(-2.0, -2.2, -2.6), ceils=(-0.4, -0.7, -1.1))

    RightPanel._on_pixel_readout(panel, (0.25, 0.5, (0.9, 0.9, 0.9)))
    panel.parade_widget.set_marker.assert_not_called()

    reading = DensitometerReading((0.1, 0.2, 0.3), 0.5, 1.0, 5.0, (1.0, 0.5, 0.0))
    RightPanel._on_densitometer(panel, reading)
    x, *tones = panel.parade_widget.set_marker.call_args[0][0]
    assert x == 0.25
    np.testing.assert_allclose(tones, scan_density_axis((0.4, 1.45, 2.6)))

    RightPanel._on_densitometer(panel, None)
    panel.parade_widget.set_marker.assert_called_with(None)


def test_the_parade_hover_draws_greens_level_across_all_panels(qapp) -> None:
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    widget.resize(300, 200)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[:, -1, 0] = 1.0
    widget.set_data(counts)
    widget.set_marker((0.5, 0.8, 0.35, 0.1))

    img = widget.grab().toImage()

    assert _row_across(img, int(200 - 0.35 * 200))


def _scale_panel(view: str, negative_peek: bool, saved: dict) -> MagicMock:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.view_btn.currentData.return_value = view
    panel.controller.state = AppState(negative_peek=negative_peek)
    panel.controller.session.repo.get_global_setting.side_effect = lambda key, default=None: saved.get(key, default)
    panel._scope_log = lambda: False
    panel._scan_parade_log = lambda: RightPanel._scan_parade_log(panel)
    return panel


def test_the_scan_parade_keeps_its_own_scale_log_at_first() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    scan = _scale_panel("parade", True, {"histogram_log_scale": False})
    RightPanel._sync_scale_btn(scan)
    scan.parade_widget.set_log_scale.assert_called_once_with(True)
    scan.scale_btn.setChecked.assert_called_once_with(True)

    printed = _scale_panel("parade", False, {"histogram_log_scale": False, "scan_parade_log": True})
    RightPanel._sync_scale_btn(printed)
    printed.parade_widget.set_log_scale.assert_called_once_with(False)
    printed.scale_btn.setChecked.assert_called_once_with(False)


def test_log_in_peek_negative_saves_only_the_scan_parades_scale() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = _scale_panel("parade", True, {})

    RightPanel._on_histogram_scale(panel, False)

    panel.controller.session.repo.save_global_setting.assert_called_once_with("scan_parade_log", False)
    panel.parade_widget.set_log_scale.assert_called_once_with(False)
    panel.curve_widget.set_log_scale.assert_not_called()


def test_the_parade_overlay_adds_the_channels_on_one_plot(qapp) -> None:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    widget.resize(WAVEFORM_COLS, WAVEFORM_BINS)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[:, 64, :128] = 4.0  # left half: R, G and B agree
    counts[0, 80, 128:] = 4.0  # right half: red above green and blue
    counts[1:, 64, 128:] = 4.0
    widget.set_overlay(True)
    widget.set_data(counts)

    img = widget.grab().toImage()
    agree = QColor(img.pixel(60, WAVEFORM_BINS - 1 - 64))
    red = QColor(img.pixel(200, WAVEFORM_BINS - 1 - 80))
    cyan = QColor(img.pixel(200, WAVEFORM_BINS - 1 - 64))

    assert min(agree.red(), agree.green(), agree.blue()) > 200
    assert red.red() > 200 and red.green() < 40 and red.blue() < 40
    assert cyan.red() < 40 and cyan.green() > 200 and cyan.blue() > 200


def test_the_overlay_choice_persists_and_shows_only_with_the_parade() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    RightPanel._on_parade_overlay(panel, True)
    panel.controller.session.repo.save_global_setting.assert_called_once_with("parade_overlay", True)
    panel._sync_parade_overlay.assert_called_once_with()

    panel.view_btn.currentData.return_value = "waveform"
    RightPanel._on_analysis_view(panel, 1)
    panel._sync_parade_tools.assert_called_once_with()


def test_the_parade_tools_show_with_the_parade_and_base_relative_only_on_the_scan() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    for view, peek, overlay, base in (("waveform", True, False, False), ("parade", False, True, False), ("parade", True, True, True)):
        panel = MagicMock()
        panel.controller.state = AppState(negative_peek=peek)
        panel.view_btn.currentData.return_value = view

        RightPanel._sync_parade_tools(panel)

        panel.parade_overlay_btn.setVisible.assert_called_once_with(overlay)
        panel.scan_base_btn.setVisible.assert_called_once_with(base)


def test_the_overlay_waits_for_base_relative_on_the_scan() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    for peek, base, on in ((True, False, False), (True, True, True), (False, False, True)):
        panel = MagicMock()
        panel.controller.state = AppState(negative_peek=peek)
        panel.parade_overlay_btn.isChecked.return_value = True
        panel.scan_base_btn.isChecked.return_value = base

        RightPanel._sync_parade_overlay(panel)

        panel.parade_overlay_btn.setEnabled.assert_called_once_with(on)


def test_base_relative_persists_and_renames_the_caption() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel
    from negpy.features.exposure.normalization import LogNegativeBounds

    panel = MagicMock()
    panel.controller.state = AppState(negative_peek=True)
    panel._scan_bounds = LogNegativeBounds(floors=(-2.0, -2.2, -2.6), ceils=(-0.4, -0.7, -1.1))
    panel.scan_base_btn.isChecked.return_value = True
    panel._scan_marks_and_caption = lambda: RightPanel._scan_marks_and_caption(panel)
    panel._push_scan_domain = lambda negative: RightPanel._push_scan_domain(panel, negative)
    panel._scan_base_density = None

    RightPanel._on_scan_base(panel, True)

    panel.controller.session.repo.save_global_setting.assert_called_once_with("scan_parade_base_relative", True)
    panel.parade_widget.set_base_relative.assert_called_once_with(True)
    assert panel.parade_widget.set_scan_domain.call_args[0][2] == "Scan density above thin end"

    for source, caption in (("picked", "Scan density above film base"), ("read", "Scan density above rebate base")):
        panel._scan_base_density = np.array([0.2, 0.6, 0.9])
        panel._scan_base_source = source
        RightPanel._on_scan_base(panel, True)
        args, kwargs = panel.parade_widget.set_scan_domain.call_args
        assert args[2] == caption
        np.testing.assert_allclose(kwargs["base"], scan_density_axis(np.array([0.2, 0.6, 0.9])))


def test_the_overlay_is_off_and_grayed_under_a_color_vision_palette() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    for vision, on in (("standard", True), ("protan_deutan", False), ("achromat", False)):
        panel = MagicMock()
        panel.controller.state = AppState(color_vision=vision)
        panel.parade_overlay_btn.isChecked.return_value = True
        panel.scan_base_btn.isChecked.return_value = False

        RightPanel._sync_parade_overlay(panel)

        panel.parade_overlay_btn.setEnabled.assert_called_once_with(on)
        panel.parade_widget.set_overlay.assert_called_once_with(on)


def test_the_split_parade_names_its_panels(qapp) -> None:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.widgets.charts import WaveformWidget

    def top_right_lit(widget) -> list[bool]:
        img = widget.grab().toImage()
        return [
            any(QColor(img.pixel(x, y)).lightness() > 60 for x in range(r.right() - 14, r.right() - 2) for y in range(3, 16))
            for r in widget._panel_rects(3)
        ]

    widget = WaveformWidget(parade=True)
    widget.resize(300, 120)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[:, 0, 0] = 1.0
    widget.set_data(counts)

    assert top_right_lit(widget) == [True, True, True]
    widget.set_overlay(True)
    assert not any(top_right_lit(widget)[1:])


def _scan_parade_peaks(prefer_gpu: bool, crop_preview_full: bool):
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    x = np.linspace(0.0, 1.0, 512, dtype=np.float32)[None, :, None]
    img = np.ascontiguousarray(np.broadcast_to(0.03 + 0.8 * x, (200, 512, 3)).astype(np.float32))
    cfg = WorkspaceConfig()
    cfg = replace(cfg, geometry=replace(cfg.geometry, crop_rect=(0.5, 0.0, 1.0, 1.0)))
    tag = f"cropfull|{prefer_gpu}|{crop_preview_full}"
    _out, metrics = ImageProcessor().run_pipeline(
        img, cfg, tag, render_size_ref=512, prefer_gpu=prefer_gpu, readback_metrics=True, crop_preview_full=crop_preview_full
    )
    if prefer_gpu:
        _require_gpu(metrics)
    return metrics["density_parade"][1].argmax(axis=0)


def test_the_scan_parade_spans_the_whole_frame_while_the_crop_tool_shows_it() -> None:
    cropped = _scan_parade_peaks(prefer_gpu=False, crop_preview_full=False)
    full = _scan_parade_peaks(prefer_gpu=False, crop_preview_full=True)

    # Column 0 is the frame's dense left edge only when the parade spans the whole frame.
    left_d = -np.log10(0.03)
    assert abs(float(scan_density_axis(left_d)) * WAVEFORM_BINS - full[0]) <= 1
    assert cropped[0] > full[0] + 20


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
def test_wgsl_scan_parade_spans_the_whole_frame_like_the_cpu() -> None:
    gpu = _scan_parade_peaks(prefer_gpu=True, crop_preview_full=True)
    cpu = _scan_parade_peaks(prefer_gpu=False, crop_preview_full=True)
    assert np.abs(gpu.astype(int) - cpu.astype(int)).max() <= 1


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
def test_wgsl_scan_parade_matches_the_cpu_on_a_slide_with_a_camera_matrix() -> None:
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.features.process.models import ProcessMode
    from negpy.services.rendering.image_processor import ImageProcessor

    cam_xyz = [[0.5271, -0.0712, -0.0347], [-0.6153, 1.3653, 0.2763], [-0.1601, 0.2366, 0.7242]]
    grad = np.linspace(0.02, 0.9, 300, dtype=np.float32)
    img = np.stack([grad, grad[::-1] * 0.8, np.full_like(grad, 0.3)], axis=-1)
    img = np.ascontiguousarray(np.repeat(img[None], 200, axis=0))
    cfg = WorkspaceConfig()
    cfg = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, white_point_offset=0.05))
    proc = ImageProcessor()

    kw = dict(render_size_ref=300, readback_metrics=True, cam_xyz=cam_xyz, camera_wb=[1856.0, 1024.0, 1744.0])
    _c, cpu_metrics = proc.run_pipeline(img, cfg, "slide|cpu", prefer_gpu=False, **kw)
    _g, gpu_metrics = proc.run_pipeline(img, cfg, "slide|gpu", prefer_gpu=True, **kw)

    _require_gpu(gpu_metrics)
    gpu = gpu_metrics["density_parade"]
    cpu = cpu_metrics["density_parade"]
    assert cpu.sum() == gpu.sum()
    for c in range(3):
        assert float(np.abs(cpu[c] / cpu[c].sum() - gpu[c] / gpu[c].sum()).sum()) < 0.02


def test_an_export_render_skips_the_scan_parade() -> None:
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    img = np.ascontiguousarray(np.full((40, 60, 3), 0.4, dtype=np.float32))
    proc = ImageProcessor()

    _o, preview = proc.run_pipeline(img, WorkspaceConfig(), "skip|pv", render_size_ref=60, prefer_gpu=False)
    _o, export = proc.run_pipeline(img, WorkspaceConfig(), "skip|ex", render_size_ref=60, prefer_gpu=False, cache_stages=False)

    assert preview.get("density_parade") is not None
    assert export.get("density_parade") is None


def test_the_panel_builds_with_a_saved_view_past_the_curve(qapp) -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    controller = MagicMock()
    controller.state = AppState()
    saved = {"analysis_view": "parade"}
    controller.session.repo.get_global_setting.side_effect = lambda key, default=None: saved.get(key, default)
    try:
        panel = RightPanel(controller)
    except AttributeError as exc:  # pragma: no cover - the regression
        raise AssertionError(f"restoring the saved view reached an unbuilt widget: {exc}")
    assert panel.view_btn.currentData() == "parade"
    assert panel.chart_stack.currentWidget() is panel.parade_widget


def test_the_scan_parade_reads_out_each_channels_range() -> None:
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[:, 0, 0] = 1.0
    widget.set_data(counts)
    assert widget._scan_ranges() is None

    ends = np.array([[2.35, 0.82], [2.40, 0.48], [2.88, 0.69]])
    widget.set_scan_domain(True, scan_density_axis(ends))

    np.testing.assert_allclose(widget._scan_ranges(), (1.53, 1.92, 2.19), atol=1e-9)


def test_full_scale_columns_count_clipped_scan_pixels_per_channel() -> None:
    img = np.full((10, 512, 3), 0.5, dtype=np.float32)
    img[:, 256:, 0] = 1.0  # red clips on the right half
    img[:4, :2, 2] = 0.995

    cols = scan_full_scale_columns(img)

    assert cols.shape == (3, WAVEFORM_COLS)
    assert np.flatnonzero(cols[0]).tolist() == list(range(128, 256))
    assert cols[0, 200] == 10 * 2
    assert not cols[1].any()
    assert np.flatnonzero(cols[2]).tolist() == [0] and cols[2, 0] == 4 * 2
    assert scan_full_scale_columns(img, roi=(0, 10, 0, 256))[0].sum() == 0
    assert scan_full_scale_columns(None) is None


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
def test_wgsl_full_scale_columns_match_the_cpu() -> None:
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    grad = np.linspace(0.05, 0.9, 300, dtype=np.float32)
    img = np.stack([grad, grad[::-1], np.full_like(grad, 0.4)], axis=-1)
    img = np.ascontiguousarray(np.repeat(img[None], 200, axis=0))
    img[:, 200:, 0] = 1.0
    proc = ImageProcessor()
    cfg = WorkspaceConfig()

    _c, cpu_metrics = proc.run_pipeline(img, cfg, "fs|cpu", render_size_ref=300, prefer_gpu=False, readback_metrics=True)
    _g, gpu_metrics = proc.run_pipeline(img, cfg, "fs|gpu", render_size_ref=300, prefer_gpu=True, readback_metrics=True)

    _require_gpu(gpu_metrics)
    gpu = gpu_metrics["scan_full_scale"]
    cpu = cpu_metrics["scan_full_scale"]
    assert cpu[0].any()
    assert np.array_equal(cpu > 0, gpu > 0)


def _top_tick(img, x: int) -> bool:
    from PyQt6.QtGui import QColor

    return QColor(img.pixel(x, 2)).name() == QColor(palette_for("standard").pair[1]).name()


def test_the_scan_parade_ticks_full_scale_from_the_scan_before_the_unmix(qapp) -> None:
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    widget.resize(3 * 100, WAVEFORM_BINS)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[:, 100, :] = 4.0  # the unmix moved the clipped red off the top bin
    widget.set_data(counts)
    full = np.zeros((3, WAVEFORM_COLS))
    full[0, :32] = 3.0

    widget.set_scan_domain(True, np.array([[0.2, 0.7]] * 3), full_scale=full)
    img = widget.grab().toImage()

    assert _top_tick(img, 5) and not _top_tick(img, 60) and not _top_tick(img, 106)


def test_the_scan_parade_overlays_only_when_base_relative(qapp) -> None:
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    widget.set_overlay(True)
    widget.set_scan_domain(True, np.array([[0.2, 0.7], [0.1, 0.6], [0.0, 0.5]]))
    assert widget._split()

    widget.set_base_relative(True)
    assert not widget._split()
    widget.set_scan_domain(False)
    assert not widget._split()


def test_base_relative_moves_each_channels_thin_mark_to_the_top(qapp) -> None:
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[0, 96, 0] = 1.0  # at red's thin mark
    counts[2, 64, 0] = 1.0  # at blue's thin mark
    counts[2, 120, 0] = 2.0  # thinner than blue's base: pinned at the top
    widget.set_data(counts)
    widget.set_scan_domain(True, np.array([[0.25, 96 / WAVEFORM_BINS], [0.1, 0.6], [0.0, 64 / WAVEFORM_BINS]]))
    widget.set_base_relative(True)

    shown = widget._shown_panels()

    assert shown[0, -1, 0] == 1.0
    assert shown[2, -1, 0] == 3.0
    np.testing.assert_allclose(widget._shown_marks(widget._scan_marks)[:, 1], 1.0, atol=1.0 / WAVEFORM_BINS)


def test_base_relative_puts_the_film_base_at_the_top_before_the_thin_mark(qapp) -> None:
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[1, 100, 0] = 1.0  # green's base
    counts[1, 90, 0] = 1.0  # green's thinnest picture tone, below its base
    widget.set_data(counts)
    marks = np.array([[0.2, 0.9], [0.2, 90 / WAVEFORM_BINS], [0.2, 0.6]])
    base = np.array([0.95, 100 / WAVEFORM_BINS, 0.7])
    widget.set_base_relative(True)
    widget.set_scan_domain(True, marks, base=base)

    shown = widget._shown_panels()

    assert shown[1, -1, 0] == 1.0
    assert shown[1, WAVEFORM_BINS - 1 - 10, 0] == 1.0
    np.testing.assert_allclose(widget._shown_marks(base), 1.0, atol=1.0 / WAVEFORM_BINS)


def test_the_scan_parade_draws_the_film_base_line(qapp) -> None:
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    widget.resize(300, 200)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[:, -1, 0] = 1.0
    widget.set_data(counts)
    widget.set_scan_domain(True, np.array([[0.2, 0.7]] * 3))
    y = int(200 - 0.85 * 200)

    def dots(img) -> int:
        from PyQt6.QtGui import QColor

        from negpy.desktop.view.styles.theme import THEME

        lit = QColor(THEME.bg_input).lightness() + 20
        return sum(QColor(img.pixel(x, y)).lightness() > lit for x in range(img.width()))

    assert dots(widget.grab().toImage()) < 10
    widget.set_scan_domain(True, np.array([[0.2, 0.7]] * 3), base=np.full(3, 0.85))
    assert dots(widget.grab().toImage()) > 60  # a dotted line across all three panels


def test_both_engines_report_the_picked_film_base_as_scan_density() -> None:
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    grad = np.linspace(0.05, 0.6, 300, dtype=np.float32)
    img = np.ascontiguousarray(np.repeat(np.stack([grad, grad * 0.8, grad * 0.5], axis=-1)[None], 200, axis=0))
    base_log = (-0.1, -0.3, -0.5)
    proc = ImageProcessor()
    kw = dict(render_size_ref=300, readback_metrics=True, film_base=base_log)

    _c, cpu = proc.run_pipeline(img, WorkspaceConfig(), "fb|cpu", prefer_gpu=False, **kw)
    np.testing.assert_allclose(cpu["film_base_density"], (0.1, 0.3, 0.5), atol=1e-6)
    if GPUDevice.get().is_available:
        _g, gpu = proc.run_pipeline(img, WorkspaceConfig(), "fb|gpu", prefer_gpu=True, **kw)
        _require_gpu(gpu)
        np.testing.assert_allclose(gpu["film_base_density"], cpu["film_base_density"], atol=1e-6)


def _unmix_cfg():
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig

    cfg = WorkspaceConfig()
    # A strong positive cross term pulls a clipped red off full scale wherever green is dense.
    matrix = (1.0, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return replace(cfg, process=replace(cfg.process, crosstalk_strength=1.0, crosstalk_matrix=matrix))


def test_full_scale_ticks_read_the_scan_before_the_unmix() -> None:
    from negpy.services.rendering.image_processor import ImageProcessor

    img = np.full((200, 300, 3), (0.5, 0.03, 0.4), dtype=np.float32)
    img[:, 150:, 0] = 1.0
    proc = ImageProcessor()
    kw = dict(render_size_ref=300, readback_metrics=True)

    _c, cpu = proc.run_pipeline(img, _unmix_cfg(), "fsu|cpu", prefer_gpu=False, **kw)

    right = slice(WAVEFORM_COLS // 2 + 1, WAVEFORM_COLS)
    assert cpu["scan_full_scale"][0, right].all()
    assert not cpu["density_parade"][0, -1, right].any()  # the unmixed red sits below the top bin
    if GPUDevice.get().is_available:
        _g, gpu = proc.run_pipeline(img, _unmix_cfg(), "fsu|gpu", prefer_gpu=True, **kw)
        _require_gpu(gpu)
        assert np.array_equal(gpu["scan_full_scale"] > 0, cpu["scan_full_scale"] > 0)
        for c in range(3):
            g, k = gpu["density_parade"][c], cpu["density_parade"][c]
            assert float(np.abs(g / g.sum() - k / k.sum()).sum()) < 0.02


def test_the_film_base_reads_through_the_unmix_on_both_engines() -> None:
    from negpy.features.exposure.normalization import effective_crosstalk_matrix
    from negpy.services.rendering.image_processor import ImageProcessor

    grad = np.linspace(0.05, 0.6, 300, dtype=np.float32)
    img = np.ascontiguousarray(np.repeat(np.stack([grad, grad * 0.8, grad * 0.5], axis=-1)[None], 200, axis=0))
    base_log = (-0.1, -0.3, -0.5)
    cfg = _unmix_cfg()
    matrix = effective_crosstalk_matrix(cfg.process, cfg.process.process_mode)
    want = -(matrix @ np.array(base_log))
    proc = ImageProcessor()
    kw = dict(render_size_ref=300, readback_metrics=True, film_base=base_log)

    _c, cpu = proc.run_pipeline(img, cfg, "fbu|cpu", prefer_gpu=False, **kw)
    np.testing.assert_allclose(cpu["film_base_density"], want, atol=1e-5)
    if GPUDevice.get().is_available:
        _g, gpu = proc.run_pipeline(img, cfg, "fbu|gpu", prefer_gpu=True, **kw)
        _require_gpu(gpu)
        np.testing.assert_allclose(gpu["film_base_density"], want, atol=1e-5)


def test_a_slide_reports_no_film_base_after_a_negative() -> None:
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.features.process.models import ProcessMode
    from negpy.services.rendering.image_processor import ImageProcessor

    img = np.ascontiguousarray(np.full((60, 90, 3), 0.4, dtype=np.float32))
    proc = ImageProcessor()
    kw = dict(render_size_ref=90, readback_metrics=True, film_base=(-0.1, -0.3, -0.5), prefer_gpu=False)
    slide = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, process_mode=ProcessMode.E6))

    _o, neg = proc.run_pipeline(img, WorkspaceConfig(), "stale|neg", **kw)
    _o, pos = proc.run_pipeline(img, slide, "stale|pos", **kw)

    assert neg["film_base_density"] is not None
    assert "film_base_density" in pos and pos["film_base_density"] is None


def test_the_film_base_survives_a_drag_on_the_cpu() -> None:
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    img = np.ascontiguousarray(np.full((60, 90, 3), 0.4, dtype=np.float32))
    proc = ImageProcessor()
    kw = dict(render_size_ref=90, film_base=(-0.1, -0.3, -0.5), prefer_gpu=False)

    proc.run_pipeline(img, WorkspaceConfig(), "drag|x", readback_metrics=False, **kw)
    _o, release = proc.run_pipeline(img, WorkspaceConfig(), "drag|x", readback_metrics=True, **kw)

    np.testing.assert_allclose(release["film_base_density"], (0.1, 0.3, 0.5), atol=1e-6)


def test_base_relative_keeps_a_base_at_full_scale_in_place(qapp) -> None:
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    widget.set_data(np.ones((3, WAVEFORM_BINS, WAVEFORM_COLS)))
    widget.set_base_relative(True)
    widget.set_scan_domain(True, np.array([[0.2, 0.9]] * 3), base=np.array([1.0, 1.2, 0.5]))

    shifts = widget._base_shifts()

    assert shifts[0] == 0 and shifts[1] == 0 and shifts[2] > 0
