"""The Analysis waveform: print luma per image column, binned the same way on both engines,
and the chart view that shows it."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.session import AppState
from negpy.features.exposure.analysis import WAVEFORM_BINS, WAVEFORM_COLOR_SCALE, WAVEFORM_COLS, waveform, waveform_color
from negpy.infrastructure.gpu.device import GPUDevice


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

    from negpy.desktop.view.styles.theme import THEME
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget()
    widget.resize(256, 64)
    counts = np.zeros((WAVEFORM_BINS, WAVEFORM_COLS))
    counts[0, :64] = 5.0  # left quarter at paper black
    counts[64, 64:] = 5.0
    widget.set_data(counts)

    img = widget.grab().toImage()

    assert QColor(img.pixel(20, 61)).name() == QColor(THEME.clip_warning).name()
    assert QColor(img.pixel(200, 61)).name() != QColor(THEME.clip_warning).name()

    widget.set_log_scale(True)
    assert widget._image is None
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
