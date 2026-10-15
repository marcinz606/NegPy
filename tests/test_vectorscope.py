"""The Analysis vectorscope: polar a*b* binning on both engines, the display lookup, and the
chroma scale that is kept apart per film mode."""

import unittest
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.session import AppState
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.analysis import (
    SCOPE_CHROMA_BINS,
    SCOPE_HUE_BINS,
    scope_chroma,
    scope_radius,
    vectorscope_density,
    vectorscope_histogram,
)
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.gpu.device import GPUDevice


def test_a_gray_frame_lands_in_the_first_chroma_bin() -> None:
    counts = vectorscope_histogram(np.full((20, 30, 3), 0.5, dtype=np.float32))

    assert counts.shape == (SCOPE_HUE_BINS, SCOPE_CHROMA_BINS)
    assert counts.sum() == 600
    assert counts[:, 0].sum() == 600


def test_a_saturated_red_sits_far_out_at_the_red_hue() -> None:
    from negpy.kernel.image.logic import rgb_to_lab_working

    counts = vectorscope_histogram(np.tile(np.array([[[0.9, 0.1, 0.1]]], dtype=np.float32), (4, 4, 1)))
    hue_bin, chroma_bin = np.unravel_index(int(counts.argmax()), counts.shape)

    lab = rgb_to_lab_working(np.array([[0.9, 0.1, 0.1]], dtype=np.float32) ** 2.19921875)[0]
    expected_hue = int((np.arctan2(lab[2], lab[1]) + np.pi) / (2 * np.pi) * SCOPE_HUE_BINS)
    assert abs(int(hue_bin) - expected_hue) <= 1
    assert chroma_bin > SCOPE_CHROMA_BINS // 2


def test_the_chroma_scale_round_trips() -> None:
    for log in (False, True):
        c = np.array([0.0, 2.0, 5.0, 20.0, 60.0])
        assert np.allclose(scope_chroma(scope_radius(c, log), log), c)


def test_the_display_keeps_every_pixel_and_puts_neutral_at_the_center() -> None:
    rng = np.random.default_rng(1)
    counts = vectorscope_histogram(rng.random((80, 80, 3)).astype(np.float32))
    gray = vectorscope_histogram(np.full((10, 10, 3), 0.4, dtype=np.float32))

    for log in (False, True):
        assert round(vectorscope_density(counts, 160, log).sum()) == 6400
        density = vectorscope_density(gray, 160, log)
        row, col = np.unravel_index(int(density.argmax()), density.shape)
        assert {int(row), int(col)} <= {79, 80}


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestVectorscopeParity(unittest.TestCase):
    def test_wgsl_bins_match_the_cpu(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        rng = np.random.default_rng(5)
        img = rng.random((200, 300, 3)).astype(np.float32) * 0.85 + 0.05
        proc = ImageProcessor()
        cfg = WorkspaceConfig()

        cpu_out, _ = proc.run_pipeline(img, cfg, "scope|cpu", render_size_ref=300, prefer_gpu=False, readback_metrics=True)
        _gpu_out, gpu_metrics = proc.run_pipeline(img, cfg, "scope|gpu", render_size_ref=300, prefer_gpu=True, readback_metrics=True)

        gpu_scope = gpu_metrics.get("vectorscope")
        if gpu_scope is None:
            self.skipTest("render fell back to the CPU engine")
        cpu_scope = vectorscope_histogram(cpu_out)
        self.assertEqual(cpu_scope.sum(), gpu_scope.sum())
        a = cpu_scope / cpu_scope.sum()
        b = gpu_scope / gpu_scope.sum()
        self.assertLess(float(np.abs(a - b).sum()), 0.01)


def test_vectorscope_widget_paints_both_scales(qapp) -> None:
    from negpy.desktop.view.widgets.charts import VectorscopeWidget

    widget = VectorscopeWidget()
    widget.resize(200, 160)
    rng = np.random.default_rng(2)
    widget.set_data(vectorscope_histogram(rng.random((40, 40, 3)).astype(np.float32)))
    widget.grab()
    assert widget._image is not None

    widget.set_log_scale(True)
    assert widget._image is None
    widget.grab()
    widget.set_data(None)
    widget.grab()


def _panel(view: str, mode: ProcessMode, saved: dict) -> MagicMock:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.view_btn.currentData.return_value = view
    panel.controller.state = AppState(config=replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, process_mode=mode)))
    panel.controller.session.repo.get_global_setting.side_effect = lambda key, default=None: saved.get(key, default)
    panel._scope_scale_key = lambda: RightPanel._scope_scale_key(panel)
    panel._scope_log = lambda: RightPanel._scope_log(panel)
    return panel


def test_the_vectorscope_scale_defaults_to_log_for_bw_and_lin_for_color() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    bw = _panel("vectorscope", ProcessMode.BW, {"histogram_log_scale": False})
    RightPanel._sync_scale_btn(bw)
    bw.scale_btn.setCurrentIndex.assert_called_once_with(1)
    bw.vectorscope_widget.set_log_scale.assert_called_once_with(True)

    color = _panel("vectorscope", ProcessMode.C41, {"histogram_log_scale": True})
    RightPanel._sync_scale_btn(color)
    color.scale_btn.setCurrentIndex.assert_called_once_with(0)

    curve = _panel("curve", ProcessMode.BW, {"histogram_log_scale": False})
    RightPanel._sync_scale_btn(curve)
    curve.scale_btn.setCurrentIndex.assert_called_once_with(0)


def test_lin_log_on_the_vectorscope_saves_the_film_modes_own_scale() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = _panel("vectorscope", ProcessMode.BW, {})

    RightPanel._on_histogram_scale(panel, 0)

    panel.controller.session.repo.save_global_setting.assert_called_once_with("vectorscope_log_bw", False)
    panel.vectorscope_widget.set_log_scale.assert_called_once_with(False)
    panel.curve_widget.set_log_scale.assert_not_called()
