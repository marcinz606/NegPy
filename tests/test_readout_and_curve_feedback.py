import unittest
from dataclasses import replace

import numpy as np

from negpy.desktop.view.widgets.charts import PhotometricCurveWidget
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.exposure.models import ExposureConfig


def test_histogram_marker_set_clear_and_paint() -> None:
    from negpy.features.exposure.analysis import output_histogram

    widget = PhotometricCurveWidget()
    widget.resize(256, 100)
    widget.update_curve(ExposureConfig())
    widget.set_output_histogram(output_histogram(np.random.rand(64, 64, 3).astype(np.float32)))

    widget.set_marker((10, 128, 250))
    assert widget._marker == (10, 128, 250)
    widget.grab()  # exercises paintEvent with the marker + output histogram

    widget.set_marker(None)
    assert widget._marker is None
    widget.grab()


def test_curve_ghost_frozen_across_updates_and_cleared() -> None:
    widget = PhotometricCurveWidget()
    widget.resize(200, 120)
    config = ExposureConfig()
    widget.update_curve(config)
    original_pts = list(widget._curve_pts)

    widget.set_active_param("toe")
    assert widget._ghost_pts == original_pts
    assert widget._active_param == "toe"

    widget.update_curve(replace(config, toe=0.8, grade=80.0))
    assert widget._curve_pts != original_pts
    assert widget._ghost_pts == original_pts  # ghost stays frozen at drag start
    widget.grab()  # exercises ghost + zone emphasis painting

    widget.set_active_param("")
    assert widget._active_param is None
    assert widget._ghost_pts == []
    assert widget._ghost_pivot is None


def test_slider_emits_drag_signals() -> None:
    slider = CompactSlider("Test", 0.0, 1.0, 0.5)
    events: list[str] = []
    slider.dragStarted.connect(lambda: events.append("start"))
    slider.dragEnded.connect(lambda: events.append("end"))

    slider.slider.sliderPressed.emit()
    slider.slider.sliderReleased.emit()
    assert events == ["start", "end"]


class TestGPUHistogramEncodedDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from negpy.infrastructure.gpu.device import GPUDevice
        from negpy.services.rendering.gpu_engine import GPUEngine

        cls.gpu = GPUDevice.get()
        cls.engine = GPUEngine() if cls.gpu.is_available else None

    def setUp(self):
        if self.engine is None:
            self.skipTest("GPU not available")

    def test_histogram_bins_match_encoded_output(self):
        """The metrics shader must bin the same display-encoded values as the
        readback buffer (base_positive) — not the scene-linear toning output."""
        from negpy.domain.models import WorkspaceConfig

        ramp = np.tile(np.linspace(0.05, 0.95, 256, dtype=np.float32)[None, :, None], (64, 1, 3))
        res, metrics = self.engine.process(ramp, WorkspaceConfig())
        hist = metrics["histogram_raw"]

        bins = np.arange(256, dtype=np.float64)

        def centroid(counts: np.ndarray) -> float:
            counts = counts.astype(np.float64)
            return float((bins * counts).sum() / max(1.0, counts.sum()))

        for c in range(3):
            cpu_counts, _ = np.histogram(res[..., c], bins=256, range=(0, 1))
            self.assertLess(
                abs(centroid(cpu_counts) - centroid(hist[c])),
                2.0,
                msg=f"channel {c}: GPU histogram domain diverges from encoded output",
            )
