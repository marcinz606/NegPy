"""Peek Negative applies no curve, so the Analysis chart's print histogram, curve and zone
strip would describe a print that was never made; only the pre-curve density histogram
still applies."""

from unittest.mock import MagicMock

from negpy.desktop.view.sidebar.right_panel import RightPanel
from negpy.desktop.view.widgets.charts import PhotometricCurveWidget


def _panel_stub(negative_peek: bool) -> MagicMock:
    panel = MagicMock()
    panel.controller.state.negative_peek = negative_peek
    panel.controller.state.flat_peek = False
    panel.curve_widget = MagicMock()
    panel.zone_strip = MagicMock()
    return panel


def test_a_negative_peek_shows_only_the_density_histogram() -> None:
    panel = _panel_stub(negative_peek=True)
    metrics = {"histogram_density": [1.0, 2.0], "histogram_raw": object()}

    RightPanel._update_histograms(panel, metrics)

    panel.curve_widget.set_density_histogram.assert_called_once_with(metrics["histogram_density"])
    panel.curve_widget.set_output_histogram.assert_called_once_with(None)
    panel.curve_widget.set_show_print.assert_called_once_with(False)
    panel.zone_strip.setVisible.assert_called_once_with(False)
    assert panel._clip_fracs == (None, None)


def test_a_normal_render_still_shows_the_print(monkeypatch) -> None:
    import numpy as np

    panel = _panel_stub(negative_peek=False)
    metrics = {"histogram_density": None, "histogram_raw": np.zeros((4, 4, 3), dtype=np.float32)}

    RightPanel._update_histograms(panel, metrics)

    panel.curve_widget.set_show_print.assert_called_once_with(True)
    panel.curve_widget.set_output_histogram.assert_called_once()
    assert panel.curve_widget.set_output_histogram.call_args[0][0] is not None


def test_show_print_false_leaves_the_density_histogram_paintable() -> None:
    """set_show_print only gates the print-derived traces; the widget must not blank
    entirely just because there is no curve to draw."""
    import numpy as np

    from negpy.features.exposure.analysis import DENSITY_HIST_BINS

    widget = PhotometricCurveWidget()
    widget.resize(200, 120)
    widget._curve_pts = [(0.0, 0.0), (1.0, 1.0)]
    bins = np.zeros((4, DENSITY_HIST_BINS))
    bins[:, 10] = 1.0
    widget.set_density_histogram(bins)
    widget.set_show_print(False)
    widget.set_channel_density(True)

    assert widget._show_print is False
    # paintEvent must not raise with print traces suppressed but density data present.
    widget.grab()
