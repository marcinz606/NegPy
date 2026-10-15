"""The per-channel density histogram is a Peek Negative feature: everywhere else the print's
own output histogram already carries color information, so the merged luma trace stays."""

from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.view.sidebar.right_panel import RightPanel
from negpy.features.exposure.analysis import DENSITY_HIST_BINS


def _panel_stub(negative_peek: bool) -> MagicMock:
    panel = MagicMock()
    panel.controller.state.negative_peek = negative_peek
    panel.controller.state.flat_peek = False
    panel.curve_widget = MagicMock()
    panel.zone_strip = MagicMock()
    return panel


def test_channel_density_follows_negative_peek() -> None:
    panel = _panel_stub(negative_peek=True)
    metrics = {
        "histogram_density": np.zeros((4, DENSITY_HIST_BINS)),
        "histogram_raw": np.full((4, 4, 3), 0.5, dtype=np.float32),
    }

    RightPanel._update_histograms(panel, metrics)

    panel.curve_widget.set_channel_density.assert_called_once_with(True)


def test_channel_density_off_for_a_normal_render() -> None:
    panel = _panel_stub(negative_peek=False)
    metrics = {
        "histogram_density": np.zeros((4, DENSITY_HIST_BINS)),
        "histogram_raw": np.full((4, 4, 3), 0.5, dtype=np.float32),
    }

    RightPanel._update_histograms(panel, metrics)

    panel.curve_widget.set_channel_density.assert_called_once_with(False)


def test_the_metering_line_reads_the_analysis_clip_fractions() -> None:
    panel = _panel_stub(negative_peek=False)
    bins = np.zeros((4, 256))
    bins[:3, 1:255] = 10.0
    bins[0, 255] = 20.0  # 2% of red blown
    metrics = {"histogram_density": np.zeros((4, DENSITY_HIST_BINS)), "histogram_raw": bins}

    RightPanel._update_histograms(panel, metrics)

    low, high = panel.controls_panel.process_sidebar.set_clipping.call_args.args
    assert low == 0.0 and abs(high - 20 / (254 * 10 + 20)) < 1e-9


def test_a_negative_peek_blanks_the_metering_line_too() -> None:
    panel = _panel_stub(negative_peek=True)
    metrics = {"histogram_density": np.zeros((4, DENSITY_HIST_BINS)), "histogram_raw": np.zeros((4, 256))}

    RightPanel._update_histograms(panel, metrics)

    panel.controls_panel.process_sidebar.set_clipping.assert_called_once_with(None, None)
