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
