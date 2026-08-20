"""Signal wiring of the right panel's analysis refresh.

_paint_negative_peek emits image_updated only, never metrics_available, so the
image_updated path must refresh the histograms itself or entering Peek Negative
leaves the chart in print mode.
"""

from unittest.mock import MagicMock

from negpy.desktop.view.sidebar.right_panel import RightPanel


def _panel_stub(last_metrics: dict) -> MagicMock:
    panel = MagicMock()
    panel.controller.session.state.last_metrics = last_metrics
    panel.controller.state.flat_peek = False
    panel.controller.state.negative_peek = True
    panel._clip_fracs = (None, None)
    return panel


def test_update_analysis_refreshes_histograms() -> None:
    metrics = {"interactive": False, "histogram_density": [1.0]}
    panel = _panel_stub(metrics)

    RightPanel._update_analysis(panel)

    panel._update_histograms.assert_called_once_with(metrics)


def test_update_analysis_skips_mid_gesture_frames() -> None:
    panel = _panel_stub({"interactive": True})

    RightPanel._update_analysis(panel)

    panel._update_histograms.assert_not_called()
