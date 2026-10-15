"""The hovered pixel on the waveform and the vectorscope: the controller's readout carries its
position, the panel turns it into a column, a tone and an a*b*, and each scope rings it."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from negpy.features.exposure.analysis import WAVEFORM_BINS, WAVEFORM_COLS, scope_point, vectorscope_histogram


def test_scope_point_reads_tone_and_color() -> None:
    luma, a, b = scope_point((0.5, 0.5, 0.5))
    assert luma == pytest.approx(0.5, abs=1e-3)
    assert abs(a) < 0.5 and abs(b) < 0.5

    _luma, a, _b = scope_point((0.9, 0.2, 0.2))
    assert a > 30


def test_the_controller_readout_carries_the_content_position() -> None:
    from negpy.desktop.controller import AppController

    controller = MagicMock()
    controller._pending_cursor_nx, controller._pending_cursor_ny = 0.25, 0.75
    controller.canvas.get_pixel_rgb.return_value = (0.2, 0.4, 0.6)

    AppController._emit_pixel_readout(controller)

    controller.pixel_readout.emit.assert_called_once_with((0.25, 0.75, (0.2, 0.4, 0.6)))

    AppController.on_cursor_left(controller)
    controller.pixel_readout.emit.assert_called_with(None)


def test_the_panel_marks_both_scopes_and_clears_them() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    RightPanel._on_pixel_readout(panel, (1.3, 0.5, (0.5, 0.5, 0.5)))

    column, luma = panel.waveform_widget.set_marker.call_args[0][0]
    assert column == 1.0
    assert luma == pytest.approx(0.5, abs=1e-3)
    a, b = panel.vectorscope_widget.set_marker.call_args[0][0]
    assert abs(a) < 0.5 and abs(b) < 0.5

    RightPanel._on_pixel_readout(panel, None)
    panel.waveform_widget.set_marker.assert_called_with(None)
    panel.vectorscope_widget.set_marker.assert_called_with(None)


def _bright(img, x: float, y: float) -> bool:
    """A ring pixel within a pixel of (x, y): antialiasing puts the stroke between pixel centers."""
    from PyQt6.QtGui import QColor

    return any(QColor(img.pixel(int(x) + dx, int(y))).lightness() > 200 for dx in (-1, 0, 1))


def test_the_waveform_rings_the_hovered_column_and_tone(qapp) -> None:
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget()
    widget.resize(200, 100)
    counts = np.zeros((WAVEFORM_BINS, WAVEFORM_COLS))
    counts[10, :] = 1.0  # a faint trace far from the ring
    widget.set_data(counts)
    widget.set_marker((0.5, 0.5))

    img = widget.grab().toImage()

    assert _bright(img, 100 + 4, 50)
    assert not _bright(img, 160 + 4, 50)


def test_the_vectorscope_rings_the_hovered_color_on_the_active_scale(qapp) -> None:
    from negpy.desktop.view.widgets.charts import VectorscopeWidget
    from negpy.features.exposure.analysis import scope_radius

    widget = VectorscopeWidget()
    widget.resize(200, 200)
    widget.set_data(vectorscope_histogram(np.full((8, 8, 3), 0.5, dtype=np.float32)))
    rim = (200 - widget._LABEL_ROOM) / 2.0

    for log in (False, True):
        widget.set_log_scale(log)
        widget.set_marker((20.0, 0.0))  # +a: straight right of the center
        img = widget.grab().toImage()
        x = 100 + float(scope_radius(20.0, log)) * rim
        assert _bright(img, x + 4, 100), f"log={log}"
