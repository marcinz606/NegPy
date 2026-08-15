"""Grain focuser: the acutance figure and the sample window.

The acutance measure is the part that can be quietly wrong. A plain standard deviation reads
high on a smooth gradient — the opposite of sharp — so the σ trap gets its own test.
"""

from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QImage, QPainter, QPixmap

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import CanvasOverlay, loupe_src_rect
from negpy.features.exposure.analysis import loupe_acutance


def _rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray.astype(np.float32)[:, :, None], 3, axis=2)


def _hard_edge(n: int = 64) -> np.ndarray:
    g = np.zeros((n, n), dtype=np.float32)
    g[:, n // 2 :] = 1.0
    return _rgb(g)


def _blurred_edge(n: int = 64) -> np.ndarray:
    return _rgb(cv2.GaussianBlur(_hard_edge(n)[..., 0], (0, 0), 3.0))


def _ramp(n: int = 64) -> np.ndarray:
    return _rgb(np.tile(np.linspace(0.0, 1.0, n, dtype=np.float32), (n, 1)))


def test_sharper_patches_score_higher():
    flat = loupe_acutance(_rgb(np.full((64, 64), 0.5, dtype=np.float32)))
    blurred = loupe_acutance(_blurred_edge())
    hard = loupe_acutance(_hard_edge())

    assert flat == 0.0
    assert hard > blurred > flat


def test_a_smooth_ramp_does_not_read_as_sharp():
    """The σ trap: a ramp and a hard edge can share a standard deviation, but only one is
    sharp. This fails the moment loupe_acutance is 'simplified' back to .std()."""
    ramp, hard = _ramp(), _hard_edge()
    # Same spread, by construction of a 0..1 ramp against a 0/1 step.
    assert abs(float(ramp[..., 0].std()) - 0.29) < 0.02
    assert abs(float(hard[..., 0].std()) - 0.50) < 0.02

    assert loupe_acutance(ramp) < loupe_acutance(hard) / 10.0


def test_degenerate_patches_score_zero():
    assert loupe_acutance(np.zeros((2, 2, 3), dtype=np.float32)) == 0.0  # too small
    assert loupe_acutance(np.zeros((8, 8), dtype=np.float32)) == 0.0  # not H×W×3
    assert loupe_acutance(np.zeros((8, 8, 1), dtype=np.float32)) == 0.0  # single channel


def test_the_sample_window_is_centred_when_it_fits():
    rect = loupe_src_rect(200, 200, 100.0, 100.0, 40.0)
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (80.0, 80.0, 40.0, 40.0)


def test_the_sample_window_shifts_rather_than_leaving_the_buffer():
    """A partly out-of-bounds source rect blits garbage, so it slides inside instead."""
    for cx, cy in ((0.0, 0.0), (200.0, 0.0), (0.0, 200.0), (200.0, 200.0), (-50.0, 300.0)):
        rect = loupe_src_rect(200, 200, cx, cy, 40.0)
        assert rect.width() == 40.0 and rect.height() == 40.0
        assert rect.left() >= 0.0 and rect.top() >= 0.0
        assert rect.right() <= 200.0 and rect.bottom() <= 200.0


def test_a_window_bigger_than_the_buffer_is_clamped_to_it():
    rect = loupe_src_rect(30, 50, 15.0, 25.0, 400.0)
    assert rect.width() == 30.0 and rect.height() == 30.0
    assert rect.left() >= 0.0 and rect.right() <= 30.0
    assert rect.top() >= 0.0 and rect.bottom() <= 50.0


def _loupe_overlay(on: bool = True) -> CanvasOverlay:
    state = AppState()
    state.grain_focuser = on
    overlay = CanvasOverlay(state)
    overlay._view_rect = QRectF(0, 0, 400, 300)
    overlay._current_size = (400, 300)
    overlay._qimage = QImage(400, 300, QImage.Format.Format_RGB888)
    overlay._qimage.fill(0x808080)
    overlay._display_buffer = np.full((300, 400, 3), 0.5, dtype=np.float32)
    overlay._mouse_pos = QPointF(200, 150)
    return overlay


def _paint(overlay: CanvasOverlay):
    pixmap = QPixmap(400, 300)
    painter = QPainter(pixmap)
    with patch.object(overlay, "_draw_grain_loupe") as spy:
        overlay._draw_ui(painter)
    painter.end()
    return spy


def test_it_draws_when_on_and_the_cursor_is_over_the_frame():
    assert _paint(_loupe_overlay()).called


def test_the_flag_off_draws_nothing():
    assert not _paint(_loupe_overlay(on=False)).called


@pytest.mark.parametrize("kind", ["tone", "color"])
def test_either_proof_suppresses_it(kind):
    """A proof mosaic is a different image than _qimage, so magnifying it there would show
    pixels that are not on screen. Both the tone strip and the color ring do this."""
    overlay = _loupe_overlay()
    overlay.state.test_strip = True
    overlay.state.test_strip_kind = kind
    assert not _paint(overlay).called


def test_the_raw_ir_layer_suppresses_it():
    overlay = _loupe_overlay()
    overlay.state.dust_overlay_mode = "ir"
    assert not _paint(overlay).called


def test_it_skips_drawing_when_the_cursor_has_left_the_frame():
    overlay = _loupe_overlay()
    overlay._mouse_pos = QPointF(-1.0, -1.0)
    pixmap = QPixmap(400, 300)
    painter = QPainter(pixmap)
    with patch.object(overlay, "_display_buffer") as buf:
        overlay._draw_grain_loupe(painter)
    painter.end()
    assert not buf.__getitem__.called  # bailed before sampling


def test_leaving_the_canvas_parks_the_cursor():
    """Regression: _mouse_pos went stale on leave, so the crosshair, brush ring and loupe all
    stayed drawn at the last position — visible when the cursor exits over the toolbar."""
    overlay = _loupe_overlay()
    overlay.leaveEvent(None)
    assert not overlay._view_rect.contains(overlay._mouse_pos)
