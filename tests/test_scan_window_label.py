"""Offline geometry tests for the scan-preview label.

An offset scan returns a raster shorter than the frame (the device blacks out one
pitch past the frame start, so the backend caps the window). The tile is anchored
to the NEXT scan: a current raster sits flush left and ends at the blackout
boundary; a stale one is re-placed by the offset delta and may clip off the left
edge. Widget fractions are then exactly the window fractions the batch applies.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.widgets.scan_window_label import ScanWindowLabel

if not QApplication.instance():
    _app = QApplication(sys.argv)

_PITCH = 37.83
_OFFSET = 4.8


def _label(width: int = 380, height: int = 140) -> ScanWindowLabel:
    label = ScanWindowLabel()
    label.setFixedSize(width, height)
    return label


def _offset_preview(label: ScanWindowLabel, offset_mm: float, *, slider_mm: float | None = None) -> None:
    """Feed the label what the backend returns for this offset, placed for `slider_mm`."""
    x = offset_mm / _PITCH
    y = x if slider_mm is None else slider_mm / _PITCH
    label.set_frame(QPixmap(max(1, int(round(380 * (1.0 - x)))), 140), (x - y, 1.0 - y))


def test_fresh_offset_preview_sits_flush_left_and_ends_at_the_boundary() -> None:
    label = _label()
    _offset_preview(label, _OFFSET)

    draw = label._display()
    content = label._content_rect(draw)

    assert content.left() == draw.left()
    assert content.width() / draw.width() == pytest.approx(1.0 - _OFFSET / _PITCH, abs=4e-3)


def test_raising_the_slider_slides_a_stale_raster_left_past_the_edge() -> None:
    # Previewed at 0, slider raised to _OFFSET: the same film sits _OFFSET/_PITCH
    # further left, clipping off the tile — content moves, the frame does not shrink.
    label = _label()
    _offset_preview(label, 0.0, slider_mm=_OFFSET)

    draw = label._display()
    content = label._content_rect(draw)

    assert (draw.left() - content.left()) / draw.width() == pytest.approx(_OFFSET / _PITCH, abs=3e-3)
    assert content.width() / draw.width() == pytest.approx(1.0, abs=4e-3)  # raster length unchanged
    label.grab()  # negative-start coverage must paint (clipped), not crash


def test_every_raster_ends_at_the_current_blackout_boundary() -> None:
    # Delivery ends at a fixed film position, so stale and fresh rasters share
    # one right edge: 1 - slider/pitch in next-scan coordinates.
    label = _label()
    _offset_preview(label, 0.0, slider_mm=_OFFSET)
    draw = label._display()
    stale = label._content_rect(draw)
    stale_right = stale.left() + stale.width()

    _offset_preview(label, _OFFSET)
    fresh = label._content_rect(draw)
    fresh_right = fresh.left() + fresh.width()

    assert abs(stale_right - fresh_right) <= 1
    assert (fresh_right - draw.left()) / draw.width() == pytest.approx(1.0 - _OFFSET / _PITCH, abs=4e-3)


def test_no_coverage_keeps_the_pixmap_filling_the_widget() -> None:
    label = _label()
    label.set_frame(QPixmap(380, 140))

    draw = label._display()
    assert label._content_rect(draw) == draw


def test_zero_offset_covers_the_whole_frame() -> None:
    label = _label()
    _offset_preview(label, 0.0)

    draw = label._display()
    assert label._content_rect(draw).width() == draw.width()


def test_crop_rect_round_trips_through_widget_pixels_under_an_offset() -> None:
    label = _label()
    _offset_preview(label, _OFFSET)
    draw = label._display()

    point = QPoint(draw.left() + int(0.75 * draw.width()), draw.top() + 20)
    fx, fy = label._to_fraction(point, draw)
    back = label._rect_in_widget((fx, fy, fx, fy), draw)

    assert abs(back.left() - point.x()) <= 1
