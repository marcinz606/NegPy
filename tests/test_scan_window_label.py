"""Offline geometry tests for the scan-preview label.

An offset scan returns a raster shorter than the frame (the device blacks out one
pitch past the frame start, so the backend caps the window). The label must place
that raster at its real position instead of stretching it over the whole widget —
otherwise every fraction read off the widget, crop rects included, means something
different from what the batch will scan.
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


def _offset_preview(label: ScanWindowLabel, offset_mm: float) -> None:
    """Feed the label what the backend would return for this offset."""
    span = 1.0 - offset_mm / _PITCH
    label.set_frame(QPixmap(max(1, int(round(380 * span))), 140), (offset_mm / _PITCH, 1.0))


def test_offset_preview_occupies_only_its_slice_of_the_frame() -> None:
    label = _label()
    _offset_preview(label, _OFFSET)

    draw = label._display()
    content = label._content_rect(draw)

    assert (content.left() - draw.left()) / draw.width() == pytest.approx(_OFFSET / _PITCH, abs=2e-3)
    assert content.width() / draw.width() == pytest.approx(1.0 - _OFFSET / _PITCH, abs=4e-3)


def test_a_film_column_maps_to_the_same_frame_fraction_at_any_offset() -> None:
    # The invariance the stretch broke: the middle of the delivered raster sits at a
    # known place in the frame, and reading the widget must report that place.
    label = _label()
    _offset_preview(label, _OFFSET)

    draw = label._display()
    content = label._content_rect(draw)
    fx, _fy = label._to_fraction(content.center(), draw)

    start = _OFFSET / _PITCH
    assert fx == pytest.approx(start + (1.0 - start) / 2, abs=3e-3)


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
