"""RoiImageLabel coordinate-math tests (clicks/ROI must map to full-frame fractions)."""

import sys

from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.roi_image import RoiImageLabel

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _label():
    lbl = RoiImageLabel()
    lbl.resize(200, 200)
    lbl.set_frame(QPixmap(100, 100))
    return lbl


def test_display_letterboxes_the_full_frame():
    # 100×100 frame in a 200×200 label → a centred 200×200 draw rect (square fits exactly).
    rect = _label()._display()
    assert rect is not None
    assert rect.width() == 200 and rect.height() == 200


def test_to_fraction_maps_widget_px_to_full_frame():
    draw = QRect(0, 0, 100, 100)
    # centre of the drawn frame → centre fraction
    fx, fy = RoiImageLabel._to_fraction(QPoint(50, 50), draw)
    assert abs(fx - 0.5) < 1e-9 and abs(fy - 0.5) < 1e-9
    # origin of the drawn frame → (0, 0)
    fx, fy = RoiImageLabel._to_fraction(QPoint(0, 0), draw)
    assert abs(fx - 0.0) < 1e-9 and abs(fy - 0.0) < 1e-9


def test_to_fraction_clamps_outside_the_draw_rect():
    draw = QRect(10, 10, 100, 100)
    fx, fy = RoiImageLabel._to_fraction(QPoint(500, 500), draw)
    assert fx == 1.0 and fy == 1.0


def test_crosshair_places_small_centred_patch():
    from negpy.desktop.view.sidebar.roi_image import _CROSSHAIR_ASPECT, _CROSSHAIR_FRAC

    lbl = _label()  # 100×100 frame (square → aspect 1)
    lbl._set_crosshair(0.5, 0.5)
    roi = lbl.roi()
    assert abs(roi.w - _CROSSHAIR_FRAC) < 1e-9
    assert abs(roi.h - _CROSSHAIR_ASPECT * _CROSSHAIR_FRAC) < 1e-9  # taller than wide (vertical strip)
    assert abs((roi.x + roi.w / 2) - 0.5) < 1e-9
    assert abs((roi.y + roi.h / 2) - 0.5) < 1e-9


def test_crosshair_clamps_inside_frame():
    lbl = _label()
    lbl._set_crosshair(0.99, 0.99)
    roi = lbl.roi()
    assert roi.x >= 0.0 and roi.y >= 0.0
    assert roi.x + roi.w <= 1.0 + 1e-9 and roi.y + roi.h <= 1.0 + 1e-9


def test_crosshair_is_vertical_strip_in_pixels_on_wide_frame():
    from negpy.desktop.view.sidebar.roi_image import _CROSSHAIR_ASPECT

    lbl = RoiImageLabel()
    lbl.set_frame(QPixmap(300, 200))  # 3:2 landscape
    lbl._set_crosshair(0.5, 0.5)
    roi = lbl.roi()
    # pixel-height = _CROSSHAIR_ASPECT × pixel-width → a vertical rectangle regardless of frame shape
    assert abs(roi.h * 200 - _CROSSHAIR_ASPECT * roi.w * 300) < 1e-6
    assert roi.h * 200 > roi.w * 300  # taller than wide in pixels


def test_loading_spinner_runs_then_clears_on_first_frame():
    lbl = RoiImageLabel()
    lbl.resize(200, 200)
    lbl.set_loading(True)
    assert lbl._loading and lbl._spin_timer.isActive()  # buffering spinner animating
    lbl.set_frame(QPixmap(100, 100))  # a real frame arrives
    assert not lbl._loading and not lbl._spin_timer.isActive()  # spinner auto-stops


def test_set_loading_false_stops_spinner():
    lbl = RoiImageLabel()
    lbl.set_loading(True)
    lbl.set_loading(False)
    assert not lbl._loading and not lbl._spin_timer.isActive()
