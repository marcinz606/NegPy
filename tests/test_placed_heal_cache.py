"""The heal tool repaints the overlay per mouse move, so the placed heals' uv lookups and
raster are cached: a repaint with nothing changed maps no point, and an edit or a new
render re-maps."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QImage, QPainter

from negpy.desktop.session import AppState
from negpy.desktop.view.canvas import overlay as overlay_mod
from negpy.desktop.view.canvas.overlay import CanvasOverlay


def _uv_grid(h: int = 60, w: int = 80) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([xx / (w - 1), yy / (h - 1)], -1)


def _stroke(x: float, y: float) -> tuple:
    return ([[x, y], [x + 0.05, y + 0.02], [x + 0.1, y + 0.05]], 12.0, 0.0, 0.0)


def _overlay(strokes: list) -> CanvasOverlay:
    state = AppState()
    state.config = replace(state.config, retouch=replace(state.config.retouch, manual_heal_strokes=strokes))
    state.last_metrics["uv_grid"] = _uv_grid()
    overlay = CanvasOverlay(state)
    overlay.resize(100, 100)
    overlay._view_rect = QRectF(0, 0, 100, 100)
    return overlay


def _paint(overlay: CanvasOverlay) -> QImage:
    img = QImage(100, 100, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    overlay._draw_placed_heals(painter)
    painter.end()
    return img


def _counting_lookups():
    real = overlay_mod.CoordinateMapping.map_raw_to_viewport
    calls = []

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    return calls, patch.object(overlay_mod.CoordinateMapping, "map_raw_to_viewport", side_effect=counted)


def test_repaint_with_nothing_changed_maps_no_point() -> None:
    overlay = _overlay([_stroke(0.2, 0.2), _stroke(0.5, 0.6)])
    calls, patcher = _counting_lookups()
    with patcher:
        first = _paint(overlay)
        assert len(calls) == 6
        second = _paint(overlay)
    assert len(calls) == 6
    assert first == second


def test_new_stroke_and_new_render_remap() -> None:
    overlay = _overlay([_stroke(0.2, 0.2)])
    calls, patcher = _counting_lookups()
    with patcher:
        _paint(overlay)
        state = overlay.state
        state.config = replace(
            state.config, retouch=replace(state.config.retouch, manual_heal_strokes=[_stroke(0.2, 0.2), _stroke(0.6, 0.6)])
        )
        _paint(overlay)
        assert len(calls) == 3 + 6
        state.last_metrics["uv_grid"] = _uv_grid()
        _paint(overlay)
    assert len(calls) == 3 + 6 + 6


def test_hit_test_reads_the_cached_geometry() -> None:
    overlay = _overlay([_stroke(0.2, 0.2), _stroke(0.6, 0.6)])
    _paint(overlay)
    with patch.object(overlay_mod.CoordinateMapping, "map_raw_to_viewport", side_effect=AssertionError):
        assert overlay.heal_hit_test(QPointF(60.5, 60.5)) == ("stroke", 1)
