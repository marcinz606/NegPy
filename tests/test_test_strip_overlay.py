"""Canvas side of the test strip: it paints over the content rect, and while it is up
the canvas is a picker — no tool and no pan may steal the click."""

from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent, QPainter, QPixmap

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay
from negpy.features.exposure.analysis import RING_GRID, STRIP_DENSITIES, STRIP_GRADES


def _press(pos: QPointF) -> QMouseEvent:
    return QMouseEvent(
        QEvent.Type.MouseButtonPress, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )


def _strip_overlay(up: bool = True) -> CanvasOverlay:
    state = AppState()
    state.test_strip = up
    state.test_strip_mosaic = np.zeros((80, 80, 3), dtype=np.float32)
    overlay = CanvasOverlay(state)
    overlay._view_rect = QRectF(0, 0, 100, 100)
    overlay._current_size = (100, 100)
    return overlay


def _paint(overlay: CanvasOverlay, method: str):
    pixmap = QPixmap(100, 100)
    painter = QPainter(pixmap)
    with patch.object(overlay, method) as spy:
        overlay._draw_ui(painter)
    painter.end()
    return spy


def test_a_click_picks_the_patch_under_it() -> None:
    overlay = _strip_overlay()
    picked: list = []
    overlay.test_strip_picked.connect(lambda r, c: picked.append((r, c)))

    overlay.mousePressEvent(_press(QPointF(5, 5)))
    overlay.mousePressEvent(_press(QPointF(95, 95)))

    assert picked == [(0, 0), (len(STRIP_GRADES) - 1, len(STRIP_DENSITIES) - 1)]


def test_the_strip_takes_the_click_off_an_active_tool() -> None:
    overlay = _strip_overlay()
    overlay.set_tool_mode(ToolMode.DUST_PICK)
    picked: list = []
    overlay.test_strip_picked.connect(lambda r, c: picked.append((r, c)))

    overlay.mousePressEvent(_press(QPointF(50, 50)))

    assert len(picked) == 1
    assert overlay._heal_drag_pts == []  # no heal was placed underneath


def test_no_strip_up_means_no_pick() -> None:
    overlay = _strip_overlay(up=False)
    picked: list = []
    overlay.test_strip_picked.connect(lambda r, c: picked.append((r, c)))

    overlay.mousePressEvent(_press(QPointF(50, 50)))

    assert picked == []


def test_the_strip_paints_and_supersedes_the_zone_grid() -> None:
    overlay = _strip_overlay()
    overlay.state.zones_overlay = True
    assert _paint(overlay, "_draw_test_strip").called
    # Both would claim the content rect; the strip wins while it is up.
    assert not _paint(overlay, "_draw_zone_grid").called


def test_the_zone_grid_comes_back_once_the_strip_is_gone() -> None:
    overlay = _strip_overlay(up=False)
    overlay.state.zones_overlay = True
    assert _paint(overlay, "_draw_zone_grid").called


def test_a_flat_peek_hides_the_strip_rather_than_covering_the_peek() -> None:
    overlay = _strip_overlay()
    overlay.state.flat_peek = True
    assert not _paint(overlay, "_draw_test_strip").called


def test_crop_and_analysis_modes_hide_the_strip() -> None:
    for mode in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW):
        overlay = _strip_overlay()
        overlay.set_tool_mode(mode)
        # These show the uncropped frame, so the patches would not line up with it.
        assert not _paint(overlay, "_draw_test_strip").called


def test_patch_rects_tile_the_content_rect_exactly() -> None:
    overlay = _strip_overlay()
    rect = QRectF(10, 20, 80, 60)
    patches = overlay._strip_patch_rects(rect)

    assert len(patches) == len(STRIP_GRADES) * len(STRIP_DENSITIES)
    assert sum(cell.width() * cell.height() for _, _, cell in patches) == rect.width() * rect.height()
    union = patches[0][2]
    for _, _, cell in patches[1:]:
        union = union.united(cell)
    assert union == rect


def test_clearing_the_strip_drops_the_hover_and_the_picker_cursor() -> None:
    overlay = _strip_overlay()
    overlay._strip_hover = (1, 2)
    overlay.state.test_strip = False

    overlay.on_test_strip_changed()

    assert overlay._strip_hover is None
    assert overlay._strip_cache is None


def _ring_overlay() -> CanvasOverlay:
    state = AppState()
    state.test_strip = True
    state.test_strip_kind = "colour"
    state.test_strip_mosaic = np.zeros((90, 90, 3), dtype=np.float32)
    overlay = CanvasOverlay(state)
    overlay._view_rect = QRectF(0, 0, 90, 90)
    overlay._current_size = (90, 90)
    return overlay


def test_the_ring_picks_across_its_own_grid() -> None:
    """The overlay reads the grid off the proof kind, so a click on the ring must map to the
    ring's geometry rather than the tone strip's."""
    overlay = _ring_overlay()
    picked: list = []
    overlay.test_strip_picked.connect(lambda r, c: picked.append((r, c)))

    overlay.mousePressEvent(_press(QPointF(5, 5)))
    overlay.mousePressEvent(_press(QPointF(45, 45)))
    overlay.mousePressEvent(_press(QPointF(85, 85)))

    mid = (RING_GRID[0] // 2, RING_GRID[1] // 2)
    assert picked == [(0, 0), mid, (RING_GRID[0] - 1, RING_GRID[1] - 1)]


def test_each_proof_kind_lays_out_its_own_grid() -> None:
    overlay = _ring_overlay()
    assert overlay._strip_grid() == RING_GRID
    assert len(overlay._strip_patch_rects(QRectF(0, 0, 90, 90))) == RING_GRID[0] * RING_GRID[1]
    # And the tone strip still gets its own grid off the same helper.
    overlay.state.test_strip_kind = "tone"
    assert overlay._strip_grid() == (len(STRIP_GRADES), len(STRIP_DENSITIES))
    assert len(overlay._strip_patch_rects(QRectF(0, 0, 90, 90))) == len(STRIP_GRADES) * len(STRIP_DENSITIES)
