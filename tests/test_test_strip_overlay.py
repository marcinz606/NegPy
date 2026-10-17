"""Canvas side of the test strip: it paints over the content rect, and while it is up
the canvas is a picker — no tool and no pan may steal the click."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent, QPainter, QPixmap

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay
from negpy.features.exposure.analysis import RING_GRID, STRIP_DENSITIES, STRIP_GRADES
from negpy.features.exposure.papers import profiles_for_mode
from negpy.features.process.models import ProcessMode


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


def test_the_strip_paints_and_supersedes_the_zone_map() -> None:
    overlay = _strip_overlay()
    overlay.state.zones_overlay = True
    assert _paint(overlay, "_draw_test_strip").called
    # Both would claim the content rect; the strip wins while it is up.
    assert not _paint(overlay, "_draw_zone_map").called


def test_the_zone_map_comes_back_once_the_strip_is_gone() -> None:
    overlay = _strip_overlay(up=False)
    overlay.state.zones_overlay = True
    assert _paint(overlay, "_draw_zone_map").called


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


def test_a_printed_proof_claims_the_keyboard() -> None:
    """Its controls are single-key. The sidebar button disables itself mid-print, handing focus to
    the next widget in the chain, and a spin box there swallows those keys as text."""
    overlay = _strip_overlay()
    with patch.object(overlay, "setFocus") as spy:
        overlay.on_test_strip_changed()
    assert spy.called


def test_a_dismissed_proof_does_not_grab_the_keyboard() -> None:
    overlay = _strip_overlay(up=False)
    with patch.object(overlay, "setFocus") as spy:
        overlay.on_test_strip_changed()
    assert not spy.called


def test_clearing_the_strip_drops_the_hover_and_the_picker_cursor() -> None:
    overlay = _strip_overlay()
    overlay._strip_hover = (1, 2)
    overlay.state.test_strip = False

    overlay.on_test_strip_changed()

    assert overlay._strip_hover is None
    assert overlay._strip_cache is None


def _strip_labels(overlay: CanvasOverlay):
    """(top_texts, left_texts, accented cell) the proof asked for on its last paint."""
    pixmap = QPixmap(400, 400)
    painter = QPainter(pixmap)
    with patch.object(overlay, "_draw_strip_labels") as spy:
        overlay._draw_test_strip(painter)
    painter.end()
    return spy.call_args.args[2:]


def _labelled_overlay() -> CanvasOverlay:
    """Big enough that the labels aren't dropped for want of room."""
    overlay = _strip_overlay()
    overlay._view_rect = QRectF(0, 0, 400, 400)
    overlay._current_size = (400, 400)
    cfg = overlay.state.config
    overlay.state.config = replace(cfg, exposure=replace(cfg.exposure, density=STRIP_DENSITIES[0]))
    return overlay


def test_the_unrotated_ladder_labels_density_along_the_top() -> None:
    top, left, current = _strip_labels(_labelled_overlay())

    assert top == [f"D {d:.1f}" for d in STRIP_DENSITIES]
    assert left == [f"R {g:.0f}" for g in STRIP_GRADES]
    assert current == (2, 0)  # default grade, thinnest density


def test_a_quarter_turn_moves_the_labels_with_the_patches() -> None:
    """The labels name which ladder runs along which axis. Rotate the patches without them
    and the strip claims a density ladder where the grades now are."""
    overlay = _labelled_overlay()
    overlay.state.test_strip_rotation = 1
    top, left, current = _strip_labels(overlay)

    assert top == [f"R {g:.0f}" for g in STRIP_GRADES]
    # Density now runs down the rows, densest at the top.
    assert left == [f"D {d:.1f}" for d in reversed(STRIP_DENSITIES)]
    assert current == (len(STRIP_DENSITIES) - 1, 2)


def _ring_overlay() -> CanvasOverlay:
    state = AppState()
    state.test_strip = True
    state.test_strip_kind = "color"
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


PAPER_SHORTS = [p.short for _, p in profiles_for_mode(ProcessMode.C41)]


def _paper_overlay(expanded=None) -> CanvasOverlay:
    """The paper strip on a C-41 frame printed on Endura Premier, the third band."""
    state = AppState()
    state.test_strip = True
    state.test_strip_kind = "paper"
    state.test_strip_mosaic = np.zeros((400, 400, 3), dtype=np.float32)
    state.test_strip_tiles = tuple(np.full((400, 400, 3), i / 10, dtype=np.float32) for i in range(len(PAPER_SHORTS)))
    state.test_strip_expanded = expanded
    state.config = replace(state.config, exposure=replace(state.config.exposure, paper_profile="kodak_endura"))
    overlay = CanvasOverlay(state)
    overlay._view_rect = QRectF(0, 0, 400, 400)
    overlay._current_size = (400, 400)
    return overlay


def test_the_paper_strip_names_each_band_along_the_top() -> None:
    overlay = _paper_overlay()
    assert overlay._strip_grid() == (1, len(PAPER_SHORTS))
    top, left, current = _strip_labels(overlay)

    assert top == PAPER_SHORTS
    assert left == [""]
    assert current == (0, 2)


def test_a_turned_paper_strip_names_its_bands_down_the_side() -> None:
    overlay = _paper_overlay()
    overlay.state.test_strip_rotation = 1
    top, left, current = _strip_labels(overlay)

    assert top == [""]
    assert left == list(reversed(PAPER_SHORTS))
    assert current == (len(PAPER_SHORTS) - 3, 0)


def test_a_paper_shown_full_frame_paints_its_own_render_and_no_bands() -> None:
    overlay = _paper_overlay(expanded=3)
    with patch("negpy.desktop.view.canvas.overlay.ImageConverter.to_qimage") as convert:
        convert.return_value = QPixmap(4, 4).toImage()
        assert _paint(overlay, "_draw_strip_labels").called is False
    assert convert.call_args.args[0] is overlay.state.test_strip_tiles[3]


def test_a_click_on_a_paper_shown_full_frame_picks_nothing() -> None:
    overlay = _paper_overlay(expanded=1)
    picked: list = []
    overlay.test_strip_picked.connect(lambda r, c: picked.append((r, c)))

    overlay.mousePressEvent(_press(QPointF(50, 50)))

    assert picked == []


def test_enter_keeps_the_paper_shown_full_frame_only() -> None:
    overlay = _paper_overlay()
    kept: list = []
    overlay.paper_strip_kept.connect(lambda: kept.append(1))

    overlay.finish_in_progress()
    assert kept == []
    overlay.state.test_strip_expanded = 2
    overlay.finish_in_progress()
    assert kept == [1]


def _grid_overlay() -> CanvasOverlay:
    overlay = _paper_overlay()
    overlay.state.test_strip_layout = "grid"
    overlay.state.test_strip_grid = np.zeros((400, 400, 3), dtype=np.float32)
    return overlay


def test_the_paper_grid_names_each_paper_in_its_own_cell() -> None:
    overlay = _grid_overlay()
    overlay.state.test_strip_rotation = 1  # the bands turn; the grid does not
    assert overlay._strip_grid() == (3, 3)

    pixmap = QPixmap(400, 400)
    painter = QPainter(pixmap)
    with patch.object(overlay, "_draw_cell_labels") as spy:
        overlay._draw_test_strip(painter)
    painter.end()
    texts, current = spy.call_args.args[2:]
    assert texts == PAPER_SHORTS
    assert current == 2


def test_the_paper_grid_paints_its_own_mosaic() -> None:
    overlay = _grid_overlay()
    with patch("negpy.desktop.view.canvas.overlay.ImageConverter.to_qimage") as convert:
        convert.return_value = QPixmap(4, 4).toImage()
        overlay._strip_qimage()
    assert convert.call_args.args[0] is overlay.state.test_strip_grid


def test_a_click_picks_the_grid_cell_under_it() -> None:
    overlay = _grid_overlay()
    picked: list = []
    overlay.test_strip_picked.connect(lambda r, c: picked.append((r, c)))

    overlay.mousePressEvent(_press(QPointF(350, 200)))

    assert picked == [(1, 2)]


def test_an_empty_grid_cell_is_no_target() -> None:
    overlay = _grid_overlay()
    overlay.state.test_strip_tiles = overlay.state.test_strip_tiles[:7]
    picked: list = []
    overlay.test_strip_picked.connect(lambda r, c: picked.append((r, c)))

    overlay.mousePressEvent(_press(QPointF(350, 350)))

    assert overlay._strip_cell_at(0.9, 0.9) is None
    assert overlay._strip_cell_at(0.1, 0.9) == (2, 0)
    assert picked == []
