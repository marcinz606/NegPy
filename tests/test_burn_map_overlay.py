"""Canvas side of the burn map: notation over the masks plus the instruction card, and every
condition under which it must stay out of the way."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QPainter, QPixmap

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import _BURN_TINT, _DODGE_TINT, CanvasOverlay
from negpy.features.local.models import PolygonMask

_SQUARE = ((0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6))


def _burn_overlay(up: bool = True, masks=()) -> CanvasOverlay:
    state = AppState()
    state.burn_map = up
    state.config = replace(state.config, local=replace(state.config.local, masks=tuple(masks)))
    state.current_file_hash = "f1"  # the hidden-mask store keys on it
    overlay = CanvasOverlay(state)
    overlay._view_rect = QRectF(0, 0, 400, 300)
    overlay._current_size = (400, 300)
    # _draw_local_masks needs a uv_grid; the burn map reads the polys it caches.
    grid = np.zeros((16, 16, 2), dtype=np.float32)
    grid[..., 0] = np.linspace(0.0, 1.0, 16)[None, :]
    grid[..., 1] = np.linspace(0.0, 1.0, 16)[:, None]
    state.last_metrics["uv_grid"] = grid
    return overlay


def _paint(overlay: CanvasOverlay, method: str):
    pixmap = QPixmap(400, 300)
    painter = QPainter(pixmap)
    with patch.object(overlay, method) as spy:
        overlay._draw_ui(painter)
    painter.end()
    return spy


def test_it_draws_when_the_flag_is_on() -> None:
    overlay = _burn_overlay(masks=[PolygonMask(vertices=_SQUARE, strength=1.0)])
    assert _paint(overlay, "_draw_burn_map").called


def test_the_flag_off_draws_nothing() -> None:
    overlay = _burn_overlay(up=False, masks=[PolygonMask(vertices=_SQUARE, strength=1.0)])
    assert not _paint(overlay, "_draw_burn_map").called


@pytest.mark.parametrize("kind", ["tone", "colour"])
def test_either_proof_suppresses_the_burn_map(kind: str) -> None:
    """A proof owns the content rect, and its patches are not the print the notation
    describes. Both the tone strip and the colour ring do this."""
    overlay = _burn_overlay(masks=[PolygonMask(vertices=_SQUARE, strength=1.0)])
    overlay.state.test_strip = True
    overlay.state.test_strip_kind = kind
    assert not _paint(overlay, "_draw_burn_map").called


def test_a_dodge_and_a_burn_label_are_told_apart_by_colour() -> None:
    """Each label takes the tint its own outline uses, so the print reads dodge from burn at a
    glance. One shared constant per direction is what keeps label and outline from drifting."""
    overlay = _burn_overlay(
        masks=[
            PolygonMask(vertices=_SQUARE, strength=1.0),
            PolygonMask(vertices=((0.7, 0.7), (0.9, 0.7), (0.9, 0.9), (0.7, 0.9)), strength=-2.0),
        ]
    )
    pixmap = QPixmap(400, 300)
    painter = QPainter(pixmap)
    with patch.object(QPainter, "setPen") as set_pen:
        overlay._draw_burn_map(painter)
    painter.end()

    pens = [call[0][0] for call in set_pen.call_args_list if isinstance(call[0][0], QColor)]
    assert _DODGE_TINT in pens, "the dodge label never took the dodge tint"
    assert _BURN_TINT in pens, "the burn label never took the burn tint"
    assert _DODGE_TINT != _BURN_TINT, "dodge and burn would be indistinguishable"


def test_the_loupe_coexists_with_it_and_draws_over_it() -> None:
    """Both are up at once, and the glass paints last: it shows the frame's actual pixels, so
    a label drawn on top of it would be notation over magnified pixels it doesn't describe."""
    overlay = _burn_overlay(masks=[PolygonMask(vertices=_SQUARE, strength=1.0)])
    overlay.state.grain_focuser = True
    overlay._mouse_pos = QPointF(200, 150)
    overlay._qimage = QPixmap(400, 300).toImage()
    overlay._display_buffer = np.full((300, 400, 3), 0.5, dtype=np.float32)

    order: list[str] = []
    pixmap = QPixmap(400, 300)
    painter = QPainter(pixmap)
    with (
        patch.object(overlay, "_draw_burn_map", side_effect=lambda _p: order.append("burn_map")),
        patch.object(overlay, "_draw_grain_loupe", side_effect=lambda _p: order.append("loupe")),
    ):
        overlay._draw_ui(painter)
    painter.end()

    assert order == ["burn_map", "loupe"]


def test_a_flat_peek_hides_the_burn_map() -> None:
    """The flat intent bypasses the local stage, so notation would describe work that is not
    in those pixels."""
    overlay = _burn_overlay(masks=[PolygonMask(vertices=_SQUARE, strength=1.0)])
    overlay.state.flat_peek = True
    assert not _paint(overlay, "_draw_burn_map").called


def test_crop_and_analysis_modes_hide_the_burn_map() -> None:
    for mode in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW):
        overlay = _burn_overlay(masks=[PolygonMask(vertices=_SQUARE, strength=1.0)])
        overlay.set_tool_mode(mode)
        assert not _paint(overlay, "_draw_burn_map").called, mode


def test_it_draws_the_card_even_with_no_masks() -> None:
    """Density and grade are printing instructions in their own right."""
    overlay = _burn_overlay()
    pixmap = QPixmap(400, 300)
    pixmap.fill()
    painter = QPainter(pixmap)
    overlay._draw_burn_map(painter)
    painter.end()

    img = pixmap.toImage()
    rect = overlay._content_view_rect()
    # The card plate is a dark rounded rect just inside the top-left of the content.
    probe = img.pixelColor(int(rect.x() + 20), int(rect.y() + 55))
    assert probe.red() < 120, f"expected the card's dark plate, got {probe.red()}"


def test_the_card_clears_the_hud_pill_and_before_badge() -> None:
    overlay = _burn_overlay()
    pixmap = QPixmap(400, 300)
    pixmap.fill()
    painter = QPainter(pixmap)
    overlay._draw_burn_map(painter)
    painter.end()

    img = pixmap.toImage()
    rect = overlay._content_view_rect()
    # y+12..+34 belongs to the HUD pill and the BEFORE badge; the card must start below it.
    for y in range(int(rect.y() + 12), int(rect.y() + 34)):
        assert img.pixelColor(int(rect.x() + 20), y).red() > 200, f"card intrudes at y={y}"


def test_hidden_masks_lose_their_label() -> None:
    masks = [PolygonMask(vertices=_SQUARE, strength=1.0)]
    overlay = _burn_overlay(masks=masks)
    with patch("negpy.desktop.view.canvas.overlay.polygon_label_anchor") as anchor:
        anchor.return_value = (100.0, 100.0)
        overlay._local_mask_screen_polys = [[QPointF(10, 10), QPointF(50, 10), QPointF(50, 50)]]
        pixmap = QPixmap(400, 300)
        painter = QPainter(pixmap)
        overlay._draw_burn_map(painter)
        painter.end()
        assert anchor.called

    overlay.state.local_hidden_masks = {0}
    assert overlay.state.local_hidden_masks == {0}  # the store keys on the file hash
    with patch("negpy.desktop.view.canvas.overlay.polygon_label_anchor") as anchor:
        overlay._local_mask_screen_polys = [[QPointF(10, 10), QPointF(50, 10), QPointF(50, 50)]]
        pixmap = QPixmap(400, 300)
        painter = QPainter(pixmap)
        overlay._draw_burn_map(painter)
        painter.end()
        assert not anchor.called, "a hidden mask must not be labelled"
