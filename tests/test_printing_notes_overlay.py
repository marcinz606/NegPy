"""Canvas side of the printing notes: when the map is painted, what it hatches, and the
exported sheet — which must be baked from the same frame and mapping the canvas shows."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QImage, QPainter, QPixmap

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.overlay import CanvasOverlay
from negpy.desktop.view.canvas.printing_notes import card_size, notes_sheet
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask, MaskShape

W = H = 200
BURN = LocalMask(vertices=((0.05, 0.05), (0.45, 0.05), (0.45, 0.45), (0.05, 0.45)), stops=1.0)
DODGE = LocalMask(vertices=((0.55, 0.55), (0.95, 0.55), (0.95, 0.95), (0.55, 0.95)), stops=-0.5)


def _uv_grid(h: int = H, w: int = W) -> np.ndarray:
    u, v = np.meshgrid(np.linspace(0, 1, w, dtype=np.float32), np.linspace(0, 1, h, dtype=np.float32))
    return np.ascontiguousarray(np.stack([u, v], axis=-1))


def _overlay(notes: bool = True, masks=(BURN, DODGE)) -> CanvasOverlay:
    state = AppState()
    state.printing_notes = notes
    state.config = replace(state.config, local=LocalAdjustmentsConfig(masks=tuple(masks)))
    state.last_metrics = {"uv_grid": _uv_grid()}
    overlay = CanvasOverlay(state)
    overlay._view_rect = QRectF(0, 0, W, H)
    overlay._current_size = (W, H)
    overlay._qimage = QImage(W, H, QImage.Format.Format_RGB32)
    overlay._qimage.fill(0x00808080)
    return overlay


def _paint(overlay: CanvasOverlay, method: str):
    pixmap = QPixmap(W, H)
    painter = QPainter(pixmap)
    with patch.object(overlay, method) as spy:
        overlay._draw_ui(painter)
    painter.end()
    return spy


def _to_array(img: QImage) -> np.ndarray:
    rgb = img.convertToFormat(QImage.Format.Format_RGB32)
    bits = rgb.bits()
    bits.setsize(rgb.sizeInBytes())
    return np.frombuffer(bits, np.uint8).reshape(rgb.height(), rgb.bytesPerLine() // 4, 4)[:, : rgb.width()]


def test_the_map_paints_when_the_toggle_is_on() -> None:
    assert _paint(_overlay(), "_draw_printing_notes").called
    assert not _paint(_overlay(notes=False), "_draw_printing_notes").called


def test_a_proof_or_a_tool_owning_the_canvas_hides_the_map() -> None:
    strip = _overlay()
    strip.state.test_strip = True
    strip.state.test_strip_mosaic = np.zeros((80, 80, 3), dtype=np.float32)
    assert not _paint(strip, "_draw_printing_notes").called

    peek = _overlay()
    peek.state.flat_peek = True
    assert not _paint(peek, "_draw_printing_notes").called

    for mode in (ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW):
        overlay = _overlay()
        overlay.set_tool_mode(mode)
        assert not _paint(overlay, "_draw_printing_notes").called


def test_the_compare_baseline_gets_no_map() -> None:
    """The baseline render has no masks applied, so a map on it would mark burns that
    are not in the picture underneath."""
    overlay = _overlay()
    overlay.state.compare_mode = True
    overlay.state.compare_before = np.zeros((H, W, 3), dtype=np.float32)
    assert not _paint(overlay, "_draw_printing_notes").called


def test_hidden_masks_are_still_on_the_record() -> None:
    overlay = _overlay()
    overlay.state.current_file_hash = "abc"
    overlay.state.local_hidden_masks = {0}
    sheet = overlay.printing_notes_sheet()
    assert sheet is not None
    # Both masks in the recipe, and the hidden burn still hatched on the map.
    assert "1 Burn +1 · 2 Dodge −½" in "\n".join(overlay._recipe_lines())
    assert _to_array(sheet)[40, 40].tolist() != [128, 128, 128, 255]


def test_the_sheet_hatches_the_burn_and_leaves_the_dodge_open() -> None:
    frame = QImage(W, H, QImage.Format.Format_RGB32)
    frame.fill(0x00808080)
    local = LocalAdjustmentsConfig(masks=(BURN, DODGE))

    sheet = notes_sheet(frame, None, local, _uv_grid(), [])
    arr = _to_array(sheet)

    # Sampled off-centre in both masks so the stop badge is not what is being measured.
    burn_interior = arr[20:45, 20:45]
    dodge_interior = arr[118:136, 118:136]
    grey = np.array([128, 128, 128, 255], dtype=np.uint8)
    assert (burn_interior != grey).any(axis=-1).mean() > 0.05  # hatch lines
    assert (dodge_interior != grey).any(axis=-1).mean() == 0.0  # open


def test_the_sheet_hatches_a_card_edges_full_exposure_side() -> None:
    """A gradient has no outline. The map marks the half plane with the full burn and
    leaves the other side clean."""
    frame = QImage(W, H, QImage.Format.Format_RGB32)
    frame.fill(0x00808080)
    edge = LocalMask(vertices=((0.4, 0.5), (0.8, 0.5)), stops=1.0, shape=MaskShape.GRADIENT)

    arr = _to_array(notes_sheet(frame, None, LocalAdjustmentsConfig(masks=(edge,)), _uv_grid(), []))

    grey = np.array([128, 128, 128, 255], dtype=np.uint8)
    assert (arr[20:60, 5:45] != grey).any(axis=-1).mean() > 0.05  # Hatched, behind the edge.
    assert (arr[20:60, 175:195] != grey).any(axis=-1).mean() == 0.0  # Past the fade-out.


def test_the_sheet_carries_the_recipe_in_a_band_below_the_frame() -> None:
    frame = QImage(W, H, QImage.Format.Format_RGB32)
    frame.fill(0x00808080)
    lines = ["roll1_04.tif", "Print Density 1.00"]

    sheet = notes_sheet(frame, None, LocalAdjustmentsConfig(), _uv_grid(), lines)

    assert sheet.width() == W
    assert sheet.height() == H + round(card_size(lines)[1])


def test_no_render_means_no_sheet() -> None:
    overlay = _overlay()
    overlay._qimage = None
    assert overlay.printing_notes_sheet() is None

    unmapped = _overlay()
    unmapped.state.last_metrics = {}
    assert unmapped.printing_notes_sheet() is None


def test_a_frame_without_masks_still_exports_its_recipe() -> None:
    overlay = _overlay(masks=())
    overlay.state.last_metrics = {}
    sheet = overlay.printing_notes_sheet()
    assert sheet is not None and sheet.height() > H


def test_the_card_sits_inside_the_picture() -> None:
    overlay = _overlay()
    with patch("negpy.desktop.view.canvas.overlay.paint_card") as spy:
        pixmap = QPixmap(W, H)
        painter = QPainter(pixmap)
        overlay._draw_ui(painter)
        painter.end()
    anchor: QPointF = spy.call_args[0][1]
    assert overlay._content_view_rect().contains(anchor)
