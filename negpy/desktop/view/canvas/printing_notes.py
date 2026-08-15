"""Painting for the Printing Notes overlay — the marked-up work print.

One implementation, two targets: the canvas paints into screen coordinates, the
exported sheet into the rendered frame's own pixels. Both hand `paint_map` polygons
that are already mapped, so the two can never drift apart.

Hatching marks a burn (shaded = extra exposure, the darkroom convention); a dodge is
left open. `scale` sizes pens, hatch spacing and type for the exported sheet, where a
hairline would vanish.
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QPolygonF

from negpy.features.local.logic import min_points, outline_points
from negpy.features.local.models import LocalAdjustmentsConfig, MaskShape
from negpy.services.view.coordinate_mapping import CoordinateMapping
from negpy.services.view.printing_notes import MaskNote, mask_notes

# The same amber and blue the Dodge & Burn outlines use, so a mask reads the same in the
# notes as it does while editing.
_DODGE = QColor(232, 200, 74)
_BURN = QColor(74, 143, 232)
_INK = QColor(242, 242, 242)
_CARD_BG = QColor(10, 10, 10, 195)
_BAND_BG = QColor(16, 16, 16)
_BADGE_BG = QColor(0, 0, 0, 190)

_HATCH_SPACING_PX = 9.0
_CARD_PAD_PX = 8.0
_CARD_LEADING = 1.3
_SHEET_SCALE_REF = 1400.0  # px long edge the on-screen line weights were drawn for

Poly = Tuple[List[QPointF], MaskNote]


def notes_font(scale: float = 1.0, px: float = 12.0) -> QFont:
    font = QFont()
    font.setBold(True)
    font.setPixelSize(max(9, round(px * scale)))
    return font


def _badge_anchor(poly: QPolygonF) -> QPointF:
    """Centre of the mask, or a vertex when the centre falls outside a concave shape."""
    centre = poly.boundingRect().center()
    if poly.containsPoint(centre, Qt.FillRule.OddEvenFill):
        return centre
    return poly.first()


def _hatch(painter: QPainter, poly: QPolygonF, color: QColor, scale: float) -> None:
    path = QPainterPath()
    path.addPolygon(poly)
    rect = poly.boundingRect()
    painter.save()
    painter.setClipPath(path)
    pen = QPen(color, max(1.0, scale))
    pen.setCosmetic(scale <= 1.0)
    painter.setPen(pen)
    spacing = _HATCH_SPACING_PX * scale
    x = rect.left() - rect.height()
    while x < rect.right():
        painter.drawLine(QPointF(x, rect.bottom()), QPointF(x + rect.height(), rect.top()))
        x += spacing
    painter.restore()


def _draw_badge(painter: QPainter, pos: QPointF, text: str, color: QColor, scale: float) -> None:
    font = notes_font(scale)
    metrics = QFontMetricsF(font)
    pad = 5.0 * scale
    w = metrics.horizontalAdvance(text) + 2 * pad
    h = metrics.height() + pad
    rect = QRectF(pos.x() - w / 2.0, pos.y() - h / 2.0, w, h)
    painter.save()
    painter.setFont(font)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_BADGE_BG)
    painter.drawRoundedRect(rect, 3.0 * scale, 3.0 * scale)
    painter.setPen(color)
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
    painter.restore()


def paint_map(painter: QPainter, polys: Sequence[Poly], scale: float = 1.0) -> None:
    """Outline every mask, hatch the burns, and badge each with its stop value."""
    for pts, note in polys:
        if len(pts) < 3:
            continue
        poly = QPolygonF(pts)
        color = _BURN if note.is_burn else _DODGE
        if note.is_burn:
            _hatch(painter, poly, color, scale)

        painter.save()
        pen = QPen(color, max(1.8, 1.8 * scale))
        pen.setCosmetic(scale <= 1.0)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(poly)
        painter.restore()

        _draw_badge(painter, _badge_anchor(poly), note.badge, color, scale)


def card_size(lines: Sequence[str], scale: float = 1.0) -> Tuple[float, float]:
    """(width, height) the recipe card needs for `lines`."""
    if not lines:
        return 0.0, 0.0
    metrics = QFontMetricsF(notes_font(scale))
    pad = _CARD_PAD_PX * scale
    leading = metrics.height() * _CARD_LEADING
    width = max(metrics.horizontalAdvance(line) for line in lines) + 2 * pad
    return width, leading * len(lines) + 2 * pad


def paint_card(
    painter: QPainter,
    anchor: QPointF,
    lines: Sequence[str],
    scale: float = 1.0,
    background: Optional[QColor] = _CARD_BG,
) -> QRectF:
    """The printing record, one line per row, anchored at its top-left."""
    if not lines:
        return QRectF()
    font = notes_font(scale)
    metrics = QFontMetricsF(font)
    pad = _CARD_PAD_PX * scale
    leading = metrics.height() * _CARD_LEADING
    w, h = card_size(lines, scale)
    rect = QRectF(anchor.x(), anchor.y(), w, h)

    painter.save()
    painter.setFont(font)
    if background is not None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 4.0 * scale, 4.0 * scale)
    painter.setPen(_INK)
    y = rect.top() + pad + metrics.ascent()
    for line in lines:
        painter.drawText(QPointF(rect.left() + pad, y), line)
        y += leading
    painter.restore()
    return rect


def notes_outline(shape: MaskShape, ctrl: List[QPointF], content: QRectF) -> List[QPointF]:
    """The region that a mask marks on the work print.

    A card edge has no boundary, so it shows as the half plane that gets the full
    exposure. The ramp is a soft edge, like the feather of a polygon, and stays undrawn.
    """
    if shape != MaskShape.GRADIENT:
        return [QPointF(x, y) for x, y in outline_points(shape, [(p.x(), p.y()) for p in ctrl])]
    a, b = ctrl[0], ctrl[1]
    dx, dy = b.x() - a.x(), b.y() - a.y()
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return []
    reach = math.hypot(content.width(), content.height())
    ux, uy = dx / length * reach, dy / length * reach
    px, py = -dy / length * reach, dx / length * reach
    far = QPointF(a.x() - ux, a.y() - uy)
    return [
        QPointF(a.x() - px, a.y() - py),
        QPointF(a.x() + px, a.y() + py),
        QPointF(far.x() + px, far.y() + py),
        QPointF(far.x() - px, far.y() - py),
    ]


def mapped_polys(local: LocalAdjustmentsConfig, uv_grid: Optional[np.ndarray], content: QRectF, grade: float = 0.0) -> List[Poly]:
    """The mask outlines inside `content`, with their notes."""
    polys: List[Poly] = []
    if uv_grid is None:
        return polys
    for mask, note in zip(local.masks, mask_notes(local, grade)):
        if len(mask.vertices) < min_points(mask.shape):
            continue
        ctrl = [CoordinateMapping.map_raw_to_viewport(rx, ry, uv_grid) for rx, ry in mask.vertices]
        screen = [QPointF(content.x() + nx * content.width(), content.y() + ny * content.height()) for nx, ny in ctrl]
        polys.append((notes_outline(mask.shape, screen, content), note))
    return polys


def notes_sheet(
    frame: QImage,
    content_rect: Optional[Tuple[int, int, int, int]],
    local: LocalAdjustmentsConfig,
    uv_grid: Optional[np.ndarray],
    lines: Sequence[str],
    grade: float = 0.0,
) -> QImage:
    """The rendered frame with the map drawn on it and the recipe in a band below."""
    scale = max(1.0, max(frame.width(), frame.height()) / _SHEET_SCALE_REF)
    _, card_h = card_size(lines, scale)
    band = int(round(card_h)) if lines else 0

    sheet = QImage(frame.width(), frame.height() + band, QImage.Format.Format_RGB32)
    sheet.fill(_BAND_BG)
    painter = QPainter(sheet)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawImage(0, 0, frame)
        if content_rect:
            off_x, off_y, cw, ch = content_rect
            content = QRectF(off_x, off_y, cw, ch)
        else:
            content = QRectF(0, 0, frame.width(), frame.height())
        paint_map(painter, mapped_polys(local, uv_grid, content, grade), scale)
        if lines:
            paint_card(painter, QPointF(_CARD_PAD_PX * scale, frame.height()), lines, scale, background=None)
    finally:
        painter.end()
    return sheet
