"""Triage badge: a check for a keeper, a cross for a rejected frame."""

from PyQt6.QtCore import QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen

from negpy.desktop.view.styles.templates import with_alpha
from negpy.desktop.view.styles.theme import THEME

# Every badge on a frame sits on this chip. It is neutral, as red there would read as the selection or a failure.
CHIP_FILL = with_alpha(THEME.mark_shadow, 190)
CHIP_RING = with_alpha(THEME.mark_photo, 90)


def draw_mark_badge(painter: QPainter, img_rect: QRect | QRectF, check: bool) -> None:
    rect = img_rect.toRect() if isinstance(img_rect, QRectF) else img_rect
    r = 9
    cx, cy = rect.right() - r - 4, rect.bottom() - r - 4
    painter.setPen(QPen(CHIP_RING, 1))
    painter.setBrush(CHIP_FILL)
    painter.drawEllipse(QRect(cx - r, cy - r, 2 * r, 2 * r))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor(THEME.text_primary), 2, cap=Qt.PenCapStyle.RoundCap))
    if check:
        painter.drawLine(cx - 4, cy, cx - 1, cy + 3)
        painter.drawLine(cx - 1, cy + 3, cx + 4, cy - 3)
    else:
        painter.drawLine(cx - 3, cy - 3, cx + 3, cy + 3)
        painter.drawLine(cx + 3, cy - 3, cx - 3, cy + 3)
