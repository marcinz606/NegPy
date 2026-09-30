"""The triage badge a frame carries wherever the roll is shown: a check for a keeper, a cross
for a frame left out."""

from PyQt6.QtCore import QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen

MARK_FILL = QColor(183, 28, 28, 150)  # THEME.accent_primary at ~60% alpha


def draw_mark_badge(painter: QPainter, img_rect: QRect | QRectF, check: bool) -> None:
    """The badge in the picture's bottom-right corner."""
    rect = img_rect.toRect() if isinstance(img_rect, QRectF) else img_rect
    r = 9
    cx, cy = rect.right() - r - 4, rect.bottom() - r - 4
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(MARK_FILL)
    painter.drawEllipse(QRect(cx - r, cy - r, 2 * r, 2 * r))
    painter.setPen(QPen(QColor(255, 255, 255, 230), 2, cap=Qt.PenCapStyle.RoundCap))
    if check:
        painter.drawLine(cx - 4, cy, cx - 1, cy + 3)
        painter.drawLine(cx - 1, cy + 3, cx + 4, cy - 3)
    else:
        painter.drawLine(cx - 3, cy - 3, cx + 3, cy + 3)
        painter.drawLine(cx + 3, cy - 3, cx - 3, cy + 3)
