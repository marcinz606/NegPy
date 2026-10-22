"""Flat rows a delegate paints, for the History steps, the Library's rolls and the metadata presets:
a fill for the hovered or selected row, the chip of a checked toggle on the current one, a pill, and ⋮ on hover."""

from typing import Optional

import qtawesome as qta
from PyQt6.QtCore import QPoint, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QListWidget

from negpy.desktop.view.styles.theme import THEME

ICON = 16


def more_rect(row: QRect) -> QRect:
    """Where a row draws its ⋮."""
    return QRect(row.right() - THEME.space_lg - ICON, row.center().y() - ICON // 2, ICON, ICON)


def more_press(view, event) -> Optional[QPoint]:
    """The press position when it lands on a row's ⋮, which opens that row's menu instead of picking it."""
    pos = event.position().toPoint()
    item = view.itemAt(pos)
    return pos if item is not None and more_rect(view.visualItemRect(item)).contains(pos) else None


def paint_fill(p: QPainter, rect: QRect, *, current: bool, hover: bool, selected: bool = False) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    box = QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5)
    if current:
        p.setPen(QPen(QColor(THEME.border_checked_hover if hover else THEME.border_checked), 1))
        p.setBrush(QColor(THEME.bg_checked_hover if hover else THEME.bg_checked))
        p.drawRoundedRect(box, THEME.radius_md, THEME.radius_md)
    elif selected or hover:
        p.setBrush(QColor(THEME.bg_selected if selected else THEME.bg_hover))
        p.drawRoundedRect(box, THEME.radius_md, THEME.radius_md)


def paint_pill(p: QPainter, right: int, center_y: float, text: str, font: QFont, *, accent: bool) -> int:
    """A rounded tag that ends at right; returns its left edge."""
    metrics = QFontMetrics(font)
    width = metrics.horizontalAdvance(text) + 2 * THEME.space_md
    height = metrics.height() + 2
    pill = QRectF(right - width, center_y - height / 2, width, height)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QColor(THEME.border_checked_hover if accent else THEME.border_color))
    p.drawRoundedRect(pill, height / 2, height / 2)
    p.setFont(font)
    p.setPen(QColor(THEME.text_primary if accent else THEME.text_hint))
    p.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)
    return int(pill.left())


def paint_more(p: QPainter, row: QRect) -> None:
    qta.icon("mdi6.dots-vertical", color=THEME.text_hint).paint(p, more_rect(row))


def two_line_rects(rect: QRect, x: int, right: int, name: QFontMetrics, detail: QFontMetrics) -> tuple[QRect, QRect]:
    """A name's line over a detail line, centered as a pair in the row."""
    top = rect.center().y() - (name.height() + detail.height() + THEME.space_xs) // 2
    return QRect(x, top, right - x, name.height()), QRect(x, top + name.height() + THEME.space_xs, right - x, detail.height())


class RowList(QListWidget):
    """Flat rows sized to fit, since the tab scrolls; a press on a row's ⋮, or a right click,
    asks for that row's menu instead of picking it."""

    menu_requested = pyqtSignal(QPoint)

    def __init__(self, delegate_cls, row_height: int) -> None:
        super().__init__()
        self._row_height = row_height
        self.setObjectName("flat_rows")
        self.setItemDelegate(delegate_cls(self))
        self.setMouseTracking(True)
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menu_requested)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = more_press(self, event)
        if pos is not None:
            self.menu_requested.emit(pos)
            return
        super().mousePressEvent(event)

    def fit(self) -> None:
        self.setFixedHeight(self.count() * self._row_height + 2 * self.frameWidth())
