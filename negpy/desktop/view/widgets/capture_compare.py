"""Large pictures for Assign Capture Times: the Compare row (the frames around one gap with the
phone photo between them), a picture box for the selected frame, and Quick Look over the dialog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QEvent, QObject, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.styles.templates import with_alpha
from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.system.i18n import tr

LINE_H = 18
GAP = THEME.space_lg
# Width share and height share of each role in the Compare row.
_WEIGHT = {"context": 0.6, "near": 0.6, "placed": 0.9, "frame": 2.0, "current": 2.0}
_HEIGHT = {"context": 0.5, "near": 0.5, "placed": 0.62, "frame": 1.0, "current": 1.0}


@dataclass(frozen=True)
class CompareItem:
    kind: str  # "frame" or "photo"
    ident: str  # the frame's index as text, or the photo's path
    role: str  # "context", "frame", "current", "placed" (same gap) or "near" (the next gaps)
    left: str = ""
    right: str = ""
    empty: str = ""  # shown in place of a picture: the start or end of the roll


def fit(pixmap: QPixmap, box: QRect, ratio: float) -> tuple[QRect, QPixmap]:
    """The pixmap scaled to fit the box, centered, for a screen with this pixel ratio."""
    size = QSize(max(1, round(box.width() * ratio)), max(1, round(box.height() * ratio)))
    scaled = pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    scaled.setDevicePixelRatio(ratio)
    w, h = round(scaled.width() / ratio), round(scaled.height() / ratio)
    return QRect(box.x() + (box.width() - w) // 2, box.y() + (box.height() - h) // 2, w, h), scaled


class CompareRow(QWidget):
    """A piece of the strip, large: each item takes a share of the width by its role, and the
    pictures stand on one line with their captions under them."""

    itemClicked = pyqtSignal(str, str, str)  # kind, ident, role
    quickLookRequested = pyqtSignal(str, str)  # kind, ident

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._items: list[CompareItem] = []
        self._pixmaps: dict[tuple[str, str], QPixmap] = {}
        self._rects: list[QRect] = []
        self._hover = -1
        self._selected: Optional[tuple[str, str]] = None
        self.setMouseTracking(True)
        self.setMinimumHeight(180)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_items(self, items: list[CompareItem], selected: Optional[tuple[str, str]] = None) -> None:
        self._items, self._selected, self._hover = items, selected, -1
        self._layout()
        self.update()

    def items(self) -> list[CompareItem]:
        return list(self._items)

    def set_pixmap(self, kind: str, ident: str, pixmap: QPixmap) -> None:
        self._pixmaps[(kind, ident)] = pixmap
        if any((i.kind, i.ident) == (kind, ident) for i in self._items):
            self.update()

    def hovered_item(self) -> Optional[tuple[str, str]]:
        if 0 <= self._hover < len(self._items) and not self._items[self._hover].empty:
            item = self._items[self._hover]
            return item.kind, item.ident
        return None

    def _layout(self) -> None:
        n = len(self._items)
        if not n:
            self._rects = []
            return
        total = sum(_WEIGHT[i.role] for i in self._items)
        free = max(0, self.width() - GAP * (n - 1))
        area = max(0, self.height() - LINE_H - THEME.space_sm)
        rects, x = [], 0.0
        for item in self._items:
            w = free * _WEIGHT[item.role] / total
            h = round(area * _HEIGHT[item.role])
            rects.append(QRect(round(x), area - h, max(1, round(w)), h))
            x += w + GAP
        self._rects = rects

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._layout()
        super().resizeEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(self.font())
        font.setPixelSize(THEME.font_size_small)
        p.setFont(font)
        ratio = self.devicePixelRatioF()
        caption_y = self.height() - LINE_H
        for index, (item, box) in enumerate(zip(self._items, self._rects)):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(THEME.bg_input))
            p.drawRoundedRect(QRectF(box), THEME.radius_sm, THEME.radius_sm)
            pixmap = self._pixmaps.get((item.kind, item.ident))
            if item.empty:
                p.setPen(QColor(THEME.text_hint))
                p.drawText(box, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, item.empty)
            elif pixmap is not None and not pixmap.isNull():
                p.setOpacity(0.55 if item.role in ("context", "near") and index != self._hover else 1.0)
                target, scaled = fit(pixmap, box, ratio)
                p.drawPixmap(target, scaled)
                p.setOpacity(1.0)
            self._paint_edge(p, item, box, index)
            colors = {"current": THEME.warn_amber, "placed": THEME.warn_amber, "near": THEME.warn_amber, "context": THEME.text_hint}
            color = colors.get(item.role, THEME.text_secondary)
            if self._selected == (item.kind, item.ident) and item.kind == "frame":
                color = THEME.text_primary
            metrics = QFontMetrics(p.font())
            p.setPen(QColor(color))
            half = box.width() // 2 if item.right else box.width()
            p.drawText(
                QRect(box.x(), caption_y, half, LINE_H),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                metrics.elidedText(item.left, Qt.TextElideMode.ElideRight, half),
            )
            if item.right:
                p.drawText(
                    QRect(box.x() + half, caption_y, box.width() - half, LINE_H),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    metrics.elidedText(item.right, Qt.TextElideMode.ElideLeft, box.width() - half),
                )
        p.end()

    def _paint_edge(self, p: QPainter, item: CompareItem, box: QRect, index: int) -> None:
        if item.role == "current":
            pen = QPen(QColor(THEME.warn_amber), 2)
        elif item.role in ("placed", "near"):
            pen = QPen(QColor(THEME.warn_amber), 1)
        elif self._selected == (item.kind, item.ident):
            pen = QPen(QColor(THEME.border_checked_hover), 2)
        elif index == self._hover and not item.empty:
            pen = QPen(QColor(THEME.border_focus), 1)
        else:
            return
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        half = pen.width() / 2
        p.drawRoundedRect(QRectF(box).adjusted(half, half, -half, -half), THEME.radius_sm, THEME.radius_sm)

    def _index_at(self, pos) -> int:
        return next((i for i, r in enumerate(self._rects) if r.contains(pos) and not self._items[i].empty), -1)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        index = self._index_at(event.position().toPoint())
        if index != self._hover:
            self._hover = index
            self.setCursor(Qt.CursorShape.PointingHandCursor if index >= 0 else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self._hover = -1
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        index = self._index_at(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and index >= 0:
            item = self._items[index]
            self.itemClicked.emit(item.kind, item.ident, item.role)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        index = self._index_at(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and index >= 0:
            self.quickLookRequested.emit(self._items[index].kind, self._items[index].ident)


class PictureBox(QWidget):
    """One picture, whole, in a fixed-height box. A click asks for Quick Look."""

    clicked = pyqtSignal()

    def __init__(self, height: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self.setFixedHeight(height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tr("Click to see the frame large"))

    def set_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._pixmap = pixmap
        self.update()

    def pixmap(self) -> Optional[QPixmap]:
        return self._pixmap

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(THEME.bg_input))
        p.drawRoundedRect(QRectF(self.rect()), THEME.radius_sm, THEME.radius_sm)
        if self._pixmap is not None and not self._pixmap.isNull():
            target, scaled = fit(self._pixmap, self.rect(), self.devicePixelRatioF())
            p.drawPixmap(target, scaled)
        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class QuickLook(QWidget):
    """A frame or photo as large as the dialog allows, over the dialog. The dialog handles the keys
    (Space or Esc closes, ← → step); a click beside the picture closes it."""

    closed = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._caption = ""
        self._detail = ""
        self.hide()
        parent.installEventFilter(self)

    def show_picture(self, pixmap: Optional[QPixmap], caption: str, detail: str = "") -> None:
        self._pixmap, self._caption, self._detail = pixmap, caption, detail
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self.update()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def caption(self) -> str:
        return self._caption

    def close_view(self) -> None:
        if self.isVisible():
            self.hide()
            self.closed.emit()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.parentWidget() and event.type() == QEvent.Type.Resize and self.isVisible():
            self.setGeometry(self.parentWidget().rect())
        return False

    def _picture_rect(self) -> QRect:
        margin = THEME.space_2xl
        return self.rect().adjusted(margin, margin, -margin, -(margin + 3 * LINE_H))

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), with_alpha(THEME.mark_shadow, 225))
        box = self._picture_rect()
        if self._pixmap is not None and not self._pixmap.isNull():
            target, scaled = fit(self._pixmap, box, self.devicePixelRatioF())
            p.drawPixmap(target, scaled)
        font = QFont(self.font())
        font.setPixelSize(THEME.font_size_base)
        p.setFont(font)
        y = box.bottom() + THEME.space_md
        for text, color in ((self._caption, THEME.text_primary), (self._detail, THEME.text_hint), (_hint(), THEME.text_hint)):
            if text:
                p.setPen(QColor(color))
                p.drawText(QRect(0, y, self.width(), LINE_H), Qt.AlignmentFlag.AlignCenter, text)
                y += LINE_H
        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self._picture_rect().contains(event.position().toPoint()):
            self.close_view()


def _hint() -> str:
    return tr("Space or Esc closes. Left and right arrows step along the strip.")
