from negpy.kernel.system.i18n import tr
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.styles.theme import THEME

_LONG_EDGE_PX = 160
_WIDTH_PX = 200  # wider than the map, so the zoom caption fits under a portrait frame too
_CAPTION_H = 20
_IDLE_MS = 1500
_IDLE_OPACITY = 0.45


def visible_part(frame: QRectF, viewport: QRectF) -> QRectF:
    """The part of `frame` (screen rect of the whole buffer) inside `viewport`, frame-normalized."""
    if frame.isEmpty():
        return QRectF()
    seen = frame.intersected(viewport)
    if seen.isEmpty():
        return QRectF()
    return QRectF(
        (seen.x() - frame.x()) / frame.width(),
        (seen.y() - frame.y()) / frame.height(),
        seen.width() / frame.width(),
        seen.height() / frame.height(),
    )


def map_size(aspect: float) -> QSize:
    """Map size for a frame of `aspect` (width / height), long edge fixed."""
    if aspect >= 1.0:
        return QSize(_LONG_EDGE_PX, max(1, round(_LONG_EDGE_PX / aspect)))
    return QSize(max(1, round(_LONG_EDGE_PX * aspect)), _LONG_EDGE_PX)


class CanvasNavigator(QWidget):
    """Corner map of the frame while zoomed in: the visible part outlined, the zoom under it.
    A press or drag on the map centers the view on that spot. Dims when left alone."""

    center_requested = pyqtSignal(float, float)  # frame-normalized point

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._image_key: Optional[int] = None
        self._map_size = map_size(1.5)
        self._visible = QRectF()
        self._caption = ""
        self._opacity = 1.0
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setInterval(_IDLE_MS)
        self._idle_timer.timeout.connect(self._dim)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tr("Navigator: click or drag to move the view"))
        self.hide()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(_WIDTH_PX, self._map_size.height() + _CAPTION_H)

    def set_view(self, image: Optional[QImage], aspect: float, visible: QRectF) -> None:
        size = map_size(aspect)
        key = None if image is None else image.cacheKey()
        if size != self._map_size or key != self._image_key:
            self._map_size = size
            self._image_key = key
            self._image = (
                None
                if image is None or image.isNull()
                else image.scaled(size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        if visible != self._visible:
            self._visible = QRectF(visible)
            self.wake()
        self.update()

    def set_caption(self, text: str) -> None:
        if text != self._caption:
            self._caption = text
            self.update()

    def wake(self) -> None:
        self._opacity = 1.0
        self._idle_timer.start()
        self.update()

    def _dim(self) -> None:
        if not self.underMouse():
            self._opacity = _IDLE_OPACITY
            self.update()

    def _map_rect(self) -> QRectF:
        w = float(self._map_size.width())
        return QRectF(self.width() - w, 0.0, w, float(self._map_size.height()))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self._opacity)
        rect = self._map_rect()
        if self._image is not None:
            painter.drawImage(rect, self._image)
        else:
            painter.fillRect(rect, QColor(THEME.canvas_bg_dark_grey))

        v = QRectF(
            rect.x() + self._visible.x() * rect.width(),
            rect.y() + self._visible.y() * rect.height(),
            self._visible.width() * rect.width(),
            self._visible.height() * rect.height(),
        )
        # Dim what the canvas does not show, so the outline reads over any picture.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 130))
        painter.drawRect(QRectF(rect.left(), rect.top(), rect.width(), v.top() - rect.top()))
        painter.drawRect(QRectF(rect.left(), v.bottom(), rect.width(), rect.bottom() - v.bottom()))
        painter.drawRect(QRectF(rect.left(), v.top(), v.left() - rect.left(), v.height()))
        painter.drawRect(QRectF(v.right(), v.top(), rect.right() - v.right(), v.height()))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1.5))
        painter.drawRect(v)
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1.0))
        painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))

        if self._caption:
            painter.setOpacity(1.0)
            w = min(float(self.width()), painter.fontMetrics().horizontalAdvance(self._caption) + 2 * THEME.space_lg)
            cap = QRectF(self.width() - w, rect.bottom() + THEME.space_xs, w, float(_CAPTION_H - THEME.space_xs))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 140))
            painter.drawRoundedRect(cap, THEME.radius_md, THEME.radius_md)
            painter.setPen(QColor(THEME.text_secondary))
            painter.drawText(cap, Qt.AlignmentFlag.AlignCenter, self._caption)

    def _request_center(self, pos: QPointF) -> None:
        rect = self._map_rect()
        nx = min(max((pos.x() - rect.x()) / rect.width(), 0.0), 1.0)
        ny = min(max((pos.y() - rect.y()) / rect.height(), 0.0), 1.0)
        self.center_requested.emit(nx, ny)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self._map_rect().contains(event.position()):
            event.ignore()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.wake()
            self._request_center(event.position())
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._request_center(event.position())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        event.accept()

    def wheelEvent(self, event) -> None:  # noqa: N802
        # The canvas below zooms on the wheel; the map is too small to own it.
        event.ignore()

    def enterEvent(self, event) -> None:  # noqa: N802
        self.wake()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._idle_timer.start()
        super().leaveEvent(event)
