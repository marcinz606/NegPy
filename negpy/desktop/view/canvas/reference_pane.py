from typing import Callable, Optional

from PyQt6.QtCore import QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.canvas.overlay import draw_view_badge
from negpy.desktop.view.styles.templates import icon_button
from negpy.desktop.view.styles.theme import THEME


class ReferencePane(QWidget):
    """A frame pinned beside the canvas, to match the rest of a roll against."""

    closed = pyqtSignal()

    def __init__(self, background: Callable[[], QColor]):
        super().__init__()
        self._background = background
        self._image: Optional[QImage] = None
        self._name = ""
        self.setMinimumWidth(160)
        self.close_btn = icon_button("fa5s.times", "Close the reference")
        self.close_btn.setParent(self)
        self.close_btn.clicked.connect(self.closed)

    @property
    def image(self) -> Optional[QImage]:
        return self._image

    def set_reference(self, image: Optional[QImage], name: str = "") -> None:
        self._image = image
        self._name = name
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.close_btn.move(self.width() - self.close_btn.width() - THEME.space_lg, THEME.space_lg)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._background())
        if self._image is None or self._image.isNull():
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        area = QRectF(self.rect()).adjusted(THEME.space_2xl, THEME.space_2xl, -THEME.space_2xl, -THEME.space_2xl)
        scale = min(area.width() / self._image.width(), area.height() / self._image.height())
        width, height = self._image.width() * scale, self._image.height() * scale
        painter.drawImage(QRectF(area.center().x() - width / 2, area.center().y() - height / 2, width, height), self._image)
        text = f"REFERENCE · {self._name}" if self._name else "REFERENCE"
        draw_view_badge(painter, text, THEME.space_xl, THEME.space_xl, painter.fontMetrics().horizontalAdvance(text) + 24.0)
