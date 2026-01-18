import numpy as np
from typing import Optional, Tuple
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QImage, QMouseEvent, QColor, QPen
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from src.desktop.converters import ImageConverter
from src.desktop.session import ToolMode, AppState


class ImageCanvas(QWidget):
    """
    High-performance image display with high-precision overlay rendering.
    """

    clicked = pyqtSignal(float, float)
    crop_completed = pyqtSignal(float, float, float, float)

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._qimage: Optional[QImage] = None
        self._display_rect: QRectF = QRectF()
        self._content_rect: Optional[Tuple[int, int, int, int]] = None

        # Interaction State
        self._crop_active: bool = False
        self._crop_p1: Optional[QPointF] = None
        self._crop_p2: Optional[QPointF] = None
        self._tool_mode: ToolMode = ToolMode.NONE
        self._mouse_pos: QPointF = QPointF()

        self.setMouseTracking(True)

    def set_tool_mode(self, mode: ToolMode) -> None:
        self._tool_mode = mode
        if mode != ToolMode.CROP_MANUAL:
            self._crop_p1 = None
            self._crop_p2 = None
        self.update()

    def update_buffer(
        self,
        buffer: np.ndarray,
        color_space: str,
        content_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self._content_rect = content_rect
        self._qimage = ImageConverter.to_qimage(buffer, color_space)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # 1. Clear Widget
        painter.fillRect(self.rect(), QColor(15, 15, 15))

        if not self._qimage:
            return

        # 2. Calculate Layout
        img_size = self._qimage.size()
        widget_size = self.size()
        ratio = min(
            widget_size.width() / img_size.width(),
            widget_size.height() / img_size.height(),
        )
        new_w, new_h = int(img_size.width() * ratio), int(img_size.height() * ratio)
        x = (widget_size.width() - new_w) // 2
        y = (widget_size.height() - new_h) // 2
        self._display_rect = QRectF(x, y, new_w, new_h)

        # 3. Draw Base Image
        painter.drawImage(self._display_rect, self._qimage)

        # 4. Draw Widget-Space UI (Crop / Guides)
        self._draw_widget_ui(painter)

    def _draw_widget_ui(self, painter: QPainter) -> None:
        # Active Crop
        if self._crop_p1 and self._crop_p2:
            rect = (
                QRectF(self._crop_p1, self._crop_p2)
                .normalized()
                .intersected(self._display_rect)
            )
            painter.setBrush(QColor(0, 0, 0, 180))
            painter.setPen(Qt.PenStyle.NoPen)
            d = self._display_rect
            painter.drawRect(QRectF(d.x(), d.y(), d.width(), rect.y() - d.y()))
            painter.drawRect(
                QRectF(d.x(), rect.bottom(), d.width(), d.bottom() - rect.bottom())
            )
            painter.drawRect(QRectF(d.x(), rect.y(), rect.x() - d.x(), rect.height()))
            painter.drawRect(
                QRectF(rect.right(), rect.y(), d.right() - rect.right(), rect.height())
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine))
            painter.drawRect(rect)

        # Crosshair
        if self._tool_mode != ToolMode.NONE and self._display_rect.contains(
            self._mouse_pos
        ):
            painter.setPen(QPen(QColor(255, 255, 255, 80), 1, Qt.PenStyle.DotLine))
            painter.drawLine(
                QPointF(self._display_rect.x(), self._mouse_pos.y()),
                QPointF(self._display_rect.right(), self._mouse_pos.y()),
            )
            painter.drawLine(
                QPointF(self._mouse_pos.x(), self._display_rect.y()),
                QPointF(self._mouse_pos.x(), self._display_rect.bottom()),
            )

    def _map_to_image_coords(self, pos: QPointF) -> Optional[Tuple[float, float]]:
        if self._display_rect.isEmpty() or not self._display_rect.contains(pos):
            return None
        nb_x = (pos.x() - self._display_rect.x()) / self._display_rect.width()
        nb_y = (pos.y() - self._display_rect.y()) / self._display_rect.height()
        if self._content_rect:
            bw, bh = self._qimage.width(), self._qimage.height()
            cx, cy, cw, ch = self._content_rect
            nx_min, ny_min = cx / bw, cy / bh
            nx_max, ny_max = (cx + cw) / bw, (cy + ch) / bh
            nx = (nb_x - nx_min) / max(1e-5, (nx_max - nx_min))
            ny = (nb_y - ny_min) / max(1e-5, (ny_max - ny_min))
            return float(np.clip(nx, 0, 1)), float(np.clip(ny, 0, 1))
        return float(nb_x), float(nb_y)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        coords = self._map_to_image_coords(event.position())
        if coords:
            self.clicked.emit(*coords)
            if self._tool_mode == ToolMode.CROP_MANUAL:
                self._crop_active = True
                self._crop_p1, self._crop_p2 = event.position(), event.position()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse_pos = event.position()
        if self._crop_active:
            pos = event.position()
            ratio_str = self.state.config.geometry.autocrop_ratio

            if ratio_str == "Free":
                # Constrain p2 to display_rect
                nx = max(
                    self._display_rect.left(), min(self._display_rect.right(), pos.x())
                )
                ny = max(
                    self._display_rect.top(), min(self._display_rect.bottom(), pos.y())
                )
                self._crop_p2 = QPointF(nx, ny)
            else:
                try:
                    # Constrain p2 to respect aspect ratio relative to p1
                    w_r, h_r = map(float, ratio_str.split(":"))
                    target_ratio = w_r / h_r

                    dx = pos.x() - self._crop_p1.x()
                    dy = pos.y() - self._crop_p1.y()

                    if abs(dx) > abs(dy) * target_ratio:
                        # DX is dominant
                        dy = (abs(dx) / target_ratio) * (1 if dy >= 0 else -1)
                    else:
                        # DY is dominant
                        dx = (abs(dy) * target_ratio) * (1 if dx >= 0 else -1)

                    # Ensure p2 stays within display_rect while keeping ratio
                    limit_x = (
                        self._display_rect.left()
                        if dx < 0
                        else self._display_rect.right()
                    )
                    limit_y = (
                        self._display_rect.top()
                        if dy < 0
                        else self._display_rect.bottom()
                    )

                    scale_x = (
                        abs(limit_x - self._crop_p1.x()) / abs(dx) if dx != 0 else 1.0
                    )
                    scale_y = (
                        abs(limit_y - self._crop_p1.y()) / abs(dy) if dy != 0 else 1.0
                    )

                    scale = min(scale_x, scale_y)
                    if scale < 1.0:
                        dx *= scale
                        dy *= scale

                    self._crop_p2 = QPointF(
                        self._crop_p1.x() + dx, self._crop_p1.y() + dy
                    )
                except Exception:
                    self._crop_p2 = pos
            self.update()
        else:
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._crop_active:
            r = (
                QRectF(self._crop_p1, self._crop_p2)
                .normalized()
                .intersected(self._display_rect)
            )
            if r.width() > 5 and r.height() > 5:
                c1, c2 = (
                    self._map_to_image_coords(r.topLeft()),
                    self._map_to_image_coords(r.bottomRight()),
                )
                if c1 and c2:
                    self.crop_completed.emit(c1[0], c1[1], c2[0], c2[1])
            self._crop_active = False
            self._crop_p1, self._crop_p2 = None, None
            self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update()
