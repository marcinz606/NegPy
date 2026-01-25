import numpy as np
import sys
from typing import Optional, Tuple
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QImage, QMouseEvent, QColor, QPen
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QSize
from src.desktop.converters import ImageConverter
from src.desktop.session import ToolMode, AppState
from src.desktop.view.widgets.overlays import ImageInfoOverlay
from src.desktop.view.styles.theme import THEME
from src.kernel.system.config import APP_CONFIG


class CanvasOverlay(QWidget):
    """
    Transparent overlay for image interaction (crop, guides) and CPU rendering fallback.
    """

    clicked = pyqtSignal(float, float)
    crop_completed = pyqtSignal(float, float, float, float)

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._qimage: Optional[QImage] = None
        self._current_size: Optional[Tuple[int, int]] = None
        self._display_rect: QRectF = QRectF()
        self._content_rect: Optional[Tuple[int, int, int, int]] = None

        self.overlay = ImageInfoOverlay(self)

        # Interaction State
        self._crop_active: bool = False
        self._crop_p1: Optional[QPointF] = None
        self._crop_p2: Optional[QPointF] = None
        self._tool_mode: ToolMode = ToolMode.NONE
        self._mouse_pos: QPointF = QPointF()

        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_tool_mode(self, mode: ToolMode) -> None:
        self._tool_mode = mode
        if mode != ToolMode.CROP_MANUAL:
            self._crop_p1 = None
            self._crop_p2 = None
        self.update()

    def update_buffer(
        self,
        buffer: Optional[np.ndarray],
        color_space: str,
        content_rect: Optional[Tuple[int, int, int, int]] = None,
        gpu_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Updates the coordinate system reference.
        If buffer is provided, it's CPU mode. If gpu_size is provided, it's GPU mode.
        """
        self._content_rect = content_rect
        if buffer is not None:
            self._qimage = ImageConverter.to_qimage(buffer, color_space)
            self._current_size = (self._qimage.width(), self._qimage.height())
        else:
            self._qimage = None
            self._current_size = gpu_size
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        # macOS trail fix: aggressive clear the dirty region with transparency
        if sys.platform == "darwin":
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.fillRect(event.rect(), Qt.GlobalColor.transparent)
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        size = None
        if self._qimage:
            size = self._qimage.size()
        elif self._current_size:
            size = QSize(self._current_size[0], self._current_size[1])

        if size:
            widget_size = self.size()
            ratio = min(
                widget_size.width() / size.width(),
                widget_size.height() / size.height(),
            )
            new_w, new_h = int(size.width() * ratio), int(size.height() * ratio)
            x = (widget_size.width() - new_w) // 2
            y = (widget_size.height() - new_h) // 2
            self._display_rect = QRectF(x, y, new_w, new_h)

            if self._qimage:
                painter.drawImage(self._display_rect, self._qimage)

        self._draw_widget_ui(painter)

    def _draw_widget_ui(self, painter: QPainter) -> None:
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

        if self._tool_mode != ToolMode.NONE and self._display_rect.contains(
            self._mouse_pos
        ):
            if self._tool_mode == ToolMode.DUST_PICK:
                self._draw_brush(painter)
            else:
                painter.setPen(QPen(QColor(255, 255, 255, 80), 1, Qt.PenStyle.DotLine))
                painter.drawLine(
                    QPointF(self._display_rect.x(), self._mouse_pos.y()),
                    QPointF(self._display_rect.right(), self._mouse_pos.y()),
                )
                painter.drawLine(
                    QPointF(self._mouse_pos.x(), self._display_rect.y()),
                    QPointF(self._mouse_pos.x(), self._display_rect.bottom()),
                )

    def _draw_brush(self, painter: QPainter) -> None:
        conf = self.state.config.retouch

        max_screen_dim = max(self._display_rect.width(), self._display_rect.height())
        radius = (
            conf.manual_dust_size / APP_CONFIG.preview_render_size
        ) * max_screen_dim

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(Qt.GlobalColor.white, 1.0, Qt.PenStyle.SolidLine))
        painter.drawEllipse(self._mouse_pos, radius, radius)

        accent = QColor(THEME.accent_primary)
        accent.setAlpha(60)
        painter.setBrush(accent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self._mouse_pos, radius, radius)

    def _map_to_image_coords(self, pos: QPointF) -> Optional[Tuple[float, float]]:
        if self._display_rect.isEmpty() or not self._display_rect.contains(pos):
            return None
        nb_x = (pos.x() - self._display_rect.x()) / self._display_rect.width()
        nb_y = (pos.y() - self._display_rect.y()) / self._display_rect.height()

        if self._content_rect and self._current_size:
            bw, bh = self._current_size
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
        self.overlay.resize(self.size())
        self.update()

    def update_overlay(
        self, filename: str, res: str, colorspace: str, extra: str
    ) -> None:
        self.overlay.update_info(filename, res, colorspace, extra)
