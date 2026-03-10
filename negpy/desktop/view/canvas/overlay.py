import numpy as np
import sys
from typing import Optional, Tuple
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QImage, QMouseEvent, QColor, QPen
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QSize
from negpy.desktop.converters import ImageConverter
from negpy.desktop.session import ToolMode, AppState
from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.system.config import APP_CONFIG


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
        self._content_rect: Optional[Tuple[int, int, int, int]] = None

        # Interaction State
        self._crop_active: bool = False
        self._crop_p1: Optional[QPointF] = None
        self._crop_p2: Optional[QPointF] = None
        self._tool_mode: ToolMode = ToolMode.NONE
        self._mouse_pos: QPointF = QPointF()

        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

        self._view_rect: QRectF = QRectF()

        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        if sys.platform == "win32":
            self.setAttribute(Qt.WidgetAttribute.WA_StaticContents, False)

    def set_transform(self, zoom: float, px: float, py: float) -> None:
        self.zoom_level = zoom
        self.pan_x = px
        self.pan_y = py
        self._recalc_view_rect()
        self.update()

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
        self._content_rect = content_rect
        if buffer is not None:
            self._qimage = ImageConverter.to_qimage(buffer, color_space)
            self._current_size = (self._qimage.width(), self._qimage.height())
        else:
            self._qimage = None
            self._current_size = gpu_size

        self._recalc_view_rect()
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._recalc_view_rect()
        self.update()

    def _recalc_view_rect(self) -> None:
        size = None
        if self._qimage:
            size = self._qimage.size()
        elif self._current_size:
            size = QSize(self._current_size[0], self._current_size[1])

        if not size:
            self._view_rect = QRectF()
            return

        # No margins - use full widget dimensions
        w, h = self.width(), self.height()
        img_w, img_h = size.width(), size.height()

        scale_fit = min(w / img_w, h / img_h)
        total_scale = scale_fit * self.zoom_level

        final_w = img_w * total_scale
        final_h = img_h * total_scale

        center_x = (w / 2) + (self.pan_x * w)
        center_y = (h / 2) + (self.pan_y * h)

        self._view_rect = QRectF(center_x - (final_w / 2), center_y - (final_h / 2), final_w, final_h)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        if sys.platform in ("darwin", "win32"):
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.fillRect(event.rect(), Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if not self._view_rect.isEmpty():
            if self._qimage:
                painter.drawImage(self._view_rect, self._qimage)

        self._draw_ui(painter)

    def _draw_ui(self, painter: QPainter) -> None:
        if self._view_rect.isEmpty():
            return

        visible_rect = self._view_rect

        if self._crop_active and self._crop_p1 and self._crop_p2:
            rect = QRectF(self._crop_p1, self._crop_p2).normalized().intersected(visible_rect)

            painter.setBrush(QColor(0, 0, 0, 180))
            painter.setPen(Qt.PenStyle.NoPen)
            d = visible_rect

            painter.drawRect(d.intersected(QRectF(d.x(), d.y(), d.width(), rect.y() - d.y())))
            painter.drawRect(d.intersected(QRectF(d.x(), rect.bottom(), d.width(), d.bottom() - rect.bottom())))
            painter.drawRect(d.intersected(QRectF(d.x(), rect.y(), rect.x() - d.x(), rect.height())))
            painter.drawRect(d.intersected(QRectF(rect.right(), rect.y(), d.right() - rect.right(), rect.height())))

            pen = QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen)
            painter.drawRect(rect)

        if self._tool_mode != ToolMode.NONE and visible_rect.contains(self._mouse_pos):
            if self._tool_mode == ToolMode.DUST_PICK:
                self._draw_brush(painter)
            else:
                pen = QPen(QColor(255, 255, 255, 80), 1, Qt.PenStyle.DotLine)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawLine(QPointF(visible_rect.x(), self._mouse_pos.y()), QPointF(visible_rect.right(), self._mouse_pos.y()))
                painter.drawLine(QPointF(self._mouse_pos.x(), visible_rect.top()), QPointF(self._mouse_pos.x(), visible_rect.bottom()))

    def _draw_brush(self, painter: QPainter) -> None:
        conf = self.state.config.retouch
        max_screen_dim = max(self._view_rect.width(), self._view_rect.height())
        radius = (conf.manual_dust_size / (2.0 * APP_CONFIG.preview_render_size)) * max_screen_dim

        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(Qt.GlobalColor.white, 1.0, Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawEllipse(self._mouse_pos, radius, radius)

        accent = QColor(THEME.accent_primary)
        accent.setAlpha(60)
        painter.setBrush(accent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self._mouse_pos, radius, radius)

    def _map_to_image_coords(self, screen_pos: QPointF) -> Optional[Tuple[float, float]]:
        if self._view_rect.isEmpty() or not self._view_rect.contains(screen_pos):
            return None

        nb_x = (screen_pos.x() - self._view_rect.x()) / self._view_rect.width()
        nb_y = (screen_pos.y() - self._view_rect.y()) / self._view_rect.height()

        return float(np.clip(nb_x, 0, 1)), float(np.clip(nb_y, 0, 1))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self.zoom_level > 1.0 and self._tool_mode == ToolMode.NONE
        ):
            self.parent()._is_panning = True
            self.parent()._last_mouse_pos = event.position()
            self.parent().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        coords = self._map_to_image_coords(event.position())
        if coords:
            self.clicked.emit(*coords)
            if self._tool_mode == ToolMode.CROP_MANUAL:
                self._crop_active = True
                self._crop_p1, self._crop_p2 = event.position(), event.position()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse_pos = event.position()

        if self.parent()._is_panning:
            delta = event.position() - self.parent()._last_mouse_pos
            self.parent()._last_mouse_pos = event.position()
            self.parent().pan_offset += QPointF(delta.x() / self.width(), delta.y() / self.height())
            self.parent()._sync_transform()
            event.accept()
            return

        if self._crop_active:
            ratio_str = self.state.config.geometry.autocrop_ratio
            if ratio_str == "Free":
                self._crop_p2 = event.position()
            else:
                try:
                    w_r, h_r = map(float, ratio_str.split(":"))
                    target_ratio = w_r / h_r
                    dx = event.position().x() - self._crop_p1.x()
                    dy = event.position().y() - self._crop_p1.y()
                    if abs(dx) > abs(dy) * target_ratio:
                        dy = (abs(dx) / target_ratio) * (1 if dy >= 0 else -1)
                    else:
                        dx = (abs(dy) * target_ratio) * (1 if dx >= 0 else -1)
                    self._crop_p2 = QPointF(self._crop_p1.x() + dx, self._crop_p1.y() + dy)
                except Exception:
                    self._crop_p2 = event.position()
            self.update()
        else:
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.parent()._is_panning:
            self.parent()._is_panning = False
            self.parent().setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        if self._crop_active:
            r = QRectF(self._crop_p1, self._crop_p2).normalized()
            r = r.intersected(self._view_rect)

            if r.width() > 5 and r.height() > 5:
                c1 = self._map_to_image_coords(r.topLeft())
                c2 = self._map_to_image_coords(r.bottomRight())
                if c1 and c2:
                    self.crop_completed.emit(c1[0], c1[1], c2[0], c2[1])
            self._crop_active = False
            self._crop_p1, self._crop_p2 = None, None
            self.update()

    def update_overlay(self, filename: str, res: str, colorspace: str, extra: str, edits: int = 0) -> None:
        self.update()
