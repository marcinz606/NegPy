import sys
from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from negpy.desktop.converters import ImageConverter
from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.styles.theme import THEME
from negpy.features.geometry.logic import translate_manual_crop_rect
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.view.coordinate_mapping import CoordinateMapping


class CanvasOverlay(QWidget):
    """
    Transparent overlay for image interaction (crop, guides) and CPU rendering fallback.
    """

    clicked = pyqtSignal(float, float)
    crop_completed = pyqtSignal(float, float, float, float)
    crop_translated = pyqtSignal(float, float, float, float)
    cursor_moved = pyqtSignal(float, float)
    cursor_left = pyqtSignal()

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
        self._move_active: bool = False
        self._move_press_raw: Optional[Tuple[float, float]] = None
        self._move_orig_rect: Optional[Tuple[float, float, float, float]] = None
        self._move_last_emitted: Optional[Tuple[float, float, float, float]] = None
        self._move_uv_grid: Optional[np.ndarray] = None
        self._tool_mode: ToolMode = ToolMode.NONE
        self._mouse_pos: QPointF = QPointF()

        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

        self._view_rect: QRectF = QRectF()

        self._buffer_overlay_ratio: float = 0.0
        self._buffer_overlay_visible: bool = False
        self._buffer_hide_timer = QTimer(self)
        self._buffer_hide_timer.setSingleShot(True)
        self._buffer_hide_timer.timeout.connect(self._hide_buffer_overlay)

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

    def show_analysis_buffer(self, ratio: float) -> None:
        self._buffer_overlay_ratio = max(0.0, min(ratio, 0.3))
        self._buffer_overlay_visible = True
        self._buffer_hide_timer.start(1000)
        self.update()

    def _hide_buffer_overlay(self) -> None:
        self._buffer_overlay_visible = False
        self.update()

    def set_tool_mode(self, mode: ToolMode) -> None:
        self._tool_mode = mode
        if mode != ToolMode.CROP_MANUAL:
            self._crop_p1 = None
            self._crop_p2 = None
        if mode != ToolMode.CROP_MOVE:
            self._move_active = False
            self._move_press_raw = None
            self._move_orig_rect = None
            self._move_last_emitted = None
            self._move_uv_grid = None
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

        if size is None or size.isNull():
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

        parent_bg = getattr(self.parent(), "_bg_color", QColor("#050505"))
        if not getattr(self.parent(), "gpu_widget", None) or not self.parent().gpu_widget.isVisible():
            painter.fillRect(event.rect(), parent_bg)

        if sys.platform in ("darwin", "win32"):
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            if getattr(self.parent(), "gpu_widget", None) and self.parent().gpu_widget.isVisible():
                painter.fillRect(event.rect(), Qt.GlobalColor.transparent)
            else:
                painter.fillRect(event.rect(), parent_bg)
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

        if (
            self._crop_active
            and self._crop_p1 is not None
            and not self._crop_p1.isNull()
            and self._crop_p2 is not None
            and not self._crop_p2.isNull()
        ):
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

        if self._buffer_overlay_visible and self._buffer_overlay_ratio > 1e-4 and not self._crop_active:
            d = visible_rect
            margin_w = d.width() * self._buffer_overlay_ratio
            margin_h = d.height() * self._buffer_overlay_ratio
            inner = QRectF(d.x() + margin_w, d.y() + margin_h, d.width() - 2 * margin_w, d.height() - 2 * margin_h)

            painter.setBrush(QColor(0, 0, 0, 140))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(QRectF(d.x(), d.y(), d.width(), margin_h))
            painter.drawRect(QRectF(d.x(), inner.bottom(), d.width(), margin_h))
            painter.drawRect(QRectF(d.x(), inner.y(), margin_w, inner.height()))
            painter.drawRect(QRectF(inner.right(), inner.y(), margin_w, inner.height()))

            pen = QPen(QColor(THEME.accent_primary), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen)
            painter.drawRect(inner)

        if self._tool_mode != ToolMode.NONE and visible_rect.contains(self._mouse_pos):
            if self._tool_mode == ToolMode.DUST_PICK:
                self._draw_brush(painter)
            else:
                pen = QPen(QColor(255, 255, 255, 80), 1, Qt.PenStyle.DotLine)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawLine(QPointF(visible_rect.x(), self._mouse_pos.y()), QPointF(visible_rect.right(), self._mouse_pos.y()))
                painter.drawLine(QPointF(self._mouse_pos.x(), visible_rect.top()), QPointF(self._mouse_pos.x(), visible_rect.bottom()))

        if getattr(self.state, "compare_mode", False):
            self._draw_compare_badge(painter, visible_rect)

    def _draw_compare_badge(self, painter: QPainter, visible_rect: QRectF) -> None:
        badge = QRectF(visible_rect.x() + 12, visible_rect.y() + 12, 78, 22)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(badge, 4, 4)
        painter.setPen(QColor(THEME.accent_primary))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, "BEFORE")

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

    def _raw_from_screen_with_grid(self, screen_pos: QPointF, uv_grid: np.ndarray) -> Optional[Tuple[float, float]]:
        if self._view_rect.isEmpty():
            return None
        nb_x = float(np.clip((screen_pos.x() - self._view_rect.x()) / self._view_rect.width(), 0.0, 1.0))
        nb_y = float(np.clip((screen_pos.y() - self._view_rect.y()) / self._view_rect.height(), 0.0, 1.0))
        return CoordinateMapping.map_click_to_raw(nb_x, nb_y, uv_grid)

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
                px = np.clip(event.position().x(), self._view_rect.left(), self._view_rect.right())
                py = np.clip(event.position().y(), self._view_rect.top(), self._view_rect.bottom())
                self._crop_p1 = QPointF(px, py)
                self._crop_p2 = QPointF(px, py)
            elif self._tool_mode == ToolMode.CROP_MOVE:
                orig_rect = self.state.config.geometry.manual_crop_rect
                with self.state.metrics_lock:
                    uv_grid = self.state.last_metrics.get("uv_grid")
                if orig_rect is not None and uv_grid is not None:
                    self._move_uv_grid = uv_grid
                    press_raw = self._raw_from_screen_with_grid(event.position(), uv_grid)
                    if press_raw is not None:
                        self._move_active = True
                        self._move_press_raw = press_raw
                        self._move_orig_rect = orig_rect
                        self._move_last_emitted = orig_rect
                        self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse_pos = event.position()

        coords = self._map_to_image_coords(event.position())
        if coords is not None:
            self.cursor_moved.emit(*coords)
        else:
            self.cursor_left.emit()

        if self.parent()._is_panning:
            delta = event.position() - self.parent()._last_mouse_pos
            self.parent()._last_mouse_pos = event.position()
            self.parent().pan_offset += QPointF(delta.x() / self.width(), delta.y() / self.height())
            self.parent()._sync_transform()
            event.accept()
            return

        if self._move_active and self._move_press_raw is not None and self._move_orig_rect is not None and self._move_uv_grid is not None:
            curr_raw = self._raw_from_screen_with_grid(event.position(), self._move_uv_grid)
            if curr_raw is not None:
                fine = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                sensitivity = 0.2 if fine else 0.5
                dx = (curr_raw[0] - self._move_press_raw[0]) * sensitivity
                dy = (curr_raw[1] - self._move_press_raw[1]) * sensitivity
                new_rect = translate_manual_crop_rect(self._move_orig_rect, dx, dy)
                if self._move_last_emitted is None or any(abs(a - b) > 5e-4 for a, b in zip(new_rect, self._move_last_emitted)):
                    self._move_last_emitted = new_rect
                    self.crop_translated.emit(*new_rect)
            event.accept()
            return

        if self._crop_active:
            mx = np.clip(event.position().x(), self._view_rect.left(), self._view_rect.right())
            my = np.clip(event.position().y(), self._view_rect.top(), self._view_rect.bottom())

            ratio_str = self.state.config.geometry.autocrop_ratio
            if ratio_str == "Free":
                self._crop_p2 = QPointF(mx, my)
            else:
                try:
                    w_r, h_r = map(float, ratio_str.split(":"))
                    target_ratio = w_r / h_r

                    dx = mx - self._crop_p1.x()
                    dy = my - self._crop_p1.y()

                    if abs(dx) > abs(dy) * target_ratio:
                        dx = abs(dy) * target_ratio * (1 if dx >= 0 else -1)
                    else:
                        dy = abs(dx) / target_ratio * (1 if dy >= 0 else -1)

                    self._crop_p2 = QPointF(self._crop_p1.x() + dx, self._crop_p1.y() + dy)
                except Exception:
                    self._crop_p2 = QPointF(mx, my)
            self.update()
        else:
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.parent()._is_panning:
            self.parent()._is_panning = False
            self.parent().reset_tool_cursor()
            event.accept()
            return

        if self._move_active:
            self._move_active = False
            self._move_press_raw = None
            self._move_orig_rect = None
            self._move_last_emitted = None
            self._move_uv_grid = None
            self.unsetCursor()
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

    def leaveEvent(self, event) -> None:
        self.cursor_left.emit()
        super().leaveEvent(event)

    def update_overlay(self, filename: str, res: str, colorspace: str, extra: str, edits: int = 0) -> None:
        self.update()
