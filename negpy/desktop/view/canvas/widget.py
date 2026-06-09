from typing import TYPE_CHECKING, Any, Optional, Tuple
import math
import sys
from PyQt6.QtWidgets import QStackedLayout, QMenu, QWidget, QPinchGesture, QGestureEvent
from PyQt6.QtGui import QMouseEvent, QNativeGestureEvent, QPainter, QColor, QWheelEvent
from PyQt6.QtCore import QEvent, pyqtSignal, Qt, QPointF
from negpy.desktop.session import ToolMode, AppState
from negpy.desktop.view.canvas.gpu_widget import GPUCanvasWidget
from negpy.desktop.view.canvas.overlay import CanvasOverlay
from negpy.desktop.view.canvas.pixel_readout import PixelReadoutOverlay
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.infrastructure.gpu.resources import GPUTexture
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger

if TYPE_CHECKING:
    from negpy.desktop.controller import AppController

logger = get_logger(__name__)


def clamp_canvas_zoom_level(zoom: float) -> float:
    zmin, zmax = APP_CONFIG.canvas_zoom_min, APP_CONFIG.canvas_zoom_max
    return max(zmin, min(zoom, zmax))


# One "notch" (typical mouse wheel) ≈ 120/8°; matches prior fixed 1.1 / 0.9 per notch.
WHEEL_ZOOM_NOTCH = 1.1
# Trackpad: map pixel delta to notch-equivalents (tuned for ~smooth steps).
_WHEEL_PIXELS_PER_NOTCH = 64.0

_TOOL_CURSORS: dict[ToolMode, Qt.CursorShape] = {
    ToolMode.NONE: Qt.CursorShape.ArrowCursor,
    ToolMode.WB_PICK: Qt.CursorShape.PointingHandCursor,
    ToolMode.CROP_MANUAL: Qt.CursorShape.CrossCursor,
    ToolMode.CROP_MOVE: Qt.CursorShape.OpenHandCursor,
    ToolMode.DUST_PICK: Qt.CursorShape.BlankCursor,
}
# Do not apply more than this many notch-equivalents in a single event (huge flings).
_WHEEL_MAX_NOTCHES = 4.0


def wheel_notch_delta(event: QWheelEvent) -> float:
    """
    Signed "notch" count: >0 = zoom in, 0 = no vertical scroll intent.
    angleDelta preferred; pixelDelta when y-angle is 0. OS ``inverted`` honored.
    Result is negated at the end so a natural trackpad (scroll down) zooms in, matching photo viewers.
    """
    d = int(event.angleDelta().y())
    if d != 0:
        u = float(d) / 120.0
    else:
        pdy = int(event.pixelDelta().y())
        if pdy == 0:
            return 0.0
        u = float(pdy) / _WHEEL_PIXELS_PER_NOTCH
    if event.inverted():
        u = -u
    if u == 0.0:
        return 0.0
    c = _WHEEL_MAX_NOTCHES
    u = max(-c, min(c, u))
    return -u


def apply_wheel_zoom_notches(zoom: float, notch_u: float) -> float:
    """Clamped zoom after one wheel event (notch_u from ``wheel_notch_delta``)."""
    return clamp_canvas_zoom_level(zoom * (WHEEL_ZOOM_NOTCH**notch_u))


class ImageCanvas(QWidget):
    """
    Main viewport container using QStackedLayout to layer GPU and UI overlays.
    """

    clicked = pyqtSignal(float, float)
    crop_completed = pyqtSignal(float, float, float, float)
    crop_translated = pyqtSignal(float, float, float, float)
    zoom_changed = pyqtSignal(float)
    cursor_position_changed = pyqtSignal(float, float)
    cursor_left_canvas = pyqtSignal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._controller: Optional["AppController"] = None
        self.setMouseTracking(True)

        if sys.platform == "win32":
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
            self.setAttribute(Qt.WidgetAttribute.WA_StaticContents, False)
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        self._last_mouse_pos = QPointF(0, 0)
        self._is_panning = False
        self._bg_color = QColor("#050505")
        self._last_buffer: Any = None

        self.root_layout = QStackedLayout(self)
        self.root_layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self.root_layout.setContentsMargins(0, 0, 0, 0)

        # Acceleration layer
        self.gpu_widget = GPUCanvasWidget(self)
        gpu = GPUDevice.get()
        if gpu.is_available:
            try:
                self.gpu_widget.initialize_gpu(gpu.device, gpu.adapter)
            except Exception as e:
                logger.error(f"Hardware viewport acceleration failed: {e}")
        self.root_layout.addWidget(self.gpu_widget)

        # UI Overlay layer
        self.overlay = CanvasOverlay(state, self)
        self.root_layout.addWidget(self.overlay)

        self.overlay.clicked.connect(self.clicked.emit)
        self.overlay.crop_completed.connect(self.crop_completed.emit)
        self.overlay.crop_translated.connect(self.crop_translated.emit)
        self.overlay.cursor_moved.connect(self.cursor_position_changed.emit)
        self.overlay.cursor_left.connect(self.cursor_left_canvas.emit)

        # Pixel readout overlay — absolute child, not in layout
        self.pixel_readout_overlay = PixelReadoutOverlay(self)
        self.pixel_readout_overlay.raise_()

        self.cursor_position_changed.connect(lambda x, y: self.pixel_readout_overlay.setVisible(True))
        self.cursor_left_canvas.connect(lambda: self.pixel_readout_overlay.setVisible(False))

        self.grabGesture(Qt.GestureType.PinchGesture)

    def set_tool_mode(self, mode: ToolMode) -> None:
        self.setCursor(_TOOL_CURSORS.get(mode, Qt.CursorShape.ArrowCursor))
        self.overlay.set_tool_mode(mode)

    def reset_tool_cursor(self) -> None:
        self.setCursor(_TOOL_CURSORS.get(self.state.active_tool, Qt.CursorShape.ArrowCursor))

    def set_controller(self, controller: "AppController") -> None:
        self._controller = controller

    def set_zoom(self, zoom: float) -> None:
        """Sets zoom level directly (from toolbar)."""
        self.zoom_level = clamp_canvas_zoom_level(zoom)
        if self.zoom_level <= 1.0:
            self.pan_offset = QPointF(0, 0)
        self._sync_transform()

    def fit_to_window(self) -> None:
        """Fit image to the visible viewport (zoom to fill, reset pan)."""
        metrics = self.state.last_metrics
        buf = metrics.get("base_positive")
        if buf is None:
            self.set_zoom(1.0)
            return
        import numpy as np

        if isinstance(buf, np.ndarray):
            img_h, img_w = buf.shape[:2]
        elif isinstance(buf, GPUTexture):
            img_w, img_h = buf.width, buf.height
        else:
            self.set_zoom(1.0)
            return
        vw, vh = max(1, self.width()), max(1, self.height())
        zoom = min(vw / max(1, img_w), vh / max(1, img_h))
        self.zoom_level = clamp_canvas_zoom_level(zoom)
        self.pan_offset = QPointF(0, 0)
        self._sync_transform()

    def set_background_color(self, r: float, g: float, b: float) -> None:
        """Update canvas background color (0–1 linear values)."""
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        self._bg_color = QColor(hex_color)
        self.gpu_widget.set_background_color(r, g, b)
        self.update()

    def paintEvent(self, event) -> None:
        """Draw background only if GPU is not active to prevent covering it."""
        if not self.gpu_widget.isVisible():
            painter = QPainter(self)
            painter.fillRect(event.rect(), self._bg_color)

    def clear(self) -> None:
        """Total viewport reset."""
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        self._last_buffer = None
        self.gpu_widget.clear()
        self.overlay.update_buffer(None, "sRGB", None)

    def get_pixel_rgb(self, nx: float, ny: float) -> Optional[Tuple[float, float, float]]:
        """Returns the displayed sRGB triplet in 0..1 at normalized image coords, or None."""
        import numpy as np

        buf = self._last_buffer
        if buf is None:
            return None
        if isinstance(buf, GPUTexture):
            w, h = buf.width, buf.height
            x = int(max(0, min(w - 1, nx * w)))
            y = int(max(0, min(h - 1, ny * h)))
            try:
                arr = buf.readback_region(x, y, 1, 1)
            except Exception:
                return None
            return (float(arr[0, 0, 0]), float(arr[0, 0, 1]), float(arr[0, 0, 2]))
        if isinstance(buf, np.ndarray):
            h, w = buf.shape[:2]
            x = int(max(0, min(w - 1, nx * w)))
            y = int(max(0, min(h - 1, ny * h)))
            px = buf[y, x]
            scale = 1.0 / 255.0 if buf.dtype == np.uint8 else 1.0
            px = np.atleast_1d(px)
            if px.shape[0] == 1:
                v = float(px[0]) * scale
                return (v, v, v)
            return (float(px[0]) * scale, float(px[1]) * scale, float(px[2]) * scale)
        return None

    @staticmethod
    def _scale_from_native_zoom_value(v: float) -> float | None:
        """Map QNativeGestureEvent (Zoom) ``value`` to a multiplicative scale step."""
        v = float(v)
        if not math.isfinite(v) or abs(v) < 1e-9:
            return None
        if abs(1.0 - v) < 0.15:  # ~0.85–1.15, treat as a direct factor
            k = v
        elif -0.5 < v < 0.5:  # small per-frame delta
            k = 1.0 + v
        else:
            k = 1.0 + v
        if not math.isfinite(k) or k < 0.1:
            return None
        return min(k, 4.0) if k > 1.0 else max(k, 0.25)  # single-event factor bounds

    def _commit_anchored_zoom(self, new_zoom: float, anchor: QPointF) -> None:
        """Applies a zoom level with pan adjusted so `anchor` stays under the same image point."""
        old_zoom = self.zoom_level
        if new_zoom == old_zoom:
            return
        if new_zoom > 1.0 and old_zoom > 0:
            dx = anchor.x() / max(1, self.width()) - 0.5
            dy = anchor.y() / max(1, self.height()) - 0.5
            k = new_zoom / old_zoom
            self.pan_offset = QPointF(
                self.pan_offset.x() * k - dx * (k - 1.0),
                self.pan_offset.y() * k - dy * (k - 1.0),
            )
        self.zoom_level = new_zoom
        if self.zoom_level <= 1.0:
            self.pan_offset = QPointF(0, 0)
        self._sync_transform()

    def _apply_scale_at(self, scale_k: float, anchor: QPointF) -> bool:
        """
        Multiplies current zoom by ``scale_k`` (e.g. pinch), clamped. Returns True if the view changed.
        """
        if not math.isfinite(scale_k) or scale_k <= 0.0 or abs(scale_k - 1.0) < 1e-6:
            return False
        zmin = APP_CONFIG.canvas_zoom_min
        zmax = APP_CONFIG.canvas_zoom_max
        old = self.zoom_level
        if (old >= zmax and scale_k > 1.0) or (old <= zmin and scale_k < 1.0):
            return False
        new = clamp_canvas_zoom_level(old * scale_k)
        if new == old:
            return False
        self._commit_anchored_zoom(new, anchor)
        return True

    def event(self, e: QEvent) -> bool:
        t = e.type()
        if t == QEvent.Type.Gesture and isinstance(e, QGestureEvent):
            if self._try_pinch_gesture(e):
                return True
        if t == QEvent.Type.NativeGesture and isinstance(e, QNativeGestureEvent):
            if self._try_native_pinch_zoom(e):
                return True
        return super().event(e)

    def _try_pinch_gesture(self, ev: QGestureEvent) -> bool:
        g = ev.gesture(Qt.GestureType.PinchGesture)
        if g is None or not isinstance(g, QPinchGesture):
            return False
        st = g.state()
        if st in (Qt.GestureState.GestureFinished, Qt.GestureState.GestureCanceled):
            ev.setAccepted(g, True)
            return True
        if st == Qt.GestureState.GestureStarted:
            ev.setAccepted(g, True)
            return True
        if st != Qt.GestureState.GestureUpdated:
            return False
        k = float(g.lastScaleFactor())
        if not math.isfinite(k) or k <= 0.0 or abs(k - 1.0) < 1e-6:
            ev.setAccepted(g, True)
            return True
        anchor = g.centerPoint()
        w = ev.widget()
        if w is not None and w is not self:
            anchor = w.mapTo(self, anchor)
        if self._apply_scale_at(k, anchor):
            ev.setAccepted(g, True)
            return True
        ev.setAccepted(g, True)
        return True

    def _try_native_pinch_zoom(self, n: QNativeGestureEvent) -> bool:
        if n.gestureType() != Qt.NativeGestureType.ZoomNativeGesture:
            return False
        if n.isBeginEvent() or n.isEndEvent():
            n.accept()
            return True
        n.accept()
        if not n.isUpdateEvent():
            return True
        k = self._scale_from_native_zoom_value(n.value())
        if k is not None and abs(k - 1.0) >= 1e-6:
            self._apply_scale_at(k, n.position())
        return True

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handles zooming anchored on the mouse cursor position."""
        u = wheel_notch_delta(event)
        if u == 0.0:
            event.ignore()
            return

        zmin = APP_CONFIG.canvas_zoom_min
        zmax = APP_CONFIG.canvas_zoom_max
        old_zoom = self.zoom_level
        if (old_zoom >= zmax and u > 0.0) or (old_zoom <= zmin and u < 0.0):
            event.accept()
            return

        new_zoom = apply_wheel_zoom_notches(old_zoom, u)
        if new_zoom == old_zoom:
            event.accept()
            return

        self._commit_anchored_zoom(new_zoom, event.position())
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self.zoom_level > 1.0 and self.state.active_tool == ToolMode.NONE
        ):
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_panning:
            delta = event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()
            self.pan_offset += QPointF(delta.x() / self.width(), delta.y() / self.height())
            self._sync_transform()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._is_panning:
            self._is_panning = False
            self.reset_tool_cursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _sync_transform(self) -> None:
        """Propagates zoom/pan to sub-widgets."""
        self.gpu_widget.set_transform(self.zoom_level, self.pan_offset.x(), self.pan_offset.y())
        self.overlay.set_transform(self.zoom_level, self.pan_offset.x(), self.pan_offset.y())
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def update_buffer(
        self,
        buffer: Any,
        color_space: str,
        content_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """
        Switches between CPU and GPU rendering paths.
        """
        self._last_buffer = buffer
        if self.state.gpu_enabled and isinstance(buffer, GPUTexture):
            self.gpu_widget.show()
            self.gpu_widget.update_texture(buffer)
            self.overlay.update_buffer(None, color_space, content_rect, gpu_size=(buffer.width, buffer.height))
            self.overlay.show()
            self.overlay.raise_()
            self.overlay.update()
        else:
            self.gpu_widget.hide()
            self.overlay.update_buffer(buffer, color_space, content_rect)
            self.overlay.show()
            self.overlay.raise_()

    def update_overlay(self, filename: str, res: str, colorspace: str, extra: str, edits: int = 0) -> None:
        self.overlay.update_overlay(filename, res, colorspace, extra, edits)

    def contextMenuEvent(self, event) -> None:
        if self.state.selected_file_idx < 0 or self._controller is None:
            event.ignore()
            return

        menu = QMenu(self)
        act_wb = menu.addAction("Pick WB  Shift+W")
        act_wb.triggered.connect(lambda: self._controller.set_active_tool(ToolMode.WB_PICK))  # type: ignore[union-attr]
        act_dust = menu.addAction("Pick Dust  Shift+D")
        act_dust.triggered.connect(lambda: self._controller.set_active_tool(ToolMode.DUST_PICK))  # type: ignore[union-attr]
        menu.addSeparator()
        act_copy = menu.addAction("Copy Settings  Ctrl+C")
        act_copy.triggered.connect(self._controller.session.copy_settings)  # type: ignore[union-attr]
        act_copy_bounds = menu.addAction("Copy Settings + Bounds  Ctrl+Shift+C")
        act_copy_bounds.triggered.connect(self._controller.session.copy_settings_with_bounds)  # type: ignore[union-attr]
        act_paste = menu.addAction("Paste Settings  Ctrl+V")
        act_paste.triggered.connect(self._controller.session.paste_settings)  # type: ignore[union-attr]
        act_paste.setEnabled(self.state.clipboard is not None)
        menu.addSeparator()
        act_reset = menu.addAction("Reset View")
        act_reset.triggered.connect(self.fit_to_window)
        menu.exec(event.globalPos())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.pixel_readout_overlay._reposition()
