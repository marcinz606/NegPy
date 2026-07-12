from typing import TYPE_CHECKING, Any, Optional, Tuple
import math
import sys
from PyQt6.QtWidgets import QStackedLayout, QMenu, QWidget, QPinchGesture, QGestureEvent
from PyQt6.QtGui import QCursor, QMouseEvent, QNativeGestureEvent, QPainter, QColor, QWheelEvent
from PyQt6.QtCore import QEvent, pyqtSignal, Qt, QPointF
from negpy.desktop.session import ToolMode, AppState
from negpy.desktop.view.canvas.gpu_widget import GPUCanvasWidget
from negpy.desktop.view.canvas.hud import CanvasHud
from negpy.desktop.view.canvas.overlay import CanvasOverlay
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
    ToolMode.DUST_PICK: Qt.CursorShape.BlankCursor,
    ToolMode.LOCAL_DRAW: Qt.CursorShape.CrossCursor,
    ToolMode.ANALYSIS_DRAW: Qt.CursorShape.CrossCursor,
    ToolMode.STRAIGHTEN: Qt.CursorShape.CrossCursor,
}

_scratch_pen_cursor: Optional[QCursor] = None


def _cursor_for_tool(mode: ToolMode) -> QCursor | Qt.CursorShape:
    """Cursor for a tool mode. The scratch tool gets a pen-nib cursor so it's
    obvious the canvas is in click-points line-drawing mode (built lazily —
    QCursor pixmaps need a live QGuiApplication)."""
    if mode == ToolMode.SCRATCH_PICK:
        global _scratch_pen_cursor
        if _scratch_pen_cursor is None:
            import qtawesome as qta

            # Rotated 90°: the glyph's nib swings from bottom-left to top-left,
            # tail to lower-right — reads like a normal pointer, tip up-left.
            pix = qta.icon("fa5s.pen-nib", color="white", rotated=90).pixmap(18, 18)
            _scratch_pen_cursor = QCursor(pix, 2, 2)
        return _scratch_pen_cursor
    return _TOOL_CURSORS.get(mode, Qt.CursorShape.ArrowCursor)


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
    crop_rect_changed = pyqtSignal(float, float, float, float, bool)
    crop_rotation_changed = pyqtSignal(float, bool)
    crop_confirmed = pyqtSignal()
    analysis_rect_changed = pyqtSignal(float, float, float, float, bool)
    analysis_confirmed = pyqtSignal()
    zoom_changed = pyqtSignal(float)
    cursor_position_changed = pyqtSignal(float, float)
    cursor_left_canvas = pyqtSignal()
    lasso_completed = pyqtSignal(list)
    scratch_completed = pyqtSignal(list)
    straighten_completed = pyqtSignal(float)
    local_mask_selected = pyqtSignal(int)
    local_mask_edited = pyqtSignal(int, list)
    local_vertex_deleted = pyqtSignal(int, int)

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
        self.overlay.crop_rect_changed.connect(self.crop_rect_changed.emit)
        self.overlay.crop_rotation_changed.connect(self.crop_rotation_changed.emit)
        self.overlay.crop_confirmed.connect(self.crop_confirmed.emit)
        self.overlay.analysis_rect_changed.connect(self.analysis_rect_changed.emit)
        self.overlay.analysis_confirmed.connect(self.analysis_confirmed.emit)
        self.overlay.cursor_moved.connect(self.cursor_position_changed.emit)
        self.overlay.cursor_left.connect(self.cursor_left_canvas.emit)
        self.overlay.lasso_completed.connect(self.lasso_completed.emit)
        self.overlay.scratch_completed.connect(self.scratch_completed.emit)
        self.overlay.straighten_completed.connect(self.straighten_completed.emit)
        self.overlay.local_mask_selected.connect(self.local_mask_selected.emit)
        self.overlay.local_mask_edited.connect(self.local_mask_edited.emit)
        self.overlay.local_vertex_deleted.connect(self.local_vertex_deleted.emit)

        self.hud = CanvasHud(self)
        self._floating_toolbar: Optional[QWidget] = None

        self.grabGesture(Qt.GestureType.PinchGesture)

    def set_floating_toolbar(self, toolbar: QWidget) -> None:
        """Adopts the action toolbar as a floating pill anchored to the canvas bottom."""
        toolbar.setParent(self)
        toolbar.show()
        self._floating_toolbar = toolbar
        self._layout_floating_widgets()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_floating_widgets()
        # The fit scale moves with the viewport, so the true-zoom % readout must refresh.
        self.zoom_changed.emit(self.zoom_level)

    def _layout_floating_widgets(self) -> None:
        self.hud.setGeometry(self.rect())
        tb = self._floating_toolbar
        if tb is not None:
            if hasattr(tb, "set_available_width"):
                tb.set_available_width(self.width())
            size = tb.sizeHint()
            tb.setGeometry((self.width() - size.width()) // 2, self.height() - size.height(), size.width(), size.height())
        self._raise_floating_widgets()

    def _raise_floating_widgets(self) -> None:
        self.hud.raise_()
        if self._floating_toolbar is not None:
            self._floating_toolbar.raise_()

    def set_tool_mode(self, mode: ToolMode) -> None:
        self.setCursor(_cursor_for_tool(mode))
        self.overlay.set_tool_mode(mode)

    def reset_tool_cursor(self) -> None:
        self.setCursor(_cursor_for_tool(self.state.active_tool))

    def set_controller(self, controller: "AppController") -> None:
        self._controller = controller

    def set_zoom(self, zoom: float) -> None:
        """Sets zoom level directly (from toolbar)."""
        self.zoom_level = clamp_canvas_zoom_level(zoom)
        if self.zoom_level <= 1.0:
            self.pan_offset = QPointF(0, 0)
        self._sync_transform()

    def _image_dims(self) -> Optional[Tuple[int, int]]:
        """Current rendered image size as (width, height), or None if nothing is shown."""
        buf = self.state.last_metrics.get("base_positive")
        if buf is None:
            return None
        import numpy as np

        if isinstance(buf, np.ndarray):
            return int(buf.shape[1]), int(buf.shape[0])
        if isinstance(buf, GPUTexture):
            return int(buf.width), int(buf.height)
        return None

    def _fit_scale(self) -> Optional[float]:
        """Device-pixel scale the shader applies at zoom_level 1.0 — its fit ratio
        min(viewport / image). True pixel zoom = zoom_level * _fit_scale()."""
        dims = self._image_dims()
        if not dims:
            return None
        img_w, img_h = dims
        dpr = self.devicePixelRatioF()
        vw = max(1.0, self.width() * dpr)
        vh = max(1.0, self.height() * dpr)
        return min(vw / max(1, img_w), vh / max(1, img_h))

    def current_zoom_percent(self) -> int:
        """True pixel zoom percentage (100 = 1 image pixel : 1 device pixel)."""
        fs = self._fit_scale() or 1.0
        return int(round(self.zoom_level * fs * 100.0))

    def fit_to_window(self) -> None:
        """Fit the image to the viewport (zoom_level 1.0 is the shader's fit); reset pan."""
        self.set_zoom(1.0)

    def zoom_to_percent(self, percent: float) -> None:
        """Set the true pixel zoom to ``percent`` (100 = 1 image pixel : 1 device pixel)."""
        fs = self._fit_scale()
        if not fs:
            self.set_zoom(1.0)
            return
        zmin = APP_CONFIG.canvas_zoom_min
        # Allow above the normal wheel max so true 100% stays reachable for images much
        # larger than the viewport (where 1 / fit_scale exceeds canvas_zoom_max).
        self.zoom_level = max(zmin, (percent / 100.0) / fs)
        if self.zoom_level <= 1.0:
            self.pan_offset = QPointF(0, 0)
        self._sync_transform()

    def zoom_to_original(self) -> None:
        """Zoom to true 1:1 pixels (100%)."""
        self.zoom_to_percent(100.0)

    def set_monitor_profile(self, monitor_icc_bytes: Optional[bytes]) -> None:
        """Forward the detected monitor ICC profile to the GPU display path."""
        self.gpu_widget.set_monitor_profile(monitor_icc_bytes)

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

        # User preference: reverse scroll-to-zoom direction (set in Customize Shortcuts).
        if getattr(self.state, "invert_zoom_scroll", False):
            u = -u

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
        monitor_icc_bytes: Optional[bytes] = None,
    ) -> None:
        """
        Switches between CPU and GPU rendering paths.

        ``monitor_icc_bytes`` drives the CPU display transform (working→monitor); it
        must be None when ``buffer`` is already in display space (e.g. a baked soft
        proof) to avoid a double conversion. The GPU path manages its own display LUT.
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
            self.overlay.update_buffer(buffer, color_space, content_rect, monitor_icc_bytes=monitor_icc_bytes)
            self.overlay.show()
            self.overlay.raise_()
        self._raise_floating_widgets()
        # The image resolution can change between renders (e.g. toggling HQ), which
        # shifts the fit scale, so refresh the true-zoom % readout for the new buffer.
        self.zoom_changed.emit(self.zoom_level)

    def update_overlay(self, filename: str, res: str, colorspace: str, extra: str, edits: int = 0) -> None:
        self.overlay.update_overlay(filename, res, colorspace, extra, edits)

    def contextMenuEvent(self, event) -> None:
        if self.state.selected_file_idx < 0 or self._controller is None:
            event.ignore()
            return

        # While a heal tool is live, the menu serves that tool — the general
        # settings menu would be noise mid-retouch.
        if self.state.active_tool in (ToolMode.DUST_PICK, ToolMode.SCRATCH_PICK):
            self._exec_retouch_menu(event)
            return

        # Right-click on a selected mask's vertex deletes that point (no menu).
        if self.state.active_tool in (ToolMode.NONE, ToolMode.LOCAL_DRAW) and self.overlay.try_delete_local_vertex(QPointF(event.pos())):
            event.accept()
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

    def _exec_retouch_menu(self, event) -> None:
        """Context menu while the heal or scratch tool is active."""
        controller = self._controller
        assert controller is not None
        conf = self.state.config.retouch
        num_heals = len(conf.manual_dust_spots) + len(conf.manual_heal_strokes)
        pos = QPointF(event.pos())

        menu = QMenu(self)

        if self.state.active_tool == ToolMode.SCRATCH_PICK:
            act_confirm = menu.addAction("Confirm Scratch  Enter")
            act_confirm.triggered.connect(self.overlay.confirm_scratch)
            act_confirm.setEnabled(self.overlay.has_scratch_points())
            act_point = menu.addAction("Undo Last Point  Backspace")
            act_point.triggered.connect(self.overlay.undo_last_scratch_point)
            act_point.setEnabled(self.overlay.has_scratch_points())
            menu.addSeparator()

        hit = self.overlay.heal_hit_test(pos)
        if hit is not None:
            kind, index = hit
            act_delete = menu.addAction("Delete This Heal")
            act_delete.triggered.connect(lambda _=False, k=kind, i=index: controller.delete_heal(k, i))
            menu.addSeparator()

        act_undo = menu.addAction("Undo Last Heal  Ctrl+Z")
        act_undo.triggered.connect(controller.undo_last_retouch)
        act_undo.setEnabled(num_heals > 0)
        act_clear = menu.addAction("Clear All Heals…")
        act_clear.triggered.connect(controller.clear_retouch)
        act_clear.setEnabled(num_heals > 0)
        menu.exec(event.globalPos())
