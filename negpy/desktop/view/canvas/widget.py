from typing import TYPE_CHECKING, Optional, Tuple, Any
import sys
from PyQt6.QtWidgets import QWidget, QStackedLayout, QMenu
from PyQt6.QtGui import QPainter, QColor, QMouseEvent
from PyQt6.QtCore import pyqtSignal, Qt, QPointF
from negpy.desktop.session import ToolMode, AppState
from negpy.desktop.view.canvas.gpu_widget import GPUCanvasWidget
from negpy.desktop.view.canvas.overlay import CanvasOverlay
from negpy.desktop.view.canvas.pixel_readout import PixelReadoutOverlay
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.infrastructure.gpu.resources import GPUTexture
from negpy.kernel.system.logging import get_logger

if TYPE_CHECKING:
    from negpy.desktop.controller import AppController

logger = get_logger(__name__)


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

    def set_tool_mode(self, mode: ToolMode) -> None:
        self.overlay.set_tool_mode(mode)

    def set_controller(self, controller: "AppController") -> None:
        self._controller = controller

    def set_zoom(self, zoom: float) -> None:
        """Sets zoom level directly (from toolbar)."""
        self.zoom_level = max(0.25, min(zoom, 4.0))
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
        zoom = max(0.25, min(zoom, 4.0))
        self.zoom_level = zoom
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

    def wheelEvent(self, event) -> None:
        """Handles zooming anchored on the mouse cursor position."""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9

        old_zoom = self.zoom_level
        new_zoom = max(0.25, min(old_zoom * zoom_factor, 4.0))

        if new_zoom > 1.0 and old_zoom > 0:
            # Cursor offset from widget center, in normalized units [-0.5, 0.5]
            cursor = event.position()
            dx = cursor.x() / max(1, self.width()) - 0.5
            dy = cursor.y() / max(1, self.height()) - 0.5
            # Adjust pan so the image point under cursor stays fixed
            k = new_zoom / old_zoom
            self.pan_offset = QPointF(
                self.pan_offset.x() * k - dx * (k - 1),
                self.pan_offset.y() * k - dy * (k - 1),
            )

        self.zoom_level = new_zoom
        if self.zoom_level <= 1.0:
            self.pan_offset = QPointF(0, 0)

        self._sync_transform()
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
            self.setCursor(Qt.CursorShape.ArrowCursor)
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
        act_wb = menu.addAction(tooltip_with_shortcut("Pick WB", "pick_wb"))
        act_wb.triggered.connect(lambda: self._controller.set_active_tool(ToolMode.WB_PICK))  # type: ignore[union-attr]
        act_dust = menu.addAction(tooltip_with_shortcut("Pick Dust", "pick_dust"))
        act_dust.triggered.connect(lambda: self._controller.set_active_tool(ToolMode.DUST_PICK))  # type: ignore[union-attr]
        menu.addSeparator()
        act_copy = menu.addAction("Copy Settings  Ctrl+C")
        act_copy.triggered.connect(self._controller.session.copy_settings)  # type: ignore[union-attr]
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
