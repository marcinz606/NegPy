import sys
from typing import Any, Optional, Tuple

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QStackedLayout, QWidget

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.canvas.gpu_widget import GPUCanvasWidget
from negpy.desktop.view.canvas.overlay import CanvasOverlay
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.infrastructure.gpu.resources import GPUTexture
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


class ImageCanvas(QWidget):
    """
    Main viewport container using QStackedLayout to layer GPU and UI overlays.
    """

    clicked = pyqtSignal(float, float)
    crop_completed = pyqtSignal(float, float, float, float)
    zoom_changed = pyqtSignal(float)

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setMouseTracking(True)

        # Windows stability: The container is a native window to help with resize sync,
        # but we use a solid background to prevent composition ghosting.
        if sys.platform == "win32":
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
            self.setAttribute(Qt.WidgetAttribute.WA_StaticContents, False)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        self._last_mouse_pos = QPointF(0, 0)
        self._is_panning = False

        self.root_layout = QStackedLayout(self)
        self.root_layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self.root_layout.setContentsMargins(0, 0, 0, 0)

        self.setStyleSheet("""
            ImageCanvas {
                background-color: #050505;
            }
        """)

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

    def set_tool_mode(self, mode: ToolMode) -> None:
        self.overlay.set_tool_mode(mode)

    def set_zoom(self, zoom: float) -> None:
        """Sets zoom level directly (from toolbar)."""
        self.zoom_level = max(1.0, min(zoom, 4.0))
        if self.zoom_level == 1.0:
            self.pan_offset = QPointF(0, 0)
        self._sync_transform()

    def paintEvent(self, event) -> None:
        """Ensure a solid background on Windows to prevent ghosting."""
        if sys.platform == "win32":
            painter = QPainter(self)
            painter.fillRect(event.rect(), QColor("#050505"))
        else:
            super().paintEvent(event)

    def clear(self) -> None:
        """Total viewport reset."""
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        self.gpu_widget.clear()
        self.overlay.update_buffer(None, "sRGB", None)

    def wheelEvent(self, event) -> None:
        """Handles zooming centered on the mouse cursor."""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9

        old_zoom = self.zoom_level
        self.zoom_level = max(1.0, min(self.zoom_level * zoom_factor, 4.0))

        if self.zoom_level == 1.0:
            self.pan_offset = QPointF(0, 0)
        elif self.zoom_level != old_zoom:
            pass

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

    def mouseMoveEvent(self, event) -> None:
        if self._is_panning:
            delta = event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()
            self.pan_offset += QPointF(delta.x() / self.width(), delta.y() / self.height())
            self._sync_transform()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
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
        if self.state.gpu_enabled and isinstance(buffer, GPUTexture):
            self.gpu_widget.show()
            self.gpu_widget.update_texture(buffer)
            # Pass texture size so overlay can calculate projection rect
            self.overlay.update_buffer(None, color_space, content_rect, gpu_size=(buffer.width, buffer.height))
            self.overlay.raise_()
        else:
            self.gpu_widget.hide()
            self.overlay.update_buffer(buffer, color_space, content_rect)

    def update_overlay(self, filename: str, res: str, colorspace: str, extra: str, edits: int = 0) -> None:
        self.overlay.update_overlay(filename, res, colorspace, extra, edits)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
