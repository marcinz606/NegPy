from typing import Optional, Tuple, Any
import numpy as np
from PyQt6.QtWidgets import QWidget, QStackedLayout
from PyQt6.QtCore import pyqtSignal, Qt
from negpy.desktop.session import ToolMode, AppState
from negpy.desktop.view.canvas.gpu_widget import GPUCanvasWidget
from negpy.desktop.view.canvas.overlay import CanvasOverlay
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.infrastructure.gpu.resources import GPUTexture
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


class ImageCanvas(QWidget):
    """
    Unified viewport orchestrator.
    Manages hardware acceleration layers and SVG/Qt UI overlays.
    """

    clicked = pyqtSignal(float, float)
    crop_completed = pyqtSignal(float, float, float, float)

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.root_layout = QStackedLayout(self)
        self.root_layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self.root_layout.setContentsMargins(16, 16, 16, 16)

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

        self.overlay.clicked.connect(self.clicked)
        self.overlay.crop_completed.connect(self.crop_completed)

    def set_tool_mode(self, mode: ToolMode) -> None:
        self.overlay.set_tool_mode(mode)

    def clear(self) -> None:
        """Total viewport reset."""
        self.gpu_widget.clear()
        self.overlay.update_buffer(None, "sRGB", None)

    def update_buffer(
        self,
        buffer: Any,
        color_space: str,
        content_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """Updates the active viewport with a CPU or GPU buffer."""
        if isinstance(buffer, np.ndarray):
            self.gpu_widget.hide()
            self.overlay.update_buffer(buffer, color_space, content_rect)
            self.overlay.show()
            self.overlay.raise_()
        elif isinstance(buffer, GPUTexture):
            self.overlay.update_buffer(None, color_space, content_rect, gpu_size=(buffer.width, buffer.height))
            self.gpu_widget.update_texture(buffer)
            self.gpu_widget.show()
            self.overlay.show()
            self.overlay.raise_()
        else:
            self.gpu_widget.hide()
            self.overlay.update_buffer(None, color_space, content_rect)

    def update_overlay(self, filename: str, res: str, colorspace: str, extra: str) -> None:
        self.overlay.update_overlay(filename, res, colorspace, extra)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
