from typing import Optional, Tuple, Any
import numpy as np
from PyQt6.QtWidgets import QWidget, QStackedLayout
from PyQt6.QtCore import pyqtSignal
from src.desktop.session import ToolMode, AppState
from src.desktop.view.canvas.gpu_widget import GPUCanvasWidget
from src.desktop.view.canvas.overlay import CanvasOverlay
from src.infrastructure.gpu.device import GPUDevice
from src.infrastructure.gpu.resources import GPUTexture


class ImageCanvas(QWidget):
    """
    Main canvas widget that orchestrates GPU rendering and UI overlays.
    """

    clicked = pyqtSignal(float, float)
    crop_completed = pyqtSignal(float, float, float, float)

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        # We use a layout for the GPU widget, but the Overlay will be manually managed
        # to ensure it stays on top of the native window container.
        self.root_layout = QStackedLayout(self)
        self.root_layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self.root_layout.setContentsMargins(0, 0, 0, 0)

        # 1. GPU Layer
        self.gpu_widget = GPUCanvasWidget(self)
        gpu = GPUDevice.get()
        if gpu.is_available:
            try:
                self.gpu_widget.initialize_gpu(gpu.device, gpu.adapter)
            except Exception as e:
                print(f"GPU Init Error: {e}")
        self.root_layout.addWidget(self.gpu_widget)

        # 2. Overlay Layer (Handles CPU rendering + UI)
        # We add it to the layout but also ensure it can be raised
        self.overlay = CanvasOverlay(state, self)
        self.root_layout.addWidget(self.overlay)

        # Connect signals
        self.overlay.clicked.connect(self.clicked)
        self.overlay.crop_completed.connect(self.crop_completed)

    def set_tool_mode(self, mode: ToolMode) -> None:
        self.overlay.set_tool_mode(mode)

    def clear(self) -> None:
        """
        Clears the canvas content.
        """
        self.gpu_widget.clear()
        self.overlay.update_buffer(None, "sRGB", None)

    def update_buffer(
        self,
        buffer: Any,
        color_space: str,
        content_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """
        Updates the displayed image. buffer can be np.ndarray (CPU) or GPUTexture wrapper (GPU).
        """
        if isinstance(buffer, np.ndarray):
            # CPU Mode
            self.gpu_widget.hide()
            self.overlay.update_buffer(buffer, color_space, content_rect)
            self.overlay.show()
            self.overlay.raise_()
        elif isinstance(buffer, GPUTexture):
            # GPU Mode
            size = (buffer.width, buffer.height)
            # Sync overlay sizing with GPU result size
            self.overlay.update_buffer(None, color_space, content_rect, gpu_size=size)
            self.gpu_widget.update_texture(buffer)
            self.gpu_widget.show()
            self.overlay.show()
            self.overlay.raise_()  # Force overlay to top of native window
        else:
            self.gpu_widget.hide()
            self.overlay.update_buffer(None, color_space, content_rect)

    def update_overlay(
        self, filename: str, res: str, colorspace: str, extra: str
    ) -> None:
        self.overlay.update_overlay(filename, res, colorspace, extra)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Ensure overlay covers the whole area
        self.overlay.setGeometry(self.rect())
        self.gpu_widget.setGeometry(self.rect())
