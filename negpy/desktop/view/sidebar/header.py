from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from negpy.desktop.controller import AppController
from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.system.paths import get_resource_path
from negpy.kernel.system.version import get_app_version
from negpy.infrastructure.gpu.device import GPUDevice


class SidebarHeader(QWidget):
    """
    Top header for the sidebar containing logo, version and hardware settings.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.session = controller.session
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 0)
        layout.setSpacing(5)

        header = QHBoxLayout()
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_label = QLabel()
        icon_pix = QPixmap(get_resource_path("media/icons/icon.png"))
        if not icon_pix.isNull():
            icon_label.setPixmap(
                icon_pix.scaled(
                    32,
                    32,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        name_label = QLabel("NegPy")
        name_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {THEME.text_primary}; margin-left: 5px;")

        header.addWidget(icon_label)
        header.addWidget(name_label)
        layout.addLayout(header)

        self.ver_label = QLabel(f"v{get_app_version()}")
        self.ver_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ver_label.setStyleSheet(f"font-size: 14px; color: {THEME.text_secondary}; font-weight: bold;")
        layout.addWidget(self.ver_label)

        gpu_available = GPUDevice.get().is_available

        gpu_container = QHBoxLayout()
        gpu_container.setContentsMargins(10, 5, 10, 5)
        gpu_container.setSpacing(10)

        self.gpu_checkbox = QCheckBox("GPU Acceleration")
        self.gpu_checkbox.setStyleSheet(f"color: {THEME.text_secondary}; font-size: 12px; font-weight: bold;")

        if gpu_available:
            self.gpu_checkbox.setChecked(self.session.state.gpu_enabled)
        else:
            self.gpu_checkbox.setEnabled(False)
            self.gpu_checkbox.setChecked(False)
            self.gpu_checkbox.setToolTip("GPU not available on this hardware")

        self.gpu_checkbox.toggled.connect(self._on_gpu_toggled)

        gpu_container.addWidget(self.gpu_checkbox)
        layout.addLayout(gpu_container)

    def _on_gpu_toggled(self, checked: bool) -> None:
        if checked != self.session.state.gpu_enabled:
            self.session.set_gpu_enabled(checked)
