from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from negpy.desktop.controller import AppController
from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.system.paths import get_resource_path
from negpy.kernel.system.version import get_app_version


class SidebarHeader(QWidget):
    """
    Top header for the sidebar containing the logo and version.
    """

    def __init__(self, controller: AppController):
        super().__init__()
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
