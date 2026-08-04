import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QToolButton,
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap

from negpy.desktop.controller import AppController
from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.system.paths import get_resource_path
from negpy.kernel.system.version import get_app_version


class SidebarHeader(QWidget):
    """
    Top header for the sidebar containing the logo and version.

    The branding collapses to a single chevron row so the film strip can claim the
    space; the caller persists ``expanded_changed`` (see the sidebar sections).
    """

    expanded_changed = pyqtSignal(bool)

    def __init__(self, controller: AppController, expanded: bool = True):
        super().__init__()
        self._init_ui()
        self.set_expanded(expanded)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 0)
        layout.setSpacing(2)

        self.toggle_button = QToolButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setIconSize(QSize(10, 10))
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setStyleSheet("QToolButton { border: none; background: transparent; padding: 0px; }")
        self.toggle_button.toggled.connect(self.set_expanded)

        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.addStretch()
        toggle_row.addWidget(self.toggle_button)
        layout.addLayout(toggle_row)

        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(5)

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
        body_layout.addLayout(header)

        self.ver_label = QLabel(f"v{get_app_version()}")
        self.ver_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ver_label.setStyleSheet(f"font-size: 14px; color: {THEME.text_secondary}; font-weight: bold;")
        body_layout.addWidget(self.ver_label)

        layout.addWidget(self.body)

    def is_expanded(self) -> bool:
        return self.toggle_button.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        if self.toggle_button.isChecked() != expanded:
            self.toggle_button.blockSignals(True)
            self.toggle_button.setChecked(expanded)
            self.toggle_button.blockSignals(False)
        self.body.setVisible(expanded)
        self.toggle_button.setIcon(qta.icon("fa5s.chevron-up" if expanded else "fa5s.chevron-down", color=THEME.text_muted))
        self.toggle_button.setToolTip("Hide the NegPy logo and version" if expanded else "Show the NegPy logo and version")
        self.expanded_changed.emit(expanded)
