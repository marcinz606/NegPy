import os
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QComboBox,
    QCheckBox,
    QRadioButton,
    QHBoxLayout,
    QLabel,
    QGroupBox,
)
from src.desktop.controller import AppController
from src.infrastructure.display.color_mgmt import ColorService


class ICCSidebar(QWidget):
    """
    Panel for custom ICC profile application and soft-proofing.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.session = controller.session
        self.state = controller.state

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 5)

        # Profile Selection
        available = ColorService.get_available_profiles()
        self.profiles = ["None"] + available

        self.profile_combo = QComboBox()
        self.profile_combo.addItems([os.path.basename(p) for p in self.profiles])

        current = self.state.icc_profile_path
        if current:
            self.profile_combo.setCurrentText(os.path.basename(current))
        else:
            self.profile_combo.setCurrentText("None")

        # Direction (Input/Output)
        self.mode_group = QGroupBox("Direction")
        mode_layout = QHBoxLayout(self.mode_group)
        self.radio_input = QRadioButton("Input")
        self.radio_output = QRadioButton("Output")

        if self.state.icc_invert:
            self.radio_input.setChecked(True)
        else:
            self.radio_output.setChecked(True)

        mode_layout.addWidget(self.radio_input)
        mode_layout.addWidget(self.radio_output)

        # Export Toggle
        self.apply_export_check = QCheckBox("Apply to Export")
        self.apply_export_check.setChecked(self.state.apply_icc_to_export)

        layout.addWidget(QLabel("Profile:"))
        layout.addWidget(self.profile_combo)
        layout.addWidget(self.mode_group)
        layout.addWidget(self.apply_export_check)

    def _connect_signals(self) -> None:
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        self.radio_input.toggled.connect(self._on_mode_changed)
        self.apply_export_check.toggled.connect(self._on_apply_changed)

    def _on_profile_changed(self, index: int) -> None:
        path = self.profiles[index]
        self.state.icc_profile_path = path if path != "None" else None
        self.controller.request_render()

    def _on_mode_changed(self) -> None:
        self.state.icc_invert = self.radio_input.isChecked()
        self.controller.request_render()

    def _on_apply_changed(self, checked: bool) -> None:
        self.state.apply_icc_to_export = checked
        self.controller.request_render()

    def sync_ui(self) -> None:
        """
        Updates widgets from current state.
        """
        self.profile_combo.blockSignals(True)
        self.radio_input.blockSignals(True)
        self.radio_output.blockSignals(True)
        self.apply_export_check.blockSignals(True)

        try:
            path = self.state.icc_profile_path
            if path:
                self.profile_combo.setCurrentText(os.path.basename(path))
            else:
                self.profile_combo.setCurrentText("None")

            if self.state.icc_invert:
                self.radio_input.setChecked(True)
            else:
                self.radio_output.setChecked(True)

            self.apply_export_check.setChecked(self.state.apply_icc_to_export)
        finally:
            self.profile_combo.blockSignals(False)
            self.radio_input.blockSignals(False)
            self.radio_output.blockSignals(False)
            self.apply_export_check.blockSignals(False)
