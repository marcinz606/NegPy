from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QHBoxLayout,
    QLineEdit,
)
from src.desktop.controller import AppController
from src.services.assets.presets import Presets
from src.domain.models import WorkspaceConfig


class PresetsSidebar(QWidget):
    """
    Panel for saving and loading editing presets.
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

        # Load Row
        row_load = QHBoxLayout()
        self.preset_combo = QComboBox()
        self._refresh_presets()

        self.load_btn = QPushButton("Load")
        row_load.addWidget(self.preset_combo, stretch=1)
        row_load.addWidget(self.load_btn)

        layout.addLayout(row_load)

        # Save Row
        row_save = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("New Preset Name")
        self.save_btn = QPushButton("Save")

        row_save.addWidget(self.name_input, stretch=1)
        row_save.addWidget(self.save_btn)

        layout.addLayout(row_save)

    def _connect_signals(self) -> None:
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.save_btn.clicked.connect(self._on_save_clicked)

    def _on_load_clicked(self) -> None:
        name = self.preset_combo.currentText()
        if not name or not self.state.current_file_hash:
            return

        p_settings = Presets.load_preset(name)
        if p_settings:
            current_dict = self.state.config.to_dict()
            current_dict.update(p_settings)
            new_config = WorkspaceConfig.from_flat_dict(current_dict)

            self.session.update_config(new_config)
            self.controller.request_render()

    def _on_save_clicked(self) -> None:
        name = self.name_input.text()
        if not name or not self.state.current_file_hash:
            return

        Presets.save_preset(name, self.state.config)
        self._refresh_presets()
        self.name_input.clear()

    def _refresh_presets(self) -> None:
        self.preset_combo.clear()
        self.preset_combo.addItems(Presets.list_presets())

    def sync_ui(self) -> None:
        """
        Updates widgets from current state.
        """
        self._refresh_presets()
