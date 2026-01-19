from PyQt6.QtWidgets import (
    QComboBox,
    QPushButton,
    QHBoxLayout,
    QLineEdit,
)
import qtawesome as qta
from src.desktop.view.sidebar.base import BaseSidebar
from src.services.assets.presets import Presets
from src.domain.models import WorkspaceConfig
from src.desktop.view.styles.theme import THEME


class PresetsSidebar(BaseSidebar):
    """
    Panel for saving and loading editing presets.
    """

    def _init_ui(self) -> None:
        # Load Row
        row_load = QHBoxLayout()
        self.preset_combo = QComboBox()
        self._refresh_presets()

        self.load_btn = QPushButton(" Load")
        self.load_btn.setIcon(qta.icon("fa5s.upload", color=THEME.text_primary))

        row_load.addWidget(self.preset_combo, stretch=1)
        row_load.addWidget(self.load_btn)
        self.layout.addLayout(row_load)

        # Save Row
        row_save = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("New Preset Name")
        self.save_btn = QPushButton(" Save")
        self.save_btn.setIcon(qta.icon("fa5s.save", color=THEME.text_primary))

        row_save.addWidget(self.name_input, stretch=1)
        row_save.addWidget(self.save_btn)
        self.layout.addLayout(row_save)

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
            self.controller.session.update_config(new_config)
            self.controller.request_render()

    def _on_save_clicked(self) -> None:
        name = self.name_input.text()
        if not name or not self.state.current_file_hash:
            return

        Presets.save_preset(name, self.state.config)
        self._refresh_presets()
        self.name_input.clear()

    def _refresh_presets(self) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(Presets.list_presets())
        self.preset_combo.blockSignals(False)

    def sync_ui(self) -> None:
        self._refresh_presets()
