from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
)
import qtawesome as qta
from negpy.desktop.settings_catalog import (
    CATALOG,
    preset_summary,
    selected_flat_dict,
)
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog
from negpy.domain.models import WorkspaceConfig
from negpy.services.assets.presets import Presets

_PRESET_EXCLUDED_SECTIONS = frozenset({"Crop", "Rotation"})

# "Replace edits" resets only the look sections; per-frame geometry, frame
# metadata and export prefs stay (dust/heal/masks aren't catalog rows at all).
_REPLACE_KEPT_SECTIONS = frozenset({"Crop", "Rotation", "Metadata", "Export"})
_LOOK_ROWS = tuple(r for title, rows in CATALOG if title not in _REPLACE_KEPT_SECTIONS for r in rows)


class PresetsSidebar(BaseSidebar):
    """
    Panel for saving and applying editing presets.
    """

    def _init_ui(self) -> None:
        self.preset_list = QListWidget()
        self.preset_list.setObjectName("preset_list")
        self.preset_list.setMaximumHeight(180)
        self._refresh_presets()
        self.layout.addWidget(self.preset_list)

        row = QHBoxLayout()
        self.apply_btn = QPushButton(" Apply")
        self.apply_btn.setIcon(qta.icon("fa5s.check", color=THEME.text_primary))
        self.apply_btn.setToolTip("Apply the selected preset to the current image (or double-click a preset)")

        self.save_btn = QPushButton(" Save…")
        self.save_btn.setIcon(qta.icon("fa5s.save", color=THEME.text_primary))
        self.save_btn.setToolTip("Pick which of the current settings to store as a new preset")

        self.edit_btn = QPushButton()
        self.edit_btn.setIcon(qta.icon("fa5s.pen", color=THEME.text_primary))
        self.edit_btn.setToolTip("Edit the selected preset — rename it or change which settings it stores")
        self.edit_btn.setFixedWidth(32)

        self.delete_btn = QPushButton()
        self.delete_btn.setIcon(qta.icon("fa5s.trash", color=THEME.text_primary))
        self.delete_btn.setToolTip("Delete the selected preset")
        self.delete_btn.setFixedWidth(32)

        row.addWidget(self.apply_btn, stretch=1)
        row.addWidget(self.save_btn, stretch=1)
        row.addWidget(self.edit_btn)
        row.addWidget(self.delete_btn)
        self.layout.addLayout(row)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.preset_list.itemDoubleClicked.connect(self._apply_preset)
        self.apply_btn.clicked.connect(self._apply_preset)
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.edit_btn.clicked.connect(self._on_edit_clicked)
        self.delete_btn.clicked.connect(self._on_delete_clicked)

    def _current_name(self) -> str:
        item = self.preset_list.currentItem()
        return item.text() if item is not None else ""

    def _apply_preset(self) -> None:
        name = self._current_name()
        if not name or not self.state.current_file_hash:
            return
        preset_cfg = self._preset_config(name)
        if preset_cfg is None:
            return

        visible = self.controller.session.asset_model.visible_actual_indices()
        sel_count = len([i for i in set(self.state.selected_indices) if i in visible])
        dlg = GranularSettingsDialog(
            self,
            preset_cfg,
            name,
            show_scope=True,
            show_current=True,
            show_apply_mode=True,
            sel_count=sel_count,
            roll_count=len(visible),
            exclude_sections=_PRESET_EXCLUDED_SECTIONS,
        )
        dlg.setWindowTitle("Apply Preset")
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        rows = dlg.selected()
        if dlg.apply_mode() == "replace":
            rows = list(_LOOK_ROWS) + rows
        if self.controller.session.apply_preset_fields(preset_cfg, rows, dlg.scope()):
            self.controller.request_render()

    def _preset_config(self, name: str) -> WorkspaceConfig | None:
        """The preset's stored fields over defaults, so pickers show the preset's
        own values, not the current image's."""
        data = Presets.load_preset(name)
        if not data:
            return None
        base = WorkspaceConfig().to_dict()
        base.update(data)
        return WorkspaceConfig.from_flat_dict(base)

    def _on_save_clicked(self) -> None:
        if not self.state.current_file_hash:
            return
        dlg = GranularSettingsDialog(
            self,
            self.state.config,
            "current settings",
            ask_name=True,
            exclude_sections=_PRESET_EXCLUDED_SECTIONS,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            Presets.save_preset(dlg.name(), selected_flat_dict(self.state.config, dlg.selected()))
            self._refresh_presets(force=True)

    def _on_edit_clicked(self) -> None:
        name = self._current_name()
        cfg = self._preset_config(name) if name else None
        if cfg is None:
            return
        dlg = GranularSettingsDialog(self, cfg, name, ask_name=True, exclude_sections=_PRESET_EXCLUDED_SECTIONS)
        dlg.setWindowTitle("Edit Preset")
        dlg.set_name(name)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_name = dlg.name()
            Presets.save_preset(new_name, selected_flat_dict(cfg, dlg.selected()))
            if new_name != name:
                Presets.delete_preset(name)
            self._refresh_presets(force=True)

    def _on_delete_clicked(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        name = self._current_name()
        if not name:
            return
        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Delete preset '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            Presets.delete_preset(name)
            self._refresh_presets(force=True)

    def _refresh_presets(self, force: bool = False) -> None:
        # sync_ui fires on every state sync; skip the rebuild (N file reads)
        # unless the name set actually changed. force covers same-name overwrites.
        names = sorted(Presets.list_presets())
        if not force and names == [self.preset_list.item(i).text() for i in range(self.preset_list.count())]:
            return
        self.preset_list.clear()
        for name in names:
            item = QListWidgetItem(name)
            data = Presets.load_preset(name)
            if data:
                item.setToolTip(wrap_tooltip(preset_summary(data)))
            self.preset_list.addItem(item)

    def sync_ui(self) -> None:
        self._refresh_presets()
