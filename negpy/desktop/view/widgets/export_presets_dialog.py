import uuid

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.export_settings_form import ExportSettingsForm
from negpy.domain.models import ExportPreset


class ExportPresetsDialog(QDialog):
    """Modal dialog for managing export presets."""

    presets_changed = pyqtSignal(list)  # emits updated list[ExportPreset]

    def __init__(self, presets: list, parent=None):
        super().__init__(parent)
        self._presets: list[ExportPreset] = [self._copy_preset(p) for p in presets]
        self._selected_idx: int = -1
        self._updating_form = False

        self.setWindowTitle("Export Presets")
        self.resize(860, 620)
        self._init_ui()
        if self._presets:
            self._select_row(0)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left: preset list + action buttons
        left = QWidget()
        left.setFixedWidth(220)
        left.setStyleSheet(f"background: {THEME.bg_panel}; border-right: 1px solid {THEME.border_primary};")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        list_label = QLabel("PRESETS")
        list_label.setStyleSheet(f"color: {THEME.text_muted}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        left_layout.addWidget(list_label)

        self.preset_list = QListWidget()
        self.preset_list.setStyleSheet(f"""
            QListWidget {{ background: {THEME.bg_dark}; border: 1px solid {THEME.border_primary}; }}
            QListWidget::item {{ padding: 8px; color: {THEME.text_primary}; }}
            QListWidget::item:selected {{ background: #2a2a2a; color: white; }}
        """)
        self.preset_list.currentRowChanged.connect(self._on_list_selection_changed)
        left_layout.addWidget(self.preset_list)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton()
        self.add_btn.setIcon(qta.icon("fa5s.plus", color=THEME.text_primary))
        self.add_btn.setToolTip("Add preset")
        self.add_btn.setFixedWidth(36)
        self.add_btn.clicked.connect(self._add_preset)

        self.dup_btn = QPushButton()
        self.dup_btn.setIcon(qta.icon("fa5s.copy", color=THEME.text_primary))
        self.dup_btn.setToolTip("Duplicate preset")
        self.dup_btn.setFixedWidth(36)
        self.dup_btn.clicked.connect(self._duplicate_preset)

        self.del_btn = QPushButton()
        self.del_btn.setIcon(qta.icon("fa5s.trash-alt", color=THEME.text_primary))
        self.del_btn.setToolTip("Delete preset")
        self.del_btn.setFixedWidth(36)
        self.del_btn.clicked.connect(self._delete_preset)

        self.up_btn = QPushButton()
        self.up_btn.setIcon(qta.icon("fa5s.arrow-up", color=THEME.text_primary))
        self.up_btn.setToolTip("Move up")
        self.up_btn.setFixedWidth(36)
        self.up_btn.clicked.connect(self._move_up)

        self.down_btn = QPushButton()
        self.down_btn.setIcon(qta.icon("fa5s.arrow-down", color=THEME.text_primary))
        self.down_btn.setToolTip("Move down")
        self.down_btn.setFixedWidth(36)
        self.down_btn.clicked.connect(self._move_down)

        for btn in (self.add_btn, self.dup_btn, self.del_btn, self.up_btn, self.down_btn):
            btn_row.addWidget(btn)
        btn_row.addStretch()
        left_layout.addLayout(btn_row)

        root.addWidget(left)

        # Right: edit form in a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {THEME.bg_dark}; }}")

        form_widget = QWidget()
        form_widget.setStyleSheet(f"background: {THEME.bg_dark};")
        self._form_layout = QVBoxLayout(form_widget)
        self._form_layout.setContentsMargins(20, 20, 20, 20)
        self._form_layout.setSpacing(14)

        self._build_form()
        self._form_layout.addStretch()

        scroll.setWidget(form_widget)
        root.addWidget(scroll)

        self._rebuild_list()

    def _build_form(self) -> None:
        fl = self._form_layout

        self._no_preset_label = QLabel("No preset selected. Add one with the + button.")
        self._no_preset_label.setStyleSheet(f"color: {THEME.text_muted};")
        fl.addWidget(self._no_preset_label)

        self._form_container = QWidget()
        form = QVBoxLayout(self._form_container)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(14)

        # Name & enabled
        row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Preset name")
        self.name_edit.textChanged.connect(self._on_name_changed)
        self.enabled_check = QCheckBox("Enabled")
        self.enabled_check.stateChanged.connect(self._on_enabled_changed)
        row.addWidget(self.name_edit)
        row.addWidget(self.enabled_check)
        form.addLayout(row)

        # Shared FORMAT / SIZE / COLOR / DESTINATION rows.
        self.form = ExportSettingsForm()
        self.form.changed.connect(self._on_form_changed)
        form.addWidget(self.form)

        fl.addWidget(self._form_container)

    # ------------------------------------------------------------------
    # List management
    # ------------------------------------------------------------------

    def _rebuild_list(self) -> None:
        self.preset_list.blockSignals(True)
        self.preset_list.clear()
        for p in self._presets:
            item = QListWidgetItem(p.name)
            item.setCheckState(Qt.CheckState.Checked if p.enabled else Qt.CheckState.Unchecked)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self.preset_list.addItem(item)
        self.preset_list.blockSignals(False)

        has = len(self._presets) > 0
        self._no_preset_label.setVisible(not has)
        self._form_container.setVisible(has)

    def _select_row(self, idx: int) -> None:
        if 0 <= idx < len(self._presets):
            self._selected_idx = idx
            self.preset_list.blockSignals(True)
            self.preset_list.setCurrentRow(idx)
            self.preset_list.blockSignals(False)
            self._populate_form(self._presets[idx])

    def _on_list_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._presets):
            return
        # Check if enabled state toggled via checkbox
        item = self.preset_list.item(row)
        if item and self._selected_idx == row:
            enabled = item.checkState() == Qt.CheckState.Checked
            if self._presets[row].enabled != enabled:
                self._presets[row].enabled = enabled
                self.enabled_check.setChecked(enabled)
                self._emit_changed()
                return
        self._select_row(row)

    # ------------------------------------------------------------------
    # Form population and change handling
    # ------------------------------------------------------------------

    def _populate_form(self, preset: ExportPreset) -> None:
        self._updating_form = True
        try:
            self.name_edit.setText(preset.name)
            self.enabled_check.setChecked(preset.enabled)
            self.form.load(preset.to_dict())
        finally:
            self._updating_form = False

    def _on_name_changed(self, text: str) -> None:
        if self._updating_form or self._selected_idx < 0:
            return
        self._presets[self._selected_idx].name = text
        item = self.preset_list.item(self._selected_idx)
        if item:
            item.setText(text)
        self._emit_changed()

    def _on_enabled_changed(self, _state: int) -> None:
        if self._updating_form or self._selected_idx < 0:
            return
        self._presets[self._selected_idx].enabled = self.enabled_check.isChecked()
        item = self.preset_list.item(self._selected_idx)
        if item:
            item.setCheckState(Qt.CheckState.Checked if self.enabled_check.isChecked() else Qt.CheckState.Unchecked)
        self._emit_changed()

    def _on_form_changed(self) -> None:
        if self._updating_form or self._selected_idx < 0:
            return
        self._write_form_to_preset(self._presets[self._selected_idx])
        self._emit_changed()

    def _write_form_to_preset(self, preset: ExportPreset) -> None:
        vals = self.form.values()
        preset.export_fmt = vals["export_fmt"]
        preset.jpeg_quality = vals["jpeg_quality"]
        preset.export_resolution_mode = vals["export_resolution_mode"]
        preset.export_print_size = vals["export_print_size"]
        preset.export_dpi = vals["export_dpi"]
        preset.export_target_long_edge_px = vals["export_target_long_edge_px"]
        preset.paper_aspect_ratio = vals["paper_aspect_ratio"]
        preset.output_mode = vals["output_mode"]
        preset.output_subfolder = vals["output_subfolder"]
        preset.output_path = vals["output_path"]
        preset.filename_pattern = vals["filename_pattern"] or "{{ original_name }}"
        preset.overwrite = vals["overwrite"]
        preset.export_color_space = vals["export_color_space"]
        preset.icc_input_path = vals["icc_input_path"]
        preset.icc_output_path = vals["icc_output_path"]

    # ------------------------------------------------------------------
    # Preset actions
    # ------------------------------------------------------------------

    def _add_preset(self) -> None:
        preset = ExportPreset(id=str(uuid.uuid4()), name="New Preset")
        self._presets.append(preset)
        self._rebuild_list()
        self._select_row(len(self._presets) - 1)
        self._emit_changed()

    def _duplicate_preset(self) -> None:
        if self._selected_idx < 0:
            return
        src = self._presets[self._selected_idx]
        dup = ExportPreset.from_dict({**src.to_dict(), "id": str(uuid.uuid4()), "name": f"{src.name} Copy"})
        self._presets.insert(self._selected_idx + 1, dup)
        self._rebuild_list()
        self._select_row(self._selected_idx + 1)
        self._emit_changed()

    def _delete_preset(self) -> None:
        if self._selected_idx < 0 or not self._presets:
            return
        self._presets.pop(self._selected_idx)
        new_idx = min(self._selected_idx, len(self._presets) - 1)
        self._rebuild_list()
        if new_idx >= 0:
            self._select_row(new_idx)
        else:
            self._selected_idx = -1
            self._no_preset_label.setVisible(True)
            self._form_container.setVisible(False)
        self._emit_changed()

    def _move_up(self) -> None:
        idx = self._selected_idx
        if idx <= 0:
            return
        self._presets[idx - 1], self._presets[idx] = self._presets[idx], self._presets[idx - 1]
        self._rebuild_list()
        self._select_row(idx - 1)
        self._emit_changed()

    def _move_down(self) -> None:
        idx = self._selected_idx
        if idx < 0 or idx >= len(self._presets) - 1:
            return
        self._presets[idx], self._presets[idx + 1] = self._presets[idx + 1], self._presets[idx]
        self._rebuild_list()
        self._select_row(idx + 1)
        self._emit_changed()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_preset(p: ExportPreset) -> ExportPreset:
        return ExportPreset.from_dict(p.to_dict())

    def _emit_changed(self) -> None:
        self.presets_changed.emit(list(self._presets))
