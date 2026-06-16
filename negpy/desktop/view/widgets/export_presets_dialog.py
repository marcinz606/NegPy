import os
import uuid

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.theme import THEME
from negpy.domain.models import (
    AspectRatio,
    ColorSpace,
    ExportFormat,
    ExportPreset,
    ExportPresetOutputMode,
    ExportResolutionMode,
)
from negpy.infrastructure.display.color_mgmt import ColorService
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry


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
        self.enabled_check.stateChanged.connect(lambda _: self._on_field_changed())
        row.addWidget(self.name_edit)
        row.addWidget(self.enabled_check)
        form.addLayout(row)

        form.addWidget(self._section("FORMAT"))

        fmt_row = QHBoxLayout()
        fmt_label = QLabel("Format")
        fmt_label.setFixedWidth(90)
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems([f.value for f in ExportFormat])
        self.fmt_combo.currentTextChanged.connect(self._on_fmt_changed)
        fmt_row.addWidget(fmt_label)
        fmt_row.addWidget(self.fmt_combo)
        form.addLayout(fmt_row)

        quality_row = QHBoxLayout()
        quality_label = QLabel("JPEG Quality")
        quality_label.setFixedWidth(90)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(90)
        self.quality_spin.valueChanged.connect(lambda _: self._on_field_changed())
        quality_row.addWidget(quality_label)
        quality_row.addWidget(self.quality_spin)
        quality_row.addStretch()
        self._quality_container = QWidget()
        self._quality_container.setLayout(quality_row)
        form.addWidget(self._quality_container)

        form.addWidget(self._section("SIZE"))

        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self.mode_original_btn = QPushButton("Original")
        self.mode_print_btn = QPushButton("Print")
        self.mode_target_px_btn = QPushButton("Pixels")
        btn_style = f"font-size: {THEME.font_size_base}px; padding: 8px;"
        for btn in (self.mode_original_btn, self.mode_print_btn, self.mode_target_px_btn):
            btn.setCheckable(True)
            btn.setStyleSheet(btn_style)
            mode_row.addWidget(btn)
        mode_row.addStretch()
        self.mode_btn_group = QButtonGroup(self)
        self.mode_btn_group.setExclusive(True)
        self.mode_btn_group.addButton(self.mode_original_btn, 0)
        self.mode_btn_group.addButton(self.mode_print_btn, 1)
        self.mode_btn_group.addButton(self.mode_target_px_btn, 2)
        self.mode_btn_group.idToggled.connect(self._on_mode_toggled)
        form.addLayout(mode_row)

        # Print mode controls
        self._print_container = QWidget()
        print_inner = QHBoxLayout(self._print_container)
        print_inner.setContentsMargins(0, 0, 0, 0)
        vbox_size = QVBoxLayout()
        vbox_size.addWidget(QLabel('Size <span style="color: #666; font-size: 10px;">cm</span>'))
        self.size_input = QDoubleSpinBox()
        self.size_input.setRange(1.0, 500.0)
        self.size_input.setValue(30.0)
        self.size_input.valueChanged.connect(lambda _: self._on_field_changed())
        vbox_size.addWidget(self.size_input)
        vbox_dpi = QVBoxLayout()
        vbox_dpi.addWidget(QLabel("DPI"))
        self.dpi_input = QSpinBox()
        self.dpi_input.setRange(72, 4800)
        self.dpi_input.setValue(300)
        self.dpi_input.valueChanged.connect(lambda _: self._on_field_changed())
        vbox_dpi.addWidget(self.dpi_input)
        print_inner.addLayout(vbox_size)
        print_inner.addLayout(vbox_dpi)
        print_inner.addStretch()
        form.addWidget(self._print_container)

        # Target px controls
        self._target_px_container = QWidget()
        target_px_inner = QVBoxLayout(self._target_px_container)
        target_px_inner.setContentsMargins(0, 0, 0, 0)
        target_px_inner.addWidget(QLabel('Long edge <span style="color: #666; font-size: 10px;">px</span>'))
        self.target_px_input = QSpinBox()
        self.target_px_input.setRange(256, 32768)
        self.target_px_input.setValue(2000)
        self.target_px_input.valueChanged.connect(lambda _: self._on_field_changed())
        target_px_inner.addWidget(self.target_px_input)
        form.addWidget(self._target_px_container)

        ratio_row = QHBoxLayout()
        ratio_label = QLabel("Paper ratio")
        ratio_label.setFixedWidth(90)
        self.ratio_combo = QComboBox()
        ratios = [AspectRatio.ORIGINAL] + [r.value for r in AspectRatio if r != AspectRatio.ORIGINAL]
        self.ratio_combo.addItems(ratios)
        self.ratio_combo.currentTextChanged.connect(lambda _: self._on_field_changed())
        ratio_row.addWidget(ratio_label)
        ratio_row.addWidget(self.ratio_combo)
        form.addLayout(ratio_row)

        form.addWidget(self._section("OUTPUT"))

        output_mode_row = QHBoxLayout()
        output_mode_label = QLabel("Folder")
        output_mode_label.setFixedWidth(90)
        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItem("Subfolder of source", ExportPresetOutputMode.SUBFOLDER_OF_SOURCE)
        self.output_mode_combo.addItem("Same as source", ExportPresetOutputMode.SAME_AS_SOURCE)
        self.output_mode_combo.addItem("Absolute path", ExportPresetOutputMode.ABSOLUTE)
        self.output_mode_combo.currentIndexChanged.connect(self._on_output_mode_changed)
        output_mode_row.addWidget(output_mode_label)
        output_mode_row.addWidget(self.output_mode_combo)
        form.addLayout(output_mode_row)

        # Subfolder name (shown for SUBFOLDER_OF_SOURCE)
        self._subfolder_container = QWidget()
        sf_inner = QHBoxLayout(self._subfolder_container)
        sf_inner.setContentsMargins(0, 0, 0, 0)
        sf_label = QLabel("Subfolder")
        sf_label.setFixedWidth(90)
        self.subfolder_edit = QLineEdit()
        self.subfolder_edit.setPlaceholderText("e.g. TIFF")
        self.subfolder_edit.textChanged.connect(lambda _: self._on_field_changed())
        sf_inner.addWidget(sf_label)
        sf_inner.addWidget(self.subfolder_edit)
        form.addWidget(self._subfolder_container)

        # Absolute path (shown for ABSOLUTE)
        self._abspath_container = QWidget()
        ap_inner = QHBoxLayout(self._abspath_container)
        ap_inner.setContentsMargins(0, 0, 0, 0)
        ap_label = QLabel("Path")
        ap_label.setFixedWidth(90)
        self.abspath_edit = QLineEdit()
        self.abspath_edit.textChanged.connect(lambda _: self._on_field_changed())
        self.abspath_browse_btn = QPushButton()
        self.abspath_browse_btn.setIcon(qta.icon("fa5s.folder-open", color=THEME.text_primary))
        self.abspath_browse_btn.setFixedWidth(36)
        self.abspath_browse_btn.clicked.connect(self._browse_output_path)
        ap_inner.addWidget(ap_label)
        ap_inner.addWidget(self.abspath_edit)
        ap_inner.addWidget(self.abspath_browse_btn)
        form.addWidget(self._abspath_container)

        filename_row = QHBoxLayout()
        fn_label = QLabel("Filename")
        fn_label.setFixedWidth(90)
        self.filename_edit = QLineEdit()
        self.filename_edit.setToolTip(
            "Jinja2 template. Variables:\n"
            "{{ original_name }}, {{ colorspace }}, {{ format }},\n"
            "{{ paper_ratio }}, {{ size }}, {{ dpi }}, {{ target_px }},\n"
            "{{ border }}, {{ date }}"
        )
        self.filename_edit.textChanged.connect(lambda _: self._on_field_changed())
        filename_row.addWidget(fn_label)
        filename_row.addWidget(self.filename_edit)
        form.addLayout(filename_row)

        self.overwrite_check = QCheckBox("Overwrite existing files")
        self.overwrite_check.stateChanged.connect(lambda _: self._on_field_changed())
        form.addWidget(self.overwrite_check)

        form.addWidget(self._section("COLOR"))

        # Output profiles
        enum_mapped = {ColorSpaceRegistry.get_icc_path(cs.value) for cs in ColorSpace}
        enum_mapped.discard(None)
        custom_profiles = [p for p in ColorService.get_available_profiles() if p not in enum_mapped]

        cs_row = QHBoxLayout()
        cs_label = QLabel("Color space")
        cs_label.setFixedWidth(90)
        self.color_space_combo = QComboBox()
        self.color_space_combo.addItems([cs.value for cs in ColorSpace])
        self.color_space_combo.currentTextChanged.connect(lambda _: self._on_field_changed())
        cs_row.addWidget(cs_label)
        cs_row.addWidget(self.color_space_combo)
        form.addLayout(cs_row)

        icc_row = QHBoxLayout()
        icc_label = QLabel("Output ICC")
        icc_label.setFixedWidth(90)
        self._icc_profiles = ["None"] + custom_profiles
        self.icc_output_combo = QComboBox()
        self.icc_output_combo.addItems([os.path.basename(p) for p in self._icc_profiles])
        self.icc_output_combo.setToolTip("Custom output ICC profile (overrides color space)")
        self.icc_output_combo.currentIndexChanged.connect(lambda _: self._on_field_changed())
        icc_row.addWidget(icc_label)
        icc_row.addWidget(self.icc_output_combo)
        form.addLayout(icc_row)

        fl.addWidget(self._form_container)

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {THEME.text_muted}; font-size: 10px; font-weight: bold; "
            f"letter-spacing: 1px; border-bottom: 1px solid {THEME.border_primary}; padding-bottom: 2px;"
        )
        return lbl

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
            self.fmt_combo.setCurrentText(preset.export_fmt)
            self.quality_spin.setValue(preset.jpeg_quality)
            self._select_mode_button(preset.export_resolution_mode)
            self._update_mode_visibility(preset.export_resolution_mode)
            self.size_input.setValue(preset.export_print_size)
            self.dpi_input.setValue(preset.export_dpi)
            self.target_px_input.setValue(preset.export_target_long_edge_px)
            self.ratio_combo.setCurrentText(preset.paper_aspect_ratio)

            # Output mode
            idx = self.output_mode_combo.findData(preset.output_mode)
            if idx >= 0:
                self.output_mode_combo.setCurrentIndex(idx)
            self._update_output_mode_visibility(preset.output_mode)
            self.subfolder_edit.setText(preset.output_subfolder)
            self.abspath_edit.setText(preset.output_path)
            self.filename_edit.setText(preset.filename_pattern)
            self.overwrite_check.setChecked(preset.overwrite)
            self.color_space_combo.setCurrentText(preset.export_color_space)

            icc_name = os.path.basename(preset.icc_output_path) if preset.icc_output_path else "None"
            self.icc_output_combo.setCurrentText(icc_name)

            self._quality_container.setVisible(preset.export_fmt == ExportFormat.JPEG)
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

    def _on_fmt_changed(self, fmt: str) -> None:
        if self._updating_form or self._selected_idx < 0:
            return
        self._quality_container.setVisible(fmt == ExportFormat.JPEG)
        self._on_field_changed()

    def _on_mode_toggled(self, _id: int, checked: bool) -> None:
        if not checked or self._updating_form:
            return
        mode = self._mode_by_id()[_id]
        self._update_mode_visibility(mode)
        self._on_field_changed()

    def _on_output_mode_changed(self, _idx: int) -> None:
        if self._updating_form:
            return
        mode = self.output_mode_combo.currentData()
        self._update_output_mode_visibility(mode)
        self._on_field_changed()

    def _on_field_changed(self) -> None:
        if self._updating_form or self._selected_idx < 0:
            return
        self._write_form_to_preset(self._presets[self._selected_idx])
        item = self.preset_list.item(self._selected_idx)
        if item:
            item.setText(self._presets[self._selected_idx].name)
            item.setCheckState(Qt.CheckState.Checked if self._presets[self._selected_idx].enabled else Qt.CheckState.Unchecked)
        self._emit_changed()

    def _write_form_to_preset(self, preset: ExportPreset) -> None:
        preset.name = self.name_edit.text() or "Untitled"
        preset.enabled = self.enabled_check.isChecked()
        preset.export_fmt = self.fmt_combo.currentText()
        preset.jpeg_quality = self.quality_spin.value()
        preset.export_resolution_mode = self._current_mode_value()
        preset.export_print_size = self.size_input.value()
        preset.export_dpi = self.dpi_input.value()
        preset.export_target_long_edge_px = self.target_px_input.value()
        preset.paper_aspect_ratio = self.ratio_combo.currentText()
        mode_data = self.output_mode_combo.currentData()
        preset.output_mode = mode_data if mode_data else ExportPresetOutputMode.SAME_AS_SOURCE
        preset.output_subfolder = self.subfolder_edit.text()
        preset.output_path = self.abspath_edit.text()
        preset.filename_pattern = self.filename_edit.text() or "{{ original_name }}"
        preset.overwrite = self.overwrite_check.isChecked()
        preset.export_color_space = self.color_space_combo.currentText()
        icc_idx = self.icc_output_combo.currentIndex()
        preset.icc_output_path = self._icc_profiles[icc_idx] if icc_idx > 0 else None

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

    def _browse_output_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.abspath_edit.text())
        if path:
            self.abspath_edit.setText(path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_preset(p: ExportPreset) -> ExportPreset:
        return ExportPreset.from_dict(p.to_dict())

    def _emit_changed(self) -> None:
        self.presets_changed.emit(list(self._presets))

    _MODE_BY_ID = {
        0: ExportResolutionMode.ORIGINAL.value,
        1: ExportResolutionMode.PRINT.value,
        2: ExportResolutionMode.TARGET_PX.value,
    }
    _ID_BY_MODE = {v: k for k, v in _MODE_BY_ID.items()}

    @classmethod
    def _mode_by_id(cls) -> dict:
        return cls._MODE_BY_ID

    def _current_mode_value(self) -> str:
        return self._MODE_BY_ID.get(self.mode_btn_group.checkedId(), ExportResolutionMode.ORIGINAL.value)

    def _select_mode_button(self, mode_value: str) -> None:
        btn_id = self._ID_BY_MODE.get(mode_value, 0)
        btn = self.mode_btn_group.button(btn_id)
        if btn:
            btn.setChecked(True)

    def _update_mode_visibility(self, mode_value: str) -> None:
        self._print_container.setVisible(mode_value == ExportResolutionMode.PRINT.value)
        self._target_px_container.setVisible(mode_value == ExportResolutionMode.TARGET_PX.value)

    def _update_output_mode_visibility(self, mode) -> None:
        self._subfolder_container.setVisible(mode == ExportPresetOutputMode.SUBFOLDER_OF_SOURCE)
        self._abspath_container.setVisible(mode == ExportPresetOutputMode.ABSOLUTE)
