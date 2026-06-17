import os

import qtawesome as qta
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.domain.models import AspectRatio, ColorSpace, ExportFormat, ExportResolutionMode
from negpy.infrastructure.display.color_mgmt import ColorService
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry


class ExportSidebar(BaseSidebar):
    """
    Panel for export settings, presets and batch processing.
    """

    @staticmethod
    def _icc_row_label(text: str) -> QLabel:
        """Fixed-width row label so the ICC-section dropdowns line up."""
        label = QLabel(text)
        label.setFixedWidth(52)
        return label

    def _init_ui(self) -> None:
        self.layout.setSpacing(10)
        conf = self.state.config.export

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(500)
        self.update_timer.timeout.connect(self._persist_all_export_settings)

        self._add_presets_section()
        self._add_contact_sheet_section()

        self.layout.addWidget(section_subheader("ICC"))

        # Soft proof: when off, Output/Input ICC affect export only (preview = the edit
        # on the monitor); when on, the preview simulates the Output profile.
        self.soft_proof_checkbox = QCheckBox("Soft proof")
        self.soft_proof_checkbox.setChecked(self.state.soft_proof_enabled)
        self.soft_proof_checkbox.setToolTip("Simulate the Output profile (incl. paper/printer) in the preview")
        self.layout.addWidget(self.soft_proof_checkbox)

        # Drop bundled profiles that already back a color-space enum (e.g.
        # AdobeCompat-v4.icc ↔ "Adobe RGB") so the Output list isn't duplicated.
        enum_mapped = {ColorSpaceRegistry.get_icc_path(cs.value) for cs in ColorSpace}
        enum_mapped.discard(None)
        custom_profiles = [p for p in ColorService.get_available_profiles() if p not in enum_mapped]

        # Input: source profile correction.
        self.input_profiles = ["None"] + custom_profiles
        self.input_combo = QComboBox()
        self.input_combo.addItems([os.path.basename(p) for p in self.input_profiles])
        self.input_combo.setToolTip("Source/input ICC profile — soft-proofed in the preview")
        in_path = self.state.icc_input_path
        self.input_combo.setCurrentText(os.path.basename(in_path) if in_path else "None")
        in_row = QHBoxLayout()
        in_row.addWidget(self._icc_row_label("Input"))
        in_row.addWidget(self.input_combo)
        self.layout.addLayout(in_row)

        # Output: one selector for a color space (→ export_color_space) or a custom
        # ICC profile (→ icc_output_path).
        self.output_spaces = [cs.value for cs in ColorSpace]
        self.output_profiles = custom_profiles
        self.output_map = list(self.output_spaces) + list(self.output_profiles)
        self.output_combo = QComboBox()
        self.output_combo.addItems(self.output_spaces + [os.path.basename(p) for p in self.output_profiles])
        self.output_combo.setToolTip("Output color space or custom ICC profile — soft-proofed in the preview")
        out_path = self.state.icc_output_path
        self.output_combo.setCurrentText(os.path.basename(out_path) if out_path else conf.export_color_space)
        out_row = QHBoxLayout()
        out_row.addWidget(self._icc_row_label("Output"))
        out_row.addWidget(self.output_combo)
        self.layout.addLayout(out_row)

        # Display: monitor profile the preview is shown on (preview only, not export).
        self.display_spaces = [
            ColorSpace.SRGB.value,
            ColorSpace.P3_D65.value,
            ColorSpace.ADOBE_RGB.value,
            ColorSpace.REC2020.value,
            ColorSpace.PROPHOTO.value,
        ]
        self.display_map = [None] + self.display_spaces
        self.display_combo = QComboBox()
        self.display_combo.addItems(["As detected"] + self.display_spaces)
        self.display_combo.setToolTip("Monitor profile the preview is displayed on (affects preview only, not export)")
        override = self.state.monitor_profile_override
        self.display_combo.setCurrentText(override if override in self.display_spaces else "As detected")
        disp_row = QHBoxLayout()
        disp_row.addWidget(self._icc_row_label("Display"))
        disp_row.addWidget(self.display_combo)
        self.layout.addLayout(disp_row)

        self.display_detected_label = QLabel()
        self.display_detected_label.setStyleSheet(f"color: {THEME.text_muted}; font-size: 10px;")
        self.layout.addWidget(self.display_detected_label)
        self._refresh_display_info()

        self.layout.addWidget(section_subheader("FORMAT"))

        fmt_row = QHBoxLayout()
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems([f.value for f in ExportFormat])
        self.fmt_combo.setCurrentText(conf.export_fmt)
        self.fmt_combo.setToolTip("File format")
        self.ratio_combo = QComboBox()
        ratios = [AspectRatio.ORIGINAL] + [r.value for r in AspectRatio if r != AspectRatio.ORIGINAL]
        self.ratio_combo.addItems(ratios)
        self.ratio_combo.setCurrentText(conf.paper_aspect_ratio)
        self.ratio_combo.setToolTip("Paper aspect ratio")
        fmt_row.addWidget(self.fmt_combo)
        fmt_row.addWidget(self.ratio_combo)
        self.layout.addLayout(fmt_row)

        self.layout.addWidget(section_subheader("SIZE"))

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
        self.mode_btn_group = QButtonGroup(self)
        self.mode_btn_group.setExclusive(True)
        self.mode_btn_group.addButton(self.mode_original_btn, 0)
        self.mode_btn_group.addButton(self.mode_print_btn, 1)
        self.mode_btn_group.addButton(self.mode_target_px_btn, 2)
        self.layout.addLayout(mode_row)

        # PRINT mode: cm + DPI
        self.print_size_container = QWidget()
        print_layout = QVBoxLayout(self.print_size_container)
        print_layout.setContentsMargins(0, 0, 0, 0)
        print_row = QHBoxLayout()

        vbox_size = QVBoxLayout()
        size_label = QLabel('Size <span style="color: #666666; font-size: 10px;">cm</span>')
        vbox_size.addWidget(size_label)
        self.size_input = QDoubleSpinBox()
        self.size_input.setRange(1.0, 500.0)
        self.size_input.setValue(conf.export_print_size)
        vbox_size.addWidget(self.size_input)

        vbox_dpi = QVBoxLayout()
        vbox_dpi.addWidget(QLabel("DPI"))
        self.dpi_input = QSpinBox()
        self.dpi_input.setRange(72, 4800)
        self.dpi_input.setValue(conf.export_dpi)
        vbox_dpi.addWidget(self.dpi_input)

        print_row.addLayout(vbox_size)
        print_row.addLayout(vbox_dpi)
        print_layout.addLayout(print_row)
        self.layout.addWidget(self.print_size_container)

        # TARGET_PX mode: long edge in pixels
        self.target_px_container = QWidget()
        target_px_layout = QVBoxLayout(self.target_px_container)
        target_px_layout.setContentsMargins(0, 0, 0, 0)
        target_px_layout.addWidget(QLabel('Long edge <span style="color: #666666; font-size: 10px;">px</span>'))
        self.target_px_input = QSpinBox()
        self.target_px_input.setRange(256, 32768)
        self.target_px_input.setValue(conf.export_target_long_edge_px)
        target_px_layout.addWidget(self.target_px_input)
        self.layout.addWidget(self.target_px_container)

        self._select_mode_button(conf.export_resolution_mode)
        self._update_mode_visibility(conf.export_resolution_mode)

        self.layout.addWidget(section_subheader("DESTINATION"))

        self.pattern_input = QLineEdit(conf.filename_pattern)
        self.pattern_input.setPlaceholderText("Filename Pattern...")
        self.pattern_input.setToolTip(
            "Jinja2 Template. Available variables:\n"
            "- {{ original_name }}\n"
            "- {{ colorspace }}\n"
            "- {{ format }} (JPEG/TIFF/PNG)\n"
            "- {{ paper_ratio }}\n"
            "- {{ size }} (e.g. 20cm; PRINT mode only)\n"
            "- {{ dpi }} (PRINT mode only)\n"
            "- {{ target_px }} (e.g. 2000px; TARGET_PX mode only)\n"
            "- {{ border }} ('border' or empty)\n"
            "- {{ date }} (YYYYMMDD)"
        )
        self.layout.addWidget(self.pattern_input)

        checkbox_row = QHBoxLayout()
        self.overwrite_checkbox = QCheckBox("Overwrite existing files")
        self.overwrite_checkbox.setChecked(conf.overwrite)
        self.same_as_source_checkbox = QCheckBox("Same folder as source")
        self.same_as_source_checkbox.setChecked(conf.same_as_source)
        checkbox_row.addWidget(self.overwrite_checkbox)
        checkbox_row.addWidget(self.same_as_source_checkbox)
        self.layout.addLayout(checkbox_row)

        path_layout = QHBoxLayout()
        self.path_input = QLineEdit(conf.export_path)
        self.path_input.setToolTip("Export folder")
        self.path_input.setDisabled(conf.same_as_source)
        self.browse_btn = QPushButton()
        self.browse_btn.setIcon(qta.icon("fa5s.folder-open", color=THEME.text_primary))
        self.browse_btn.setFixedWidth(40)
        self.browse_btn.setToolTip("Choose export folder")
        self.browse_btn.setDisabled(conf.same_as_source)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_btn)
        self.layout.addLayout(path_layout)

        self.layout.addWidget(section_subheader("BATCH"))

        self.apply_all_btn = QPushButton(" Sync export settings")
        self.apply_all_btn.setFixedHeight(40)
        self.apply_all_btn.setCheckable(True)
        self.apply_all_btn.setChecked(True)
        self.apply_all_btn.setToolTip("Apply current export settings (Size, DPI, Border) to all files")
        self._update_apply_all_style(True)
        self.layout.addWidget(self.apply_all_btn)

        self.batch_export_btn = QPushButton(" Export All")
        self.batch_export_btn.setObjectName("batch_export_btn")
        self.batch_export_btn.setFixedHeight(40)
        self.batch_export_btn.setIcon(qta.icon("fa5s.images", color=THEME.text_primary))
        self.layout.addWidget(self.batch_export_btn)

        self.layout.addStretch()

        self._rebuild_preset_rows()

    def _connect_signals(self) -> None:
        self.soft_proof_checkbox.toggled.connect(self.controller.set_soft_proof)
        self.input_combo.currentIndexChanged.connect(self._on_input_changed)
        self.output_combo.currentIndexChanged.connect(self._on_output_changed)
        self.display_combo.currentIndexChanged.connect(self._on_display_changed)
        self.controller.monitor_profile_changed.connect(self._refresh_display_info)

        self.fmt_combo.currentTextChanged.connect(lambda _: self.update_timer.start())
        self.ratio_combo.currentTextChanged.connect(lambda _: self.update_timer.start())
        self.mode_btn_group.idToggled.connect(self._on_mode_toggled)
        self.size_input.valueChanged.connect(lambda _: self.update_timer.start())
        self.dpi_input.valueChanged.connect(lambda _: self.update_timer.start())
        self.target_px_input.valueChanged.connect(lambda _: self.update_timer.start())

        self.browse_btn.clicked.connect(self._on_browse_clicked)
        self.pattern_input.textChanged.connect(lambda _: self.update_timer.start())
        self.path_input.textChanged.connect(lambda _: self.update_timer.start())
        self.overwrite_checkbox.stateChanged.connect(lambda _: self.update_timer.start())
        self.same_as_source_checkbox.stateChanged.connect(self._on_same_as_source_toggled)

        self.manage_presets_btn.clicked.connect(self._open_presets_dialog)
        self.export_presets_btn.clicked.connect(self.controller.request_preset_export)

        self.apply_all_btn.toggled.connect(self._update_apply_all_style)
        self.batch_export_btn.clicked.connect(
            lambda: self.controller.request_batch_export(override_settings=self.apply_all_btn.isChecked())
        )
        self.contact_sheet_btn.clicked.connect(self.controller.request_contact_sheet)

    # --- Presets -------------------------------------------------------------

    def _add_presets_section(self) -> None:
        """Collapsible PRESETS section pinned to the top of the panel."""
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        self._presets_scroll = QScrollArea()
        self._presets_scroll.setWidgetResizable(True)
        self._presets_scroll.setMaximumHeight(140)
        self._presets_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._presets_scroll.setStyleSheet(f"QScrollArea {{ border: 1px solid {THEME.border_primary}; background: {THEME.bg_dark}; }}")
        self._presets_container = QWidget()
        self._presets_container.setStyleSheet(f"background: {THEME.bg_dark};")
        self._presets_inner = QVBoxLayout(self._presets_container)
        self._presets_inner.setContentsMargins(4, 4, 4, 4)
        self._presets_inner.setSpacing(2)
        self._presets_scroll.setWidget(self._presets_container)
        content_layout.addWidget(self._presets_scroll)

        self._no_presets_label = QLabel("No presets — click Manage to add some.")
        self._no_presets_label.setStyleSheet(f"color: {THEME.text_muted}; font-size: 10px;")
        self._no_presets_label.setWordWrap(True)
        self._presets_inner.addWidget(self._no_presets_label)
        self._preset_checkboxes: list[QCheckBox] = []

        preset_btn_row = QHBoxLayout()
        self.manage_presets_btn = QPushButton(" Manage")
        self.manage_presets_btn.setObjectName("manage_presets_btn")
        self.manage_presets_btn.setIcon(qta.icon("fa5s.sliders-h", color=THEME.text_primary))
        self.export_presets_btn = QPushButton(" Export Presets")
        self.export_presets_btn.setObjectName("export_presets_btn")
        self.export_presets_btn.setIcon(qta.icon("fa5s.layer-group", color=THEME.text_primary))
        self.export_presets_btn.setToolTip("Export the current file with every enabled preset")
        preset_btn_row.addWidget(self.manage_presets_btn)
        preset_btn_row.addWidget(self.export_presets_btn)
        content_layout.addLayout(preset_btn_row)

        repo = self.controller.session.repo
        expanded = bool(repo.get_global_setting("section_expanded_export_presets", default=True))
        section = CollapsibleSection("Presets", expanded=expanded, icon=qta.icon("fa5s.layer-group", color="#aaa"))
        section.set_content(content)
        section.expanded_changed.connect(lambda checked: repo.save_global_setting("section_expanded_export_presets", checked))
        self.layout.addWidget(section)

    # --- Contact sheet -------------------------------------------------------

    def _add_contact_sheet_section(self) -> None:
        """Collapsible CONTACT SHEET section: layout settings + the render button."""
        conf = self.state.config.export

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        def _labeled_spinbox(label: str, value: int, lo: int, hi: int) -> QSpinBox:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(value)
            spin.valueChanged.connect(lambda _: self.update_timer.start())
            row.addWidget(spin)
            content_layout.addLayout(row)
            return spin

        self.cs_cell_px_input = _labeled_spinbox("Cell px", conf.contact_sheet_cell_px, 100, 4000)
        self.cs_gap_input = _labeled_spinbox("Gap px", conf.contact_sheet_gap, 0, 200)
        self.cs_margin_input = _labeled_spinbox("Margin px", conf.contact_sheet_margin, 0, 500)
        self.cs_max_tiles_input = _labeled_spinbox("Max tiles", conf.contact_sheet_max_tiles, 1, 200)

        self.contact_sheet_btn = QPushButton(" Export contact sheet")
        self.contact_sheet_btn.setObjectName("contact_sheet_btn")
        self.contact_sheet_btn.setFixedHeight(40)
        self.contact_sheet_btn.setIcon(qta.icon("fa5s.th", color=THEME.text_primary))
        self.contact_sheet_btn.setToolTip("Render all visible frames into a contact sheet")
        content_layout.addWidget(self.contact_sheet_btn)

        repo = self.controller.session.repo
        expanded = bool(repo.get_global_setting("section_expanded_contact_sheet", default=False))
        section = CollapsibleSection("Contact Sheet", expanded=expanded, icon=qta.icon("fa5s.th", color="#aaa"))
        section.set_content(content)
        section.expanded_changed.connect(lambda checked: repo.save_global_setting("section_expanded_contact_sheet", checked))
        self.layout.addWidget(section)

    def _rebuild_preset_rows(self) -> None:
        """Rebuild the preset checkbox list from state."""
        for cb in self._preset_checkboxes:
            self._presets_inner.removeWidget(cb)
            cb.deleteLater()
        self._preset_checkboxes.clear()

        presets = self.state.export_presets
        self._no_presets_label.setVisible(not presets)

        for i, preset in enumerate(presets):
            cb = QCheckBox(preset.name)
            cb.setChecked(preset.enabled)
            cb.setStyleSheet(f"color: {THEME.text_primary};")
            cb.stateChanged.connect(lambda state, idx=i: self._on_preset_toggled(idx, state))
            self._presets_inner.addWidget(cb)
            self._preset_checkboxes.append(cb)

        self._presets_inner.addStretch()

    def _on_preset_toggled(self, idx: int, state: int) -> None:
        presets = self.state.export_presets
        if 0 <= idx < len(presets):
            presets[idx].enabled = state == Qt.CheckState.Checked.value
            self.controller.session.save_export_presets()

    def _open_presets_dialog(self) -> None:
        from negpy.desktop.view.widgets.export_presets_dialog import ExportPresetsDialog

        dlg = ExportPresetsDialog(self.state.export_presets, parent=self)
        dlg.presets_changed.connect(self._on_presets_changed)
        dlg.exec()

    def _on_presets_changed(self, presets: list) -> None:
        self.state.export_presets = presets
        self.controller.session.save_export_presets()
        self._rebuild_preset_rows()

    # --- Current export settings ---------------------------------------------

    def _update_apply_all_style(self, checked: bool) -> None:
        """Toggle checked appearance for the Sync export settings button."""
        if checked:
            self.apply_all_btn.setStyleSheet("""
                QPushButton {
                    background-color: #222222;
                    color: white;
                    font-weight: bold;
                    border: 2px solid #555555;
                    border-radius: 4px;
                }
            """)
            self.apply_all_btn.setIcon(qta.icon("fa5s.clone", color="white"))
        else:
            self.apply_all_btn.setStyleSheet("font-weight: bold;")
            self.apply_all_btn.setIcon(qta.icon("fa5s.clone", color=THEME.text_primary))

    def _persist_all_export_settings(self) -> None:
        """Collects all UI values and performs a single debounced config update."""
        self.update_config_section(
            "export",
            persist=True,
            render=True,
            export_fmt=self.fmt_combo.currentText(),
            export_color_space=self._current_export_color_space(),
            paper_aspect_ratio=self.ratio_combo.currentText(),
            export_resolution_mode=self._current_mode_value(),
            export_print_size=self.size_input.value(),
            export_dpi=self.dpi_input.value(),
            export_target_long_edge_px=self.target_px_input.value(),
            filename_pattern=self.pattern_input.text(),
            export_path=self.path_input.text(),
            overwrite=self.overwrite_checkbox.isChecked(),
            same_as_source=self.same_as_source_checkbox.isChecked(),
            contact_sheet_cell_px=self.cs_cell_px_input.value(),
            contact_sheet_gap=self.cs_gap_input.value(),
            contact_sheet_margin=self.cs_margin_input.value(),
            contact_sheet_max_tiles=self.cs_max_tiles_input.value(),
        )

    def _on_input_changed(self, index: int) -> None:
        path = self.input_profiles[index]
        self.state.icc_input_path = path if path != "None" else None
        self.controller.session.save_icc_prefs()
        self.controller.request_render()

    def _on_output_changed(self, index: int) -> None:
        if index < len(self.output_spaces):
            # Color space: persist synchronously (not debounced) so this render resolves
            # the effective output from the new export_color_space.
            self.state.icc_output_path = None
            self.controller.session.save_icc_prefs()
            self._persist_all_export_settings()
        else:
            # Custom output profile.
            self.state.icc_output_path = self.output_map[index]
            self.controller.session.save_icc_prefs()
            self.controller.request_render()

    def _on_display_changed(self, index: int) -> None:
        self.controller.set_monitor_override(self.display_map[index])

    def _refresh_display_info(self) -> None:
        """Update the 'As detected' label with the live detected monitor profile."""
        from negpy.infrastructure.display.color_mgmt import profile_description

        desc = profile_description(self.state.monitor_icc_detected_bytes)
        self.display_detected_label.setText(f"Detected: {desc}")
        self.display_combo.setItemText(0, f"As detected ({desc})")

    def _current_export_color_space(self) -> str:
        """Output dropdown value when a standard space is selected; else keep prior."""
        idx = self.output_combo.currentIndex()
        if 0 <= idx < len(self.output_spaces):
            return self.output_map[idx]
        return self.state.config.export.export_color_space

    _MODE_BY_ID = {
        0: ExportResolutionMode.ORIGINAL.value,
        1: ExportResolutionMode.PRINT.value,
        2: ExportResolutionMode.TARGET_PX.value,
    }
    _ID_BY_MODE = {v: k for k, v in _MODE_BY_ID.items()}

    def _current_mode_value(self) -> str:
        return self._MODE_BY_ID.get(self.mode_btn_group.checkedId(), ExportResolutionMode.PRINT.value)

    def _select_mode_button(self, mode_value: str) -> None:
        idx = self._ID_BY_MODE.get(mode_value, 1)
        btn = self.mode_btn_group.button(idx)
        if btn is not None:
            btn.setChecked(True)

    def _update_mode_visibility(self, mode_value: str) -> None:
        self.print_size_container.setVisible(mode_value == ExportResolutionMode.PRINT.value)
        self.target_px_container.setVisible(mode_value == ExportResolutionMode.TARGET_PX.value)

    def _on_mode_toggled(self, _id: int, checked: bool) -> None:
        if not checked:
            return
        self._update_mode_visibility(self._current_mode_value())
        self.update_timer.start()

    def _on_same_as_source_toggled(self) -> None:
        checked = self.same_as_source_checkbox.isChecked()
        self.path_input.setDisabled(checked)
        self.browse_btn.setDisabled(checked)
        self.update_timer.start()

    def _on_browse_clicked(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "Select Export Directory", self.state.config.export.export_path)
        if path:
            self.path_input.setText(path)

    def sync_ui(self) -> None:
        conf = self.state.config.export
        self.block_signals(True)
        try:
            self.soft_proof_checkbox.setChecked(self.state.soft_proof_enabled)
            self.fmt_combo.setCurrentText(conf.export_fmt)
            in_path = self.state.icc_input_path
            self.input_combo.setCurrentText(os.path.basename(in_path) if in_path else "None")
            out_path = self.state.icc_output_path
            self.output_combo.setCurrentText(os.path.basename(out_path) if out_path else conf.export_color_space)
            override = self.state.monitor_profile_override
            self.display_combo.setCurrentText(override if override in self.display_spaces else "As detected")
            self._refresh_display_info()
            self.ratio_combo.setCurrentText(conf.paper_aspect_ratio)
            self._select_mode_button(conf.export_resolution_mode)
            self._update_mode_visibility(conf.export_resolution_mode)
            self.size_input.setValue(conf.export_print_size)
            self.dpi_input.setValue(conf.export_dpi)
            self.target_px_input.setValue(conf.export_target_long_edge_px)
            self.pattern_input.setText(conf.filename_pattern)
            self.path_input.setText(conf.export_path)
            self.overwrite_checkbox.setChecked(conf.overwrite)
            self.same_as_source_checkbox.setChecked(conf.same_as_source)
            self.path_input.setDisabled(conf.same_as_source)
            self.browse_btn.setDisabled(conf.same_as_source)
            self.cs_cell_px_input.setValue(conf.contact_sheet_cell_px)
            self.cs_gap_input.setValue(conf.contact_sheet_gap)
            self.cs_margin_input.setValue(conf.contact_sheet_margin)
            self.cs_max_tiles_input.setValue(conf.contact_sheet_max_tiles)
        finally:
            self.block_signals(False)

        self._rebuild_preset_rows()

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.soft_proof_checkbox,
            self.fmt_combo,
            self.input_combo,
            self.output_combo,
            self.display_combo,
            self.ratio_combo,
            self.mode_original_btn,
            self.mode_print_btn,
            self.mode_target_px_btn,
            self.size_input,
            self.dpi_input,
            self.target_px_input,
            self.pattern_input,
            self.path_input,
            self.overwrite_checkbox,
            self.same_as_source_checkbox,
            self.cs_cell_px_input,
            self.cs_gap_input,
            self.cs_margin_input,
            self.cs_max_tiles_input,
        ]
        for w in widgets:
            w.blockSignals(blocked)
