import qtawesome as qta
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
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
from negpy.desktop.view.widgets.export_settings_form import ExportSettingsForm
from negpy.domain.models import ColorSpace


class ExportSidebar(BaseSidebar):
    """
    Panel for export settings, presets and batch processing.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(10)

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(500)
        self.update_timer.timeout.connect(self._persist_all_export_settings)

        self._add_presets_section()
        self._add_contact_sheet_section()

        # Shared FORMAT / SIZE / COLOR / DESTINATION rows.
        self.form = ExportSettingsForm()
        self.form.load(self._config_to_form_values())
        self.layout.addWidget(self.form)

        self._add_preview_section()
        self._add_batch_section()

        self.layout.addStretch()

        self._rebuild_preset_rows()

    def _connect_signals(self) -> None:
        self.form.changed.connect(self.update_timer.start)
        self.form.changed.connect(self._refresh_proof_mismatch_warning)

        self.soft_proof_checkbox.toggled.connect(self.controller.set_soft_proof)
        self.soft_proof_checkbox.toggled.connect(self._refresh_proof_mismatch_warning)
        self.display_combo.currentIndexChanged.connect(self._on_display_changed)
        self.controller.monitor_profile_changed.connect(self._refresh_display_info)

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

    # --- Preview (soft proof + monitor profile, preview only) ----------------

    def _add_preview_section(self) -> None:
        self.layout.addWidget(section_subheader("PREVIEW"))

        # Soft proof: on by default so the preview is true to export. When off,
        # Output/Input ICC and the export color space affect export only, not
        # the preview — i.e. exported colors may differ from what's shown.
        self.soft_proof_checkbox = QCheckBox("Soft proof (preview matches export)")
        self.soft_proof_checkbox.setChecked(self.state.soft_proof_enabled)
        self.soft_proof_checkbox.setToolTip(
            "Simulate the export color space and Output profile (incl. paper/printer) in the "
            "preview, so what you see matches what you'll get. Turn off only to preview at full "
            "gamut regardless of the export target."
        )
        self.layout.addWidget(self.soft_proof_checkbox)

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
        disp_label = QLabel("Display")
        disp_label.setFixedWidth(52)
        disp_row.addWidget(disp_label)
        disp_row.addWidget(self.display_combo)
        self.layout.addLayout(disp_row)

        self.display_detected_label = QLabel()
        self.display_detected_label.setStyleSheet(f"color: {THEME.text_muted}; font-size: 10px;")
        self.layout.addWidget(self.display_detected_label)
        self._refresh_display_info()

        # Warns when the preview won't reflect the export's gamut clamp (soft
        # proof off + export space narrower than the working space).
        self.proof_mismatch_label = QLabel("Soft proof is off — preview won't show the export's color clipping")
        self.proof_mismatch_label.setWordWrap(True)
        self.proof_mismatch_label.setStyleSheet(f"color: {THEME.accent_edited}; font-size: 10px;")
        self.layout.addWidget(self.proof_mismatch_label)
        self._refresh_proof_mismatch_warning()

    # --- Batch ---------------------------------------------------------------

    def _add_batch_section(self) -> None:
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

    def _config_to_form_values(self) -> dict:
        """Build the form's value dict from the export config + ICC AppState."""
        conf = self.state.config.export
        return {
            "export_fmt": conf.export_fmt,
            "jpeg_quality": conf.jpeg_quality,
            "export_resolution_mode": conf.export_resolution_mode,
            "paper_aspect_ratio": conf.paper_aspect_ratio,
            "export_print_size": conf.export_print_size,
            "export_dpi": conf.export_dpi,
            "export_target_long_edge_px": conf.export_target_long_edge_px,
            "output_mode": conf.output_mode,
            "output_subfolder": conf.output_subfolder,
            "output_path": conf.export_path,
            "filename_pattern": conf.filename_pattern,
            "overwrite": conf.overwrite,
            "export_color_space": conf.export_color_space,
            "icc_input_path": self.state.icc_input_path,
            "icc_output_path": self.state.icc_output_path,
        }

    def _persist_all_export_settings(self) -> None:
        """Collects all UI values and performs a single debounced config update."""
        vals = self.form.values()

        # ICC paths live in AppState (injected at export time), not the config.
        self.state.icc_input_path = vals["icc_input_path"]
        self.state.icc_output_path = vals["icc_output_path"]
        self.controller.session.save_icc_prefs()

        self.update_config_section(
            "export",
            persist=True,
            render=True,
            export_fmt=vals["export_fmt"],
            jpeg_quality=vals["jpeg_quality"],
            export_color_space=vals["export_color_space"],
            paper_aspect_ratio=vals["paper_aspect_ratio"],
            export_resolution_mode=vals["export_resolution_mode"],
            export_print_size=vals["export_print_size"],
            export_dpi=vals["export_dpi"],
            export_target_long_edge_px=vals["export_target_long_edge_px"],
            output_mode=vals["output_mode"],
            output_subfolder=vals["output_subfolder"],
            export_path=vals["output_path"],
            filename_pattern=vals["filename_pattern"],
            overwrite=vals["overwrite"],
            contact_sheet_cell_px=self.cs_cell_px_input.value(),
            contact_sheet_gap=self.cs_gap_input.value(),
            contact_sheet_margin=self.cs_margin_input.value(),
            contact_sheet_max_tiles=self.cs_max_tiles_input.value(),
        )

    def _on_display_changed(self, index: int) -> None:
        self.controller.set_monitor_override(self.display_map[index])

    def _refresh_display_info(self) -> None:
        """Update the 'As detected' label with the live detected monitor profile."""
        from negpy.infrastructure.display.color_mgmt import profile_description

        desc = profile_description(self.state.monitor_icc_detected_bytes)
        self.display_detected_label.setText(f"Detected: {desc}")
        self.display_combo.setItemText(0, f"As detected ({desc})")

    def _refresh_proof_mismatch_warning(self) -> None:
        """Show a hint when soft proof is off and export will clamp to a
        narrower/different color space than the preview is shown in, so the
        preview can't be trusted to predict the exported colors."""
        from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE

        export_cs = self.form.values()["export_color_space"]
        mismatch = (
            not self.soft_proof_checkbox.isChecked() and export_cs != ColorSpace.SAME_AS_SOURCE.value and export_cs != WORKING_COLOR_SPACE
        )
        self.proof_mismatch_label.setVisible(mismatch)

    def sync_ui(self) -> None:
        conf = self.state.config.export
        self.block_signals(True)
        try:
            self.form.load(self._config_to_form_values())
            self.soft_proof_checkbox.setChecked(self.state.soft_proof_enabled)
            override = self.state.monitor_profile_override
            self.display_combo.setCurrentText(override if override in self.display_spaces else "As detected")
            self._refresh_display_info()
            self.cs_cell_px_input.setValue(conf.contact_sheet_cell_px)
            self.cs_gap_input.setValue(conf.contact_sheet_gap)
            self.cs_margin_input.setValue(conf.contact_sheet_margin)
            self.cs_max_tiles_input.setValue(conf.contact_sheet_max_tiles)
        finally:
            self.block_signals(False)

        self._refresh_proof_mismatch_warning()
        self._rebuild_preset_rows()

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.soft_proof_checkbox,
            self.display_combo,
            self.cs_cell_px_input,
            self.cs_gap_input,
            self.cs_margin_input,
            self.cs_max_tiles_input,
        ]
        for w in widgets:
            w.blockSignals(blocked)
