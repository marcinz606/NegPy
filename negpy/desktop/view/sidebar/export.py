import os

import qtawesome as qta
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QActionGroup
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.widgets.contact_sheet_colors_dialog import ContactSheetColorsDialog
from negpy.desktop.view.styles.templates import (
    default_button_height,
    field_label,
    hint_label,
    labeled_toggle_qss,
    section_subheader,
    set_hint_kind,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.widgets.export_settings_form import ExportSettingsForm, constrain_combo
from negpy.desktop.view.widgets.split_button import make_split_button
from negpy.domain.models import ColorSpace, preset_display_name
from negpy.services.export.contact_sheet_templates import ContactSheetLayout, ContactSheetTemplates


class ExportSidebar(BaseSidebar):
    """
    Panel for export settings, presets and batch processing.
    """

    SIDE_MARGIN = THEME.space_xl

    def _init_ui(self) -> None:
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(500)
        self.update_timer.timeout.connect(self._persist_all_export_settings)

        # Task-flow order: the Print/Flat decision reframes the whole form, so it
        # comes first with the primary Export action right under it; the form
        # follows; occasional tools (presets, contact sheet, preview) sit
        # collapsed at the bottom.
        self._add_flat_master_section()
        self._add_export_section()

        # Shared FORMAT / SIZE / COLOR / DESTINATION rows.
        self.form = ExportSettingsForm()
        self.form.load(self._config_to_form_values())
        self.layout.addWidget(self.form)

        self._add_presets_section()
        self._add_sidecars_section()
        self._add_contact_sheet_section()
        self._add_printing_notes_section()
        self._add_preview_section()
        self._sync_flat_enabled()

        self.layout.addStretch()

        self._rebuild_preset_rows()
        self._refresh_export_enabled()

    def _connect_signals(self) -> None:
        self.controller.flush_export_settings = self._flush_export_settings

        self.form.changed.connect(self.update_timer.start)
        self.form.changed.connect(self._refresh_proof_mismatch_warning)
        self.form.changed.connect(self._refresh_export_enabled)

        self.soft_proof_checkbox.toggled.connect(self.controller.set_soft_proof)
        self.soft_proof_checkbox.toggled.connect(self._refresh_proof_mismatch_warning)
        self.display_combo.currentIndexChanged.connect(self._on_display_changed)
        self.controller.monitor_profile_changed.connect(self._refresh_display_info)

        self.manage_presets_btn.clicked.connect(self._open_presets_dialog)
        self.export_presets_btn.clicked.connect(self._on_export_presets_clicked)
        self.export_main_btn.clicked.connect(self._on_export_clicked)

        self.intent_btn_group.idToggled.connect(self._on_flat_output_toggled)
        self.flat_peek_btn.toggled.connect(lambda checked: self.controller.toggle_flat_peek(force=checked))
        self.flat_bake_btn.clicked.connect(self.controller.request_batch_normalization)
        self.controller.flat_output_changed.connect(self._on_flat_output_changed)
        self.controller.linear_output_changed.connect(self._on_linear_output_changed)
        self.controller.flat_peek_changed.connect(self._on_flat_peek_changed)

        self.contact_sheet_btn.clicked.connect(self.controller.request_contact_sheet)
        self.printing_notes_btn.clicked.connect(self.controller.request_printing_notes_export)
        self.printing_notes_preview_btn.toggled.connect(lambda checked: self.controller.toggle_printing_notes(force=checked))
        self.controller.printing_notes_changed.connect(self._on_printing_notes_changed)
        self.cs_save_template_btn.clicked.connect(self._on_save_contact_sheet_template)
        self.cs_delete_template_btn.clicked.connect(self._on_delete_contact_sheet_template)
        self.cs_template_combo.currentTextChanged.connect(self._on_contact_sheet_template_changed)

        self.sidecars_enabled_btn.toggled.connect(lambda _: self.update_timer.start())
        self.export_sidecars_btn.clicked.connect(self.controller.export_edit_sidecars)

    # --- Presets -------------------------------------------------------------

    def _add_presets_section(self) -> None:
        """Collapsible PRESETS section pinned to the top of the panel."""
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        self._presets_container = QWidget()
        # Border-only grouping — a filled box reads as a darker plate behind the
        # buttons against the section card.
        self._presets_container.setStyleSheet(f"border: 1px solid {THEME.border_primary}; background: transparent;")
        self._presets_inner = QVBoxLayout(self._presets_container)
        self._presets_inner.setContentsMargins(4, 4, 4, 4)
        self._presets_inner.setSpacing(2)
        content_layout.addWidget(self._presets_container)

        self._no_presets_label = hint_label("No presets — click Manage to add some.")
        self._presets_inner.addWidget(self._no_presets_label)
        self._preset_checkboxes: list[QCheckBox] = []

        preset_btn_row = QHBoxLayout()
        self.manage_presets_btn = QPushButton(" Manage")
        self.manage_presets_btn.setObjectName("manage_presets_btn")
        self.manage_presets_btn.setIcon(qta.icon("fa5s.sliders-h", color=THEME.text_primary))
        self.manage_presets_btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        preset_menu, self._preset_scope_actions = self._build_scope_menu(self._PRESET_SCOPES, self._set_preset_scope)
        self._export_presets_menu = preset_menu

        self.export_presets_group, self.export_presets_btn, self.export_presets_menu_btn = make_split_button(
            " Export Presets", "fa5s.layer-group", preset_menu, primary=True
        )
        self.export_presets_menu_btn.setToolTip("Choose what the Export Presets button does")

        saved = self.controller.session.repo.get_global_setting("preset_export_scope", "current")
        self._set_preset_scope(saved if saved in self._PRESET_SCOPES else "current", persist=False)
        preset_btn_row.addWidget(self.manage_presets_btn, 0)
        preset_btn_row.addWidget(self.export_presets_group, 1)
        content_layout.addLayout(preset_btn_row)

        repo = self.controller.session.repo
        expanded = bool(repo.get_global_setting("section_expanded_export_presets", default=False))
        self._presets_section = CollapsibleSection("Presets", expanded=expanded, icon=qta.icon("fa5s.layer-group", color="#aaa"))
        self._presets_section.set_content(content)
        self._presets_section.expanded_changed.connect(lambda checked: repo.save_global_setting("section_expanded_export_presets", checked))
        self.layout.addWidget(self._presets_section)

    # --- Printing notes ------------------------------------------------------

    def _add_printing_notes_section(self) -> None:
        """Collapsible PRINTING NOTES section: the canvas preview toggle + the export."""
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        self.printing_notes_preview_btn = self._tool_toggle(
            "fa5s.eye",
            "Preview",
            "Show the marked-up work print over the frame: burns hatched, dodges open, each mask "
            "labelled with its value in stops, plus a card with the print recipe. Display only.",
        )
        self.printing_notes_preview_btn.setChecked(self.state.printing_notes)
        self.printing_notes_preview_btn.setFixedHeight(default_button_height())

        self.printing_notes_btn = QPushButton(" Export")
        self.printing_notes_btn.setObjectName("printing_notes_btn")
        self.printing_notes_btn.setProperty("primary", True)
        self.printing_notes_btn.setFixedHeight(default_button_height())
        self.printing_notes_btn.setIcon(qta.icon("mdi.playlist-edit", color="white"))
        self.printing_notes_btn.setToolTip(
            "Save this frame as a marked-up work print — the map plus the print recipe below it — as its "
            "own JPEG in the export folder. The print itself is untouched. Resolution follows the "
            "preview, so turn HQ on for a full-resolution sheet."
        )

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.printing_notes_preview_btn, 1)
        btn_row.addWidget(self.printing_notes_btn, 1)
        content_layout.addLayout(btn_row)

        repo = self.controller.session.repo
        expanded = bool(repo.get_global_setting("section_expanded_printing_notes", default=False))
        self.printing_notes_section = CollapsibleSection(
            "Printing Notes", expanded=expanded, icon=qta.icon("mdi.playlist-edit", color="#aaa")
        )
        self.printing_notes_section.setToolTip("The printer's record for this frame: dodge/burn map + print recipe.")
        self.printing_notes_section.set_content(content)
        self.printing_notes_section.expanded_changed.connect(
            lambda checked: repo.save_global_setting("section_expanded_printing_notes", checked)
        )
        self.layout.addWidget(self.printing_notes_section)

    def _on_printing_notes_changed(self, active: bool) -> None:
        self.printing_notes_preview_btn.blockSignals(True)
        self.printing_notes_preview_btn.setChecked(active)
        self.printing_notes_preview_btn.blockSignals(False)

    # --- Contact sheet -------------------------------------------------------

    def _add_contact_sheet_section(self) -> None:
        """Collapsible CONTACT SHEET section: layout settings + the render button."""
        conf = self.state.config.export
        self._cs_syncing = False

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        template_row = QHBoxLayout()
        template_label = field_label("Template")
        template_label.setFixedWidth(90)
        template_row.addWidget(template_label)
        self.cs_template_combo = QComboBox()
        constrain_combo(self.cs_template_combo)
        self.cs_template_combo.setToolTip(
            "Layout preset from .toml files in NegPy/contact_sheets (see docs/CONTACT_SHEET_TEMPLATES.md). "
            "Edits to the spinboxes below are saved to the active template."
        )
        template_row.addWidget(self.cs_template_combo)
        content_layout.addLayout(template_row)

        self.cs_delete_template_btn = QPushButton()
        self.cs_delete_template_btn.setIcon(qta.icon("fa5s.trash", color=THEME.text_primary))
        self.cs_delete_template_btn.setToolTip("Delete the selected template (Default can't be deleted)")
        self.cs_delete_template_btn.setFixedWidth(32)
        template_row.addWidget(self.cs_delete_template_btn)

        self.cs_save_template_btn = QPushButton(" Save as template")
        self.cs_save_template_btn.setIcon(qta.icon("fa5s.save", color=THEME.text_primary))
        self.cs_save_template_btn.setToolTip("Save the current layout as a new named template file")
        content_layout.addWidget(self.cs_save_template_btn)

        initial_layout = self._contact_sheet_layout_for_config(conf)
        self._cs_background_color = initial_layout.background_color
        self._cs_label_color = initial_layout.label_color

        def _labeled_spinbox(label: str, value: int, lo: int, hi: int) -> QSpinBox:
            row = QHBoxLayout()
            row.addWidget(field_label(label))
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(value)
            spin.valueChanged.connect(self._on_contact_sheet_layout_changed)
            row.addWidget(spin)
            content_layout.addLayout(row)
            return spin

        self.cs_cell_px_input = _labeled_spinbox("Cell px", initial_layout.cell_px, 100, 4000)
        self.cs_gap_input = _labeled_spinbox("Gap px", initial_layout.gap, 0, 200)
        self.cs_margin_input = _labeled_spinbox("Margin px", initial_layout.margin, 0, 500)
        self.cs_max_tiles_input = _labeled_spinbox("Max tiles", initial_layout.max_tiles, 1, 200)

        self.cs_show_labels_checkbox = QCheckBox("Show filenames")
        self.cs_show_labels_checkbox.setChecked(initial_layout.show_labels)
        self.cs_show_labels_checkbox.setToolTip("Print each frame's original filename below its thumbnail")
        self.cs_show_labels_checkbox.stateChanged.connect(self._on_contact_sheet_settings_changed)
        content_layout.addWidget(self.cs_show_labels_checkbox)

        colors_row = QHBoxLayout()
        colors_label = field_label("Colours")
        colors_label.setFixedWidth(90)
        colors_row.addWidget(colors_label)
        self.cs_colors_btn = QPushButton(" Choose…")
        self.cs_colors_btn.setToolTip("Background and label colours")
        self._update_cs_colors_btn_tooltip()
        self.cs_colors_btn.clicked.connect(self._on_cs_colors_clicked)
        colors_row.addWidget(self.cs_colors_btn, 1)
        content_layout.addLayout(colors_row)

        self._refresh_contact_sheet_templates()
        saved_template = conf.contact_sheet_template.strip()
        if saved_template and saved_template in ContactSheetTemplates.list_templates():
            self.cs_template_combo.setCurrentText(saved_template)
        else:
            self.cs_template_combo.setCurrentText(ContactSheetTemplates.DEFAULT_NAME)

        cs_path_row = QHBoxLayout()
        cs_path_label = field_label("Path")
        cs_path_label.setFixedWidth(90)
        cs_path_row.addWidget(cs_path_label)
        self.cs_output_path_edit = QLineEdit(conf.contact_sheet_output_path)
        self.cs_output_path_edit.setPlaceholderText("Uses export destination")
        self.cs_output_path_edit.setToolTip(
            "Folder for contact sheet JPEGs. Leave empty to follow the export destination (same as source or absolute export path)."
        )
        self.cs_output_path_edit.textChanged.connect(lambda _: self.update_timer.start())
        self.cs_output_path_browse_btn = QPushButton()
        self.cs_output_path_browse_btn.setIcon(qta.icon("fa5s.folder-open", color=THEME.text_primary))
        self.cs_output_path_browse_btn.setFixedWidth(40)
        self.cs_output_path_browse_btn.setToolTip("Choose contact sheet output folder")
        self.cs_output_path_browse_btn.clicked.connect(self._browse_contact_sheet_output_path)
        cs_path_row.addWidget(self.cs_output_path_edit)
        cs_path_row.addWidget(self.cs_output_path_browse_btn)
        content_layout.addLayout(cs_path_row)

        self.contact_sheet_btn = QPushButton(" Export contact sheet")
        self.contact_sheet_btn.setObjectName("contact_sheet_btn")
        self.contact_sheet_btn.setProperty("primary", True)
        self.contact_sheet_btn.setFixedHeight(default_button_height())
        self.contact_sheet_btn.setIcon(qta.icon("fa5s.th", color="white"))
        self.contact_sheet_btn.setToolTip("Render all visible frames into a contact sheet")
        content_layout.addWidget(self.contact_sheet_btn)

        repo = self.controller.session.repo
        expanded = bool(repo.get_global_setting("section_expanded_contact_sheet", default=False))
        self.contact_sheet_section = CollapsibleSection("Contact Sheet", expanded=expanded, icon=qta.icon("fa5s.th", color="#aaa"))
        self.contact_sheet_section.setToolTip("Render a contact sheet of display previews. Independent of flat master export.")
        self.contact_sheet_section.set_content(content)
        self.contact_sheet_section.expanded_changed.connect(
            lambda checked: repo.save_global_setting("section_expanded_contact_sheet", checked)
        )
        self.layout.addWidget(self.contact_sheet_section)

    def _browse_contact_sheet_output_path(self) -> None:
        start = self.cs_output_path_edit.text().strip() or self.state.config.export.export_path
        path = QFileDialog.getExistingDirectory(self, "Select Contact Sheet Output Folder", start)
        if path:
            self.cs_output_path_edit.setText(path)

    def _update_cs_colors_btn_tooltip(self) -> None:
        self.cs_colors_btn.setToolTip(f"Background {self._cs_background_color}, labels {self._cs_label_color}")

    def _on_cs_colors_clicked(self) -> None:
        try:
            dlg = ContactSheetColorsDialog(self._cs_background_color, self._cs_label_color, self)
            if not dlg.exec():
                return
            bg, label = dlg.colors()
        except Exception as exc:
            QMessageBox.critical(self, "Contact Sheet Colours", f"Could not open colour picker:\n{exc}")
            return
        self._cs_background_color = bg
        self._cs_label_color = label
        self._update_cs_colors_btn_tooltip()
        self._on_contact_sheet_settings_changed()

    def _contact_sheet_layout_for_config(self, conf) -> ContactSheetLayout:
        saved_template = conf.contact_sheet_template.strip()
        if saved_template and saved_template in ContactSheetTemplates.list_templates():
            layout = ContactSheetTemplates.get_layout(saved_template)
            if layout is not None:
                return layout
        return ContactSheetTemplates.default_layout_from_export(conf)

    def _refresh_contact_sheet_templates(self) -> None:
        profiles = ContactSheetTemplates.list_templates()
        if profiles != [self.cs_template_combo.itemText(i) for i in range(self.cs_template_combo.count())]:
            current = self.cs_template_combo.currentText()
            self.cs_template_combo.blockSignals(True)
            self.cs_template_combo.clear()
            self.cs_template_combo.addItems(profiles)
            idx = self.cs_template_combo.findText(current)
            self.cs_template_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.cs_template_combo.blockSignals(False)

    def _current_contact_sheet_layout(self) -> ContactSheetLayout:
        return ContactSheetLayout(
            cell_px=self.cs_cell_px_input.value(),
            gap=self.cs_gap_input.value(),
            margin=self.cs_margin_input.value(),
            max_tiles=self.cs_max_tiles_input.value(),
            show_labels=self.cs_show_labels_checkbox.isChecked(),
            background_color=self._cs_background_color,
            label_color=self._cs_label_color,
        )

    def _apply_contact_sheet_layout(self, layout: ContactSheetLayout) -> None:
        self._cs_syncing = True
        try:
            self.cs_cell_px_input.setValue(layout.cell_px)
            self.cs_gap_input.setValue(layout.gap)
            self.cs_margin_input.setValue(layout.margin)
            self.cs_max_tiles_input.setValue(layout.max_tiles)
            self.cs_show_labels_checkbox.setChecked(layout.show_labels)
            self._cs_background_color = layout.background_color
            self._cs_label_color = layout.label_color
            self._update_cs_colors_btn_tooltip()
        finally:
            self._cs_syncing = False

    def _contact_sheet_template_name(self) -> str:
        name = self.cs_template_combo.currentText()
        if name == ContactSheetTemplates.DEFAULT_NAME:
            return ""
        return name

    def _on_contact_sheet_layout_changed(self, _value: int) -> None:
        self._on_contact_sheet_settings_changed()

    def _on_contact_sheet_settings_changed(self) -> None:
        if self._cs_syncing:
            return
        self.update_timer.start()

    def _sync_active_contact_sheet_template(self) -> None:
        """Write spinbox layout back to the active template (Default snapshot or .toml file)."""
        if self._cs_syncing:
            return
        layout = self._current_contact_sheet_layout()
        template_name = self._contact_sheet_template_name()
        if template_name:
            try:
                ContactSheetTemplates.save(template_name, layout)
            except OSError as exc:
                QMessageBox.critical(
                    self,
                    "Contact Sheet Template",
                    f'Could not update template "{template_name}":\n{exc}',
                )
                return

    def _contact_sheet_persist_kwargs(self) -> dict:
        layout = self._current_contact_sheet_layout()
        template_name = self._contact_sheet_template_name()
        kwargs = {
            **ContactSheetTemplates.active_layout_field_updates(layout),
            "contact_sheet_output_path": self.cs_output_path_edit.text(),
            "contact_sheet_template": template_name,
        }
        if not template_name:
            kwargs.update(ContactSheetTemplates.default_layout_field_updates(layout))
        return kwargs

    def _on_contact_sheet_template_changed(self, name: str) -> None:
        if self._cs_syncing:
            return
        if name == ContactSheetTemplates.DEFAULT_NAME:
            layout = ContactSheetTemplates.default_layout_from_export(self.state.config.export)
            self._apply_contact_sheet_layout(layout)
            self.update_config_section(
                "export",
                persist=True,
                render=False,
                contact_sheet_template="",
                **ContactSheetTemplates.active_layout_field_updates(layout),
                **ContactSheetTemplates.default_layout_field_updates(layout),
            )
            return
        layout = ContactSheetTemplates.get_layout(name)
        if layout is None:
            return
        self._apply_contact_sheet_layout(layout)
        self.update_config_section(
            "export",
            persist=True,
            render=False,
            contact_sheet_template=name,
            **ContactSheetTemplates.active_layout_field_updates(layout),
        )

    def _on_delete_contact_sheet_template(self) -> None:
        name = self.cs_template_combo.currentText()
        if not name or name == ContactSheetTemplates.DEFAULT_NAME:
            return
        reply = QMessageBox.question(
            self,
            "Delete Contact Sheet Template",
            f'Delete template "{name}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        ContactSheetTemplates.delete(name)
        self._refresh_contact_sheet_templates()
        # Fall back to Default explicitly — the refresh keeps signals blocked.
        self.cs_template_combo.blockSignals(True)
        self.cs_template_combo.setCurrentText(ContactSheetTemplates.DEFAULT_NAME)
        self.cs_template_combo.blockSignals(False)
        self._on_contact_sheet_template_changed(ContactSheetTemplates.DEFAULT_NAME)

    def _on_save_contact_sheet_template(self) -> None:
        current = self.cs_template_combo.currentText()
        default_text = current if current != ContactSheetTemplates.DEFAULT_NAME else ""
        name, ok = QInputDialog.getText(self, "Save Contact Sheet Template", "Template name:", text=default_text)
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name == ContactSheetTemplates.DEFAULT_NAME:
            QMessageBox.warning(self, "Save Contact Sheet Template", '"Default" is reserved. Choose another name.')
            return

        path = ContactSheetTemplates.path_for_name(name)
        if os.path.exists(path) or ContactSheetTemplates.template_exists(name):
            reply = QMessageBox.question(
                self,
                "Overwrite Template",
                f'A template named "{name}" already exists. Replace it?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        layout = self._current_contact_sheet_layout()
        try:
            ContactSheetTemplates.save(name, layout)
        except OSError as exc:
            QMessageBox.critical(self, "Save Contact Sheet Template", f"Could not write template:\n{exc}")
            return

        self._refresh_contact_sheet_templates()
        self.cs_template_combo.blockSignals(True)
        self.cs_template_combo.setCurrentText(name)
        self.cs_template_combo.blockSignals(False)
        self.update_config_section(
            "export",
            persist=True,
            render=False,
            contact_sheet_template=name,
            **ContactSheetTemplates.active_layout_field_updates(layout),
        )

    def _add_flat_master_section(self) -> None:
        """Output-intent override: Print (default) or Flat digital intermediate."""
        self.layout.addWidget(section_subheader("OUTPUT INTENT"))

        # Contain the whole intent block (toggle + format + peek/bake + hints) so
        # it reads as one unit. objectName-scoped so the border doesn't cascade.
        container = QWidget()
        container.setObjectName("flat_intent_box")
        container.setStyleSheet(f"#flat_intent_box {{ border: 1px solid {THEME.border_primary}; background: transparent; }}")
        box = QVBoxLayout(container)
        box.setContentsMargins(6, 6, 6, 6)
        box.setSpacing(6)

        intent_row = QHBoxLayout()
        intent_row.setSpacing(4)
        self.intent_print_btn = QPushButton("Print")
        self.intent_flat_btn = QPushButton("Flat")
        self.intent_flat_btn.setToolTip(
            "Export a flat, neutral, low-contrast master that keeps maximum tonal and colour "
            "information for editing in Lightroom, Darktable or Photoshop. Skips the creative "
            "print look (auto density/grade, cast removal, lab effects, toning, vignette) and "
            "writes a wide-gamut, high-bit-depth file. Your in-app preview is unaffected."
        )
        self.intent_linear_btn = QPushButton("Linear")
        self.intent_linear_btn.setToolTip(
            "Export the raw decoded sensor data as an untagged 16-bit TIFF, before any "
            "NegPy processing (no normalization, exposure, lab, toning, color management). "
            "Supported for Pakon RAW and LinearRaw DNG (SilverFast/VueScan) files."
        )
        for btn in (self.intent_print_btn, self.intent_flat_btn, self.intent_linear_btn):
            btn.setCheckable(True)
            btn.setStyleSheet(labeled_toggle_qss())
            intent_row.addWidget(btn)
        self.intent_btn_group = QButtonGroup(self)
        self.intent_btn_group.setExclusive(True)
        self.intent_btn_group.addButton(self.intent_print_btn, 0)
        self.intent_btn_group.addButton(self.intent_flat_btn, 1)
        self.intent_btn_group.addButton(self.intent_linear_btn, 2)
        if self.state.linear_output:
            self.intent_linear_btn.setChecked(True)
        elif self.state.flat_output:
            self.intent_flat_btn.setChecked(True)
        else:
            self.intent_print_btn.setChecked(True)
        box.addLayout(intent_row)

        peek_bake_row = QHBoxLayout()
        peek_bake_row.setSpacing(4)
        self.flat_peek_btn = self._tool_toggle(
            "fa5s.eye", "Preview Flat", "Temporarily show the flat master in the canvas (does not change your edit)"
        )
        self.flat_peek_btn.setChecked(self.state.flat_peek)
        self.flat_bake_btn = QPushButton(" Roll Baseline")
        self.flat_bake_btn.setIcon(qta.icon("fa5s.link", color=THEME.text_primary))
        self.flat_bake_btn.setToolTip(
            "Measure every visible frame's exposure bounds and apply their shared average, so flat "
            "masters render consistently across the roll."
        )
        peek_bake_row.addWidget(self.flat_peek_btn)
        peek_bake_row.addWidget(self.flat_bake_btn)
        box.addLayout(peek_bake_row)

        self.flat_hint_label = hint_label(
            "Exports a flat 16-bit TIFF master in the selected color space at full resolution by default. "
            "Choose Print or Pixels below to downscale."
        )
        box.addWidget(self.flat_hint_label)

        # Roll-consistency nudge: a flat master is only identical across frames
        # once the roll shares one normalization baseline (locked bounds). Until
        # then, per-frame auto bounds make each frame's tones drift.
        self.flat_roll_warning = hint_label("For consistent masters across a roll, lock one baseline for every frame.", kind="warning")
        box.addWidget(self.flat_roll_warning)

        self.linear_hint_label = hint_label(
            "Exports the loader's decoded buffer as an untagged 16-bit file. "
            "No pipeline processing, no color management, no scaling. "
            "Pakon RAW, LinearRaw DNG (SilverFast/VueScan), and camera RAW."
        )
        box.addWidget(self.linear_hint_label)

        fmt_row = QHBoxLayout()
        fmt_row.setContentsMargins(0, 0, 0, 0)
        fmt_row.addWidget(field_label("Format"))
        self.linear_format_combo = QComboBox()
        self.linear_format_combo.addItem("TIFF", "tiff")
        self.linear_format_combo.addItem("JPEG XL (lossless)", "jxl")
        idx = self.linear_format_combo.findData(self.state.linear_format)
        if idx >= 0:
            self.linear_format_combo.setCurrentIndex(idx)
        self.linear_format_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        fmt_row.addWidget(self.linear_format_combo)
        self.linear_format_row = QWidget()
        self.linear_format_row.setLayout(fmt_row)
        self.linear_format_row.setVisible(False)
        box.addWidget(self.linear_format_row)
        self.linear_format_combo.currentIndexChanged.connect(self._on_linear_format_changed)

        self.linear_jxl_effort_slider = CompactSlider("Effort", 1, 9, 7, step=1, precision=1)
        self.linear_jxl_effort_slider.label.setToolTip("Encoder effort: higher = slower, smaller file")
        self.linear_jxl_effort_slider.setValue(self.state.linear_jxl_effort)
        self.linear_jxl_effort_slider.setVisible(False)
        self.linear_jxl_effort_slider.valueChanged.connect(self._on_linear_jxl_effort_changed)
        box.addWidget(self.linear_jxl_effort_slider)

        expansion_row = QHBoxLayout()
        expansion_row.setContentsMargins(0, 0, 0, 0)
        self.linear_expansion_label = field_label("Expansion")
        expansion_row.addWidget(self.linear_expansion_label)
        self.linear_expansion_combo = QComboBox()
        self.linear_expansion_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        expansion_row.addWidget(self.linear_expansion_combo)
        self.linear_expansion_row = QWidget()
        self.linear_expansion_row.setLayout(expansion_row)
        self.linear_expansion_row.setVisible(False)
        box.addWidget(self.linear_expansion_row)
        self.linear_expansion_hint = hint_label("Leave at the default unless you know why you need to change it.")
        self.linear_expansion_hint.setVisible(False)
        box.addWidget(self.linear_expansion_hint)
        self.linear_expansion_combo.currentIndexChanged.connect(self._on_linear_expansion_changed)

        self.linear_corrections_label = field_label("Corrections")
        self.linear_corrections_label.setVisible(False)
        box.addWidget(self.linear_corrections_label)

        self.linear_wb_checkbox = QCheckBox("Apply white balance")
        self.linear_wb_checkbox.setToolTip("Multiply by the as-shot WB gains before writing")
        self.linear_wb_checkbox.setChecked(self.state.linear_apply_wb)
        self.linear_wb_checkbox.setVisible(False)
        self.linear_wb_checkbox.toggled.connect(self._on_linear_correction_changed)
        box.addWidget(self.linear_wb_checkbox)

        self.linear_flatfield_checkbox = QCheckBox("Apply flatfield")
        self.linear_flatfield_checkbox.setToolTip("Apply the flatfield gain correction")
        self.linear_flatfield_checkbox.setChecked(self.state.linear_apply_flatfield)
        self.linear_flatfield_checkbox.setVisible(False)
        self.linear_flatfield_checkbox.toggled.connect(self._on_linear_correction_changed)
        box.addWidget(self.linear_flatfield_checkbox)

        self.linear_sensor_checkbox = QCheckBox("Apply sensor correction")
        self.linear_sensor_checkbox.setToolTip("Apply the sensor crosstalk unmixing matrix")
        self.linear_sensor_checkbox.setChecked(self.state.linear_apply_sensor)
        self.linear_sensor_checkbox.setVisible(False)
        self.linear_sensor_checkbox.toggled.connect(self._on_linear_correction_changed)
        box.addWidget(self.linear_sensor_checkbox)

        self.linear_ice_checkbox = QCheckBox("Apply ICE dust removal")
        self.linear_ice_checkbox.setToolTip("Apply IR-based dust and scratch correction")
        self.linear_ice_checkbox.setChecked(self.state.linear_apply_ice)
        self.linear_ice_checkbox.setVisible(False)
        self.linear_ice_checkbox.toggled.connect(self._on_linear_correction_changed)
        box.addWidget(self.linear_ice_checkbox)

        gamma_row = QHBoxLayout()
        gamma_row.setContentsMargins(0, 0, 0, 0)
        self.linear_gamma_label = field_label("Input gamma")
        gamma_row.addWidget(self.linear_gamma_label)
        self.linear_gamma_combo = QComboBox()
        self.linear_gamma_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        gamma_row.addWidget(self.linear_gamma_combo)
        self.linear_gamma_row = QWidget()
        self.linear_gamma_row.setLayout(gamma_row)
        self.linear_gamma_row.setVisible(False)
        box.addWidget(self.linear_gamma_row)
        self.linear_gamma_hint = hint_label("Select the gamma encoding of the input TIFF so it can be linearized before export.")
        self.linear_gamma_hint.setVisible(False)
        box.addWidget(self.linear_gamma_hint)
        self.linear_gamma_combo.currentIndexChanged.connect(self._on_linear_gamma_changed)

        self.linear_corrections_hint = hint_label(
            "Corrections are baked in and cannot be undone from the exported file. Re-export from the original RAW to get uncorrected data."
        )
        self.linear_corrections_hint.setVisible(False)
        box.addWidget(self.linear_corrections_hint)

        self.layout.addWidget(container)

    def _sync_flat_enabled(self) -> None:
        flat_on = self.intent_flat_btn.isChecked()
        linear_on = self.intent_linear_btn.isChecked()
        if hasattr(self, "form"):
            self.form.set_flat_mode(flat_on)
            self.form.setVisible(not linear_on)
        self.flat_hint_label.setVisible(flat_on)
        self.flat_peek_btn.setVisible(flat_on)
        self.linear_hint_label.setVisible(linear_on)
        if hasattr(self, "linear_format_row"):
            self.linear_format_row.setVisible(linear_on)
        if hasattr(self, "linear_jxl_effort_slider"):
            self.linear_jxl_effort_slider.setVisible(linear_on and self.state.linear_format == "jxl")
        if hasattr(self, "linear_expansion_row"):
            self.linear_expansion_row.setVisible(linear_on)
            self.linear_expansion_hint.setVisible(linear_on)
            if linear_on:
                self._refresh_linear_expansion_combo()
        if hasattr(self, "linear_gamma_row") and not linear_on:
            self.linear_gamma_row.setVisible(False)
            self.linear_gamma_hint.setVisible(False)
        if hasattr(self, "linear_corrections_label") and not linear_on:
            self.linear_corrections_label.setVisible(False)
            self.linear_wb_checkbox.setVisible(False)
            self.linear_flatfield_checkbox.setVisible(False)
            self.linear_sensor_checkbox.setVisible(False)
            self.linear_ice_checkbox.setVisible(False)
            self.linear_corrections_hint.setVisible(False)
        if hasattr(self, "_presets_section"):
            self._presets_section.setVisible(not linear_on)
        if hasattr(self, "_sidecars_section"):
            self._sidecars_section.setVisible(not linear_on)
        self._sync_flat_roll_warning()
        if hasattr(self, "form"):
            self._refresh_export_enabled()

    def _sync_flat_roll_warning(self) -> None:
        """Show the roll-baseline nudge only when flat output is on and the roll
        doesn't yet share a locked normalization baseline."""
        on = self.intent_flat_btn.isChecked()
        proc = self.state.config.process
        # Flat-master roll consistency needs both axes baselined across the roll.
        locked = proc.use_luma_average and proc.use_colour_average and proc.is_locked_initialized
        show = on and not locked
        self.flat_roll_warning.setVisible(show)
        self.flat_bake_btn.setVisible(show)

    def _on_flat_output_toggled(self, btn_id: int, checked: bool) -> None:
        if checked:
            if btn_id == 2:
                self.controller.set_linear_output(True)
            else:
                self.controller.set_linear_output(False)
                self.controller.set_flat_output(btn_id == 1)
            self._sync_flat_enabled()

    def _on_flat_output_changed(self, enabled: bool) -> None:
        self.intent_btn_group.blockSignals(True)
        if enabled:
            self.intent_flat_btn.setChecked(True)
        elif not self.state.linear_output:
            self.intent_print_btn.setChecked(True)
        self.intent_btn_group.blockSignals(False)
        self._sync_flat_enabled()

    def _on_linear_output_changed(self, enabled: bool) -> None:
        self.intent_btn_group.blockSignals(True)
        if enabled:
            self.intent_linear_btn.setChecked(True)
        elif not self.state.flat_output:
            self.intent_print_btn.setChecked(True)
        self.intent_btn_group.blockSignals(False)
        self._sync_flat_enabled()

    _EXPANSION_OPTIONS: dict[str, list[tuple[str, float | None]]] = {
        "pakon": [("4× (default)", None), ("2×", 2.0), ("Off", 1.0)],
        "pakon_f335": [("Off (default)", None), ("2×", 2.0), ("4×", 4.0)],
        "dng": [("Off (default)", None), ("2×", 2.0), ("4×", 4.0)],
        "nef": [],
        "fff": [],
        "noritsu": [("16× (default)", None), ("8×", 8.0), ("Off", 1.0)],
        "camera": [],
        "tiff": [("Off (default)", None), ("2×", 2.0), ("4×", 4.0)],
        "unsupported": [],
    }

    def _refresh_linear_expansion_combo(self) -> None:
        from negpy.services.export.linear_output import linear_output_source_type

        path = self.state.current_file_path or ""
        source_type = linear_output_source_type(path) if path else "unsupported"
        options = self._EXPANSION_OPTIONS.get(source_type, [])

        combo = self.linear_expansion_combo
        combo.blockSignals(True)
        combo.clear()
        if not options:
            combo.addItem("N/A")
            combo.setEnabled(False)
        else:
            for label, _val in options:
                combo.addItem(label)
            combo.setEnabled(True)
            current = self.state.linear_expansion
            for i, (_label, val) in enumerate(options):
                if val == current:
                    combo.setCurrentIndex(i)
                    break
            else:
                combo.setCurrentIndex(0)
        combo.blockSignals(False)
        self._current_expansion_source_type = source_type

        needs_gamma = source_type in ("tiff", "nef")
        self.linear_gamma_row.setVisible(needs_gamma)
        self.linear_gamma_hint.setVisible(needs_gamma)
        if needs_gamma:
            self._refresh_linear_gamma_combo()

        is_camera = source_type == "camera"
        has_ir = self.state.has_ir
        show_corrections = is_camera or has_ir
        self.linear_corrections_label.setVisible(show_corrections)
        self.linear_wb_checkbox.setVisible(is_camera)
        self.linear_flatfield_checkbox.setVisible(is_camera)
        self.linear_sensor_checkbox.setVisible(is_camera)
        self.linear_ice_checkbox.setVisible(has_ir)
        self.linear_ice_checkbox.setEnabled(has_ir)
        if not has_ir:
            self.linear_ice_checkbox.setToolTip("Source has no IR channel")
        else:
            self.linear_ice_checkbox.setToolTip("Apply IR-based dust and scratch correction")

        has_flatfield = bool(self.state.config.flatfield.apply and self.state.config.flatfield.profile_id)
        self.linear_flatfield_checkbox.setEnabled(has_flatfield)
        if not has_flatfield:
            self.linear_flatfield_checkbox.setToolTip("No flatfield profile configured")
        else:
            self.linear_flatfield_checkbox.setToolTip("Apply the flatfield gain correction")

        has_matrix = self.state.config.process.sensor_matrix is not None
        self.linear_sensor_checkbox.setEnabled(has_matrix)
        if not has_matrix:
            self.linear_sensor_checkbox.setToolTip("No sensor correction matrix configured")
        else:
            self.linear_sensor_checkbox.setToolTip("Apply the sensor crosstalk unmixing matrix")

        any_on = (
            self.state.linear_apply_wb
            or (self.state.linear_apply_flatfield and has_flatfield)
            or (self.state.linear_apply_sensor and has_matrix)
            or (self.state.linear_apply_ice and has_ir)
        )
        self.linear_corrections_hint.setVisible(show_corrections and any_on)

    def _on_linear_expansion_changed(self, index: int) -> None:
        source_type = getattr(self, "_current_expansion_source_type", "unsupported")
        options = self._EXPANSION_OPTIONS.get(source_type, [])
        if 0 <= index < len(options):
            self.state.linear_expansion = options[index][1]

    def _refresh_linear_gamma_combo(self) -> None:
        from negpy.services.export.linear_output import TIFF_GAMMA_OPTIONS

        combo = self.linear_gamma_combo
        combo.blockSignals(True)
        combo.clear()
        for key, label in TIFF_GAMMA_OPTIONS:
            combo.addItem(label, key)
        current = self.state.linear_gamma_key
        for i, (key, _label) in enumerate(TIFF_GAMMA_OPTIONS):
            if key == current:
                combo.setCurrentIndex(i)
                break
        else:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _on_linear_format_changed(self, index: int) -> None:
        fmt = self.linear_format_combo.itemData(index)
        if fmt:
            self.state.linear_format = str(fmt)
            self.controller.session.save_flat_output_prefs()
        if hasattr(self, "linear_jxl_effort_slider"):
            self.linear_jxl_effort_slider.setVisible(self.state.linear_format == "jxl")

    def _on_linear_jxl_effort_changed(self, value: float) -> None:
        self.state.linear_jxl_effort = int(value)
        self.controller.session.save_flat_output_prefs()

    def _on_linear_gamma_changed(self, index: int) -> None:
        from negpy.services.export.linear_output import TIFF_GAMMA_OPTIONS

        if 0 <= index < len(TIFF_GAMMA_OPTIONS):
            self.state.linear_gamma_key = TIFF_GAMMA_OPTIONS[index][0]
            self.controller.session.save_flat_output_prefs()

    def _on_linear_correction_changed(self, _checked: bool) -> None:
        self.state.linear_apply_wb = self.linear_wb_checkbox.isChecked()
        self.state.linear_apply_flatfield = self.linear_flatfield_checkbox.isChecked()
        self.state.linear_apply_sensor = self.linear_sensor_checkbox.isChecked()
        self.state.linear_apply_ice = self.linear_ice_checkbox.isChecked()
        self.controller.session.save_flat_output_prefs()
        any_on = (
            self.state.linear_apply_wb or self.state.linear_apply_flatfield or self.state.linear_apply_sensor or self.state.linear_apply_ice
        )
        self.linear_corrections_hint.setVisible(any_on)

    def _on_flat_peek_changed(self, active: bool) -> None:
        self.flat_peek_btn.blockSignals(True)
        self.flat_peek_btn.setChecked(active)
        self.flat_peek_btn.blockSignals(False)

    # --- Preview (soft proof + monitor profile, preview only) ----------------

    def _add_preview_section(self) -> None:
        # Preview-only controls (no effect on export). Collapsed by default and
        # parked below the export action so it doesn't split the form from Export.
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

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
        content_layout.addWidget(self.soft_proof_checkbox)

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
        constrain_combo(self.display_combo)
        self.display_combo.setToolTip("Monitor profile the preview is displayed on (affects preview only, not export)")
        override = self.state.monitor_profile_override
        self.display_combo.setCurrentText(override if override in self.display_spaces else "As detected")
        disp_row = QHBoxLayout()
        disp_label = field_label("Display")
        disp_label.setFixedWidth(90)
        disp_row.addWidget(disp_label)
        disp_row.addWidget(self.display_combo)
        content_layout.addLayout(disp_row)

        self.display_detected_label = hint_label("")
        content_layout.addWidget(self.display_detected_label)
        self._refresh_display_info()

        # Warns when the preview won't reflect the export's gamut clamp (soft
        # proof off + export space narrower than the working space).
        self.proof_mismatch_label = hint_label("Soft proof is off — preview won't show the export's color clipping", kind="warning")
        content_layout.addWidget(self.proof_mismatch_label)
        self._refresh_proof_mismatch_warning()

        repo = self.controller.session.repo
        expanded = bool(repo.get_global_setting("section_expanded_export_preview", default=False))
        section = CollapsibleSection("Preview", expanded=expanded, icon=qta.icon("fa5s.eye", color="#aaa"))
        section.set_content(content)
        section.expanded_changed.connect(lambda checked: repo.save_global_setting("section_expanded_export_preview", checked))
        self.layout.addWidget(section)

    # --- Edit sidecars -------------------------------------------------------

    def _add_sidecars_section(self) -> None:
        """Collapsible EXPORT EDITS SIDECARS section: on-export toggle + manual export, side by side."""
        conf = self.state.config.export

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        btn_row = QHBoxLayout()

        self.sidecars_enabled_btn = self._small_toggle(
            "fa5s.file-export",
            "Save on export",
            conf.export_sidecars_enabled,
            "When on, every export also writes a .negpy edit sidecar next to each source frame. Edits stay in the database too.",
        )
        btn_row.addWidget(self.sidecars_enabled_btn)

        self.export_sidecars_btn = QPushButton(" Export sidecars")
        self.export_sidecars_btn.setObjectName("export_sidecars_btn")
        self.export_sidecars_btn.setProperty("primary", True)
        self.export_sidecars_btn.setFixedHeight(default_button_height())
        self.export_sidecars_btn.setIcon(qta.icon("fa5s.file-code", color="white"))
        self.export_sidecars_btn.setToolTip("Write edit sidecars for all visible frames now")
        btn_row.addWidget(self.export_sidecars_btn)

        content_layout.addLayout(btn_row)

        repo = self.controller.session.repo
        expanded = bool(repo.get_global_setting("section_expanded_export_sidecars", default=False))
        self._sidecars_section = CollapsibleSection("Sidecars", expanded=expanded, icon=qta.icon("fa5s.file-export", color="#aaa"))
        self._sidecars_section.setToolTip("Optional plain-file copies of edits next to your sources, for archival. SQLite stays primary.")
        self._sidecars_section.set_content(content)
        self._sidecars_section.expanded_changed.connect(
            lambda checked: repo.save_global_setting("section_expanded_export_sidecars", checked)
        )
        self.layout.addWidget(self._sidecars_section)

    # --- Batch ---------------------------------------------------------------

    # Sticky scopes: the chevron menu picks what a split button does; only the
    # button itself triggers an export. key -> (menu label, button label, tooltip)
    _EXPORT_SCOPES = {
        "current": (
            "Export current frame",
            " Export Current Frame",
            "Export the current frame with the settings below  Ctrl+E",
        ),
        "selected": (
            "Export selected frames",
            " Export Selected",
            "Export the selected filmstrip frames using the settings below",
        ),
        "all": (
            "Export all visible frames",
            " Export All",
            "Export every visible frame using the settings below",
        ),
    }

    # The two "all visible" scopes (current vs saved per-frame settings) collapsed
    # into one when per-frame export settings were retired.
    _RETIRED_EXPORT_SCOPES = {"all_current": "all", "all_saved": "all"}

    _PRESET_SCOPES = {
        "current": (
            "Export current frame",
            " Export Presets",
            "Export the current frame with every enabled preset",
        ),
        "selected": (
            "Export selected frames",
            " Export Presets (selected)",
            "Export the selected filmstrip frames with every enabled preset",
        ),
        "all": (
            "Export all visible frames",
            " Export Presets (all)",
            "Export every visible frame with every enabled preset",
        ),
    }

    def _build_scope_menu(self, scopes: dict, on_select) -> tuple[QMenu, dict]:
        """Checkable, exclusive scope menu for a sticky split button."""
        menu = QMenu(self)
        group = QActionGroup(menu)
        group.setExclusive(True)
        actions = {}
        for key, (label, _btn_label, tooltip) in scopes.items():
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setToolTip(tooltip)
            act.triggered.connect(lambda _checked=False, k=key: on_select(k))
            group.addAction(act)
            actions[key] = act
        return menu, actions

    def _add_export_section(self) -> None:
        """Primary Export action: the menu arrow selects the sticky scope."""
        menu, self._export_scope_actions = self._build_scope_menu(self._EXPORT_SCOPES, self._set_export_scope)
        self._export_menu = menu

        container, self.export_main_btn, self.export_menu_btn = make_split_button(" Export", "fa5s.check-circle", menu, primary=True)
        self.export_menu_btn.setToolTip("Choose what the Export button does")
        self.layout.addWidget(container)

        saved = self.controller.session.repo.get_global_setting("export_scope", "current")
        saved = self._RETIRED_EXPORT_SCOPES.get(saved, saved)
        self._set_export_scope(saved if saved in self._EXPORT_SCOPES else "current", persist=False)

    def _set_export_scope(self, key: str, persist: bool = True) -> None:
        self._export_scope = key
        _label, btn_label, tooltip = self._EXPORT_SCOPES[key]
        self._export_scope_actions[key].setChecked(True)
        self.export_main_btn.setText(btn_label)
        self.export_main_btn.setToolTip(tooltip)
        if persist:
            self.controller.session.repo.save_global_setting("export_scope", key)

    def _flush_export_settings(self) -> None:
        """Stop the debounce timer and write the form into state immediately."""
        self.update_timer.stop()
        self._persist_all_export_settings()

    def _on_export_clicked(self) -> None:
        self._flush_export_settings()
        if self.state.linear_output:
            scope = self._export_scope
            if scope == "all":
                files = [
                    self.state.uploaded_files[i]
                    for i in self.controller.session.asset_model.visible_actual_indices_ordered()
                    if not self.state.uploaded_files[i].get("excluded")
                ]
                self.controller.request_linear_output_export(files=files)
            elif scope == "selected":
                files = [
                    self.state.uploaded_files[i]
                    for i in self.state.selected_indices
                    if 0 <= i < len(self.state.uploaded_files) and not self.state.uploaded_files[i].get("excluded")
                ]
                self.controller.request_linear_output_export(files=files)
            else:
                self.controller.request_linear_output_export()
            return
        scope = self._export_scope
        if scope == "selected":
            self.controller.request_export_selected()
        elif scope == "all":
            self.controller.request_batch_export()
        else:
            self.controller.request_export()

    def _set_preset_scope(self, key: str, persist: bool = True) -> None:
        self._preset_scope = key
        _label, btn_label, tooltip = self._PRESET_SCOPES[key]
        self._preset_scope_actions[key].setChecked(True)
        self.export_presets_btn.setText(btn_label)
        self.export_presets_btn.setToolTip(tooltip)
        if persist:
            self.controller.session.repo.save_global_setting("preset_export_scope", key)

    def _on_export_presets_clicked(self) -> None:
        self._flush_export_settings()
        scope = self._preset_scope
        if scope == "selected":
            self.controller.request_preset_export_selected()
        elif scope == "all":
            self.controller.request_preset_batch_export()
        else:
            self.controller.request_preset_export()

    def _rebuild_preset_rows(self) -> None:
        """Rebuild the preset checkbox list from state."""
        for cb in self._preset_checkboxes:
            self._presets_inner.removeWidget(cb)
            cb.deleteLater()
        self._preset_checkboxes.clear()

        presets = self.state.export_presets
        self._no_presets_label.setVisible(not presets)

        for i, preset in enumerate(presets):
            cb = QCheckBox(preset_display_name(preset))
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

    def _config_to_form_values(self) -> dict:
        """Build the form's value dict from the export config + ICC AppState."""
        conf = self.state.config.export
        return {
            "export_fmt": conf.export_fmt,
            "jpeg_quality": conf.jpeg_quality,
            "jxl_lossless": conf.jxl_lossless,
            "jxl_distance": conf.jxl_distance,
            "jxl_effort": conf.jxl_effort,
            "webp_quality": conf.webp_quality,
            "webp_lossless": conf.webp_lossless,
            "webp_method": conf.webp_method,
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

        self._sync_active_contact_sheet_template()
        cs_kwargs = self._contact_sheet_persist_kwargs()

        self.update_config_section(
            "export",
            persist=True,
            render=True,
            export_fmt=vals["export_fmt"],
            jpeg_quality=vals["jpeg_quality"],
            jxl_lossless=vals["jxl_lossless"],
            jxl_distance=vals["jxl_distance"],
            jxl_effort=vals["jxl_effort"],
            webp_quality=vals["webp_quality"],
            webp_lossless=vals["webp_lossless"],
            webp_method=vals["webp_method"],
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
            export_sidecars_enabled=self.sidecars_enabled_btn.isChecked(),
            **cs_kwargs,
        )

    def _on_display_changed(self, index: int) -> None:
        self.controller.set_monitor_override(self.display_map[index])

    def _refresh_display_info(self) -> None:
        """Update the 'As detected' label with the live detected monitor profile.

        When detection fails (no profile), warn in red prompting a manual pick.
        """
        from negpy.infrastructure.display.color_mgmt import profile_description

        detected = self.state.monitor_icc_detected_bytes
        desc = profile_description(detected)
        self.display_combo.setItemText(0, f"As detected ({desc})")
        if detected is None:
            self.display_detected_label.setText("Auto-detection failed — select your monitor's color space above.")
            set_hint_kind(self.display_detected_label, "error")
        else:
            self.display_detected_label.setText(f"Detected: {desc}")
            set_hint_kind(self.display_detected_label, "muted")

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

    def _refresh_export_enabled(self) -> None:
        """Disable the Export action when the current format/colour-space pairing
        can't be encoded (JPEG XL only tags a subset of colour spaces)."""
        linear_on = self.state.linear_output
        if linear_on:
            from negpy.services.export.linear_output import is_linear_output_supported

            path = self.state.current_file_path or ""
            supported = bool(path) and is_linear_output_supported(path)
            self.export_main_btn.setEnabled(supported)
            self.export_menu_btn.setEnabled(True)
            if hasattr(self, "linear_expansion_row"):
                self._refresh_linear_expansion_combo()
        else:
            blocked = self.form.is_export_blocked()
            self.export_main_btn.setEnabled(not blocked)
            self.export_menu_btn.setEnabled(not blocked)

    def sync_ui(self) -> None:
        conf = self.state.config.export
        self.block_signals(True)
        try:
            self.form.load(self._config_to_form_values())
            self.soft_proof_checkbox.setChecked(self.state.soft_proof_enabled)
            override = self.state.monitor_profile_override
            self.display_combo.setCurrentText(override if override in self.display_spaces else "As detected")
            self._refresh_display_info()
            layout = self._contact_sheet_layout_for_config(conf)
            self.cs_cell_px_input.setValue(layout.cell_px)
            self.cs_gap_input.setValue(layout.gap)
            self.cs_margin_input.setValue(layout.margin)
            self.cs_max_tiles_input.setValue(layout.max_tiles)
            self.cs_show_labels_checkbox.setChecked(layout.show_labels)
            self._cs_background_color = layout.background_color
            self._cs_label_color = layout.label_color
            self._update_cs_colors_btn_tooltip()
            self.cs_output_path_edit.setText(conf.contact_sheet_output_path)
            self.sidecars_enabled_btn.setChecked(conf.export_sidecars_enabled)
            self.printing_notes_preview_btn.setChecked(self.state.printing_notes)
            self._refresh_contact_sheet_templates()
            saved_template = conf.contact_sheet_template.strip()
            if saved_template and saved_template in ContactSheetTemplates.list_templates():
                self.cs_template_combo.setCurrentText(saved_template)
            else:
                self.cs_template_combo.setCurrentText(ContactSheetTemplates.DEFAULT_NAME)
            if self.state.linear_output:
                self.intent_linear_btn.setChecked(True)
            elif self.state.flat_output:
                self.intent_flat_btn.setChecked(True)
            else:
                self.intent_print_btn.setChecked(True)
            self.flat_peek_btn.setChecked(self.state.flat_peek)
            self.linear_wb_checkbox.setChecked(self.state.linear_apply_wb)
            self.linear_flatfield_checkbox.setChecked(self.state.linear_apply_flatfield)
            self.linear_sensor_checkbox.setChecked(self.state.linear_apply_sensor)
            self.linear_ice_checkbox.setChecked(self.state.linear_apply_ice)
            self._refresh_linear_gamma_combo()
        finally:
            self.block_signals(False)

        self._sync_flat_enabled()

        self._refresh_proof_mismatch_warning()
        self._refresh_export_enabled()
        self._rebuild_preset_rows()

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.soft_proof_checkbox,
            self.display_combo,
            self.cs_cell_px_input,
            self.cs_gap_input,
            self.cs_margin_input,
            self.cs_max_tiles_input,
            self.cs_show_labels_checkbox,
            self.cs_colors_btn,
            self.cs_output_path_edit,
            self.cs_template_combo,
            self.sidecars_enabled_btn,
            self.flat_peek_btn,
            self.printing_notes_preview_btn,
            self.linear_wb_checkbox,
            self.linear_flatfield_checkbox,
            self.linear_sensor_checkbox,
            self.linear_ice_checkbox,
            self.linear_gamma_combo,
        ]
        for w in widgets:
            w.blockSignals(blocked)
        self.intent_btn_group.blockSignals(blocked)
