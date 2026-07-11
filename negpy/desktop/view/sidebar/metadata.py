import qtawesome as qta
from dataclasses import asdict, replace
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, hint_label
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.gear_library_dialog import GearLibraryDialog
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.features.metadata.gear_logic import metadata_from_gear
from negpy.features.metadata.gear_models import GearLibrary
from negpy.features.metadata.payload import build_metadata_payload
from negpy.services.assets.gear import GearProfiles

FORMAT_OPTIONS = ["35mm", "120", "4×5", "8×10", "110", "Other"]
PUSH_PULL_OPTIONS = ["Push +3", "Push +2", "Push +1", "Normal", "Pull -1", "Pull -2", "Pull -3"]
PUSH_PULL_VALUES = [3, 2, 1, 0, -1, -2, -3]


class MetadataSidebar(BaseSidebar):
    """Panel for analog gear metadata written to exported files."""

    SIDE_MARGIN = THEME.space_xl

    def _init_ui(self) -> None:
        conf = self.state.config.metadata
        self._gear_library: GearLibrary = GearProfiles.load_library()

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(500)
        self.update_timer.timeout.connect(self._persist_all_metadata_settings)

        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(100)
        self.preview_timer.timeout.connect(self._update_preview)

        self._dirty = False
        self._exif_locked = {"exposure": True}

        self.protect_check = QCheckBox("Protect original metadata")
        self.protect_check.setChecked(conf.protect_original_metadata)
        self.protect_check.setToolTip(
            "When enabled, NegPy copies EXIF and XMP from the source file onto exports "
            "without adding or changing metadata. Gear and process fields are ignored."
        )
        self.layout.addWidget(self.protect_check)

        self._metadata_controls = QWidget()
        controls = QVBoxLayout(self._metadata_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(THEME.space_lg)

        # ── ANALOG GEAR ──────────────────────────────────────────────────
        gear_body, gear = self._card_body()
        gear.addWidget(hint_label("Type in any field to search the gear library."))

        preset_row = QHBoxLayout()
        preset_row.setSpacing(THEME.space_sm)
        gear.addWidget(field_label("Preset"))
        self.preset_combo = SearchableGearCombo(placeholder="Search presets…")
        self.preset_combo.setToolTip("Reusable camera + lens + film combination. Click and type to search.")
        preset_row.addWidget(self.preset_combo, 1)
        self.preset_clear_btn = QPushButton("Clear")
        self.preset_clear_btn.setToolTip("Clear gear preset selection")
        preset_row.addWidget(self.preset_clear_btn)
        gear.addLayout(preset_row)

        gear.addWidget(field_label("Camera"))
        self.camera_combo = SearchableGearCombo(placeholder="Search cameras…")
        self.camera_combo.setToolTip("Original film camera body. Click and type to search.")
        gear.addWidget(self.camera_combo)

        gear.addWidget(field_label("Lens"))
        self.lens_combo = SearchableGearCombo(placeholder="Search lenses…")
        self.lens_combo.setToolTip("Original lens used on the film camera. Click and type to search.")
        gear.addWidget(self.lens_combo)

        gear.addWidget(field_label("Film stock"))
        self.film_stock_combo = SearchableGearCombo(placeholder="Search film stocks…")
        self.film_stock_combo.setToolTip("Film stock used for the original capture. Click and type to search.")
        gear.addWidget(self.film_stock_combo)

        self.manage_btn = QPushButton(" Manage…")
        self.manage_btn.setIcon(qta.icon("fa5s.cog", color=THEME.text_primary))
        self.manage_btn.setToolTip("Edit cameras, lenses, film stocks, and gear presets")
        gear.addWidget(self.manage_btn)
        controls.addWidget(self._card("Analog Gear", "gear", gear_body, "fa5s.camera-retro"))

        # ── PROCESS ──────────────────────────────────────────────────────
        proc_body, proc = self._card_body()
        proc.addWidget(field_label("Format"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(FORMAT_OPTIONS)
        if conf.format in FORMAT_OPTIONS:
            self.format_combo.setCurrentText(conf.format)
        proc.addWidget(self.format_combo)

        self.format_other_edit = QLineEdit()
        self.format_other_edit.setPlaceholderText("e.g. 6×7")
        self.format_other_edit.setText(conf.format_other)
        self.format_other_edit.setVisible(conf.format == "Other")
        proc.addWidget(self.format_other_edit)

        proc.addWidget(field_label("Developer"))
        self.developer_edit = QLineEdit()
        self.developer_edit.setPlaceholderText("e.g. D-76 1+1")
        self.developer_edit.setText(conf.developer)
        proc.addWidget(self.developer_edit)

        proc.addWidget(field_label("Push / Pull"))
        self.push_pull_combo = QComboBox()
        self.push_pull_combo.addItems(PUSH_PULL_OPTIONS)
        idx = PUSH_PULL_VALUES.index(conf.push_pull) if conf.push_pull in PUSH_PULL_VALUES else 3
        self.push_pull_combo.setCurrentIndex(idx)
        proc.addWidget(self.push_pull_combo)
        controls.addWidget(self._card("Process", "process", proc_body, "fa5s.flask"))

        # ── SCANNING ─────────────────────────────────────────────────────
        scan_body, scan = self._card_body()
        scan.addWidget(field_label("Scanning"))
        self.scanning_edit = QLineEdit()
        self.scanning_edit.setPlaceholderText("e.g. DSLR copy-stand scan")
        self.scanning_edit.setText(conf.scanning)
        scan.addWidget(self.scanning_edit)

        self.sync_check = QCheckBox("Sync custom metadata to all files in batch export")
        self.sync_check.setChecked(conf.sync_to_batch)
        scan.addWidget(self.sync_check)
        controls.addWidget(self._card("Scanning", "scanning", scan_body, "mdi6.scanner"))

        # ── EXPOSURE ─────────────────────────────────────────────────────
        exp_body, exp = self._card_body()
        exp.addWidget(hint_label("Optional original capture exposure — click 🔓 to edit"))

        self.exposure_label = field_label("Exposure")
        exp.addWidget(self.exposure_label)
        self.exposure_edit = self._make_exif_field("exposure", exp)
        controls.addWidget(self._card("Exposure", "exposure", exp_body, "fa5s.stopwatch"))

        self._refresh_gear_combos()
        controls.addStretch()
        self.layout.addWidget(self._metadata_controls, 1)

        # ── METADATA PREVIEW ─────────────────────────────────────────────
        self.preview_content = QWidget()
        preview_layout = QVBoxLayout(self.preview_content)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)

        preview_hint = hint_label("Written to exported files on export.")
        preview_layout.addWidget(preview_hint)

        self.preview_rows = QVBoxLayout()
        self.preview_rows.setSpacing(2)
        preview_layout.addLayout(self.preview_rows)

        self.preview_empty = hint_label("Select gear or enter process metadata to see a preview.")
        preview_layout.addWidget(self.preview_empty)

        self.preview_section = CollapsibleSection("Metadata preview", expanded=True)
        self.preview_section.set_content(self.preview_content)
        self.layout.addWidget(self.preview_section)

        self._set_metadata_controls_enabled(not conf.protect_original_metadata)

    def _card_body(self) -> tuple[QWidget, QVBoxLayout]:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_md)
        return body, layout

    def _card(self, title: str, key: str, content: QWidget, icon_name: str) -> CollapsibleSection:
        repo = self.controller.session.repo
        setting = f"section_expanded_metadata_{key}"
        expanded = bool(repo.get_global_setting(setting, default=True))
        section = CollapsibleSection(title, expanded=expanded, icon=qta.icon(icon_name, color="#aaa"))
        section.set_content(content)
        section.expanded_changed.connect(lambda checked, s=setting: repo.save_global_setting(s, checked))
        return section

    def _make_exif_field(self, key: str, layout: QVBoxLayout) -> QLineEdit:
        row = QHBoxLayout()
        row.setSpacing(THEME.space_sm)

        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setPlaceholderText("—")
        self._apply_lock_style(edit, locked=True)

        lock_btn = QToolButton()
        lock_btn.setCheckable(True)
        lock_btn.setToolTip("Unlock to edit")
        self._update_lock_icon(lock_btn, locked=True)
        lock_btn.toggled.connect(lambda checked, k=key, e=edit, b=lock_btn: self._toggle_exif_lock(k, e, b, checked))

        row.addWidget(edit)
        row.addWidget(lock_btn)
        layout.addLayout(row)
        setattr(self, f"_{key}_lock_btn", lock_btn)
        return edit

    def _set_metadata_controls_enabled(self, enabled: bool) -> None:
        self._metadata_controls.setEnabled(enabled)

    def _apply_lock_style(self, edit: QLineEdit, locked: bool) -> None:
        if locked:
            edit.setStyleSheet(f"color: {THEME.text_secondary};")
            edit.setReadOnly(True)
        else:
            edit.setStyleSheet(f"color: {THEME.text_primary};")
            edit.setReadOnly(False)

    def _update_lock_icon(self, btn: QToolButton, locked: bool) -> None:
        icon_name = "fa5s.lock" if locked else "fa5s.lock-open"
        color = THEME.text_muted if locked else THEME.text_primary
        btn.setIcon(qta.icon(icon_name, color=color))

    def _toggle_exif_lock(self, key: str, edit: QLineEdit, btn: QToolButton, checked: bool) -> None:
        locked = not checked
        self._exif_locked[key] = locked
        self._apply_lock_style(edit, locked=locked)
        self._update_lock_icon(btn, locked=locked)
        if not locked:
            edit.setFocus()
        else:
            self._update_exif_display()
        self._mark_dirty()

    def _connect_signals(self) -> None:
        self.protect_check.toggled.connect(self._on_protect_toggled)
        self.preset_combo.selection_changed.connect(self._on_preset_changed)
        self.preset_clear_btn.clicked.connect(self._on_preset_clear)
        self.camera_combo.selection_changed.connect(self._on_gear_changed)
        self.lens_combo.selection_changed.connect(self._on_gear_changed)
        self.film_stock_combo.selection_changed.connect(self._on_gear_changed)
        self.manage_btn.clicked.connect(self._open_gear_library)

        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.format_other_edit.textChanged.connect(self._mark_dirty)
        self.developer_edit.textChanged.connect(self._mark_dirty)
        self.push_pull_combo.currentIndexChanged.connect(self._mark_dirty)
        self.scanning_edit.textChanged.connect(self._mark_dirty)
        self.sync_check.toggled.connect(self._mark_dirty)
        self.exposure_edit.textChanged.connect(self._mark_dirty)

        self.controller.session.file_selected.connect(self._on_file_selected)

    def _on_protect_toggled(self, checked: bool) -> None:
        self._set_metadata_controls_enabled(not checked)
        self.update_config_section(
            "metadata",
            persist=True,
            render=False,
            readback_metrics=False,
            protect_original_metadata=checked,
        )
        self._schedule_preview()

    def _refresh_gear_combos(self, *, force: bool = False) -> None:
        conf = self.state.config.metadata
        self._gear_library = GearProfiles.load_library()
        library = self._gear_library

        def should_refresh(combo: SearchableGearCombo) -> bool:
            return force or not combo.is_editing()

        if should_refresh(self.preset_combo):
            self.preset_combo.blockSignals(True)
            self.preset_combo.set_gear_items(
                library.gear_presets,
                conf.gear_preset_id or "",
                lambda p: p.display_name or "Unnamed preset",
                library,
            )
            self.preset_combo.blockSignals(False)

        if should_refresh(self.camera_combo):
            self.camera_combo.set_gear_items(
                library.cameras,
                conf.camera_id or "",
                lambda c: c.resolved_display_name,
            )

        if should_refresh(self.lens_combo):
            self.lens_combo.set_gear_items(
                library.lenses,
                conf.lens_id or "",
                lambda lens: lens.resolved_display_name,
            )

        if should_refresh(self.film_stock_combo):
            self.film_stock_combo.set_gear_items(
                library.film_stocks,
                conf.film_stock_id or "",
                lambda stock: stock.resolved_display_name,
            )

    def _on_preset_changed(self, _preset_id: str = "") -> None:
        preset_id = self.preset_combo.selected_id()
        if not preset_id:
            return
        self._dirty = False
        new_meta = metadata_from_gear(
            self.state.config.metadata,
            self._gear_library,
            gear_preset_id=preset_id,
        )
        self._apply_metadata_config(new_meta)

    def _on_preset_clear(self) -> None:
        cleared = replace(
            self.state.config.metadata,
            gear_preset_id="",
            camera_id="",
            lens_id="",
            film_stock_id="",
            camera_make="",
            camera_model="",
            lens_make="",
            lens_model="",
            focal_length_mm=None,
            max_aperture=None,
            film_iso=None,
            film_manufacturer="",
            film_color_type="",
            film="",
        )
        self._apply_metadata_config(cleared)

    def _on_gear_changed(self, *_args) -> None:
        sender = self.sender()
        kwargs: dict = {}
        if sender is self.camera_combo:
            kwargs["gear_preset_id"] = ""
            kwargs["camera_id"] = self.camera_combo.selected_id()
        elif sender is self.lens_combo:
            kwargs["gear_preset_id"] = ""
            kwargs["lens_id"] = self.lens_combo.selected_id()
        elif sender is self.film_stock_combo:
            kwargs["gear_preset_id"] = ""
            kwargs["film_stock_id"] = self.film_stock_combo.selected_id()
        else:
            return

        new_meta = metadata_from_gear(
            self.state.config.metadata,
            self._gear_library,
            **kwargs,
        )
        self.preset_combo.blockSignals(True)
        self.preset_combo.set_selected_id("")
        self.preset_combo.blockSignals(False)
        self._apply_metadata_config(new_meta, refresh_combos=False)

    def _apply_metadata_config(self, new_meta, *, refresh_combos: bool = True) -> None:
        self.update_config_section(
            "metadata",
            persist=True,
            render=False,
            readback_metrics=False,
            **asdict(new_meta),
        )
        if refresh_combos:
            self._refresh_gear_combos(force=True)
        self.sync_ui()
        self._schedule_preview()

    def _open_gear_library(self) -> None:
        dlg = GearLibraryDialog(self._gear_library, parent=self)
        dlg.library_changed.connect(self._on_library_changed)
        if dlg.exec():
            self._on_library_changed()

    def _on_library_changed(self) -> None:
        self._gear_library = GearProfiles.load_library()
        self._refresh_gear_combos()
        self._schedule_preview()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self.update_timer.start()
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        self.preview_timer.start()

    def _on_format_changed(self, text: str) -> None:
        self.format_other_edit.setVisible(text == "Other")
        self._mark_dirty()

    def _persist_all_metadata_settings(self) -> None:
        if not self._dirty:
            return
        self._dirty = False

        fmt = self.format_combo.currentText()
        pp_idx = self.push_pull_combo.currentIndex()

        exposure_override = ""
        if not self._exif_locked.get("exposure", True):
            exposure_override = self.exposure_edit.text().strip()

        self.update_config_section(
            "metadata",
            persist=True,
            render=False,
            readback_metrics=False,
            gear_preset_id=self.preset_combo.selected_id(),
            camera_id=self.camera_combo.selected_id(),
            lens_id=self.lens_combo.selected_id(),
            film_stock_id=self.film_stock_combo.selected_id(),
            format=fmt,
            format_other=self.format_other_edit.text().strip() if fmt == "Other" else "",
            developer=self.developer_edit.text().strip(),
            push_pull=PUSH_PULL_VALUES[pp_idx] if 0 <= pp_idx < len(PUSH_PULL_VALUES) else 0,
            scanning=self.scanning_edit.text().strip(),
            sync_to_batch=self.sync_check.isChecked(),
            exposure_override=exposure_override,
        )

    def sync_ui(self) -> None:
        if self._dirty:
            return

        conf = self.state.config.metadata

        self.block_signals(True)
        try:
            self.protect_check.setChecked(conf.protect_original_metadata)
            self._set_metadata_controls_enabled(not conf.protect_original_metadata)
            self._refresh_gear_combos()

            if conf.format in FORMAT_OPTIONS:
                self.format_combo.setCurrentText(conf.format)
            else:
                self.format_combo.setCurrentText("Other")
                self.format_other_edit.setText(conf.format_other)
            self.format_other_edit.setVisible(self.format_combo.currentText() == "Other")
            self.developer_edit.setText(conf.developer)
            idx = PUSH_PULL_VALUES.index(conf.push_pull) if conf.push_pull in PUSH_PULL_VALUES else 3
            self.push_pull_combo.setCurrentIndex(idx)
            self.scanning_edit.setText(conf.scanning)
            self.sync_check.setChecked(conf.sync_to_batch)

            if conf.exposure_override:
                self._set_exif_text_quiet("exposure", conf.exposure_override)
            else:
                self._update_exif_display()
        finally:
            self.block_signals(False)

        self._schedule_preview()

    def _set_exif_text_quiet(self, key: str, text: str) -> None:
        edit = getattr(self, f"{key}_edit", None)
        if edit is None:
            return
        edit.blockSignals(True)
        try:
            edit.setText(text)
        finally:
            edit.blockSignals(False)

    def _on_file_selected(self, _path: str) -> None:
        self._dirty = False
        self._reset_exif_locks()
        self.sync_ui()

    def _reset_exif_locks(self) -> None:
        self._exif_locked["exposure"] = True
        self._apply_lock_style(self.exposure_edit, locked=True)
        btn = getattr(self, "_exposure_lock_btn", None)
        if btn is not None:
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
            self._update_lock_icon(btn, locked=True)

    def _update_exif_display(self) -> None:
        conf = self.state.config.metadata
        if conf.exposure_override:
            self._set_exif_text_quiet("exposure", conf.exposure_override)
        else:
            self._set_exif_text_quiet("exposure", "")

    def _update_preview(self) -> None:
        while self.preview_rows.count():
            item = self.preview_rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        conf = self.state.config.metadata
        if conf.protect_original_metadata:
            self.preview_empty.setText("Original metadata will be copied from the source file on export.")
            self.preview_empty.setVisible(True)
            self.preview_section.setEnabled(True)
            return

        source_exif = None
        current_hash = self.state.current_file_hash
        if current_hash and current_hash in self.state.source_exif:
            source_exif = self.state.source_exif[current_hash]

        payload = build_metadata_payload(conf, self._gear_library, source_exif)
        sections = payload.to_preview_sections()

        self.preview_empty.setText("Select gear or enter process metadata to see a preview.")
        self.preview_empty.setVisible(not sections)
        mono = f"font-family: Consolas, monospace; font-size: {THEME.font_size_xs}px;"

        for title, rows in sections:
            header = QLabel(title)
            header.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_xs}px; font-weight: {THEME.weight_semibold};")
            self.preview_rows.addWidget(header)
            for label, value in rows:
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                lbl = QLabel(label)
                lbl.setStyleSheet(f"color: {THEME.text_muted}; {mono}")
                lbl.setFixedWidth(110)
                val = QLabel(value)
                val.setWordWrap(True)
                val.setStyleSheet(f"color: {THEME.text_primary}; {mono}")
                row_layout.addWidget(lbl)
                row_layout.addWidget(val, 1)
                self.preview_rows.addWidget(row)

    def block_signals(self, blocked: bool) -> None:
        for w in self.findChildren(QWidget):
            w.blockSignals(blocked)
