import qtawesome as qta
from dataclasses import asdict, replace
from typing import Optional
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

from negpy.desktop.settings_catalog import (
    apply_selected_fields,
    preset_config,
    preset_values,
    rows_for_keys,
)
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, hint_label, wrap_tooltip
from negpy.desktop.view.styles.fonts import mono_font_family
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.description_fields_dialog import DescriptionFieldsDialog
from negpy.desktop.view.widgets.gear_library_dialog import GearLibraryDialog
from negpy.desktop.view.widgets.location_picker_dialog import LocationPickerDialog
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.features.metadata.capture import (
    CAPTURE_DATE_HINT,
    DEV_TIME_HINT,
    format_dev_time,
    format_temperature,
    parse_capture_date,
    parse_coords,
    parse_dev_time,
    parse_temperature,
    place_summary,
)
from negpy.features.metadata.exif_read import extract_scan_from_exif
from negpy.features.metadata.gear_logic import metadata_from_gear, metadata_from_process, metadata_from_scan_setup
from negpy.features.metadata.gear_models import GearLibrary
from negpy.features.metadata.models import (
    DEFAULT_DESCRIPTION_FIELDS,
    FORMAT_OPTIONS,
    PROCESS_FIELDS,
    PUSH_PULL_LABELS,
    PUSH_PULL_VALUES,
    SCANNING_FIELDS,
    MetadataConfig,
    format_label,
    format_value,
)
from negpy.features.metadata.payload import build_metadata_payload
from negpy.services.assets.gear import GearProfiles
from negpy.services.assets.presets import MetadataPresets

PUSH_PULL_OPTIONS = [PUSH_PULL_LABELS[v] for v in PUSH_PULL_VALUES]
_LOAD_TOOLTIP = "Write the selected preset's fields onto this frame"
_CLEAR_TOOLTIPS = {
    "gear_clear_btn": ("Clear the camera, lens and film stock selections", "metadata_clear_gear"),
    "process_clear_btn": (
        "Clear the saved process and the developer, dilution, push, time and temperature it fills",
        "metadata_clear_process",
    ),
    "scan_clear_btn": ("Clear the saved scan setup and its scanning note", "metadata_clear_scanning"),
}


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
        self._description_fields: tuple[str, ...] = conf.description_fields or DEFAULT_DESCRIPTION_FIELDS

        self.protect_check = QCheckBox("Protect original metadata")
        self.protect_check.setChecked(conf.protect_original_metadata)
        self.protect_check.setToolTip(
            "When enabled, NegPy copies EXIF and XMP from the source file onto exports "
            "without adding or changing metadata. Gear and process fields are ignored."
        )
        self.layout.addWidget(self.protect_check)

        self.sync_check = QCheckBox("Sync custom metadata to all files in batch export")
        self.sync_check.setChecked(conf.sync_to_batch)
        self.sync_check.setToolTip(
            "Batch and preset exports write this frame's capture, gear and process values to every file, instead of each file's own."
        )
        self.layout.addWidget(self.sync_check)

        self._metadata_controls = QWidget()
        controls = QVBoxLayout(self._metadata_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(THEME.space_lg)

        # ── METADATA PRESETS ─────────────────────────────────────────────
        preset_body, presets = self._card_body()
        load_row = QHBoxLayout()
        load_row.setSpacing(THEME.space_sm)
        self.metadata_preset_combo = SearchableGearCombo(placeholder="Search metadata presets…")
        self.metadata_preset_combo.setToolTip("A saved set of metadata values. Click and type to search.")
        load_row.addWidget(self.metadata_preset_combo, 1)
        self.metadata_preset_load_btn = QPushButton("Load")
        load_row.addWidget(self.metadata_preset_load_btn)
        presets.addLayout(load_row)

        self.manage_btn = QPushButton(" Manage…")
        self.manage_btn.setIcon(qta.icon("fa5s.cog", color=THEME.text_primary))
        self.manage_btn.setToolTip("Save, edit and delete metadata presets, cameras, lenses and film stocks")
        presets.addWidget(self.manage_btn)
        self._refresh_metadata_presets()
        controls.addWidget(self._card("Metadata Presets", "presets", preset_body, "fa5s.magic"))

        # ── ANALOG GEAR ──────────────────────────────────────────────────
        gear_body, gear = self._card_body()
        gear.addWidget(hint_label("Type in any field to search the gear library."))

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

        self.gear_clear_btn = QPushButton("Clear")

        gear.addWidget(self.gear_clear_btn)
        controls.addWidget(self._card("Analog Gear", "gear", gear_body, "fa5s.camera-retro"))

        # ── CAPTURE ──────────────────────────────────────────────────────
        cap_body, cap = self._card_body()
        cap.addWidget(field_label("Date"))
        self.capture_date_edit = QLineEdit()
        self.capture_date_edit.setPlaceholderText(CAPTURE_DATE_HINT)
        self.capture_date_edit.setText(conf.capture_date)
        self.capture_date_edit.setToolTip(
            "When the frame was shot. Give only what you know: a year, a year and month, "
            "a date, or a date and time. An offset like +02:00 may follow a time."
        )
        cap.addWidget(self.capture_date_edit)

        cap.addWidget(field_label("Place"))
        place_row = QHBoxLayout()
        place_row.setSpacing(THEME.space_sm)
        self.place_edit = QLineEdit()
        self.place_edit.setPlaceholderText("Pick on a map, or paste coordinates")
        self.place_edit.setToolTip("Capture place. Paste a coordinate pair or a map link here, or use Map… to pick one.")
        place_row.addWidget(self.place_edit, 1)
        self.place_map_btn = self._icon_action("fa5s.map-marked-alt", "Pick the capture place on a map (contacts OpenStreetMap)")
        place_row.addWidget(self.place_map_btn)
        self.place_clear_btn = self._icon_action("fa5s.times", "Clear the capture place")
        place_row.addWidget(self.place_clear_btn)
        cap.addLayout(place_row)
        controls.addWidget(self._card("Capture", "capture", cap_body, "fa5s.clock"))

        # ── PROCESS ──────────────────────────────────────────────────────
        proc_body, proc = self._card_body()
        proc.addWidget(field_label("Saved process"))
        self.process_combo = SearchableGearCombo(placeholder="Search processes…")
        self.process_combo.setToolTip("A saved development recipe. Picking one fills Developer and Push / Pull.")
        proc.addWidget(self.process_combo)

        proc.addWidget(field_label("Format"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(FORMAT_OPTIONS)
        self.format_combo.setCurrentText(format_label(conf.format))
        proc.addWidget(self.format_combo)

        self.format_other_edit = QLineEdit()
        self.format_other_edit.setPlaceholderText("e.g. 6×7")
        self.format_other_edit.setText(conf.format_other)
        self.format_other_edit.setVisible(conf.format == "Other")
        proc.addWidget(self.format_other_edit)

        dev_name_row = QHBoxLayout()
        dev_name_row.setSpacing(THEME.space_sm)
        developer_col = QVBoxLayout()
        developer_col.setSpacing(THEME.space_md)
        developer_col.addWidget(field_label("Developer"))
        self.developer_edit = QLineEdit()
        self.developer_edit.setPlaceholderText("e.g. D-76")
        self.developer_edit.setText(conf.developer)
        developer_col.addWidget(self.developer_edit)
        dilution_col = QVBoxLayout()
        dilution_col.setSpacing(THEME.space_md)
        dilution_col.addWidget(field_label("Dilution"))
        self.dilution_edit = QLineEdit()
        self.dilution_edit.setPlaceholderText("e.g. 1+50")
        self.dilution_edit.setText(conf.process_dilution)
        self.dilution_edit.setToolTip("Working strength, for example 1+1, 1+50 or stock.")
        dilution_col.addWidget(self.dilution_edit)
        dev_name_row.addLayout(developer_col, 2)
        dev_name_row.addLayout(dilution_col, 1)
        proc.addLayout(dev_name_row)

        proc.addWidget(field_label("Push / Pull"))
        self.push_pull_combo = QComboBox()
        self.push_pull_combo.addItems(PUSH_PULL_OPTIONS)
        idx = PUSH_PULL_VALUES.index(conf.push_pull) if conf.push_pull in PUSH_PULL_VALUES else 3
        self.push_pull_combo.setCurrentIndex(idx)
        proc.addWidget(self.push_pull_combo)

        dev_row = QHBoxLayout()
        dev_row.setSpacing(THEME.space_sm)
        time_col = QVBoxLayout()
        time_col.setSpacing(THEME.space_md)
        time_col.addWidget(field_label("Time"))
        self.dev_time_edit = QLineEdit()
        self.dev_time_edit.setPlaceholderText(DEV_TIME_HINT)
        self.dev_time_edit.setText(format_dev_time(conf.process_time_seconds))
        self.dev_time_edit.setToolTip("Development time, as mm:ss or plain minutes.")
        time_col.addWidget(self.dev_time_edit)
        temp_col = QVBoxLayout()
        temp_col.setSpacing(THEME.space_md)
        temp_col.addWidget(field_label("Temp (°C)"))
        self.dev_temp_edit = QLineEdit()
        self.dev_temp_edit.setPlaceholderText("e.g. 20")
        self.dev_temp_edit.setText(format_temperature(conf.process_temperature_c))
        temp_col.addWidget(self.dev_temp_edit)
        dev_row.addLayout(time_col, 1)
        dev_row.addLayout(temp_col, 1)
        proc.addLayout(dev_row)

        self.process_clear_btn = QPushButton("Clear")
        proc.addWidget(self.process_clear_btn)
        controls.addWidget(self._card("Process", "process", proc_body, "fa5s.flask"))

        # ── SCANNING ─────────────────────────────────────────────────────
        scan_body, scan = self._card_body()
        scan.addWidget(field_label("Saved setup"))
        self.scan_setup_combo = SearchableGearCombo(placeholder="Search scan setups…")
        self.scan_setup_combo.setToolTip("A saved digitizing setup. Picking one fills Scanning.")
        scan.addWidget(self.scan_setup_combo)

        scan.addWidget(field_label("Scanning"))
        self.scanning_edit = QLineEdit()
        self.scanning_edit.setPlaceholderText("e.g. DSLR copy-stand scan")
        self.scanning_edit.setText(conf.scanning)
        scan.addWidget(self.scanning_edit)

        roll_row = QHBoxLayout()
        roll_row.setSpacing(THEME.space_sm)
        roll_col = QVBoxLayout()
        roll_col.setSpacing(THEME.space_md)
        roll_col.addWidget(field_label("Roll"))
        self.capture_roll_edit = QLineEdit()
        self.capture_roll_edit.setPlaceholderText("e.g. Roll001")
        self.capture_roll_edit.setText(conf.capture_roll)
        self.capture_roll_edit.setToolTip("Scan capture roll name (Scanlight). Used in export filename templates as {{ roll }}.")
        roll_col.addWidget(self.capture_roll_edit)
        frame_col = QVBoxLayout()
        frame_col.setSpacing(THEME.space_md)
        frame_col.addWidget(field_label("Frame"))
        self.capture_frame_edit = QLineEdit()
        self.capture_frame_edit.setPlaceholderText("e.g. 12")
        if conf.capture_frame is not None:
            self.capture_frame_edit.setText(str(conf.capture_frame))
        self.capture_frame_edit.setToolTip("Scan capture frame number. Used in export filename templates as {{ frame }}.")
        frame_col.addWidget(self.capture_frame_edit)
        roll_row.addLayout(roll_col, 2)
        roll_row.addLayout(frame_col, 1)
        scan.addLayout(roll_row)

        self.scan_clear_btn = QPushButton("Clear")
        scan.addWidget(self.scan_clear_btn)
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

        preview_top = QHBoxLayout()
        preview_top.setContentsMargins(0, 0, 0, 0)
        preview_top.setSpacing(THEME.space_sm)
        preview_hint = hint_label("Written to exported files on export.")
        preview_top.addWidget(preview_hint, 1)
        self.description_fields_btn = QPushButton("Description…")
        self.description_fields_btn.setToolTip("Choose which fields join into EXIF ImageDescription.")
        preview_top.addWidget(self.description_fields_btn)
        preview_layout.addLayout(preview_top)

        self.preview_rows = QVBoxLayout()
        self.preview_rows.setSpacing(2)
        preview_layout.addLayout(self.preview_rows)

        self.preview_empty = hint_label("Select gear or enter process metadata to see a preview.")
        preview_layout.addWidget(self.preview_empty)

        self.preview_section = CollapsibleSection("Metadata preview", expanded=True)
        self.preview_section.set_content(self.preview_content)
        self.layout.addWidget(self.preview_section)

        # After every card: the tooltips it fills in span all of them.
        self.apply_shortcut_tooltips()
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
        self.description_fields_btn.setEnabled(enabled)
        self.sync_check.setEnabled(enabled)

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
        self.description_fields_btn.clicked.connect(self._open_description_fields)
        self.gear_clear_btn.clicked.connect(self._on_gear_clear)
        self.process_clear_btn.clicked.connect(self._on_process_clear)
        self.scan_clear_btn.clicked.connect(self._on_scanning_clear)
        self.camera_combo.selection_changed.connect(self._on_gear_changed)
        self.lens_combo.selection_changed.connect(self._on_gear_changed)
        self.film_stock_combo.selection_changed.connect(self._on_gear_changed)
        self.manage_btn.clicked.connect(self._open_gear_library)

        self.metadata_preset_combo.selection_changed.connect(self._update_metadata_preset_tooltip)
        self.metadata_preset_load_btn.clicked.connect(self._on_metadata_preset_load)

        self.capture_date_edit.textChanged.connect(self._on_capture_date_changed)
        self.place_edit.editingFinished.connect(self._on_place_edited)
        self.place_map_btn.clicked.connect(self._open_location_picker)
        self.place_clear_btn.clicked.connect(self._on_place_clear)

        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.format_other_edit.textChanged.connect(self._mark_dirty)
        self.process_combo.selection_changed.connect(self._on_process_selected)
        self.scan_setup_combo.selection_changed.connect(self._on_scan_setup_selected)
        self.developer_edit.textChanged.connect(self._on_process_edited)
        self.dilution_edit.textChanged.connect(self._on_process_edited)
        self.push_pull_combo.currentIndexChanged.connect(self._on_process_edited)
        self.dev_time_edit.textChanged.connect(self._on_dev_time_changed)
        self.dev_temp_edit.textChanged.connect(self._on_dev_temp_changed)
        self.scanning_edit.textChanged.connect(self._on_scanning_edited)
        self.capture_roll_edit.textChanged.connect(self._mark_dirty)
        self.capture_frame_edit.textChanged.connect(self._mark_dirty)
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

    def _open_description_fields(self) -> None:
        dlg = DescriptionFieldsDialog(self._description_fields, self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        self._description_fields = dlg.selected_fields()
        self.update_config_section(
            "metadata",
            persist=True,
            render=False,
            readback_metrics=False,
            description_fields=self._description_fields,
        )
        # Sticky is updated only here, not on every metadata save, so the last Description confirm
        # wins for unset frames on the roll.
        self.controller.session.repo.save_global_setting(
            "last_description_fields",
            list(self._description_fields),
        )
        self._schedule_preview()

    def _refresh_gear_combos(self, *, force: bool = False) -> None:
        conf = self.state.config.metadata
        self._gear_library = GearProfiles.load_library()
        library = self._gear_library

        def should_refresh(combo: SearchableGearCombo) -> bool:
            return force or not combo.is_editing()

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

        if should_refresh(self.process_combo):
            self.process_combo.set_gear_items(
                library.processes,
                conf.process_id or "",
                lambda process: process.resolved_display_name,
            )

        if should_refresh(self.scan_setup_combo):
            self.scan_setup_combo.set_gear_items(
                library.scan_setups,
                conf.scanning_id or "",
                lambda setup: setup.resolved_display_name,
            )

    def _on_process_selected(self, *_args) -> None:
        self._dirty = False
        self._apply_metadata_config(metadata_from_process(self.state.config.metadata, self._gear_library, self.process_combo.selected_id()))

    def _on_scan_setup_selected(self, *_args) -> None:
        self._dirty = False
        self._apply_metadata_config(
            metadata_from_scan_setup(self.state.config.metadata, self._gear_library, self.scan_setup_combo.selected_id())
        )

    def _on_dev_time_changed(self, text: str) -> None:
        self._flag_invalid(self.dev_time_edit, bool(text.strip()) and parse_dev_time(text) is None)
        self._on_process_edited()

    def _on_dev_temp_changed(self, text: str) -> None:
        self._flag_invalid(self.dev_temp_edit, bool(text.strip()) and parse_temperature(text) is None)
        self._on_process_edited()

    def _flag_invalid(self, edit: QLineEdit, invalid: bool) -> None:
        edit.setStyleSheet(f"border: 1px solid {THEME.accent_secondary};" if invalid else "")

    def _dev_time_value(self) -> Optional[int]:
        """Unreadable text keeps what is stored; blank clears. Same rule as Capture Date."""
        text = self.dev_time_edit.text().strip()
        if not text:
            return None
        parsed = parse_dev_time(text)
        return self.state.config.metadata.process_time_seconds if parsed is None else parsed

    def _dev_temp_value(self) -> Optional[float]:
        text = self.dev_temp_edit.text().strip()
        if not text:
            return None
        parsed = parse_temperature(text)
        return self.state.config.metadata.process_temperature_c if parsed is None else parsed

    def _on_process_edited(self, *_args) -> None:
        """Typing over a saved value unlinks it, so the picker never names a value that is gone."""
        self._clear_combo(self.process_combo)
        self._mark_dirty()

    def _on_scanning_edited(self, *_args) -> None:
        self._clear_combo(self.scan_setup_combo)
        self._mark_dirty()

    def _clear_combo(self, combo: SearchableGearCombo) -> None:
        combo.blockSignals(True)
        combo.set_selected_id("")
        combo.blockSignals(False)

    def _on_gear_clear(self) -> None:
        cleared = replace(
            self.state.config.metadata,
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

    def _clear_fields(self, fields: tuple[str, ...]) -> None:
        defaults = MetadataConfig()
        self._dirty = False
        self._apply_metadata_config(replace(self.state.config.metadata, **{f: getattr(defaults, f) for f in fields}))

    def _on_process_clear(self) -> None:
        self._clear_fields(PROCESS_FIELDS)

    def _on_scanning_clear(self) -> None:
        self._clear_fields(SCANNING_FIELDS)

    def _on_gear_changed(self, *_args) -> None:
        sender = self.sender()
        kwargs: dict = {}
        if sender is self.camera_combo:
            kwargs["camera_id"] = self.camera_combo.selected_id()
        elif sender is self.lens_combo:
            kwargs["lens_id"] = self.lens_combo.selected_id()
        elif sender is self.film_stock_combo:
            kwargs["film_stock_id"] = self.film_stock_combo.selected_id()
        else:
            return

        new_meta = metadata_from_gear(
            self.state.config.metadata,
            self._gear_library,
            **kwargs,
        )
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

    def apply_shortcut_tooltips(self) -> None:
        """Re-read the binding: tooltips are built before saved overrides load, and again
        whenever the shortcut editor writes a new one."""
        self.metadata_preset_load_btn.setToolTip(tooltip_with_shortcut(_LOAD_TOOLTIP, "metadata_preset_load"))
        for attr, (text, action_id) in _CLEAR_TOOLTIPS.items():
            getattr(self, attr).setToolTip(tooltip_with_shortcut(text, action_id))

    def _refresh_metadata_presets(self) -> None:
        selected = self.metadata_preset_combo.selected_id()
        names = sorted(MetadataPresets.list_presets())
        self.metadata_preset_combo.set_labeled_items([(n, n) for n in names], selected if selected in names else "")
        self._update_metadata_preset_tooltip()

    def _update_metadata_preset_tooltip(self, *_args) -> None:
        name = self.metadata_preset_combo.selected_id()
        data = MetadataPresets.load_preset(name) if name else None
        lines = "\n".join(f"{label}: {value}" for label, value in preset_values(data, "metadata")) if data else ""
        self.metadata_preset_combo.setToolTip(wrap_tooltip(lines) if lines else "A saved set of metadata values. Click and type to search.")

    def _on_metadata_preset_load(self) -> None:
        name = self.metadata_preset_combo.selected_id()
        data = MetadataPresets.load_preset(name) if name else None
        if not data:
            return
        # The form may hold edits the debounce has not written yet; the preset wins over them.
        self._dirty = False
        rows = rows_for_keys(data, "metadata")
        merged = apply_selected_fields(preset_config(data), self.state.config, rows)
        self._apply_metadata_config(merged.metadata)

    def _open_gear_library(self) -> None:
        # The dialog holds the config it was given, so the debounce has to land first or a
        # preset saved from "the current frame" misses the edit that is still pending.
        self.update_timer.stop()
        self._persist_all_metadata_settings()
        dlg = GearLibraryDialog(self._gear_library, parent=self, current_config=self.state.config)
        dlg.library_changed.connect(self._on_library_changed)
        dlg.presets_changed.connect(self._refresh_metadata_presets)
        if dlg.exec():
            self._on_library_changed()
        self._refresh_metadata_presets()

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

    def _on_capture_date_changed(self, text: str) -> None:
        valid = not text.strip() or parse_capture_date(text) is not None
        self.capture_date_edit.setStyleSheet("" if valid else f"border: 1px solid {THEME.accent_secondary};")
        self._mark_dirty()

    def _source_exif(self) -> Optional[dict]:
        current_hash = self.state.current_file_hash
        if current_hash and current_hash in self.state.source_exif:
            return self.state.source_exif[current_hash]
        return None

    def _place_text(self) -> str:
        conf = self.state.config.metadata
        return place_summary(
            conf.location_city,
            conf.location_state,
            conf.location_country,
            conf.gps_latitude,
            conf.gps_longitude,
        )

    def _apply_location(self, lat, lon, city: str, state: str, country: str) -> None:
        self.update_config_section(
            "metadata",
            persist=True,
            render=False,
            readback_metrics=False,
            gps_latitude=lat,
            gps_longitude=lon,
            location_city=city,
            location_state=state,
            location_country=country,
        )
        self._set_place_text_quiet()
        self._schedule_preview()

    def _set_place_text_quiet(self) -> None:
        self.place_edit.blockSignals(True)
        try:
            self.place_edit.setText(self._place_text())
        finally:
            self.place_edit.blockSignals(False)

    def _on_place_edited(self) -> None:
        """Typed text is only ever coordinates; place names come from the picker."""
        text = self.place_edit.text().strip()
        if not text:
            self._on_place_clear()
            return
        coords = parse_coords(text)
        if coords is None:
            self._set_place_text_quiet()
            return
        conf = self.state.config.metadata
        self._apply_location(coords[0], coords[1], conf.location_city, conf.location_state, conf.location_country)

    def _on_place_clear(self) -> None:
        self._apply_location(None, None, "", "", "")

    def _open_location_picker(self) -> None:
        conf = self.state.config.metadata
        center = None
        if conf.gps_latitude is None or conf.gps_longitude is None:
            scan = extract_scan_from_exif(self._source_exif())
            if scan.gps_latitude is not None and scan.gps_longitude is not None:
                center = (scan.gps_latitude, scan.gps_longitude)
        dlg = LocationPickerDialog(
            conf.gps_latitude,
            conf.gps_longitude,
            conf.location_city,
            conf.location_state,
            conf.location_country,
            center=center,
            parent=self,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        self._apply_location(*dlg.location())

    def _on_format_changed(self, text: str) -> None:
        self.format_other_edit.setVisible(text == "Other")
        self._mark_dirty()

    def _persist_all_metadata_settings(self) -> None:
        if not self._dirty:
            return
        self._dirty = False

        fmt = format_value(self.format_combo.currentText())
        pp_idx = self.push_pull_combo.currentIndex()

        exposure_override = ""
        if not self._exif_locked.get("exposure", True):
            exposure_override = self.exposure_edit.text().strip()

        frame_text = self.capture_frame_edit.text().strip()
        capture_frame = None
        if frame_text:
            try:
                capture_frame = int(frame_text)
            except ValueError:
                capture_frame = self.state.config.metadata.capture_frame

        date_text = self.capture_date_edit.text().strip()
        parsed_date = parse_capture_date(date_text)
        capture_date = parsed_date.xmp_text() if parsed_date else ("" if not date_text else self.state.config.metadata.capture_date)

        self.update_config_section(
            "metadata",
            persist=True,
            render=False,
            readback_metrics=False,
            capture_date=capture_date,
            camera_id=self.camera_combo.selected_id(),
            lens_id=self.lens_combo.selected_id(),
            film_stock_id=self.film_stock_combo.selected_id(),
            format=fmt,
            format_other=self.format_other_edit.text().strip() if fmt == "Other" else "",
            process_id=self.process_combo.selected_id(),
            developer=self.developer_edit.text().strip(),
            process_dilution=self.dilution_edit.text().strip(),
            push_pull=PUSH_PULL_VALUES[pp_idx] if 0 <= pp_idx < len(PUSH_PULL_VALUES) else 0,
            process_time_seconds=self._dev_time_value(),
            process_temperature_c=self._dev_temp_value(),
            scanning_id=self.scan_setup_combo.selected_id(),
            scanning=self.scanning_edit.text().strip(),
            capture_roll=self.capture_roll_edit.text().strip(),
            capture_frame=capture_frame,
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

            self.format_combo.setCurrentText(format_label(conf.format))
            self.format_other_edit.setText(conf.format_other)
            self.format_other_edit.setVisible(conf.format == "Other")
            self.capture_date_edit.setText(conf.capture_date)
            self.capture_date_edit.setStyleSheet("")
            self.place_edit.setText(self._place_text())
            self.developer_edit.setText(conf.developer)
            self.dilution_edit.setText(conf.process_dilution)
            idx = PUSH_PULL_VALUES.index(conf.push_pull) if conf.push_pull in PUSH_PULL_VALUES else 3
            self.push_pull_combo.setCurrentIndex(idx)
            self.dev_time_edit.setText(format_dev_time(conf.process_time_seconds))
            self.dev_temp_edit.setText(format_temperature(conf.process_temperature_c))
            self._flag_invalid(self.dev_time_edit, False)
            self._flag_invalid(self.dev_temp_edit, False)
            self.scanning_edit.setText(conf.scanning)
            self.capture_roll_edit.setText(conf.capture_roll)
            self.capture_frame_edit.setText("" if conf.capture_frame is None else str(conf.capture_frame))
            self.sync_check.setChecked(conf.sync_to_batch)
            self._description_fields = conf.description_fields or DEFAULT_DESCRIPTION_FIELDS

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

    def _preview_metadata_config(self):
        """MetadataConfig from the live form so preview tracks edits before debounce persist."""
        conf = self.state.config.metadata
        fmt = format_value(self.format_combo.currentText())
        pp_idx = self.push_pull_combo.currentIndex()
        exposure_override = ""
        if not self._exif_locked.get("exposure", True):
            exposure_override = self.exposure_edit.text().strip()
        else:
            exposure_override = conf.exposure_override

        parsed_date = parse_capture_date(self.capture_date_edit.text())

        return replace(
            conf,
            capture_date=parsed_date.xmp_text() if parsed_date else "",
            camera_id=self.camera_combo.selected_id(),
            lens_id=self.lens_combo.selected_id(),
            film_stock_id=self.film_stock_combo.selected_id(),
            format=fmt,
            format_other=self.format_other_edit.text().strip() if fmt == "Other" else "",
            process_id=self.process_combo.selected_id(),
            developer=self.developer_edit.text().strip(),
            process_dilution=self.dilution_edit.text().strip(),
            push_pull=PUSH_PULL_VALUES[pp_idx] if 0 <= pp_idx < len(PUSH_PULL_VALUES) else 0,
            process_time_seconds=self._dev_time_value(),
            process_temperature_c=self._dev_temp_value(),
            scanning_id=self.scan_setup_combo.selected_id(),
            scanning=self.scanning_edit.text().strip(),
            sync_to_batch=self.sync_check.isChecked(),
            exposure_override=exposure_override,
            description_fields=self._description_fields,
        )

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

        payload = build_metadata_payload(self._preview_metadata_config(), self._gear_library, self._source_exif())
        sections = payload.to_preview_sections()

        self.preview_empty.setText("Select gear or enter process metadata to see a preview.")
        self.preview_empty.setVisible(not sections)
        mono = f"font-family: {mono_font_family()}; font-size: {THEME.font_size_small}px;"

        for title, rows in sections:
            header = QLabel(title)
            header.setStyleSheet(
                f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px; font-weight: {THEME.weight_semibold};"
            )
            self.preview_rows.addWidget(header)
            for label, value in rows:
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                lbl = QLabel(label)
                lbl.setStyleSheet(f"color: {THEME.text_hint}; {mono}")
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
