"""Batch metadata tagging: edit gear, capture and process fields once, then apply
them to the current frame, a selection, or the whole roll."""

from dataclasses import replace

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.settings_catalog import CATALOG, SettingRow, preset_config, rows_for_keys
from negpy.desktop.view.styles.templates import (
    field_label,
    hint_label,
    icon_button,
    labeled_action,
    pin_dialog_default,
    wrap_tooltip,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.granular_settings_dialog import ScopeRadios, build_scope_row
from negpy.desktop.view.widgets.location_picker_dialog import LocationPickerDialog
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.domain.models import WorkspaceConfig
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
from negpy.features.metadata.gear_logic import metadata_from_gear, metadata_from_process, metadata_from_scan_setup
from negpy.features.metadata.gear_models import GearLibrary
from negpy.features.metadata.models import FORMAT_OPTIONS, MetadataConfig, PUSH_PULL_LABELS, PUSH_PULL_VALUES, format_label, format_value
from negpy.services.assets.presets import MetadataPresets

_METADATA_ROWS: dict[str, SettingRow] = {row.label: row for title, rows in CATALOG if title == "Metadata" for row in rows}
_SIMPLE_GROUPS = ("Exposure", "Protect Original Metadata", "Description Fields")


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (tuple, list)):
        return ", ".join(str(v) for v in value) if value else "—"
    return str(value) if value else "—"


class RollSettingsDialog(QDialog):
    """Groups mirror the Metadata section of settings_catalog.CATALOG, so the rows this
    hands to apply_preset_fields are the catalog's own -- no separate grouping rule to
    keep in sync. Editable groups hold real field widgets, kept in a scratch MetadataConfig
    updated the same way the Metadata panel updates its own (metadata_from_gear and friends);
    the rest carry the active frame's own value, ticked in or out like Apply Settings."""

    def __init__(
        self,
        parent,
        source_config: WorkspaceConfig,
        gear_library: GearLibrary,
        *,
        sel_count: int,
        roll_count: int,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Roll Settings")
        self.resize(440, 640)
        self._library = gear_library
        self._meta = source_config.metadata
        self._checks: dict[str, QCheckBox] = {}
        self._simple_labels: dict[str, QLabel] = {}
        self._scope_radios: ScopeRadios

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        root.addWidget(hint_label("Set values once, then apply the ticked groups to the current frame, a selection or the whole roll."))
        root.addLayout(self._build_preset_row())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_lg)
        col.addWidget(self._build_gear_group())
        col.addWidget(self._build_capture_date_group())
        col.addWidget(self._build_place_group())
        col.addWidget(self._build_process_group())
        col.addWidget(self._build_scanning_group())
        col.addWidget(self._build_roll_group())
        for label in _SIMPLE_GROUPS:
            col.addWidget(self._build_simple_group(label))
        col.addStretch()
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        scope_row, self._scope_radios = build_scope_row(self, sel_count, roll_count, show_current=True)
        # A roll-wide tag is the common case; current-frame-only is the exception a
        # user reaches for by hand, not the other way around.
        if roll_count > 0:
            self._scope_radios.roll.setChecked(True)
        root.addLayout(scope_row)
        root.addLayout(self._build_footer())
        pin_dialog_default(self.apply_btn, scope=self)

    # ── group scaffolding ───────────────────────────────────────────────

    def _group(self, label: str, content: QWidget) -> QWidget:
        row = _METADATA_ROWS[label]
        default = MetadataConfig()
        edited = any(getattr(self._meta, f) != getattr(default, f) for f in row.fields)
        box = QCheckBox(label)
        box.setChecked(edited)
        self._checks[label] = box
        wrapper = QWidget()
        col = QVBoxLayout(wrapper)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_sm)
        col.addWidget(box)
        inset = QWidget()
        inset_layout = QVBoxLayout(inset)
        inset_layout.setContentsMargins(THEME.space_xl, 0, 0, 0)
        inset_layout.addWidget(content)
        col.addWidget(inset)
        return wrapper

    def _set_meta(self, group: str, **kwargs) -> None:
        self._meta = replace(self._meta, **kwargs)
        self._checks[group].setChecked(True)

    def _build_simple_group(self, label: str) -> QWidget:
        row = _METADATA_ROWS[label]
        value = _fmt(getattr(self._meta, row.fields[0]) if len(row.fields) == 1 else tuple(getattr(self._meta, f) for f in row.fields))
        lbl = hint_label(value)
        self._simple_labels[label] = lbl
        return self._group(label, lbl)

    # ── editable groups ─────────────────────────────────────────────────

    def _build_gear_group(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_md)

        col.addWidget(field_label("Camera"))
        self.camera_combo = SearchableGearCombo(placeholder="Search cameras…")
        self.camera_combo.setToolTip(wrap_tooltip("Original film camera body. Click and type to search."))
        self.camera_combo.set_gear_items(self._library.cameras, self._meta.camera_id or "", lambda c: c.resolved_display_name)
        self.camera_combo.selection_changed.connect(lambda _id: self._on_gear_changed("camera_id", self.camera_combo))
        col.addWidget(self.camera_combo)

        col.addWidget(field_label("Lens"))
        self.lens_combo = SearchableGearCombo(placeholder="Search lenses…")
        self.lens_combo.setToolTip(wrap_tooltip("Original lens used on the film camera. Click and type to search."))
        self.lens_combo.set_gear_items(self._library.lenses, self._meta.lens_id or "", lambda x: x.resolved_display_name)
        self.lens_combo.selection_changed.connect(lambda _id: self._on_gear_changed("lens_id", self.lens_combo))
        col.addWidget(self.lens_combo)

        col.addWidget(field_label("Film stock"))
        self.film_stock_combo = SearchableGearCombo(placeholder="Search film stocks…")
        self.film_stock_combo.setToolTip(wrap_tooltip("Film stock used for the original capture. Click and type to search."))
        self.film_stock_combo.set_gear_items(self._library.film_stocks, self._meta.film_stock_id or "", lambda x: x.resolved_display_name)
        self.film_stock_combo.selection_changed.connect(lambda _id: self._on_gear_changed("film_stock_id", self.film_stock_combo))
        col.addWidget(self.film_stock_combo)

        # The stock carries the film's format, so Format lives in the Gear row, matching
        # GEAR_FIELDS (settings_catalog applies it as a unit with camera/lens/film stock).
        col.addWidget(field_label("Format"))
        self.format_combo = QComboBox()
        self.format_combo.setToolTip(wrap_tooltip("Film format written to the frame's metadata"))
        self.format_combo.addItems(FORMAT_OPTIONS)
        self.format_combo.setCurrentText(format_label(self._meta.format))
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        col.addWidget(self.format_combo)

        self.format_other_edit = QLineEdit()
        self.format_other_edit.setPlaceholderText("e.g. 6×7")
        self.format_other_edit.setToolTip(wrap_tooltip("A format the list does not carry, written as you type it"))
        self.format_other_edit.setText(self._meta.format_other)
        self.format_other_edit.setVisible(self._meta.format == "Other")
        self.format_other_edit.textEdited.connect(lambda t: self._set_meta("Analog Gear", format_other=t.strip()))
        col.addWidget(self.format_other_edit)

        return self._group("Analog Gear", body)

    def apply_detected_gear(self, **gear_ids: str) -> None:
        """Pre-fill Gear from a folder-name match, ticked like any other edit -- Cancel
        discards it, Apply writes it, same as filling the fields by hand."""
        gear_ids = {k: v for k, v in gear_ids.items() if v}
        if not gear_ids:
            return
        self._meta = metadata_from_gear(self._meta, self._library, **gear_ids)
        self._checks["Analog Gear"].setChecked(True)
        self._reload_widgets()

    def _on_gear_changed(self, field: str, combo: SearchableGearCombo) -> None:
        self._meta = metadata_from_gear(self._meta, self._library, **{field: combo.selected_id()})
        self.format_combo.setCurrentText(format_label(self._meta.format))
        self.format_other_edit.setText(self._meta.format_other)
        self.format_other_edit.setVisible(self._meta.format == "Other")
        self._checks["Analog Gear"].setChecked(True)

    def _on_format_changed(self, text: str) -> None:
        value = format_value(text)
        self.format_other_edit.setVisible(value == "Other")
        self._set_meta("Analog Gear", format=value)

    def _build_capture_date_group(self) -> QWidget:
        self.capture_date_edit = QLineEdit()
        self.capture_date_edit.setPlaceholderText(CAPTURE_DATE_HINT)
        self.capture_date_edit.setText(self._meta.capture_date)
        self.capture_date_edit.setToolTip(
            wrap_tooltip(
                "When the frames were shot. Give only what you know: a year, a year and month, "
                "a date, or a date and time. An offset like +02:00 may follow a time."
            )
        )
        self.capture_date_edit.textEdited.connect(lambda _t: self._checks["Capture Date"].setChecked(True))
        return self._group("Capture Date", self.capture_date_edit)

    def _build_place_group(self) -> QWidget:
        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(THEME.space_sm)
        self.place_edit = QLineEdit()
        self.place_edit.setPlaceholderText("Pick on a map, or paste coordinates")
        self.place_edit.setText(
            place_summary(
                self._meta.location_city,
                self._meta.location_state,
                self._meta.location_country,
                self._meta.gps_latitude,
                self._meta.gps_longitude,
            )
        )
        self.place_edit.setToolTip(
            wrap_tooltip("Capture place. Paste a coordinate pair or a map link here, or use the map button to pick one.")
        )
        self.place_edit.editingFinished.connect(self._on_place_edited)
        row.addWidget(self.place_edit, 1)
        map_btn = icon_button("fa5s.map-marked-alt", "Pick the capture place on a map (contacts OpenStreetMap)")
        map_btn.clicked.connect(self._open_location_picker)
        row.addWidget(map_btn)
        return self._group("Place", body)

    def _on_place_edited(self) -> None:
        text = self.place_edit.text().strip()
        if not text:
            return
        coords = parse_coords(text)
        if coords is None:
            return
        self._set_meta("Place", gps_latitude=coords[0], gps_longitude=coords[1])

    def _open_location_picker(self) -> None:
        dlg = LocationPickerDialog(
            self._meta.gps_latitude,
            self._meta.gps_longitude,
            self._meta.location_city,
            self._meta.location_state,
            self._meta.location_country,
            parent=self,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        lat, lon, city, state, country = dlg.location()
        self._set_meta("Place", gps_latitude=lat, gps_longitude=lon, location_city=city, location_state=state, location_country=country)
        self.place_edit.setText(place_summary(city, state, country, lat, lon))

    def _build_process_group(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_md)

        col.addWidget(field_label("Saved process"))
        self.process_combo = SearchableGearCombo(placeholder="Search processes…")
        self.process_combo.setToolTip(wrap_tooltip("A saved development recipe. Picking one fills Developer and Push / Pull."))
        self.process_combo.set_gear_items(self._library.processes, self._meta.process_id or "", lambda p: p.resolved_display_name)
        self.process_combo.selection_changed.connect(lambda _id: self._on_process_selected())
        col.addWidget(self.process_combo)

        row1 = QHBoxLayout()
        row1.setSpacing(THEME.space_sm)
        dev_col = QVBoxLayout()
        dev_col.setSpacing(THEME.space_md)
        dev_col.addWidget(field_label("Developer"))
        self.developer_edit = QLineEdit()
        self.developer_edit.setPlaceholderText("e.g. D-76")
        self.developer_edit.setToolTip(wrap_tooltip("Developer the film was processed in."))
        self.developer_edit.setText(self._meta.developer)
        self.developer_edit.textEdited.connect(lambda _t: self._checks["Process"].setChecked(True))
        dev_col.addWidget(self.developer_edit)
        dil_col = QVBoxLayout()
        dil_col.setSpacing(THEME.space_md)
        dil_col.addWidget(field_label("Dilution"))
        self.dilution_edit = QLineEdit()
        self.dilution_edit.setPlaceholderText("e.g. 1+50")
        self.dilution_edit.setToolTip(wrap_tooltip("Working strength, for example 1+1, 1+50 or stock."))
        self.dilution_edit.setText(self._meta.process_dilution)
        self.dilution_edit.textEdited.connect(lambda _t: self._checks["Process"].setChecked(True))
        dil_col.addWidget(self.dilution_edit)
        row1.addLayout(dev_col, 2)
        row1.addLayout(dil_col, 1)
        col.addLayout(row1)

        col.addWidget(field_label("Push / Pull"))
        self.push_pull_combo = QComboBox()
        self.push_pull_combo.setToolTip(wrap_tooltip("Stops the film was pushed or pulled in development."))
        self.push_pull_combo.addItems([PUSH_PULL_LABELS[v] for v in PUSH_PULL_VALUES])
        self.push_pull_combo.setCurrentIndex(
            PUSH_PULL_VALUES.index(self._meta.push_pull) if self._meta.push_pull in PUSH_PULL_VALUES else 3
        )
        self.push_pull_combo.currentIndexChanged.connect(lambda _i: self._checks["Process"].setChecked(True))
        col.addWidget(self.push_pull_combo)

        row2 = QHBoxLayout()
        row2.setSpacing(THEME.space_sm)
        time_col = QVBoxLayout()
        time_col.setSpacing(THEME.space_md)
        time_col.addWidget(field_label("Time"))
        self.dev_time_edit = QLineEdit()
        self.dev_time_edit.setPlaceholderText(DEV_TIME_HINT)
        self.dev_time_edit.setToolTip(wrap_tooltip("Development time, as mm:ss or plain minutes."))
        self.dev_time_edit.setText(format_dev_time(self._meta.process_time_seconds))
        self.dev_time_edit.textEdited.connect(lambda _t: self._checks["Process"].setChecked(True))
        time_col.addWidget(self.dev_time_edit)
        temp_col = QVBoxLayout()
        temp_col.setSpacing(THEME.space_md)
        temp_col.addWidget(field_label("Temperature (°C)"))
        self.dev_temp_edit = QLineEdit()
        self.dev_temp_edit.setPlaceholderText("e.g. 20")
        self.dev_temp_edit.setToolTip(wrap_tooltip("Development temperature, in °C."))
        self.dev_temp_edit.setText(format_temperature(self._meta.process_temperature_c))
        self.dev_temp_edit.textEdited.connect(lambda _t: self._checks["Process"].setChecked(True))
        temp_col.addWidget(self.dev_temp_edit)
        row2.addLayout(time_col, 1)
        row2.addLayout(temp_col, 1)
        col.addLayout(row2)

        return self._group("Process", body)

    def _on_process_selected(self) -> None:
        self._meta = metadata_from_process(self._meta, self._library, self.process_combo.selected_id())
        self.developer_edit.setText(self._meta.developer)
        self.dilution_edit.setText(self._meta.process_dilution)
        self.push_pull_combo.setCurrentIndex(
            PUSH_PULL_VALUES.index(self._meta.push_pull) if self._meta.push_pull in PUSH_PULL_VALUES else 3
        )
        self.dev_time_edit.setText(format_dev_time(self._meta.process_time_seconds))
        self.dev_temp_edit.setText(format_temperature(self._meta.process_temperature_c))
        self._checks["Process"].setChecked(True)

    def _build_scanning_group(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_md)
        col.addWidget(field_label("Saved setup"))
        self.scan_setup_combo = SearchableGearCombo(placeholder="Search scan setups…")
        self.scan_setup_combo.setToolTip(wrap_tooltip("A saved digitizing setup. Picking one fills Scanning."))
        self.scan_setup_combo.set_gear_items(self._library.scan_setups, self._meta.scanning_id or "", lambda s: s.resolved_display_name)
        self.scan_setup_combo.selection_changed.connect(lambda _id: self._on_scan_setup_selected())
        col.addWidget(self.scan_setup_combo)
        col.addWidget(field_label("Scanning"))
        self.scanning_edit = QLineEdit()
        self.scanning_edit.setPlaceholderText("e.g. DSLR copy-stand scan")
        self.scanning_edit.setToolTip(wrap_tooltip("How the film was digitized."))
        self.scanning_edit.setText(self._meta.scanning)
        self.scanning_edit.textEdited.connect(lambda _t: self._checks["Scanning"].setChecked(True))
        col.addWidget(self.scanning_edit)
        return self._group("Scanning", body)

    def _on_scan_setup_selected(self) -> None:
        self._meta = metadata_from_scan_setup(self._meta, self._library, self.scan_setup_combo.selected_id())
        self.scanning_edit.setText(self._meta.scanning)
        self._checks["Scanning"].setChecked(True)

    def _build_roll_group(self) -> QWidget:
        self.capture_roll_edit = QLineEdit()
        self.capture_roll_edit.setPlaceholderText("e.g. Roll001")
        self.capture_roll_edit.setText(self._meta.capture_roll)
        self.capture_roll_edit.setToolTip(wrap_tooltip("Scan capture roll name. Used in export filename templates as {{ roll }}."))
        self.capture_roll_edit.textEdited.connect(lambda _t: self._checks["Roll"].setChecked(True))
        return self._group("Roll", self.capture_roll_edit)

    # ── preset shortcut ─────────────────────────────────────────────────

    def _build_preset_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(THEME.space_sm)
        self.preset_combo = SearchableGearCombo(placeholder="Load from a metadata preset…")
        self.preset_combo.setToolTip(wrap_tooltip("A saved set of metadata values. Click and type to search."))
        self.preset_combo.set_labeled_items([(n, n) for n in sorted(MetadataPresets.list_presets())], "")
        row.addWidget(self.preset_combo, 1)
        load_btn = labeled_action("fa5s.download", " Load", "Fill these fields from the preset, and tick the groups it stores")
        load_btn.clicked.connect(self._on_load_preset)
        row.addWidget(load_btn)
        return row

    def _on_load_preset(self) -> None:
        name = self.preset_combo.selected_id()
        data = MetadataPresets.load_preset(name) if name else None
        if not data:
            return
        preset_meta = preset_config(data).metadata
        rows = rows_for_keys(data, "metadata")
        self._meta = replace(self._meta, **{f: getattr(preset_meta, f) for row in rows for f in row.fields})
        for row in rows:
            if row.label in self._checks:
                self._checks[row.label].setChecked(True)
        self._reload_widgets()

    def _reload_widgets(self) -> None:
        m = self._meta
        self.camera_combo.set_selected_id(m.camera_id or "")
        self.lens_combo.set_selected_id(m.lens_id or "")
        self.film_stock_combo.set_selected_id(m.film_stock_id or "")
        self.format_combo.setCurrentText(format_label(m.format))
        self.format_other_edit.setText(m.format_other)
        self.format_other_edit.setVisible(m.format == "Other")
        self.capture_date_edit.setText(m.capture_date)
        self.place_edit.setText(place_summary(m.location_city, m.location_state, m.location_country, m.gps_latitude, m.gps_longitude))
        self.process_combo.set_selected_id(m.process_id or "")
        self.developer_edit.setText(m.developer)
        self.dilution_edit.setText(m.process_dilution)
        self.push_pull_combo.setCurrentIndex(PUSH_PULL_VALUES.index(m.push_pull) if m.push_pull in PUSH_PULL_VALUES else 3)
        self.dev_time_edit.setText(format_dev_time(m.process_time_seconds))
        self.dev_temp_edit.setText(format_temperature(m.process_temperature_c))
        self.scan_setup_combo.set_selected_id(m.scanning_id or "")
        self.scanning_edit.setText(m.scanning)
        self.capture_roll_edit.setText(m.capture_roll)
        for label, lbl in self._simple_labels.items():
            row = _METADATA_ROWS[label]
            value = getattr(m, row.fields[0]) if len(row.fields) == 1 else tuple(getattr(m, f) for f in row.fields)
            lbl.setText(_fmt(value))

    # ── footer / result ─────────────────────────────────────────────────

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.accept)
        row.addWidget(cancel_btn)
        row.addWidget(self.apply_btn)
        return row

    def selected_rows(self) -> list[SettingRow]:
        return [_METADATA_ROWS[label] for label, box in self._checks.items() if box.isChecked()]

    def selected_config(self) -> WorkspaceConfig:
        """Snapshot the scratch metadata, parsing free-text fields the same way the
        Metadata panel does at persist time: unreadable text keeps the stored value,
        blank clears it."""
        date_text = self.capture_date_edit.text().strip()
        parsed_date = parse_capture_date(date_text)
        capture_date = parsed_date.xmp_text() if parsed_date else ("" if not date_text else self._meta.capture_date)

        time_text = self.dev_time_edit.text().strip()
        process_time_seconds = None
        if time_text:
            parsed_time = parse_dev_time(time_text)
            process_time_seconds = self._meta.process_time_seconds if parsed_time is None else parsed_time

        temp_text = self.dev_temp_edit.text().strip()
        process_temperature_c = None
        if temp_text:
            parsed_temp = parse_temperature(temp_text)
            process_temperature_c = self._meta.process_temperature_c if parsed_temp is None else parsed_temp

        meta = replace(
            self._meta,
            capture_date=capture_date,
            developer=self.developer_edit.text().strip(),
            process_dilution=self.dilution_edit.text().strip(),
            push_pull=PUSH_PULL_VALUES[self.push_pull_combo.currentIndex()],
            process_time_seconds=process_time_seconds,
            process_temperature_c=process_temperature_c,
            scanning=self.scanning_edit.text().strip(),
            capture_roll=self.capture_roll_edit.text().strip(),
            format_other=self.format_other_edit.text().strip() if self._meta.format == "Other" else "",
        )
        return replace(WorkspaceConfig(), metadata=meta)

    def scope(self) -> str:
        return self._scope_radios.value()
