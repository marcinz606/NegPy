"""Modal dialog for managing the analog gear library."""

from __future__ import annotations

from dataclasses import replace

import qtawesome as qta
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.settings_catalog import (
    NON_METADATA_SECTIONS,
    preset_config,
    preset_values,
    rows_by_id,
    rows_for_keys,
    selected_flat_dict,
)
from negpy.desktop.view.confirm import confirm_delete_named
from negpy.desktop.view.styles.templates import dialog_pane_qss, field_label, hint_label, pane_header_qss, pin_dialog_default
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog
from negpy.features.metadata.gear_logic import (
    matches_gear_filter,
    metadata_from_gear,
    metadata_from_process,
    metadata_from_scan_setup,
)
from negpy.features.metadata.capture import DEV_TIME_HINT, format_dev_time, format_temperature, parse_dev_time, parse_temperature
from negpy.features.metadata.gear_models import (
    Camera,
    DevelopmentProcess,
    FilmColorType,
    FilmFormat,
    FilmStock,
    GearLibrary,
    Lens,
    ScanSetup,
)
from negpy.features.metadata.models import FORMAT_OPTIONS, PUSH_PULL_LABELS, PUSH_PULL_VALUES, format_label, format_value
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.services.assets.gear import GearProfiles
from negpy.services.assets.presets import MetadataPresets, is_valid_preset_name, preset_fields, preset_notes, with_preset_notes

_CATEGORIES = [
    ("cameras", "Cameras"),
    ("lenses", "Lenses"),
    ("film_stocks", "Film Stocks"),
    ("processes", "Process"),
    ("scan_setups", "Scanning"),
    ("metadata_presets", "Presets"),
]

# Metadata presets are files of stored values, not library records: the form pane
# shows what one holds and the field picker edits it.
_PRESETS = "metadata_presets"

_CATEGORY_FIELDS: dict[str, frozenset[str]] = {
    "cameras": frozenset({"display_name", "make", "model", "notes"}),
    "lenses": frozenset({"display_name", "make", "lens_model", "focal", "aperture", "notes"}),
    "film_stocks": frozenset({"display_name", "manufacturer", "stock_name", "iso", "format", "color_type", "notes"}),
    "processes": frozenset({"display_name", "developer", "dilution", "push_pull", "dev_time", "dev_temp", "notes"}),
    "scan_setups": frozenset({"display_name", "scanning", "notes"}),
    _PRESETS: frozenset(),
}

_CATEGORY_SEARCH_PLACEHOLDER = {
    "cameras": "Search cameras…",
    "lenses": "Search lenses…",
    "film_stocks": "Search film stocks…",
    "processes": "Search processes…",
    "scan_setups": "Search scan setups…",
    _PRESETS: "Search presets…",
}


_PRESET_ROW_WIDGETS: dict[str, tuple[str, ...]] = {
    "metadata.camera_id": ("camera", "lens", "film_stock", "format", "format_other"),
    "metadata.developer": ("process", "developer", "dilution", "push_pull", "dev_time", "dev_temp"),
    "metadata.scanning": ("scan_setup", "scanning"),
    "metadata.capture_roll": ("roll",),
    "metadata.exposure_override": ("exposure",),
}


def _push_pull_index(value: int) -> int:
    return PUSH_PULL_VALUES.index(value) if value in PUSH_PULL_VALUES else PUSH_PULL_VALUES.index(0)


class GearLibraryDialog(QDialog):
    library_changed = pyqtSignal()
    presets_changed = pyqtSignal()

    def __init__(self, library: GearLibrary | None = None, parent=None, current_config=None):
        super().__init__(parent)
        self._library = library or GearProfiles.load_library()
        self._current_config = current_config
        self._category = "cameras"
        self._selected_idx = -1
        self._list_items: list = []
        self._updating = False

        self.setWindowTitle("Library")
        self.resize(820, 560)
        self._init_ui()
        self._select_category("cameras")
        pin_dialog_default(self._close_btn, scope=self)

    def library(self) -> GearLibrary:
        return self._library

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Category list
        left = QWidget()
        left.setFixedWidth(140)
        left.setStyleSheet(dialog_pane_qss())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        cat_label = QLabel("LIBRARY")
        cat_label.setStyleSheet(pane_header_qss())
        left_layout.addWidget(cat_label)

        self.category_list = QListWidget()
        for key, label in _CATEGORIES:
            self.category_list.addItem(QListWidgetItem(label))
        self.category_list.setProperty("keys", [k for k, _ in _CATEGORIES])
        self.category_list.currentRowChanged.connect(self._on_category_changed)
        left_layout.addWidget(self.category_list)
        root.addWidget(left)

        # Item list
        mid = QWidget()
        mid.setFixedWidth(220)
        mid.setStyleSheet(dialog_pane_qss())
        mid_layout = QVBoxLayout(mid)
        mid_layout.setContentsMargins(8, 8, 8, 8)

        self.items_label = QLabel("ITEMS")
        self.items_label.setStyleSheet(pane_header_qss())
        mid_layout.addWidget(self.items_label)

        self.item_search = QLineEdit()
        self.item_search.setPlaceholderText("Search cameras…")
        self.item_search.textChanged.connect(self._on_item_search_changed)
        mid_layout.addWidget(self.item_search)

        self.item_list = QListWidget()
        self.item_list.currentRowChanged.connect(self._on_item_changed)
        mid_layout.addWidget(self.item_list)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton()
        self.add_btn.setIcon(qta.icon("fa5s.plus", color=THEME.text_primary))
        self.add_btn.setToolTip("Add item")
        self.add_btn.clicked.connect(self._add_item)
        self.dup_btn = QPushButton()
        self.dup_btn.setIcon(qta.icon("fa5s.copy", color=THEME.text_primary))
        self.dup_btn.setToolTip("Duplicate")
        self.dup_btn.clicked.connect(self._duplicate_item)
        self.edit_btn = QPushButton()
        self.edit_btn.setIcon(qta.icon("fa5s.pen", color=THEME.text_primary))
        self.edit_btn.setToolTip("Rename the preset, or change which fields it stores")
        self.edit_btn.clicked.connect(self._edit_preset)
        self.del_btn = QPushButton()
        self.del_btn.setIcon(qta.icon("fa5s.trash-alt", color=THEME.text_primary))
        self.del_btn.setToolTip("Delete")
        self.del_btn.clicked.connect(self._delete_item)
        for b in (self.add_btn, self.dup_btn, self.edit_btn, self.del_btn):
            b.setFixedWidth(36)
            btn_row.addWidget(b)
        btn_row.addStretch()
        mid_layout.addLayout(btn_row)

        root.addWidget(mid)

        # Form: a single layout, with rows shown and hidden per category, never removeRow.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 16, 16, 16)

        self.display_name_edit = QLineEdit()
        self.make_edit = QLineEdit()
        self.model_edit = QLineEdit()
        self.lens_model_edit = QLineEdit()
        self.focal_spin = QDoubleSpinBox()
        self.focal_spin.setRange(0, 2000)
        self.focal_spin.setSuffix(" mm")
        self.aperture_spin = QDoubleSpinBox()
        self.aperture_spin.setRange(0, 64)
        self.aperture_spin.setDecimals(1)
        self.aperture_spin.setPrefix("f/")
        self.manufacturer_edit = QLineEdit()
        self.stock_name_edit = QLineEdit()
        self.iso_spin = QSpinBox()
        self.iso_spin.setRange(1, 12800)
        self.format_combo = QComboBox()
        self.format_combo.addItems([e.value for e in FilmFormat])
        self.color_combo = QComboBox()
        self.color_combo.addItems([e.value for e in FilmColorType])
        self.developer_edit = QLineEdit()
        self.developer_edit.setPlaceholderText("e.g. D-76")
        self.push_pull_combo = QComboBox()
        self.push_pull_combo.addItems([PUSH_PULL_LABELS[v] for v in PUSH_PULL_VALUES])
        self.dilution_edit = QLineEdit()
        self.dilution_edit.setPlaceholderText("e.g. 1+50, stock")
        self.dev_time_edit = QLineEdit()
        self.dev_time_edit.setPlaceholderText(DEV_TIME_HINT)
        self.dev_temp_edit = QLineEdit()
        self.dev_temp_edit.setPlaceholderText("e.g. 20")
        self.scanning_edit = QLineEdit()
        self.scanning_edit.setPlaceholderText("e.g. DSLR copy-stand scan")
        self.notes_edit = QLineEdit()

        for w in (
            self.display_name_edit,
            self.make_edit,
            self.model_edit,
            self.lens_model_edit,
            self.manufacturer_edit,
            self.stock_name_edit,
            self.developer_edit,
            self.dilution_edit,
            self.dev_time_edit,
            self.dev_temp_edit,
            self.scanning_edit,
            self.notes_edit,
        ):
            w.textChanged.connect(self._on_form_changed)
        self.focal_spin.valueChanged.connect(self._on_form_changed)
        self.aperture_spin.valueChanged.connect(self._on_form_changed)
        self.iso_spin.valueChanged.connect(self._on_form_changed)
        self.format_combo.currentIndexChanged.connect(self._on_form_changed)
        self.color_combo.currentIndexChanged.connect(self._on_form_changed)
        self.push_pull_combo.currentIndexChanged.connect(self._on_form_changed)

        self.form_panel = QWidget()
        self.form_layout = QFormLayout(self.form_panel)
        self.form_layout.setSpacing(8)
        self._form_rows: dict[str, tuple[QLabel, QWidget]] = {}
        self._register_form_row("display_name", "Display name", self.display_name_edit)
        self._register_form_row("make", "Make", self.make_edit)
        self._register_form_row("model", "Model", self.model_edit)
        self._register_form_row("lens_model", "Lens model", self.lens_model_edit)
        self._register_form_row("focal", "Focal length", self.focal_spin)
        self._register_form_row("aperture", "Max aperture", self.aperture_spin)
        self._register_form_row("manufacturer", "Manufacturer", self.manufacturer_edit)
        self._register_form_row("stock_name", "Stock name", self.stock_name_edit)
        self._register_form_row("iso", "ISO", self.iso_spin)
        self._register_form_row("format", "Format", self.format_combo)
        self._register_form_row("color_type", "Color type", self.color_combo)
        self._register_form_row("developer", "Developer", self.developer_edit)
        self._register_form_row("dilution", "Dilution", self.dilution_edit)
        self._register_form_row("push_pull", "Push / Pull", self.push_pull_combo)
        self._register_form_row("dev_time", "Time", self.dev_time_edit)
        self._register_form_row("dev_temp", "Temperature (°C)", self.dev_temp_edit)
        self._register_form_row("scanning", "Scanning", self.scanning_edit)
        self._register_form_row("notes", "Notes", self.notes_edit)

        right_layout.addWidget(self.form_panel)

        self.preset_panel = QWidget()
        preset_layout = QVBoxLayout(self.preset_panel)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(8)
        self.preset_name_label = QLabel()
        self.preset_name_label.setStyleSheet(f"color: {THEME.text_primary}; font-weight: bold;")
        self.preset_form_layout = QFormLayout()
        self.preset_form_layout.setSpacing(8)
        self._preset_rows: dict[str, tuple[QLabel, QWidget]] = {}
        self._build_preset_form()
        self.preset_fields_layout = QFormLayout()
        self.preset_fields_layout.setSpacing(8)
        self.preset_empty_label = QLabel("This preset stores nothing.")
        self.preset_empty_label.setStyleSheet(f"color: {THEME.text_secondary};")
        preset_layout.addWidget(self.preset_name_label)
        preset_layout.addLayout(self.preset_form_layout)
        preset_layout.addLayout(self.preset_fields_layout)
        preset_layout.addWidget(self.preset_empty_label)
        notes_row = QFormLayout()
        notes_row.setSpacing(8)
        self.preset_notes_edit = QLineEdit()
        self.preset_notes_edit.setPlaceholderText("Notes for this preset")
        self.preset_notes_edit.textChanged.connect(self._on_preset_notes_changed)
        notes_row.addRow(field_label("Notes"), self.preset_notes_edit)
        preset_layout.addLayout(notes_row)
        preset_layout.addWidget(hint_label("The pen chooses which fields a preset stores; these edit their values."))
        self.preset_panel.setVisible(False)
        right_layout.addWidget(self.preset_panel)
        right_layout.addStretch()

        close_row = QHBoxLayout()
        close_row.addStretch()
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.accept)
        close_row.addWidget(self._close_btn)
        right_layout.addLayout(close_row)

        root.addWidget(right)

    def _build_preset_form(self) -> None:
        self.preset_camera_combo = SearchableGearCombo(placeholder="Search cameras…")
        self.preset_lens_combo = SearchableGearCombo(placeholder="Search lenses…")
        self.preset_film_combo = SearchableGearCombo(placeholder="Search film stocks…")
        self.preset_process_combo = SearchableGearCombo(placeholder="Search processes…")
        self.preset_scan_combo = SearchableGearCombo(placeholder="Search scan setups…")
        self.preset_format_combo = QComboBox()
        self.preset_format_combo.addItems(FORMAT_OPTIONS)
        self.preset_format_other_edit = QLineEdit()
        self.preset_format_other_edit.setPlaceholderText("e.g. 6×7")
        self.preset_developer_edit = QLineEdit()
        self.preset_developer_edit.setPlaceholderText("e.g. D-76")
        self.preset_dilution_edit = QLineEdit()
        self.preset_dilution_edit.setPlaceholderText("e.g. 1+50")
        self.preset_push_combo = QComboBox()
        self.preset_push_combo.addItems([PUSH_PULL_LABELS[v] for v in PUSH_PULL_VALUES])
        self.preset_time_edit = QLineEdit()
        self.preset_time_edit.setPlaceholderText(DEV_TIME_HINT)
        self.preset_temp_edit = QLineEdit()
        self.preset_temp_edit.setPlaceholderText("e.g. 20")
        self.preset_scanning_edit = QLineEdit()
        self.preset_scanning_edit.setPlaceholderText("e.g. DSLR copy-stand scan")
        self.preset_roll_edit = QLineEdit()
        self.preset_roll_edit.setPlaceholderText("e.g. Roll001")
        self.preset_exposure_edit = QLineEdit()
        self.preset_exposure_edit.setPlaceholderText("e.g. 1/125s f/2.8 ISO 400")

        for key, label, widget in (
            ("camera", "Camera", self.preset_camera_combo),
            ("lens", "Lens", self.preset_lens_combo),
            ("film_stock", "Film stock", self.preset_film_combo),
            ("format", "Format", self.preset_format_combo),
            ("format_other", "Other format", self.preset_format_other_edit),
            ("process", "Saved process", self.preset_process_combo),
            ("developer", "Developer", self.preset_developer_edit),
            ("dilution", "Dilution", self.preset_dilution_edit),
            ("push_pull", "Push / Pull", self.preset_push_combo),
            ("dev_time", "Time", self.preset_time_edit),
            ("dev_temp", "Temperature (°C)", self.preset_temp_edit),
            ("scan_setup", "Saved setup", self.preset_scan_combo),
            ("scanning", "Scanning", self.preset_scanning_edit),
            ("roll", "Roll", self.preset_roll_edit),
            ("exposure", "Exposure", self.preset_exposure_edit),
        ):
            row_label = field_label(label)
            self.preset_form_layout.addRow(row_label, widget)
            self._preset_rows[key] = (row_label, widget)

        # A library pick re-resolves everything read from it; a typed value unlinks the pick,
        # exactly as the Metadata panel behaves.
        for combo, handler in (
            (self.preset_camera_combo, self._on_preset_gear_changed),
            (self.preset_lens_combo, self._on_preset_gear_changed),
            (self.preset_film_combo, self._on_preset_gear_changed),
            (self.preset_process_combo, self._on_preset_process_picked),
            (self.preset_scan_combo, self._on_preset_scan_picked),
        ):
            combo.selection_changed.connect(handler)
        for edit in (
            self.preset_format_other_edit,
            self.preset_developer_edit,
            self.preset_dilution_edit,
            self.preset_time_edit,
            self.preset_temp_edit,
            self.preset_scanning_edit,
            self.preset_roll_edit,
            self.preset_exposure_edit,
        ):
            edit.textChanged.connect(self._on_preset_value_changed)
        self.preset_format_combo.currentIndexChanged.connect(self._on_preset_value_changed)
        self.preset_push_combo.currentIndexChanged.connect(self._on_preset_value_changed)

    def _register_form_row(self, key: str, label_text: str, widget: QWidget) -> None:
        label = field_label(label_text)
        self.form_layout.addRow(label, widget)
        self._form_rows[key] = (label, widget)

    def _show_form_for_category(self, category: str) -> None:
        visible = _CATEGORY_FIELDS[category]
        for key, (label, widget) in self._form_rows.items():
            show = key in visible
            label.setVisible(show)
            widget.setVisible(show)

    def _current_items(self) -> list:
        if self._category == "cameras":
            return self._library.cameras
        if self._category == "lenses":
            return self._library.lenses
        if self._category == "film_stocks":
            return self._library.film_stocks
        if self._category == "processes":
            return self._library.processes
        if self._category == "scan_setups":
            return self._library.scan_setups
        return sorted(MetadataPresets.list_presets())

    def _set_current_items(self, items: list) -> None:
        if self._category == "cameras":
            self._library.cameras = items
        elif self._category == "lenses":
            self._library.lenses = items
        elif self._category == "film_stocks":
            self._library.film_stocks = items
        elif self._category == "processes":
            self._library.processes = items
        elif self._category == "scan_setups":
            self._library.scan_setups = items

    def _item_label(self, item) -> str:
        """A preset is its own name; a library record has a resolved one."""
        if isinstance(item, str):
            return item
        return item.resolved_display_name

    def _item_id(self, item) -> str:
        return item if isinstance(item, str) else item.id

    def _select_category(self, key: str) -> None:
        for i, (k, _) in enumerate(_CATEGORIES):
            if k == key:
                self.category_list.setCurrentRow(i)
                break

    def _on_category_changed(self, row: int) -> None:
        if row < 0:
            return
        self._category = _CATEGORIES[row][0]
        self.item_search.blockSignals(True)
        self.item_search.clear()
        self.item_search.setPlaceholderText(_CATEGORY_SEARCH_PLACEHOLDER.get(self._category, "Search…"))
        self.item_search.blockSignals(False)
        self._rebuild_item_list()
        self._show_form_for_category(self._category)
        is_presets = self._category == _PRESETS
        self.form_panel.setVisible(not is_presets)
        self.preset_panel.setVisible(is_presets)
        self.edit_btn.setVisible(is_presets)
        self.add_btn.setEnabled(not is_presets or self._current_config is not None)
        self.add_btn.setToolTip("Store the current frame's metadata as a preset" if is_presets else "Add item")

    def _on_item_search_changed(self, _text: str) -> None:
        self._rebuild_item_list()

    def _rebuild_item_list(self, *, select_id: str | None = None) -> None:
        all_items = self._current_items()
        selected_id = select_id
        if selected_id is None and 0 <= self._selected_idx < len(all_items):
            selected_id = self._item_id(all_items[self._selected_idx])

        query = self.item_search.text().strip()
        visible = [item for item in all_items if self._matches(item, query)]

        self._list_items = visible
        self.item_list.blockSignals(True)
        self.item_list.clear()
        for item in visible:
            self.item_list.addItem(QListWidgetItem(self._item_label(item)))

        row = -1
        if visible:
            if selected_id:
                row = next((i for i, item in enumerate(visible) if self._item_id(item) == selected_id), -1)
            if row < 0 and select_id is not None:
                row = next((i for i, item in enumerate(visible) if self._item_id(item) == select_id), 0)
            elif row < 0 and not query:
                row = 0
        self.item_list.setCurrentRow(row)
        self.item_list.blockSignals(False)

        if not visible and not query:
            self._selected_idx = -1
            self._clear_form()
        elif row >= 0:
            self._on_item_changed(row)

    def _matches(self, item, query: str) -> bool:
        if isinstance(item, str):
            return query.strip().casefold() in item.casefold()
        return matches_gear_filter(item, query)

    def _on_item_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._list_items):
            self._selected_idx = -1
            self._set_form_editable(True)
            self._clear_form()
            return
        item = self._list_items[row]
        item_id = self._item_id(item)
        all_items = self._current_items()
        self._selected_idx = next(i for i, candidate in enumerate(all_items) if self._item_id(candidate) == item_id)
        self._set_form_editable(isinstance(item, str) or not item.is_bundled)
        self._populate_form(item)

    def _parsed_or_kept(self, edit: QLineEdit, parse, current):
        """Every keystroke saves, so a half-typed "9:" must not erase the stored value.
        Blank is an explicit clear; unreadable text keeps what is stored and marks the field."""
        text = edit.text().strip()
        if not text:
            self._mark_invalid(edit, False)
            return None
        value = parse(text)
        self._mark_invalid(edit, value is None)
        return current if value is None else value

    def _mark_invalid(self, edit: QLineEdit, invalid: bool) -> None:
        edit.setStyleSheet(f"border: 1px solid {THEME.error};" if invalid else "")

    def _set_form_editable(self, enabled: bool) -> None:
        for _label, widget in self._form_rows.values():
            widget.setEnabled(enabled)
        self.del_btn.setEnabled(enabled)

    def _populate_form(self, item) -> None:
        if isinstance(item, str):
            self._populate_preset(item)
            return
        self._updating = True
        try:
            if isinstance(item, Camera):
                self.display_name_edit.setText(item.display_name)
                self.make_edit.setText(item.make)
                self.model_edit.setText(item.model)
                self.notes_edit.setText(item.notes)
            elif isinstance(item, Lens):
                self.display_name_edit.setText(item.display_name)
                self.make_edit.setText(item.make)
                self.lens_model_edit.setText(item.lens_model)
                self.focal_spin.setValue(item.focal_length_mm or 0)
                self.aperture_spin.setValue(item.max_aperture or 0)
                self.notes_edit.setText(item.notes)
            elif isinstance(item, DevelopmentProcess):
                self.display_name_edit.setText(item.display_name)
                self.developer_edit.setText(item.developer)
                self.dilution_edit.setText(item.dilution)
                self.push_pull_combo.setCurrentIndex(_push_pull_index(item.push_pull))
                self.dev_time_edit.setText(format_dev_time(item.time_seconds))
                self.dev_temp_edit.setText(format_temperature(item.temperature_c))
                self._mark_invalid(self.dev_time_edit, False)
                self._mark_invalid(self.dev_temp_edit, False)
                self.notes_edit.setText(item.notes)
            elif isinstance(item, ScanSetup):
                self.display_name_edit.setText(item.display_name)
                self.scanning_edit.setText(item.scanning)
                self.notes_edit.setText(item.notes)
            elif isinstance(item, FilmStock):
                self.display_name_edit.setText(item.display_name)
                self.manufacturer_edit.setText(item.manufacturer)
                self.stock_name_edit.setText(item.stock_name)
                self.iso_spin.setValue(item.iso)
                idx = self.format_combo.findText(item.format.value)
                if idx >= 0:
                    self.format_combo.setCurrentIndex(idx)
                idx = self.color_combo.findText(item.color_type.value)
                if idx >= 0:
                    self.color_combo.setCurrentIndex(idx)
                self.notes_edit.setText(item.notes)
        finally:
            self._updating = False

    def _populate_preset(self, name: str) -> None:
        data = MetadataPresets.load_preset(name) or {}
        stored = {r.id for r in rows_for_keys(data, "metadata")}
        editable = {w for row_id in stored for w in _PRESET_ROW_WIDGETS.get(row_id, ())}
        meta = preset_config(data).metadata

        self._updating = True
        try:
            self.preset_name_label.setText(name)
            self.preset_notes_edit.setText(preset_notes(data))
            self.preset_camera_combo.set_gear_items(self._library.cameras, meta.camera_id, lambda c: c.resolved_display_name)
            self.preset_lens_combo.set_gear_items(self._library.lenses, meta.lens_id, lambda x: x.resolved_display_name)
            self.preset_film_combo.set_gear_items(self._library.film_stocks, meta.film_stock_id, lambda f: f.resolved_display_name)
            self.preset_process_combo.set_gear_items(self._library.processes, meta.process_id, lambda p: p.resolved_display_name)
            self.preset_scan_combo.set_gear_items(self._library.scan_setups, meta.scanning_id, lambda x: x.resolved_display_name)
            self.preset_format_combo.setCurrentText(format_label(meta.format))
            self.preset_format_other_edit.setText(meta.format_other)
            self.preset_developer_edit.setText(meta.developer)
            self.preset_dilution_edit.setText(meta.process_dilution)
            self.preset_push_combo.setCurrentIndex(_push_pull_index(meta.push_pull))
            self.preset_time_edit.setText(format_dev_time(meta.process_time_seconds))
            self.preset_temp_edit.setText(format_temperature(meta.process_temperature_c))
            self.preset_scanning_edit.setText(meta.scanning)
            self.preset_roll_edit.setText(meta.capture_roll)
            self.preset_exposure_edit.setText(meta.exposure_override)
            for key, (row_label, widget) in self._preset_rows.items():
                show = key in editable and (key != "format_other" or self.preset_format_combo.currentText() == "Other")
                row_label.setVisible(show)
                widget.setVisible(show)
            self._mark_invalid(self.preset_time_edit, False)
            self._mark_invalid(self.preset_temp_edit, False)
        finally:
            self._updating = False

        # Rows with no editor: per-frame decisions, shown as they are stored.
        while self.preset_fields_layout.rowCount():
            self.preset_fields_layout.removeRow(0)
        read_only = [(label, value) for label, value in preset_values(data, "metadata") if not self._is_editable_row(label, stored)]
        for label, value in read_only:
            value_label = QLabel(value)
            value_label.setWordWrap(True)
            value_label.setStyleSheet(f"color: {THEME.text_secondary};")
            self.preset_fields_layout.addRow(field_label(label), value_label)
        self.preset_empty_label.setVisible(not stored)

    def _is_editable_row(self, label: str, stored: set[str]) -> bool:
        for row in rows_by_id().values():
            if row.label == label and row.section == "metadata":
                return row.id in _PRESET_ROW_WIDGETS and row.id in stored
        return False

    def _preset_field_update(self, data: dict, meta) -> dict:
        """The stored rows' fields, re-read from one edited MetadataConfig."""
        out = dict(data)
        for row in rows_for_keys(data, "metadata"):
            for f in row.fields:
                out[f] = getattr(meta, f)
        return out

    def _write_preset(self, meta, refresh: bool = True) -> None:
        """refresh redraws the form from what was stored, which a library pick needs (it
        resolves other fields) and typing must not have — it would rewrite the text mid-edit."""
        name = self._selected_preset()
        data = MetadataPresets.load_preset(name) if name else None
        if data is None:
            return
        fields = self._preset_field_update(preset_fields(data), meta)
        MetadataPresets.save_preset(name, with_preset_notes(fields, preset_notes(data)))
        self.presets_changed.emit()
        if refresh:
            self._populate_preset(name)

    def _preset_meta(self):
        name = self._selected_preset()
        data = MetadataPresets.load_preset(name) if name else None
        return preset_config(data).metadata if data else None

    def _on_preset_gear_changed(self, *_args) -> None:
        meta = None if self._updating else self._preset_meta()
        if meta is None:
            return
        self._write_preset(
            metadata_from_gear(
                meta,
                self._library,
                camera_id=self.preset_camera_combo.selected_id(),
                lens_id=self.preset_lens_combo.selected_id(),
                film_stock_id=self.preset_film_combo.selected_id(),
            )
        )

    def _on_preset_process_picked(self, *_args) -> None:
        meta = None if self._updating else self._preset_meta()
        if meta is not None:
            self._write_preset(metadata_from_process(meta, self._library, self.preset_process_combo.selected_id()))

    def _on_preset_scan_picked(self, *_args) -> None:
        meta = None if self._updating else self._preset_meta()
        if meta is not None:
            self._write_preset(metadata_from_scan_setup(meta, self._library, self.preset_scan_combo.selected_id()))

    def _on_preset_value_changed(self, *_args) -> None:
        meta = None if self._updating else self._preset_meta()
        if meta is None:
            return
        fmt = format_value(self.preset_format_combo.currentText())
        if "format_other" in self._preset_rows and self._preset_rows["format"][1].isVisibleTo(self.preset_panel):
            other_label, other_widget = self._preset_rows["format_other"]
            other_label.setVisible(fmt == "Other")
            other_widget.setVisible(fmt == "Other")
        developer = self.preset_developer_edit.text().strip()
        dilution = self.preset_dilution_edit.text().strip()
        push = PUSH_PULL_VALUES[self.preset_push_combo.currentIndex()]
        time_seconds = self._parsed_or_kept(self.preset_time_edit, parse_dev_time, meta.process_time_seconds)
        temperature = self._parsed_or_kept(self.preset_temp_edit, parse_temperature, meta.process_temperature_c)
        scanning = self.preset_scanning_edit.text().strip()
        # A typed value unlinks the pick it came from, as it does on the panel.
        process_id = meta.process_id
        if (developer, dilution, push, time_seconds, temperature) != (
            meta.developer,
            meta.process_dilution,
            meta.push_pull,
            meta.process_time_seconds,
            meta.process_temperature_c,
        ):
            process_id = ""
        self._write_preset(
            replace(
                meta,
                format=fmt,
                format_other=self.preset_format_other_edit.text().strip() if fmt == "Other" else "",
                developer=developer,
                process_dilution=dilution,
                push_pull=push,
                process_time_seconds=time_seconds,
                process_temperature_c=temperature,
                process_id=process_id,
                scanning=scanning,
                scanning_id="" if scanning != meta.scanning else meta.scanning_id,
                capture_roll=self.preset_roll_edit.text().strip(),
                exposure_override=self.preset_exposure_edit.text().strip(),
            ),
            refresh=False,
        )

    def _on_preset_notes_changed(self, text: str) -> None:
        name = self._selected_preset()
        data = MetadataPresets.load_preset(name) if name else None
        if self._updating or data is None:
            return
        MetadataPresets.save_preset(name, with_preset_notes(data, text))
        self.presets_changed.emit()

    def _clear_form(self) -> None:
        self.preset_name_label.setText("No preset selected")
        self._updating = True
        try:
            self.preset_notes_edit.clear()
        finally:
            self._updating = False
        while self.preset_fields_layout.rowCount():
            self.preset_fields_layout.removeRow(0)
        self.preset_empty_label.setVisible(False)
        self._updating = True
        try:
            for w in (
                self.display_name_edit,
                self.make_edit,
                self.model_edit,
                self.lens_model_edit,
                self.manufacturer_edit,
                self.stock_name_edit,
                self.developer_edit,
                self.dilution_edit,
                self.dev_time_edit,
                self.dev_temp_edit,
                self.scanning_edit,
                self.notes_edit,
            ):
                w.clear()
            self.focal_spin.setValue(0)
            self.aperture_spin.setValue(0)
            self.iso_spin.setValue(100)
        finally:
            self._updating = False

    def _on_form_changed(self, *_args) -> None:
        # Presets have no form: their pane is a summary, and the picker writes the file.
        if self._updating or self._selected_idx < 0 or self._category == _PRESETS:
            return
        items = list(self._current_items())
        item = items[self._selected_idx]

        if isinstance(item, Camera):
            item.display_name = self.display_name_edit.text().strip()
            item.make = self.make_edit.text().strip()
            item.model = self.model_edit.text().strip()
            item.notes = self.notes_edit.text().strip()
        elif isinstance(item, Lens):
            item.display_name = self.display_name_edit.text().strip()
            item.make = self.make_edit.text().strip()
            item.lens_model = self.lens_model_edit.text().strip()
            item.focal_length_mm = self.focal_spin.value() or None
            item.max_aperture = self.aperture_spin.value() or None
            item.notes = self.notes_edit.text().strip()
        elif isinstance(item, DevelopmentProcess):
            item.display_name = self.display_name_edit.text().strip()
            item.developer = self.developer_edit.text().strip()
            item.dilution = self.dilution_edit.text().strip()
            item.push_pull = PUSH_PULL_VALUES[self.push_pull_combo.currentIndex()]
            item.time_seconds = self._parsed_or_kept(self.dev_time_edit, parse_dev_time, item.time_seconds)
            item.temperature_c = self._parsed_or_kept(self.dev_temp_edit, parse_temperature, item.temperature_c)
            item.notes = self.notes_edit.text().strip()
        elif isinstance(item, ScanSetup):
            item.display_name = self.display_name_edit.text().strip()
            item.scanning = self.scanning_edit.text().strip()
            item.notes = self.notes_edit.text().strip()
        elif isinstance(item, FilmStock):
            item.display_name = self.display_name_edit.text().strip()
            item.manufacturer = self.manufacturer_edit.text().strip()
            item.stock_name = self.stock_name_edit.text().strip()
            item.iso = self.iso_spin.value()
            item.format = FilmFormat(self.format_combo.currentText())
            item.color_type = FilmColorType(self.color_combo.currentText())
            item.notes = self.notes_edit.text().strip()

        items[self._selected_idx] = item
        self._set_current_items(items)
        list_row = next((i for i, visible in enumerate(self._list_items) if self._item_id(visible) == item.id), -1)
        list_entry = self.item_list.item(list_row) if list_row >= 0 else None
        if list_entry is not None:
            list_entry.setText(self._item_label(item))
        GearProfiles.save_library(self._library)
        self.library_changed.emit()

    def _selected_preset(self) -> str:
        items = self._current_items()
        if self._category != _PRESETS or not (0 <= self._selected_idx < len(items)):
            return ""
        return str(items[self._selected_idx])

    def _name_is_usable(self, name: str, replacing: str = "") -> bool:
        """A preset name is a filename, and a rename onto another preset would replace it."""
        if not is_valid_preset_name(name):
            QMessageBox.warning(
                self,
                "Preset Name",
                'A preset name cannot contain / \\ : * ? " < > | or start or end with a dot.',
            )
            return False
        if name.casefold() == replacing.casefold() or not MetadataPresets.exists(name):
            return True
        return (
            QMessageBox.question(self, "Replace Preset", f"A preset named '{name}' already exists. Replace it?")
            == QMessageBox.StandardButton.Yes
        )

    def _new_preset_from_frame(self) -> None:
        """A preset is the current frame's metadata, minus the fields left unticked."""
        if self._current_config is None:
            return
        dlg = GranularSettingsDialog(self, self._current_config, "current metadata", ask_name=True, exclude_sections=NON_METADATA_SECTIONS)
        dlg.setWindowTitle("New Metadata Preset")
        # As when editing: which fields to store is the choice, so every row is on offer.
        dlg.show_unchanged_settings()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name = dlg.name().strip()
        if not self._name_is_usable(name):
            return
        MetadataPresets.save_preset(name, selected_flat_dict(self._current_config, dlg.selected()))
        self._rebuild_item_list(select_id=name)
        self.presets_changed.emit()

    def _edit_preset(self) -> None:
        name = self._selected_preset()
        data = MetadataPresets.load_preset(name) if name else None
        if not data:
            return
        cfg = preset_config(data)
        dlg = GranularSettingsDialog(self, cfg, name, ask_name=True, exclude_sections=NON_METADATA_SECTIONS)
        dlg.setWindowTitle("Edit Metadata Preset")
        dlg.set_name(name)
        # Editing is about which fields the preset holds, so show every row, default-valued
        # ones included, rather than making the user reveal them to add one.
        dlg.show_unchanged_settings()
        # What the preset stores, not what differs from default: a row deliberately holding
        # a default value would otherwise arrive unticked and be dropped on save.
        dlg.set_checked_rows(r.id for r in rows_for_keys(data, "metadata"))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_name = dlg.name().strip()
        if not self._name_is_usable(new_name, replacing=name):
            return
        # Fields first, under the name that exists; the rename is then one atomic move.
        MetadataPresets.save_preset(name, with_preset_notes(selected_flat_dict(cfg, dlg.selected()), preset_notes(data)))
        if new_name != name:
            MetadataPresets.rename_preset(name, new_name)
        self._rebuild_item_list(select_id=new_name)
        self.presets_changed.emit()

    def _add_item(self) -> None:
        if self._category == _PRESETS:
            self._new_preset_from_frame()
            return
        if self._category == "cameras":
            item = Camera(make="New", model="Camera")
        elif self._category == "lenses":
            item = Lens(lens_model="New lens")
        elif self._category == "processes":
            item = DevelopmentProcess(display_name="New process")
        elif self._category == "scan_setups":
            item = ScanSetup(display_name="New scan setup")
        else:
            item = FilmStock(stock_name="New stock")
        items = list(self._current_items())
        items.append(item)
        self._set_current_items(items)
        GearProfiles.save_library(self._library)
        self._rebuild_item_list(select_id=item.id)
        self.library_changed.emit()

    def _duplicate_item(self) -> None:
        if self._selected_idx < 0:
            return
        if self._category == _PRESETS:
            name = self._selected_preset()
            data = MetadataPresets.load_preset(name) if name else None
            if data is None:
                return
            existing = set(MetadataPresets.list_presets())
            copy_name = next(
                f"{name} copy{'' if i == 1 else f' {i}'}"
                for i in range(1, 100)
                if f"{name} copy{'' if i == 1 else f' {i}'}" not in existing
            )
            MetadataPresets.save_preset(copy_name, data)
            self._rebuild_item_list(select_id=copy_name)
            self.presets_changed.emit()
            return
        import copy

        items = list(self._current_items())
        dup = copy.deepcopy(items[self._selected_idx])
        from negpy.features.metadata.gear_models import _new_id

        dup.id = _new_id()
        dup.is_bundled = False
        items.append(dup)
        self._set_current_items(items)
        GearProfiles.save_library(self._library)
        self._rebuild_item_list(select_id=dup.id)
        self.library_changed.emit()

    def _delete_item(self) -> None:
        if self._selected_idx < 0:
            return
        items = self._current_items()
        kind = {"metadata_presets": "Preset", "film_stocks": "Film Stock", "scan_setups": "Scan Setup"}.get(
            self._category, dict(_CATEGORIES)[self._category].rstrip("s")
        )
        if not confirm_delete_named(self, kind, self._item_label(items[self._selected_idx])):
            return
        if self._category == _PRESETS:
            name = self._selected_preset()
            if name:
                MetadataPresets.delete_preset(name)
                self._rebuild_item_list()
                self.presets_changed.emit()
            return
        items = list(self._current_items())
        del items[self._selected_idx]
        self._set_current_items(items)
        GearProfiles.save_library(self._library)
        self._rebuild_item_list()
        self.library_changed.emit()
