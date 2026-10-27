"""Panel for managing the analog gear library -- a persistent tab, not a dialog: the
library is reference data you build up over time, reached exactly as often as Export
or Metadata, not something opened, changed once and dismissed.

Two sections: My Gear (physical gear) and Presets (saved metadata field sets). They share
one GearLibrary but otherwise diverge -- My Gear has categories and a bundled/personal
Catalog toggle that a preset has no equivalent for -- so each gets its own widget."""

from __future__ import annotations

from negpy.desktop.view.styles.templates import add_choice_items

from negpy.kernel.system.i18n import tr

from dataclasses import fields, replace
from typing import Callable, Optional

import qtawesome as qta
from PyQt6.QtCore import QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
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
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.styles.templates import (
    field_label,
    header_row,
    hint_label,
    icon_button,
    labeled_action,
    section_subheader,
    tool_toggle,
    wrap_tooltip,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection, make_section
from negpy.desktop.view.widgets.flat_rows import ICON, RowList, more_rect, paint_fill, paint_more, two_line_rects
from negpy.desktop.view.widgets.gear_catalog_dialog import GearCatalogDialog, resolve_other_gear_pick
from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.domain.models import WorkspaceConfig
from negpy.features.metadata.gear_logic import (
    CATEGORY_SEARCH_PLACEHOLDER,
    CATEGORY_SINGULAR,
    OTHER_ID,
    blank_gear_item,
    clone_into_personal,
    matches_gear_filter,
    metadata_from_gear,
    metadata_from_process,
    metadata_from_scan_setup,
    own_gear_entries,
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
    ("cameras", tr("Cameras")),
    ("lenses", tr("Lenses")),
    ("film_stocks", tr("Film Stocks")),
    ("processes", tr("Processes")),
    ("scan_setups", tr("Scan Setups")),
]

_CATEGORY_ICONS: dict[str, str] = {
    "cameras": "fa5s.camera",
    "lenses": "mdi6.camera-iris",
    "film_stocks": "fa5s.film",
    "processes": "fa5s.flask",
    "scan_setups": "mdi6.scanner",
}

_CATEGORY_FIELDS: dict[str, frozenset[str]] = {
    "cameras": frozenset({"display_name", "make", "model", "notes"}),
    "lenses": frozenset({"display_name", "make", "lens_model", "focal", "aperture", "notes"}),
    "film_stocks": frozenset({"display_name", "manufacturer", "stock_name", "iso", "format", "color_type", "notes"}),
    "processes": frozenset({"display_name", "developer", "dilution", "push_pull", "dev_time", "dev_temp", "notes"}),
    "scan_setups": frozenset({"display_name", "scanning", "notes"}),
}

_CATEGORY_PLURAL_NOUN = {
    "cameras": tr("cameras"),
    "lenses": tr("lenses"),
    "film_stocks": tr("film stocks"),
    "processes": tr("processes"),
    "scan_setups": tr("scan setups"),
}

_ADD_TOOLTIPS = {
    "cameras": tr("Add a camera you own: pick one from the built-in list, or enter your own"),
    "lenses": tr("Add a lens you own: pick one from the built-in list, or enter your own"),
    "film_stocks": tr("Add a film stock you use: pick one from the built-in list, or enter your own"),
    "processes": tr("Add a development process: pick one from the built-in list, or enter your own"),
    "scan_setups": tr("Add a scan setup: pick one from the built-in list, or enter your own"),
}

_DELETE_TOOLTIPS = {
    "cameras": tr("Remove this camera from your gear"),
    "lenses": tr("Remove this lens from your gear"),
    "film_stocks": tr("Remove this film stock from your gear"),
    "processes": tr("Remove this process from your gear"),
    "scan_setups": tr("Remove this scan setup from your gear"),
}
_DELETE_BUNDLED_TOOLTIP = tr("Built-in gear cannot be removed, only duplicated")


_PRESET_ROW_WIDGETS: dict[str, tuple[str, ...]] = {
    "metadata.camera_id": ("camera", "lens", "film_stock", "format", "format_other"),
    "metadata.developer": ("process", "developer", "dilution", "push_pull", "dev_time", "dev_temp"),
    "metadata.scanning": ("scan_setup", "scanning"),
    "metadata.capture_roll": ("roll",),
    "metadata.exposure_override": ("exposure",),
}


def _push_pull_index(value: int) -> int:
    return PUSH_PULL_VALUES.index(value) if value in PUSH_PULL_VALUES else PUSH_PULL_VALUES.index(0)


# Gear list panes: tall enough for a handful of rows, short enough to leave the
# detail form on screen beside them.
_LIST_MAX_HEIGHT = 160

_PRESET_ROW = 46
_DETAIL_ROLE = Qt.ItemDataRole.UserRole
_APPLY_TOOLTIP = tr("Write the selected preset's fields onto this frame")


class _PresetDelegate(QStyledItemDelegate):
    """A preset: its name over the fields it stores, the selected one on the warm chip, ⋮ on hover."""

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return QSize(option.rect.width(), _PRESET_ROW)

    def paint(self, p: QPainter, option, index) -> None:
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(0, 1, 0, -1)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        paint_fill(p, rect, current=selected, hover=hover)

        font = QFont(option.font)
        font.setPixelSize(THEME.font_size_base)
        small = QFont(font)
        small.setPixelSize(THEME.font_size_small)
        x = rect.left() + THEME.space_lg
        index.data(Qt.ItemDataRole.DecorationRole).paint(p, QRect(x, rect.center().y() - ICON // 2, ICON, ICON))
        x += ICON + THEME.space_lg
        right = more_rect(option.rect).left() - THEME.space_md
        name_font = QFont(font)
        name_font.setWeight(QFont.Weight(THEME.weight_semibold if selected else THEME.weight_regular))
        name_metrics, small_metrics = QFontMetrics(name_font), QFontMetrics(small)
        name_line, detail_line = two_line_rects(rect, x, right, name_metrics, small_metrics)
        p.setFont(name_font)
        p.setPen(QColor(THEME.text_primary))
        name = index.data(Qt.ItemDataRole.DisplayRole) or ""
        p.drawText(name_line, Qt.AlignmentFlag.AlignVCenter, name_metrics.elidedText(name, Qt.TextElideMode.ElideRight, name_line.width()))
        p.setFont(small)
        p.setPen(QColor(THEME.text_hint))
        detail = index.data(_DETAIL_ROLE) or ""
        p.drawText(
            detail_line, Qt.AlignmentFlag.AlignVCenter, small_metrics.elidedText(detail, Qt.TextElideMode.ElideRight, detail_line.width())
        )
        if hover:
            paint_more(p, option.rect)
        p.restore()


class GearItemsPanel(QWidget):
    """Cameras, lenses, film stocks, processes and scan setups: one searchable,
    user-extendable list per category, shared by Roll Settings, Metadata and every
    other picker in the app that offers gear. The item list defaults to what the user
    has added; the Catalog toggle brings the shipped reference models back into view."""

    library_changed = pyqtSignal()

    def __init__(self, library: GearLibrary, parent=None, *, repo=None):
        super().__init__(parent)
        self._repo = repo
        self._library = library
        self._category = "cameras"
        self._selected_idx = -1
        self._list_items: list = []
        self._updating = False
        # Off by default: the list shows personal gear only until this is switched on.
        self._show_bundled = False

        self._init_ui()
        self._select_category("cameras")

    def library(self) -> GearLibrary:
        return self._library

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(THEME.space_lg)

        root.addWidget(hint_label(tr("Cameras, lenses, film stocks and processes you own. Catalog shows the full shipped list.")))

        root.addWidget(field_label(tr("Category")))
        self.category_list = StepCombo()
        self.category_list.setToolTip(wrap_tooltip(tr("Which kind of gear this pane lists")))
        for key, label in _CATEGORIES:
            self.category_list.addItem(qta.icon(_CATEGORY_ICONS[key], color=THEME.text_primary), label, key)
        root.addWidget(self.category_list)

        self.item_search = QLineEdit()
        self.item_search.setPlaceholderText(tr("Search cameras…"))
        self.item_search.setToolTip(wrap_tooltip(tr("Filter the list below by name")))
        self.item_search.textChanged.connect(self._on_item_search_changed)
        root.addWidget(self.item_search)

        self.item_list = QListWidget()
        self.item_list.setMaximumHeight(_LIST_MAX_HEIGHT)
        self.item_list.currentRowChanged.connect(self._on_item_changed)
        root.addWidget(self.item_list)

        self.empty_hint = hint_label()
        self.empty_hint.setVisible(False)
        root.addWidget(self.empty_hint)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(THEME.space_sm)
        self.add_btn = icon_button("fa5s.plus", tr("Add item"))
        self.add_btn.clicked.connect(self._add_item)
        self.dup_btn = icon_button("fa5s.copy", tr("Duplicate"))
        self.dup_btn.clicked.connect(self._duplicate_item)
        self.del_btn = icon_button("fa5s.trash-alt", tr("Delete"))
        self.del_btn.clicked.connect(self._delete_item)
        for b in (self.add_btn, self.dup_btn, self.del_btn):
            btn_row.addWidget(b)
        btn_row.addStretch()
        self.show_catalog_btn = tool_toggle("fa5s.list-ul", tr("Catalog"), tr("Show the built-in catalog alongside your own gear"))
        self.show_catalog_btn.toggled.connect(self._on_show_catalog_toggled)
        btn_row.addWidget(self.show_catalog_btn)
        root.addLayout(btn_row)

        # Form: a single layout, with rows shown and hidden per category, never removed.
        root.addWidget(section_subheader(tr("DETAILS")))

        self.display_name_edit = QLineEdit()
        self.make_edit = QLineEdit()
        self.model_edit = QLineEdit()
        self.lens_model_edit = QLineEdit()
        self.focal_spin = QDoubleSpinBox()
        self.focal_spin.setRange(0, 2000)
        self.focal_spin.setSuffix(" mm")
        self.focal_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.aperture_spin = QDoubleSpinBox()
        self.aperture_spin.setRange(0, 64)
        self.aperture_spin.setDecimals(1)
        self.aperture_spin.setPrefix("f/")
        self.aperture_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.manufacturer_edit = QLineEdit()
        self.stock_name_edit = QLineEdit()
        self.iso_spin = QSpinBox()
        self.iso_spin.setRange(1, 12800)
        self.iso_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.format_combo = StepCombo()
        add_choice_items(self.format_combo, [e.value for e in FilmFormat])
        self.color_combo = StepCombo()
        add_choice_items(self.color_combo, [e.value for e in FilmColorType])
        self.developer_edit = QLineEdit()
        self.developer_edit.setPlaceholderText(tr("e.g. D-76"))
        self.push_pull_combo = StepCombo()
        add_choice_items(self.push_pull_combo, [PUSH_PULL_LABELS[v] for v in PUSH_PULL_VALUES])
        self.dilution_edit = QLineEdit()
        self.dilution_edit.setPlaceholderText(tr("e.g. 1+50, stock"))
        self.dev_time_edit = QLineEdit()
        self.dev_time_edit.setPlaceholderText(DEV_TIME_HINT)
        self.dev_temp_edit = QLineEdit()
        self.dev_temp_edit.setPlaceholderText(tr("e.g. 20"))
        self.scanning_edit = QLineEdit()
        self.scanning_edit.setPlaceholderText(tr("e.g. DSLR copy-stand scan"))
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
        self.form_layout = QVBoxLayout(self.form_panel)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.form_layout.setSpacing(THEME.space_md)
        self._form_rows: dict[str, tuple[QLabel, QWidget]] = {}
        row = self._register_form_row
        row("display_name", tr("Display name"), self.display_name_edit, tooltip=tr("The name this item shows in every gear list"))
        row("make", tr("Make"), self.make_edit, tooltip=tr("The maker of the camera or lens, written to the photo's metadata"))
        row("model", tr("Model"), self.model_edit, tooltip=tr("The camera model, written to the photo's metadata"))
        row("lens_model", tr("Lens model"), self.lens_model_edit, tooltip=tr("The lens model, written to the photo's metadata"))
        row("focal", tr("Focal length"), self.focal_spin, tooltip=tr("The focal length of the lens"))
        row("aperture", tr("Max aperture"), self.aperture_spin, tooltip=tr("The widest aperture of the lens"))
        row("manufacturer", tr("Manufacturer"), self.manufacturer_edit, tooltip=tr("The company that makes the film"))
        row("stock_name", tr("Stock name"), self.stock_name_edit, tooltip=tr("The name of the film, for example Portra 400"))
        row("iso", tr("ISO"), self.iso_spin, tooltip=tr("The box speed of the film"))
        row("format", tr("Film format"), self.format_combo, tooltip=tr("The film size this stock comes in"))
        row("color_type", tr("Color type"), self.color_combo, tooltip=tr("Color or black and white, negative or slide"))
        row("developer", tr("Developer"), self.developer_edit, tooltip=tr("The developer this process uses"))
        row("dilution", tr("Dilution"), self.dilution_edit, tooltip=tr("Working strength, for example 1+1, 1+50 or stock"))
        row("push_pull", tr("Push / Pull"), self.push_pull_combo, tooltip=tr("How many stops this process pushes or pulls the film"))
        row("dev_time", tr("Time"), self.dev_time_edit, tooltip=tr("Development time, as mm:ss or plain minutes"))
        row("dev_temp", tr("Temperature (°C)"), self.dev_temp_edit, tooltip=tr("Developer temperature in degrees Celsius"))
        row("scanning", tr("Scanning"), self.scanning_edit, tooltip=tr("How the film is digitized, for example a DSLR on a copy stand"))
        row("notes", tr("Notes"), self.notes_edit, tooltip=tr("Your own notes on this item"))

        root.addWidget(self.form_panel)
        root.addStretch()

        self.category_list.currentIndexChanged.connect(self._on_category_changed)

    def _register_form_row(self, key: str, label_text: str, widget: QWidget, *, tooltip: str) -> None:
        widget.setToolTip(wrap_tooltip(tooltip))
        label = field_label(label_text)
        self.form_layout.addWidget(label)
        self.form_layout.addWidget(widget)
        self._form_rows[key] = (label, widget)

    def _show_form_for_category(self, category: str) -> None:
        visible = _CATEGORY_FIELDS[category]
        for key, (label, widget) in self._form_rows.items():
            show = key in visible
            label.setVisible(show)
            widget.setVisible(show)

    def _current_items(self) -> list:
        return getattr(self._library, self._category)

    def _set_current_items(self, items: list) -> None:
        setattr(self._library, self._category, items)

    def _select_category(self, key: str) -> None:
        idx = self.category_list.findData(key)
        if idx < 0:
            return
        if self.category_list.currentIndex() == idx:
            self._on_category_changed(idx)
        else:
            self.category_list.setCurrentIndex(idx)

    def _on_category_changed(self, index: int) -> None:
        if index < 0:
            return
        self._category = self.category_list.itemData(index)
        self._show_bundled = False
        self.show_catalog_btn.blockSignals(True)
        self.show_catalog_btn.setChecked(False)
        self.show_catalog_btn.blockSignals(False)
        self.item_search.blockSignals(True)
        self.item_search.clear()
        self.item_search.setPlaceholderText(tr(CATEGORY_SEARCH_PLACEHOLDER[self._category]))
        self.item_search.blockSignals(False)
        self._rebuild_item_list()
        self._show_form_for_category(self._category)
        self._update_del_tooltip()
        self.add_btn.setToolTip(wrap_tooltip(_ADD_TOOLTIPS.get(self._category, tr("Add item"))))

    def _on_show_catalog_toggled(self, checked: bool) -> None:
        self._show_bundled = checked
        self._rebuild_item_list()

    def _on_item_search_changed(self, _text: str) -> None:
        self._rebuild_item_list()

    def _visible_items(self, all_items: list) -> list:
        """The default list is personal gear only; the catalog toggle brings the
        bundled reference entries back in."""
        return all_items if self._show_bundled else [item for item in all_items if not item.is_bundled]

    def _rebuild_item_list(self, *, select_id: str | None = None) -> None:
        all_items = self._current_items()
        selected_id = select_id
        if selected_id is None and 0 <= self._selected_idx < len(all_items):
            selected_id = all_items[self._selected_idx].id

        query = self.item_search.text().strip()
        candidates = self._visible_items(all_items)
        visible = [item for item in candidates if matches_gear_filter(item, query)]

        self._list_items = visible
        self.item_list.blockSignals(True)
        self.item_list.clear()
        for item in visible:
            self.item_list.addItem(QListWidgetItem(item.resolved_display_name))

        row = -1
        if visible:
            if selected_id:
                row = next((i for i, item in enumerate(visible) if item.id == selected_id), -1)
            if row < 0 and select_id is not None:
                row = next((i for i, item in enumerate(visible) if item.id == select_id), 0)
            elif row < 0 and not query:
                row = 0
        self.item_list.setCurrentRow(row)
        self.item_list.blockSignals(False)

        # An empty personal list reads as "nothing here yet", not a blank box.
        no_personal_gear = not candidates and not query
        self.empty_hint.setVisible(no_personal_gear)
        if no_personal_gear:
            self.empty_hint.setText(tr("You have not added any {value} yet.", value=_CATEGORY_PLURAL_NOUN[self._category]))
        self.item_list.setVisible(not no_personal_gear)

        if not visible and not query:
            self._selected_idx = -1
            self._clear_form()
        elif row >= 0:
            self._on_item_changed(row)

    def _on_item_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._list_items):
            self._selected_idx = -1
            self._set_form_editable(True)
            self._clear_form()
            self._update_del_tooltip()
            return
        item = self._list_items[row]
        all_items = self._current_items()
        self._selected_idx = next(i for i, candidate in enumerate(all_items) if candidate.id == item.id)
        self._set_form_editable(not item.is_bundled)
        self._populate_form(item)
        self._update_del_tooltip()

    def _update_del_tooltip(self) -> None:
        item = self._selected_item()
        bundled = item is not None and item.is_bundled
        text = _DELETE_BUNDLED_TOOLTIP if bundled else _DELETE_TOOLTIPS.get(self._category, tr("Delete"))
        self.del_btn.setToolTip(wrap_tooltip(text))

    def _selected_item(self):
        items = self._current_items()
        return items[self._selected_idx] if 0 <= self._selected_idx < len(items) else None

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
                idx = self.format_combo.findData(item.format.value)
                if idx >= 0:
                    self.format_combo.setCurrentIndex(idx)
                idx = self.color_combo.findData(item.color_type.value)
                if idx >= 0:
                    self.color_combo.setCurrentIndex(idx)
                self.notes_edit.setText(item.notes)
        finally:
            self._updating = False

    def _clear_form(self) -> None:
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
        if self._updating or self._selected_idx < 0:
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
            item.format = FilmFormat(self.format_combo.currentData())
            item.color_type = FilmColorType(self.color_combo.currentData())
            item.notes = self.notes_edit.text().strip()

        items[self._selected_idx] = item
        self._set_current_items(items)
        list_row = next((i for i, visible in enumerate(self._list_items) if visible.id == item.id), -1)
        list_entry = self.item_list.item(list_row) if list_row >= 0 else None
        if list_entry is not None:
            list_entry.setText(item.resolved_display_name)
        GearProfiles.save_library(self._library)
        self.library_changed.emit()

    def _add_item(self) -> None:
        catalog = [item for item in self._current_items() if item.is_bundled]
        if not catalog:
            self._add_custom_item()
            return
        dlg = GearCatalogDialog(
            self,
            tr(CATEGORY_SINGULAR[self._category]),
            catalog,
            lambda item: item.resolved_display_name,
            tr(CATEGORY_SEARCH_PLACEHOLDER[self._category]),
            repo=self._repo,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if dlg.wants_custom():
            self._add_custom_item()
            return
        picked = next((c for c in catalog if c.id == dlg.selected_id()), None)
        if picked is not None:
            self._clone_into_personal(picked)

    def _add_custom_item(self) -> None:
        """Blank-record escape hatch, offered by the catalog dialog when the user's own
        gear isn't in the shipped list."""
        item = blank_gear_item(self._category)
        items = list(self._current_items())
        items.append(item)
        self._set_current_items(items)
        GearProfiles.save_library(self._library)
        self._rebuild_item_list(select_id=item.id)
        self.library_changed.emit()
        self.display_name_edit.setFocus()
        self.display_name_edit.selectAll()

    def _clone_into_personal(self, source) -> None:
        dup = clone_into_personal(source)
        items = list(self._current_items())
        items.append(dup)
        self._set_current_items(items)
        GearProfiles.save_library(self._library)
        self._rebuild_item_list(select_id=dup.id)
        self.library_changed.emit()

    def _duplicate_item(self) -> None:
        if self._selected_idx < 0:
            return
        items = self._current_items()
        self._clone_into_personal(items[self._selected_idx])

    def _delete_item(self) -> None:
        if self._selected_idx < 0:
            return
        items = self._current_items()
        item = items[self._selected_idx]
        if item.is_bundled:
            return
        if not confirm_delete_named(self, tr(CATEGORY_SINGULAR[self._category]), item.resolved_display_name):
            return
        items = list(self._current_items())
        del items[self._selected_idx]
        self._set_current_items(items)
        GearProfiles.save_library(self._library)
        self._rebuild_item_list()
        self.library_changed.emit()


class GearPresetsPanel(QWidget):
    """Saved metadata field sets, stored as files rather than library records: the form
    pane shows what one holds and the field picker edits it. Its gear combos default to
    personal gear, same as the Metadata tab, with Other… reaching the full catalog."""

    library_changed = pyqtSignal()
    presets_changed = pyqtSignal()
    apply_requested = pyqtSignal(str)

    def __init__(
        self,
        library: GearLibrary,
        current_config_fn: Optional[Callable[[], Optional[WorkspaceConfig]]] = None,
        parent=None,
        *,
        repo=None,
    ):
        super().__init__(parent)
        self._repo = repo
        self._library = library
        # A getter, not a snapshot: this panel is built once and stays live for the
        # whole session, so "save preset from the current frame" needs whichever frame
        # is current *when Add is clicked*, not whichever was current at construction.
        self._current_config_fn = current_config_fn or (lambda: None)
        self._selected_idx = -1
        self._list_items: list[str] = []
        self._updating = False

        self._init_ui()
        self._rebuild_item_list()

    def library(self) -> GearLibrary:
        return self._library

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(THEME.space_lg)

        root.addWidget(hint_label(tr("Saved sets of metadata values. Apply one to write its fields onto this frame.")))

        search_row = QHBoxLayout()
        search_row.setSpacing(THEME.space_sm)
        self.item_search = QLineEdit()
        self.item_search.setPlaceholderText(tr("Search presets…"))
        self.item_search.setToolTip(wrap_tooltip(tr("Filter the list below by name")))
        self.item_search.textChanged.connect(self._on_item_search_changed)
        search_row.addWidget(self.item_search, 1)
        self.add_btn = icon_button("fa5s.plus", tr("Store the current frame's metadata as a preset"))
        self.add_btn.clicked.connect(self._add_item)
        search_row.addWidget(self.add_btn)
        root.addLayout(search_row)

        self.item_list = RowList(_PresetDelegate, _PRESET_ROW)
        self.item_list.setToolTip(tr("Click a preset to see what it stores, or right-click it for more"))
        self.item_list.currentRowChanged.connect(self._on_item_changed)
        self.item_list.menu_requested.connect(self._on_preset_menu)
        root.addWidget(self.item_list)

        self.empty_hint = hint_label()
        self.empty_hint.setVisible(False)
        root.addWidget(self.empty_hint)

        apply_row = QHBoxLayout()
        apply_row.addStretch()
        self.apply_btn = labeled_action("mdi6.auto-fix", tr("Apply to This Frame"), _APPLY_TOOLTIP, primary=True)
        self.apply_btn.clicked.connect(lambda: self.apply_requested.emit(self._selected_preset()))
        apply_row.addWidget(self.apply_btn)
        root.addLayout(apply_row)
        self._apply_allowed = True

        self.edit_btn = icon_button("fa5s.pen", tr("Rename the preset, or change which fields it stores"))
        self.edit_btn.clicked.connect(self._edit_preset)
        root.addLayout(header_row(section_subheader(tr("DETAILS")), self.edit_btn))

        self.preset_panel = QWidget()
        preset_layout = QVBoxLayout(self.preset_panel)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(THEME.space_lg)
        self.preset_name_label = QLabel()
        self.preset_name_label.setStyleSheet(f"color: {THEME.text_primary}; font-weight: bold;")
        self.preset_form_layout = QVBoxLayout()
        self.preset_form_layout.setSpacing(THEME.space_md)
        self._preset_rows: dict[str, tuple[QLabel, QWidget]] = {}
        self._build_preset_form()
        self.preset_fields_layout = QVBoxLayout()
        self.preset_fields_layout.setSpacing(THEME.space_md)
        self.preset_empty_label = QLabel(tr("This preset stores nothing."))
        self.preset_empty_label.setStyleSheet(f"color: {THEME.text_secondary};")
        preset_layout.addWidget(self.preset_name_label)
        preset_layout.addLayout(self.preset_form_layout)
        preset_layout.addLayout(self.preset_fields_layout)
        preset_layout.addWidget(self.preset_empty_label)
        preset_layout.addWidget(field_label(tr("Notes")))
        self.preset_notes_edit = QLineEdit()
        self.preset_notes_edit.setPlaceholderText(tr("Notes for this preset"))
        self.preset_notes_edit.textChanged.connect(self._on_preset_notes_changed)
        preset_layout.addWidget(self.preset_notes_edit)
        preset_layout.addWidget(hint_label(tr("The pen chooses which fields a preset stores; these edit their values.")))
        root.addWidget(self.preset_panel)
        root.addStretch()

        self._sync_buttons()

    def _build_preset_form(self) -> None:
        self.preset_camera_combo = SearchableGearCombo(placeholder=tr("Search cameras…"))
        self.preset_lens_combo = SearchableGearCombo(placeholder=tr("Search lenses…"))
        self.preset_film_combo = SearchableGearCombo(placeholder=tr("Search film stocks…"))
        self.preset_process_combo = SearchableGearCombo(placeholder=tr("Search processes…"))
        self.preset_scan_combo = SearchableGearCombo(placeholder=tr("Search scan setups…"))
        self.preset_format_combo = StepCombo()
        add_choice_items(self.preset_format_combo, FORMAT_OPTIONS)
        self.preset_format_other_edit = QLineEdit()
        self.preset_format_other_edit.setPlaceholderText(tr("e.g. 6×7"))
        self.preset_developer_edit = QLineEdit()
        self.preset_developer_edit.setPlaceholderText(tr("e.g. D-76"))
        self.preset_dilution_edit = QLineEdit()
        self.preset_dilution_edit.setPlaceholderText(tr("e.g. 1+50"))
        self.preset_push_combo = StepCombo()
        add_choice_items(self.preset_push_combo, [PUSH_PULL_LABELS[v] for v in PUSH_PULL_VALUES])
        self.preset_time_edit = QLineEdit()
        self.preset_time_edit.setPlaceholderText(DEV_TIME_HINT)
        self.preset_temp_edit = QLineEdit()
        self.preset_temp_edit.setPlaceholderText(tr("e.g. 20"))
        self.preset_scanning_edit = QLineEdit()
        self.preset_scanning_edit.setPlaceholderText(tr("e.g. DSLR copy-stand scan"))
        self.preset_roll_edit = QLineEdit()
        self.preset_roll_edit.setPlaceholderText(tr("e.g. Roll001"))
        self.preset_exposure_edit = QLineEdit()
        self.preset_exposure_edit.setPlaceholderText(tr("e.g. 1/125s f/2.8 ISO 400"))

        for key, label, widget, tip in (
            ("camera", tr("Camera"), self.preset_camera_combo, tr("The camera this preset writes onto a frame")),
            ("lens", tr("Lens"), self.preset_lens_combo, tr("The lens this preset writes onto a frame")),
            ("film_stock", tr("Film stock"), self.preset_film_combo, tr("The film stock this preset writes onto a frame")),
            ("format", tr("Film format"), self.preset_format_combo, tr("Film format written to the frame's metadata")),
            ("format_other", tr("Other format"), self.preset_format_other_edit, tr("The film format, when the list does not have it")),
            (
                "process",
                tr("Saved process"),
                self.preset_process_combo,
                tr("A saved development recipe that fills Developer and Push / Pull"),
            ),
            ("developer", tr("Developer"), self.preset_developer_edit, tr("The developer the film was processed in")),
            ("dilution", tr("Dilution"), self.preset_dilution_edit, tr("Working strength, for example 1+1, 1+50 or stock")),
            ("push_pull", tr("Push / Pull"), self.preset_push_combo, tr("How many stops the film was pushed or pulled in development")),
            ("dev_time", tr("Time"), self.preset_time_edit, tr("Development time, as mm:ss or plain minutes")),
            ("dev_temp", tr("Temperature (°C)"), self.preset_temp_edit, tr("Developer temperature in degrees Celsius")),
            ("scan_setup", tr("Saved setup"), self.preset_scan_combo, tr("A saved digitizing setup that fills Scanning")),
            ("scanning", tr("Scanning"), self.preset_scanning_edit, tr("How the film was digitized, for example a DSLR on a copy stand")),
            ("roll", tr("Roll"), self.preset_roll_edit, tr("Roll name, used in export file name templates as {{ roll }}")),
            (
                "exposure",
                tr("Exposure"),
                self.preset_exposure_edit,
                tr("The exposure written to the frame, for example 1/125s f/2.8 ISO 400"),
            ),
        ):
            widget.setToolTip(wrap_tooltip(tip))
            row_label = field_label(label)
            self.preset_form_layout.addWidget(row_label)
            self.preset_form_layout.addWidget(widget)
            self._preset_rows[key] = (row_label, widget)

        self._preset_combo_category = {
            id(self.preset_camera_combo): "cameras",
            id(self.preset_lens_combo): "lenses",
            id(self.preset_film_combo): "film_stocks",
            id(self.preset_process_combo): "processes",
            id(self.preset_scan_combo): "scan_setups",
        }
        self._preset_combo_field = {
            id(self.preset_camera_combo): lambda meta: meta.camera_id,
            id(self.preset_lens_combo): lambda meta: meta.lens_id,
            id(self.preset_film_combo): lambda meta: meta.film_stock_id,
            id(self.preset_process_combo): lambda meta: meta.process_id,
            id(self.preset_scan_combo): lambda meta: meta.scanning_id,
        }

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

    def on_activated(self) -> None:
        """Re-check whether a frame is open every time this tab is shown, since it may
        have changed while another tab was in front."""
        self._sync_buttons()

    def set_apply_allowed(self, allowed: bool) -> None:
        """False while Protect Original Metadata is on: the frame's own fields are ignored then."""
        self._apply_allowed = allowed
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        has_frame = self._current_config_fn() is not None
        self.add_btn.setEnabled(has_frame)
        self.apply_btn.setEnabled(has_frame and self._apply_allowed and bool(self._selected_preset()))
        self.edit_btn.setEnabled(bool(self._selected_preset()))

    def _on_preset_menu(self, pos) -> None:
        item = self.item_list.itemAt(pos)
        if item is None:
            return
        self.item_list.setCurrentItem(item)
        menu = QMenu(self)
        apply_action = menu.addAction(tr("Apply to This Frame"))
        apply_action.setEnabled(self.apply_btn.isEnabled())
        duplicate_action = menu.addAction(tr("Duplicate"))
        edit_action = menu.addAction(tr("Edit Preset…"))
        delete_action = menu.addAction(tr("Delete…"))
        chosen = menu.exec(self.item_list.mapToGlobal(pos))
        if chosen is apply_action:
            self.apply_btn.click()
        elif chosen is duplicate_action:
            self._duplicate_item()
        elif chosen is edit_action:
            self._edit_preset()
        elif chosen is delete_action:
            self._delete_item()

    def refresh_gear_combos(self) -> None:
        """Personal gear added on Items must reach an already-open preset's combos
        without the user reselecting it."""
        name = self._selected_preset()
        if name:
            self._populate_preset(name)

    def _current_items(self) -> list[str]:
        return sorted(MetadataPresets.list_presets())

    def _on_item_search_changed(self, _text: str) -> None:
        self._rebuild_item_list()

    def _rebuild_item_list(self, *, select_id: str | None = None) -> None:
        all_items = self._current_items()
        selected_id = select_id
        if selected_id is None and 0 <= self._selected_idx < len(all_items):
            selected_id = all_items[self._selected_idx]

        query = self.item_search.text().strip()
        visible = [name for name in all_items if query.casefold() in name.casefold()]

        self._list_items = visible
        self.item_list.blockSignals(True)
        self.item_list.clear()
        icon = qta.icon("mdi6.auto-fix", color=THEME.text_secondary)
        for name in visible:
            item = QListWidgetItem(icon, name)
            rows = rows_for_keys(MetadataPresets.load_preset(name) or {}, "metadata")
            item.setData(_DETAIL_ROLE, ", ".join(tr(r.label) for r in rows) or tr("Stores nothing"))
            self.item_list.addItem(item)
        self.item_list.fit()

        row = -1
        if visible:
            if selected_id:
                row = next((i for i, name in enumerate(visible) if name == selected_id), -1)
            if row < 0 and select_id is not None:
                row = next((i for i, name in enumerate(visible) if name == select_id), 0)
            elif row < 0 and not query:
                row = 0
        self.item_list.setCurrentRow(row)
        self.item_list.blockSignals(False)

        no_presets = not all_items and not query
        self.empty_hint.setVisible(no_presets)
        if no_presets:
            self.empty_hint.setText(tr("You have not added any presets yet."))
        self.item_list.setVisible(not no_presets)

        if not visible and not query:
            self._selected_idx = -1
            self._clear_form()
        elif row >= 0:
            self._on_item_changed(row)
        self._sync_buttons()

    def _on_item_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._list_items):
            self._selected_idx = -1
            self._clear_form()
        else:
            name = self._list_items[row]
            all_items = self._current_items()
            self._selected_idx = next(i for i, candidate in enumerate(all_items) if candidate == name)
            self._populate_preset(name)
        self._sync_buttons()

    def _selected_preset(self) -> str:
        items = self._current_items()
        return items[self._selected_idx] if 0 <= self._selected_idx < len(items) else ""

    def _parsed_or_kept(self, edit: QLineEdit, parse, current):
        text = edit.text().strip()
        if not text:
            self._mark_invalid(edit, False)
            return None
        value = parse(text)
        self._mark_invalid(edit, value is None)
        return current if value is None else value

    def _mark_invalid(self, edit: QLineEdit, invalid: bool) -> None:
        edit.setStyleSheet(f"border: 1px solid {THEME.error};" if invalid else "")

    def _populate_preset(self, name: str) -> None:
        data = MetadataPresets.load_preset(name) or {}
        stored = {r.id for r in rows_for_keys(data, "metadata")}
        editable = {w for row_id in stored for w in _PRESET_ROW_WIDGETS.get(row_id, ())}
        meta = preset_config(data).metadata

        self._updating = True
        try:
            self.preset_name_label.setText(name)
            self.preset_notes_edit.setText(preset_notes(data))
            self._set_own_gear_items(self.preset_camera_combo, self._library.cameras, meta.camera_id)
            self._set_own_gear_items(self.preset_lens_combo, self._library.lenses, meta.lens_id)
            self._set_own_gear_items(self.preset_film_combo, self._library.film_stocks, meta.film_stock_id)
            self._set_own_gear_items(self.preset_process_combo, self._library.processes, meta.process_id)
            self._set_own_gear_items(self.preset_scan_combo, self._library.scan_setups, meta.scanning_id)
            self.preset_format_combo.setCurrentIndex(max(0, self.preset_format_combo.findData(format_label(meta.format))))
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
                show = key in editable and (key != "format_other" or self.preset_format_combo.currentData() == "Other")
                row_label.setVisible(show)
                widget.setVisible(show)
            self._mark_invalid(self.preset_time_edit, False)
            self._mark_invalid(self.preset_temp_edit, False)
        finally:
            self._updating = False

        # Rows with no editor: per-frame decisions, shown as they are stored.
        self._clear_preset_fields_layout()
        read_only = [(label, value) for label, value in preset_values(data, "metadata") if not self._is_editable_row(label, stored)]
        for label, value in read_only:
            value_label = QLabel(value)
            value_label.setWordWrap(True)
            value_label.setStyleSheet(f"color: {THEME.text_secondary};")
            self.preset_fields_layout.addWidget(field_label(label))
            self.preset_fields_layout.addWidget(value_label)
        self.preset_empty_label.setVisible(not stored)

    def _clear_preset_fields_layout(self) -> None:
        while self.preset_fields_layout.count():
            item = self.preset_fields_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()

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
        # No sender-based dispatch: called directly (no live signal) by callers that
        # just set a combo's id and want the write-through, not just a real pick.
        for combo in (self.preset_camera_combo, self.preset_lens_combo, self.preset_film_combo):
            self._resolve_other_preset_pick(combo)
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
        self._resolve_other_preset_pick(self.preset_process_combo)
        meta = None if self._updating else self._preset_meta()
        if meta is not None:
            self._write_preset(metadata_from_process(meta, self._library, self.preset_process_combo.selected_id()))

    def _on_preset_scan_picked(self, *_args) -> None:
        self._resolve_other_preset_pick(self.preset_scan_combo)
        meta = None if self._updating else self._preset_meta()
        if meta is not None:
            self._write_preset(metadata_from_scan_setup(meta, self._library, self.preset_scan_combo.selected_id()))

    def _set_own_gear_items(self, combo: SearchableGearCombo, items, selected_id: str) -> None:
        """Personal gear only, plus the currently selected item even if it is a bundled
        catalog pick made before this filter existed. Other… is the escape hatch back to
        the full catalog, so the default search never returns gear the user doesn't own."""
        entries, search_text = own_gear_entries(items, selected_id)
        combo.set_labeled_items(
            entries,
            selected_id,
            search_fn=lambda label, item_id: search_text.get(item_id, label.casefold()),
        )

    def _resolve_other_preset_pick(self, combo: SearchableGearCombo) -> None:
        """Other… resolves to a real personal item before the caller writes the
        selection: picking a catalog model clones it into the user's own gear, Add
        Custom starts a blank one, cancelling reverts to what was selected before."""
        if combo.selected_id() != OTHER_ID or self._updating:
            return
        category = self._preset_combo_category[id(combo)]
        meta = self._preset_meta()
        previous = self._preset_combo_field[id(combo)](meta) if meta is not None else ""
        new_item = resolve_other_gear_pick(self, category, self._library, repo=self._repo)
        if new_item is None:
            self._set_own_gear_items(combo, getattr(self._library, category), previous)
            return
        items = list(getattr(self._library, category))
        items.append(new_item)
        setattr(self._library, category, items)
        GearProfiles.save_library(self._library)
        self.library_changed.emit()
        self._set_own_gear_items(combo, getattr(self._library, category), new_item.id)

    def _on_preset_value_changed(self, *_args) -> None:
        meta = None if self._updating else self._preset_meta()
        if meta is None:
            return
        fmt = format_value(self.preset_format_combo.currentData())
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
        self.preset_name_label.setText(tr("No preset selected"))
        self._updating = True
        try:
            self.preset_notes_edit.clear()
        finally:
            self._updating = False
        self._clear_preset_fields_layout()
        self.preset_empty_label.setVisible(False)

    def _name_is_usable(self, name: str, replacing: str = "") -> bool:
        """A preset name is a filename, and a rename onto another preset would replace it."""
        if not is_valid_preset_name(name):
            QMessageBox.warning(
                self,
                tr("Preset Name"),
                tr('A preset name cannot contain / \\ : * ? " < > | or start or end with a dot.'),
            )
            return False
        if name.casefold() == replacing.casefold() or not MetadataPresets.exists(name):
            return True
        return (
            QMessageBox.question(self, tr("Replace Preset"), tr("A preset named '{name}' already exists. Replace it?", name=name))
            == QMessageBox.StandardButton.Yes
        )

    def _add_item(self) -> None:
        """A preset is the current frame's metadata, minus the fields turned off."""
        current_config = self._current_config_fn()
        if current_config is None:
            return
        dlg = GranularSettingsDialog(
            self, current_config, tr("current metadata"), ask_name=True, exclude_sections=NON_METADATA_SECTIONS, repo=self._repo
        )
        dlg.setWindowTitle(tr("New Metadata Preset"))
        # As when editing: which fields to store is the choice, so every row is on offer.
        dlg.show_unchanged_settings()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name = dlg.name().strip()
        if not self._name_is_usable(name):
            return
        MetadataPresets.save_preset(name, selected_flat_dict(current_config, dlg.selected()))
        self._rebuild_item_list(select_id=name)
        self.presets_changed.emit()

    def _edit_preset(self) -> None:
        name = self._selected_preset()
        data = MetadataPresets.load_preset(name) if name else None
        if not data:
            return
        cfg = preset_config(data)
        dlg = GranularSettingsDialog(self, cfg, name, ask_name=True, exclude_sections=NON_METADATA_SECTIONS, repo=self._repo)
        dlg.setWindowTitle(tr("Edit Metadata Preset"))
        dlg.set_name(name)
        # Editing is about which fields the preset holds, so show every row, default-valued
        # ones included, rather than making the user reveal them to add one.
        dlg.show_unchanged_settings()
        # What the preset stores, not what differs from default: a row deliberately holding
        # a default value would otherwise arrive off and be dropped on save.
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

    def _duplicate_item(self) -> None:
        if self._selected_idx < 0:
            return
        name = self._selected_preset()
        data = MetadataPresets.load_preset(name) if name else None
        if data is None:
            return
        existing = set(MetadataPresets.list_presets())
        copy_name = next(
            f"{name} copy{'' if i == 1 else f' {i}'}" for i in range(1, 100) if f"{name} copy{'' if i == 1 else f' {i}'}" not in existing
        )
        MetadataPresets.save_preset(copy_name, data)
        self._rebuild_item_list(select_id=copy_name)
        self.presets_changed.emit()

    def _delete_item(self) -> None:
        if self._selected_idx < 0:
            return
        name = self._selected_preset()
        if not confirm_delete_named(self, tr("Preset"), name):
            return
        MetadataPresets.delete_preset(name)
        self._rebuild_item_list()
        self.presets_changed.emit()


class GearLibraryPanel(QWidget):
    """My Gear and Presets over one gear library, each a card the Metadata tab gives a tab of
    its own. My Gear holds physical gear (Cameras, Lenses, Film Stocks, Processes, Scan Setups),
    with a bundled/personal Catalog toggle; Presets holds saved metadata field sets, which
    have no bundled counterpart."""

    library_changed = pyqtSignal()
    presets_changed = pyqtSignal()

    def __init__(
        self,
        library: GearLibrary | None = None,
        parent=None,
        current_config_fn: Optional[Callable[[], Optional[WorkspaceConfig]]] = None,
        repo=None,
    ):
        super().__init__(parent)
        self._library = library or GearProfiles.load_library()
        self._repo = repo

        self.items = GearItemsPanel(self._library, repo=self._repo)
        self.presets = GearPresetsPanel(self._library, current_config_fn, repo=self._repo)
        self.items.library_changed.connect(self.library_changed.emit)
        self.items.library_changed.connect(self.presets.refresh_gear_combos)
        self.presets.library_changed.connect(self.library_changed.emit)
        self.presets.presets_changed.connect(self.presets_changed.emit)

        self.sections: dict[str, CollapsibleSection] = {
            "items": make_section(self._repo, tr("My Gear"), "gear_items", self.items, "fa5s.toolbox", collapsible=False),
            "presets": make_section(self._repo, tr("Presets"), "gear_presets", self.presets, "fa5s.magic", collapsible=False),
        }
        self.apply_shortcut_tooltips()

    def library(self) -> GearLibrary:
        return self._library

    def reload_library(self) -> None:
        """Re-read the library saved elsewhere, in place: both panels hold this one object."""
        fresh = GearProfiles.load_library()
        for f in fields(GearLibrary):
            setattr(self._library, f.name, getattr(fresh, f.name))
        selected = self.items._selected_item()
        self.items._rebuild_item_list(select_id=selected.id if selected is not None else None)
        self.presets.refresh_gear_combos()

    def apply_shortcut_tooltips(self) -> None:
        self.presets.apply_btn.setToolTip(tooltip_with_shortcut(_APPLY_TOOLTIP, "metadata_preset_load"))
