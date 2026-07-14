"""Modal dialog for managing the analog gear library."""

from __future__ import annotations

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

from negpy.desktop.view.styles.templates import dialog_pane_qss, field_label, pane_header_qss
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.features.metadata.gear_logic import matches_gear_filter
from negpy.features.metadata.gear_models import (
    Camera,
    FilmColorType,
    FilmFormat,
    FilmStock,
    GearLibrary,
    GearPreset,
    Lens,
)
from negpy.services.assets.gear import GearProfiles

_CATEGORIES = [
    ("cameras", "Cameras"),
    ("lenses", "Lenses"),
    ("film_stocks", "Film Stocks"),
    ("gear_presets", "Gear Presets"),
]

_CATEGORY_FIELDS: dict[str, frozenset[str]] = {
    "cameras": frozenset({"display_name", "make", "model", "notes"}),
    "lenses": frozenset({"display_name", "make", "lens_model", "focal", "aperture", "notes"}),
    "film_stocks": frozenset({"display_name", "manufacturer", "stock_name", "iso", "format", "color_type", "notes"}),
    "gear_presets": frozenset({"display_name", "preset_camera", "preset_lens", "preset_film", "notes"}),
}

_CATEGORY_SEARCH_PLACEHOLDER = {
    "cameras": "Search cameras…",
    "lenses": "Search lenses…",
    "film_stocks": "Search film stocks…",
    "gear_presets": "Search presets…",
}


class GearLibraryDialog(QDialog):
    library_changed = pyqtSignal()

    def __init__(self, library: GearLibrary | None = None, parent=None):
        super().__init__(parent)
        self._library = library or GearProfiles.load_library()
        self._category = "cameras"
        self._selected_idx = -1
        self._list_items: list = []
        self._updating = False

        self.setWindowTitle("Gear Library")
        self.resize(820, 560)
        self._init_ui()
        self._select_category("cameras")

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
        self.del_btn = QPushButton()
        self.del_btn.setIcon(qta.icon("fa5s.trash-alt", color=THEME.text_primary))
        self.del_btn.setToolTip("Delete")
        self.del_btn.clicked.connect(self._delete_item)
        for b in (self.add_btn, self.dup_btn, self.del_btn):
            b.setFixedWidth(36)
            btn_row.addWidget(b)
        btn_row.addStretch()
        mid_layout.addLayout(btn_row)

        root.addWidget(mid)

        # Form — single layout; rows are shown/hidden per category (never removeRow).
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
        self.notes_edit = QLineEdit()
        self.preset_camera_combo = SearchableGearCombo(placeholder="Search cameras…")
        self.preset_lens_combo = SearchableGearCombo(placeholder="Search lenses…")
        self.preset_film_combo = SearchableGearCombo(placeholder="Search film stocks…")

        for w in (
            self.display_name_edit,
            self.make_edit,
            self.model_edit,
            self.lens_model_edit,
            self.manufacturer_edit,
            self.stock_name_edit,
            self.notes_edit,
        ):
            w.textChanged.connect(self._on_form_changed)
        self.focal_spin.valueChanged.connect(self._on_form_changed)
        self.aperture_spin.valueChanged.connect(self._on_form_changed)
        self.iso_spin.valueChanged.connect(self._on_form_changed)
        self.format_combo.currentIndexChanged.connect(self._on_form_changed)
        self.color_combo.currentIndexChanged.connect(self._on_form_changed)
        self.preset_camera_combo.selection_changed.connect(self._on_form_changed)
        self.preset_lens_combo.selection_changed.connect(self._on_form_changed)
        self.preset_film_combo.selection_changed.connect(self._on_form_changed)

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
        self._register_form_row("color_type", "Colour type", self.color_combo)
        self._register_form_row("preset_camera", "Camera", self.preset_camera_combo)
        self._register_form_row("preset_lens", "Lens", self.preset_lens_combo)
        self._register_form_row("preset_film", "Film stock", self.preset_film_combo)
        self._register_form_row("notes", "Notes", self.notes_edit)

        right_layout.addWidget(self.form_panel)
        right_layout.addStretch()

        close_row = QHBoxLayout()
        close_row.addStretch()
        save_btn = QPushButton("Done")
        save_btn.clicked.connect(self.accept)
        close_row.addWidget(save_btn)
        right_layout.addLayout(close_row)

        root.addWidget(right)

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
        return self._library.gear_presets

    def _set_current_items(self, items: list) -> None:
        if self._category == "cameras":
            self._library.cameras = items
        elif self._category == "lenses":
            self._library.lenses = items
        elif self._category == "film_stocks":
            self._library.film_stocks = items
        else:
            self._library.gear_presets = items

    def _item_label(self, item) -> str:
        if isinstance(item, Camera):
            return item.resolved_display_name
        if isinstance(item, Lens):
            return item.resolved_display_name
        if isinstance(item, FilmStock):
            return item.resolved_display_name
        if isinstance(item, GearPreset):
            return item.display_name or "Unnamed preset"
        return str(item)

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

    def _on_item_search_changed(self, _text: str) -> None:
        self._rebuild_item_list()

    def _rebuild_item_list(self, *, select_id: str | None = None) -> None:
        all_items = self._current_items()
        selected_id = select_id
        if selected_id is None and 0 <= self._selected_idx < len(all_items):
            selected_id = all_items[self._selected_idx].id

        query = self.item_search.text().strip()
        lib = self._library if self._category == "gear_presets" else None
        visible = [item for item in all_items if matches_gear_filter(item, query, lib)]

        self._list_items = visible
        self.item_list.blockSignals(True)
        self.item_list.clear()
        for item in visible:
            self.item_list.addItem(QListWidgetItem(self._item_label(item)))

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

        if not visible and not query:
            self._selected_idx = -1
            self._clear_form()
        elif row >= 0:
            self._on_item_changed(row)

    def _refresh_preset_combos(
        self,
        *,
        camera_id: str = "",
        lens_id: str = "",
        film_id: str = "",
    ) -> None:
        self.preset_camera_combo.set_gear_items(
            self._library.cameras,
            camera_id,
            lambda camera: camera.resolved_display_name,
        )
        self.preset_lens_combo.set_gear_items(
            self._library.lenses,
            lens_id,
            lambda lens: lens.resolved_display_name,
        )
        self.preset_film_combo.set_gear_items(
            self._library.film_stocks,
            film_id,
            lambda stock: stock.resolved_display_name,
        )

    def _on_item_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._list_items):
            self._selected_idx = -1
            self._set_form_editable(True)
            self._clear_form()
            return
        item = self._list_items[row]
        all_items = self._current_items()
        self._selected_idx = next(i for i, candidate in enumerate(all_items) if candidate.id == item.id)
        self._set_form_editable(not item.is_bundled)
        self._populate_form(item)

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
            elif isinstance(item, GearPreset):
                self._refresh_preset_combos(
                    camera_id=item.camera_id,
                    lens_id=item.lens_id,
                    film_id=item.film_stock_id,
                )
                self.display_name_edit.setText(item.display_name)
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
        elif isinstance(item, FilmStock):
            item.display_name = self.display_name_edit.text().strip()
            item.manufacturer = self.manufacturer_edit.text().strip()
            item.stock_name = self.stock_name_edit.text().strip()
            item.iso = self.iso_spin.value()
            item.format = FilmFormat(self.format_combo.currentText())
            item.color_type = FilmColorType(self.color_combo.currentText())
            item.notes = self.notes_edit.text().strip()
        elif isinstance(item, GearPreset):
            item.display_name = self.display_name_edit.text().strip()
            item.camera_id = self.preset_camera_combo.selected_id()
            item.lens_id = self.preset_lens_combo.selected_id()
            item.film_stock_id = self.preset_film_combo.selected_id()
            item.notes = self.notes_edit.text().strip()

        items[self._selected_idx] = item
        self._set_current_items(items)
        list_row = next((i for i, visible in enumerate(self._list_items) if visible.id == item.id), -1)
        if list_row >= 0:
            self.item_list.item(list_row).setText(self._item_label(item))
        GearProfiles.save_library(self._library)
        self.library_changed.emit()

    def _add_item(self) -> None:
        if self._category == "cameras":
            item = Camera(make="New", model="Camera")
        elif self._category == "lenses":
            item = Lens(lens_model="New lens")
        elif self._category == "film_stocks":
            item = FilmStock(stock_name="New stock")
        else:
            item = GearPreset(display_name="New preset")
        items = list(self._current_items())
        items.append(item)
        self._set_current_items(items)
        GearProfiles.save_library(self._library)
        self._rebuild_item_list(select_id=item.id)
        self.library_changed.emit()

    def _duplicate_item(self) -> None:
        if self._selected_idx < 0:
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
        if QMessageBox.question(self, "Delete", "Delete this item?") != QMessageBox.StandardButton.Yes:
            return
        items = list(self._current_items())
        del items[self._selected_idx]
        self._set_current_items(items)
        GearProfiles.save_library(self._library)
        self._rebuild_item_list()
        self.library_changed.emit()
