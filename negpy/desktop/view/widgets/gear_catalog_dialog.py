"""Picker for the "+" flow on a Gear Library category: pick a built-in model to make
personal, or fall back to a custom entry."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from PyQt6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout

from negpy.desktop.view.styles.templates import field_label, hint_label, pin_dialog_default, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.features.metadata.gear_logic import (
    CATEGORY_SEARCH_PLACEHOLDER,
    CATEGORY_SINGULAR,
    GearItem,
    blank_gear_item,
    clone_into_personal,
)
from negpy.features.metadata.gear_models import GearLibrary


class GearCatalogDialog(QDialog):
    """Search the shipped catalog for one category. Accepting with a pick returns its id
    (the caller clones it into a personal copy); Add Custom accepts with no pick, asking
    for a blank entry instead."""

    def __init__(self, parent, singular: str, catalog: Sequence, label_fn: Callable, placeholder: str):
        super().__init__(parent)
        self._custom = False
        self.setWindowTitle(f"Add {singular}")
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        root.addWidget(field_label(singular))
        self.combo = SearchableGearCombo(placeholder=placeholder)
        self.combo.setToolTip(wrap_tooltip(f"Search the built-in {singular.lower()} catalog. Click and type to search."))
        self.combo.set_gear_items(catalog, "", label_fn)
        self.combo.selection_changed.connect(self._update_add_enabled)
        root.addWidget(self.combo)
        root.addWidget(hint_label(f"Pick the {singular.lower()} you own from the built-in list, or add a custom one."))

        root.addLayout(self._build_footer())
        self._update_add_enabled()

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        custom_btn = QPushButton("Add Custom")
        custom_btn.setToolTip(wrap_tooltip("Add a blank entry to fill in by hand, for gear the catalog does not carry"))
        custom_btn.clicked.connect(self._pick_custom)
        row.addWidget(custom_btn)
        row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setToolTip(wrap_tooltip("Close without adding anything"))
        cancel_btn.clicked.connect(self.reject)
        self.add_btn = QPushButton("Add")
        self.add_btn.setToolTip(wrap_tooltip("Add the picked entry to your own gear"))
        self.add_btn.clicked.connect(self.accept)
        row.addWidget(cancel_btn)
        row.addWidget(self.add_btn)
        pin_dialog_default(self.add_btn, cancel_btn, custom_btn)
        return row

    def _pick_custom(self) -> None:
        self._custom = True
        self.accept()

    def _update_add_enabled(self, *_args) -> None:
        self.add_btn.setEnabled(bool(self.combo.selected_id()))

    def wants_custom(self) -> bool:
        return self._custom

    def selected_id(self) -> str:
        return "" if self._custom else self.combo.selected_id()


def resolve_other_gear_pick(parent, category: str, library: GearLibrary) -> Optional[GearItem]:
    """The Other… row's flow, shared by every gear combo that defaults to personal gear:
    search the shipped catalog for this category, clone a pick into a personal item, or
    start a blank custom one. Returns the item to add to the library, or None if
    cancelled -- the caller still owns saving it and refreshing its own combo."""
    items = getattr(library, category)
    catalog = [item for item in items if item.is_bundled]
    if not catalog:
        return blank_gear_item(category)
    dlg = GearCatalogDialog(
        parent,
        CATEGORY_SINGULAR[category],
        catalog,
        lambda item: item.resolved_display_name,
        CATEGORY_SEARCH_PLACEHOLDER[category],
    )
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    if dlg.wants_custom():
        return blank_gear_item(category)
    picked = next((item for item in catalog if item.id == dlg.selected_id()), None)
    return clone_into_personal(picked) if picked is not None else None
