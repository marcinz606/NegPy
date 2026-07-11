"""Searchable gear picker — a line edit with a QCompleter dropdown.

Deliberately **not** a ``QComboBox``: editable combos constantly re-sync their
line-edit text to the current index (on focus-out, popup close, ``setCurrentIndex``),
which fought every attempt to keep search text / cleared state stable.

The popup is a :class:`QCompleter`, *not* a hand-rolled ``Qt.Popup`` window. A
``Qt.Popup`` grabs the keyboard when shown, so after the first keystroke typing
went to the popup instead of the field. ``QCompleter`` is built for this: its popup
leaves keyboard focus on the line edit, and handles Up/Down/Enter/positioning.

Single source of truth:
- ``_selected_id`` is the committed selection (``""`` means no selection).
- the :class:`QLineEdit` owns the visible text; nothing rewrites it behind the
  user's back, so clearing, re-searching and re-selecting all behave. An empty
  field *is* "no selection" — clearing the text and leaving the field removes the
  choice, so there is no separate "— None —" row.

Contract used by callers:
- ``selection_changed(str)`` — emitted only on a genuine user commit (never on
  programmatic ``set_gear_items`` / ``set_labeled_items`` / ``set_selected_id``).
- ``selected_id()`` / ``set_selected_id()`` / ``set_gear_items()`` /
  ``set_labeled_items()`` / ``is_editing()``.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from PyQt6.QtCore import QEvent, QModelIndex, QSortFilterProxyModel, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QCompleter, QHBoxLayout, QLineEdit, QToolButton, QWidget


_NONE_LABEL = "— None —"

_ID_ROLE = Qt.ItemDataRole.UserRole
_SEARCH_ROLE = Qt.ItemDataRole.UserRole + 1


class _GearFilterProxy(QSortFilterProxyModel):
    """Filters rows by a hidden search string."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._query = ""

    def set_query(self, query: str) -> None:
        needle = (query or "").strip().casefold()
        if needle == self._query:
            return
        self._query = needle
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        if not self._query:
            return True
        model = self.sourceModel()
        if model is None:
            return True
        index = model.index(source_row, 0, source_parent)
        search_text = model.data(index, _SEARCH_ROLE) or ""
        return self._query in search_text


class SearchableGearCombo(QWidget):
    """Line-edit gear picker with a type-to-filter completer dropdown."""

    selection_changed = pyqtSignal(str)

    def __init__(
        self,
        *,
        none_label: str = _NONE_LABEL,
        placeholder: str = "Search…",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._none_label = none_label
        self._entries: list[tuple[str, str, str]] = []  # (label, id, search_text)
        self._selected_id = ""
        self._updating = False

        self._line = QLineEdit(self)
        self._line.setPlaceholderText(placeholder)
        self._line.textEdited.connect(self._on_text_edited)
        self._line.editingFinished.connect(self._on_editing_finished)
        self._line.installEventFilter(self)

        self._arrow = QToolButton(self)
        self._arrow.setText("\u25be")
        self._arrow.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._arrow.setCursor(Qt.CursorShape.ArrowCursor)
        self._arrow.setFixedWidth(20)
        self._arrow.clicked.connect(self._toggle_popup)
        self._arrow.setObjectName("gear_combo_arrow")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._line, 1)
        layout.addWidget(self._arrow)

        self._model = QStandardItemModel(self)
        self._proxy = _GearFilterProxy(self)
        self._proxy.setSourceModel(self._model)

        self._completer = QCompleter(self._proxy, self)
        self._completer.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setCompletionColumn(0)
        self._completer.setCompletionRole(Qt.ItemDataRole.DisplayRole)
        self._completer.setWidget(self._line)
        self._completer.activated[QModelIndex].connect(self._on_completer_activated)
        popup = self._completer.popup()
        popup.setObjectName("gear_combo_popup")

        self._load_entries([])

    # ── public API ───────────────────────────────────────────────────────
    def set_gear_items(
        self,
        items: Sequence,
        selected_id: str,
        label_fn: Callable,
        library=None,
    ) -> None:
        """Replace list contents and selection from gear dataclasses."""
        from negpy.features.metadata.gear_logic import gear_search_text

        entries: list[tuple[str, str, str]] = []
        for item in items:
            item_id = getattr(item, "id", "") or ""
            entries.append((label_fn(item), item_id, gear_search_text(item, library)))
        self._selected_id = selected_id or ""
        self._load_entries(entries)

    def set_labeled_items(
        self,
        entries: Sequence[tuple[str, str]],
        selected_id: str,
        search_fn: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        """Set (label, id) pairs — search text defaults to the label lowercased."""
        fn = search_fn or (lambda label, _item_id: label.casefold())
        built = [(label, item_id, fn(label, item_id).casefold()) for label, item_id in entries]
        self._selected_id = selected_id or ""
        self._load_entries(built)

    def selected_id(self) -> str:
        return self._selected_id

    def set_selected_id(self, item_id: str) -> None:
        self._selected_id = item_id or ""
        self._proxy.set_query("")
        self._show_selected_text()

    def is_editing(self) -> bool:
        """True while the field has focus and shows something other than the committed label."""
        if not self._line.hasFocus():
            return False
        return self._line.text() != self._committed_text()

    def line_edit(self) -> QLineEdit:
        return self._line

    def setToolTip(self, text: str) -> None:  # noqa: N802
        super().setToolTip(text)
        self._line.setToolTip(text)

    # ── model construction ───────────────────────────────────────────────
    def _load_entries(self, entries: Sequence[tuple[str, str, str]]) -> None:
        self._entries = [(label, item_id, search.casefold()) for label, item_id, search in entries]
        self._model.clear()
        for label, item_id, search in self._entries:
            row = QStandardItem(label)
            row.setData(item_id, _ID_ROLE)
            row.setData(search, _SEARCH_ROLE)
            row.setEditable(False)
            self._model.appendRow(row)
        self._proxy.set_query("")
        self._show_selected_text()

    def _label_for_id(self, item_id: str) -> str:
        target = item_id or ""
        for label, entry_id, _search in self._entries:
            if entry_id == target:
                return label
        return ""

    def _committed_text(self) -> str:
        return self._label_for_id(self._selected_id) if self._selected_id else ""

    def _show_selected_text(self) -> None:
        """Reflect the committed selection in the field (blank => placeholder)."""
        self._updating = True
        try:
            self._line.setText(self._committed_text())
        finally:
            self._updating = False

    # ── popup ─────────────────────────────────────────────────────────────
    def _popup_visible(self) -> bool:
        popup = self._completer.popup()
        return popup is not None and popup.isVisible()

    def _show_popup(self) -> None:
        self._completer.complete()

    def _toggle_popup(self) -> None:
        if self._popup_visible():
            self._completer.popup().hide()
            return
        self._proxy.set_query("")
        self._line.setFocus()
        self._line.selectAll()
        self._completer.complete()

    def _id_from_source_index(self, source_index: QModelIndex) -> str:
        item = self._model.itemFromIndex(source_index)
        return (item.data(_ID_ROLE) or "") if item is not None else ""

    def _first_match_id(self) -> Optional[str]:
        """First real (non-None) row currently passing the filter, if any."""
        for row in range(self._proxy.rowCount()):
            source_index = self._proxy.mapToSource(self._proxy.index(row, 0))
            item_id = self._id_from_source_index(source_index)
            if item_id:
                return item_id
        return None

    def _hide_popup(self) -> None:
        popup = self._completer.popup()
        if popup is not None and popup.isVisible():
            popup.hide()

    def _on_completer_activated(self, index: QModelIndex) -> None:
        if self._updating:
            return
        item_id = ""
        if index.isValid():
            completion_model = self._completer.completionModel()
            proxy_index = completion_model.mapToSource(index)
            source_index = self._proxy.mapToSource(proxy_index)
            item_id = self._id_from_source_index(source_index)
        self._commit_id(item_id)

    # ── user interaction ─────────────────────────────────────────────────
    def _on_text_edited(self, text: str) -> None:
        if self._updating:
            return
        self._proxy.set_query(text)
        self._completer.complete()

    def _on_editing_finished(self) -> None:
        if not self._popup_visible():
            QTimer.singleShot(0, self._finalize)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self._line:
            etype = event.type()
            if etype == QEvent.Type.FocusIn:
                if self._line.text():
                    QTimer.singleShot(0, self._line.selectAll)
            elif etype == QEvent.Type.FocusOut:
                if not self._popup_visible():
                    QTimer.singleShot(0, self._finalize)
            elif etype == QEvent.Type.KeyPress and event.key() in (
                Qt.Key.Key_Return,
                Qt.Key.Key_Enter,
            ):
                # Commit the highlighted/first match ourselves. When the completer
                # popup is showing it will emit ``activated`` first (and consume the
                # key), so this only runs when the popup is not intercepting.
                self._commit_on_return()
                return True
        return super().eventFilter(watched, event)

    def _commit_on_return(self) -> None:
        text = self._line.text().strip()
        if not text or text.casefold() == self._none_label.casefold():
            self._commit_id("")
            return
        resolved = self._resolve_label(text)
        if resolved is not None:
            self._commit_id(resolved)
            return
        first = self._first_match_id()
        if first:
            self._commit_id(first)
        else:
            self._show_selected_text()

    def _finalize(self) -> None:
        """Settle the field: commit exact matches, clear on empty, else revert."""
        if self._updating:
            return
        text = self._line.text().strip()
        if not text or text.casefold() == self._none_label.casefold():
            self._commit_id("")
            return
        resolved = self._resolve_label(text)
        if resolved is not None:
            self._commit_id(resolved)
        else:
            self._show_selected_text()  # partial / unknown text reverts to selection

    def _resolve_label(self, text: str) -> Optional[str]:
        needle = text.casefold()
        if needle == self._none_label.casefold():
            return ""
        for label, item_id, _search in self._entries:
            if label.casefold() == needle:
                return item_id
        return None

    def _commit_id(self, item_id: str) -> None:
        new_id = item_id or ""
        changed = new_id != self._selected_id
        self._selected_id = new_id
        self._proxy.set_query("")
        self._hide_popup()
        self._show_selected_text()
        if changed and not self.signalsBlocked():
            self.selection_changed.emit(new_id)
