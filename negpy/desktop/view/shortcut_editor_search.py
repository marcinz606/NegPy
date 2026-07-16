"""Search index for shortcut dialogs (Customize and overview)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PyQt6.QtCore import QModelIndex, QPoint, Qt, QSortFilterProxyModel
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import QCompleter, QScrollArea, QWidget

from negpy.desktop.view.shortcut_registry import (
    REGISTRY,
    EditorRowSlider,
    ShortcutEntry,
    category_editor_rows,
)

TARGET_ROLE = Qt.ItemDataRole.UserRole
SEARCH_ROLE = Qt.ItemDataRole.UserRole + 1
BINDING_KEYS_ROLE = Qt.ItemDataRole.UserRole + 2
TEXT_SEARCH_ROLE = Qt.ItemDataRole.UserRole + 3
HIGHLIGHT_MS = 1800

RowKind = Literal["single", "slider"]


class ShortcutSearchProxy(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._query = ""

    def set_query(self, query: str) -> None:
        needle = (query or "").strip().casefold()
        if needle == self._query:
            return
        self._query = needle
        self.invalidateFilter()
        if self._query:
            self.sort(0, Qt.SortOrder.AscendingOrder)

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        if not self._query:
            return False
        model = self.sourceModel()
        if model is None:
            return False
        index = model.index(source_row, 0, source_parent)
        binding_keys = model.data(index, BINDING_KEYS_ROLE) or ""
        if self._query in binding_keys.split():
            return True
        if len(self._query) < 2:
            return False
        text = model.data(index, TEXT_SEARCH_ROLE) or ""
        return self._query in text

    def lessThan(self, source_left: QModelIndex, source_right: QModelIndex) -> bool:  # noqa: N802
        if not self._query:
            return super().lessThan(source_left, source_right)
        left_rank = self._binding_match_rank(source_left)
        right_rank = self._binding_match_rank(source_right)
        if left_rank != right_rank:
            return left_rank < right_rank
        left_label = self.sourceModel().data(source_left, Qt.ItemDataRole.DisplayRole) or ""
        right_label = self.sourceModel().data(source_right, Qt.ItemDataRole.DisplayRole) or ""
        return left_label.casefold() < right_label.casefold()

    def _binding_match_rank(self, source_index: QModelIndex) -> int:
        model = self.sourceModel()
        if model is None or not source_index.isValid():
            return 1
        keys = model.data(source_index, BINDING_KEYS_ROLE) or ""
        if self._query in keys.split():
            return 0
        return 1


def target_id_from_completer_index(
    completer: QCompleter,
    search_model: QStandardItemModel,
    search_proxy: ShortcutSearchProxy,
    index: QModelIndex,
) -> str:
    if not index.isValid():
        return ""
    completion_model = completer.completionModel()
    proxy_index = completion_model.mapToSource(index)
    source_index = search_proxy.mapToSource(proxy_index)
    if not source_index.isValid():
        return ""
    target_id = search_model.data(source_index, TARGET_ROLE)
    return str(target_id) if target_id else ""


def first_matching_target_id(
    completer: QCompleter,
    search_model: QStandardItemModel,
    search_proxy: ShortcutSearchProxy,
) -> str:
    completion_model = completer.completionModel()
    for row in range(completion_model.rowCount()):
        target_id = target_id_from_completer_index(completer, search_model, search_proxy, completion_model.index(row, 0))
        if target_id:
            return target_id
    return ""


def scroll_row_to_center(scroll: QScrollArea, row: QWidget) -> None:
    content = scroll.widget()
    if content is None:
        return
    center = row.mapTo(content, QPoint(0, row.height() // 2))
    bar = scroll.verticalScrollBar()
    viewport_h = scroll.viewport().height()
    value = center.y() - viewport_h // 2
    value = max(bar.minimum(), min(bar.maximum(), value))
    bar.setValue(value)


@dataclass(frozen=True)
class ShortcutEditorTarget:
    target_id: str
    label: str
    category: str
    search_text: str
    text_search: str
    row_kind: RowKind
    binding_keys: frozenset[str]


def binding_keys_display(keys: frozenset[str]) -> str:
    return " ".join(sorted(keys))


def normalize_binding(key: str) -> str:
    return key.strip().casefold()


def _binding_keys(bindings: dict[str, str], *action_ids: str) -> frozenset[str]:
    keys: set[str] = set()
    for action_id in action_ids:
        key = bindings.get(action_id) or REGISTRY[action_id].default_key
        if key:
            keys.add(normalize_binding(key))
    return frozenset(keys)


def action_id_for_binding(bindings: dict[str, str], portable: str) -> str | None:
    needle = normalize_binding(portable)
    for action_id, key in bindings.items():
        if key and normalize_binding(key) == needle:
            return action_id
    return None


def target_ids_for_binding(targets: list[ShortcutEditorTarget], portable: str) -> list[str]:
    needle = normalize_binding(portable)
    return [target.target_id for target in targets if needle in target.binding_keys]


def _text_tokens(*parts: str) -> str:
    return _tokens(*parts)


def _combined_search_text(text_search: str, binding_keys: frozenset[str]) -> str:
    keys = binding_keys_display(binding_keys)
    if text_search and keys:
        return f"{text_search} {keys}"
    return text_search or keys


def configure_search_completer(completer: QCompleter) -> None:
    """Keep popup rows aligned with proxy filtering (not display-label prefix matching)."""
    completer.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
    completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    completer.setCompletionColumn(0)
    completer.setCompletionRole(SEARCH_ROLE)
    completer.setFilterMode(Qt.MatchFlag.MatchContains)


def _tokens(*parts: str) -> str:
    return " ".join(p.strip() for p in parts if p and str(p).strip()).casefold()


def build_shortcut_editor_targets(bindings: dict[str, str] | None = None) -> list[ShortcutEditorTarget]:
    """Build navigable editor rows with precomputed search text (includes current bindings)."""
    resolved = bindings if bindings is not None else {}
    targets: list[ShortcutEditorTarget] = []
    seen_categories: dict[str, list[tuple[str, ShortcutEntry]]] = {}
    for action_id, entry in REGISTRY.items():
        seen_categories.setdefault(entry.category, []).append((action_id, entry))

    for category, items in seen_categories.items():
        for editor_row in category_editor_rows(items):
            if isinstance(editor_row, EditorRowSlider):
                group = editor_row.group
                inc = REGISTRY[group.inc_action]
                dec = REGISTRY[group.dec_action]
                binding_keys = _binding_keys(resolved, group.inc_action, group.dec_action)
                text_search = _text_tokens(group.label, category, inc.description, dec.description)
                targets.append(
                    ShortcutEditorTarget(
                        target_id=group.id,
                        label=group.label,
                        category=category,
                        row_kind="slider",
                        binding_keys=binding_keys,
                        search_text=_combined_search_text(text_search, binding_keys),
                        text_search=text_search,
                    )
                )
                continue

            action_id = editor_row.action_id
            entry = editor_row.entry
            binding_keys = _binding_keys(resolved, action_id)
            text_search = _text_tokens(entry.description, category)
            targets.append(
                ShortcutEditorTarget(
                    target_id=action_id,
                    label=entry.description,
                    category=category,
                    row_kind="single",
                    binding_keys=binding_keys,
                    search_text=_combined_search_text(text_search, binding_keys),
                    text_search=text_search,
                )
            )

    return targets


def filter_targets(targets: list[ShortcutEditorTarget], query: str) -> list[ShortcutEditorTarget]:
    needle = query.strip().casefold()
    if not needle:
        return list(targets)
    return [target for target in targets if needle in target.binding_keys or (len(needle) >= 2 and needle in target.text_search)]
