from PyQt6.QtCore import QEvent, QModelIndex, Qt, QTimer
from PyQt6.QtGui import QKeySequence, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QCheckBox,
    QCompleter,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.shortcut_editor_search import (
    BINDING_KEYS_ROLE,
    HIGHLIGHT_MS,
    SEARCH_ROLE,
    TARGET_ROLE,
    TEXT_SEARCH_ROLE,
    ShortcutEditorTarget,
    ShortcutSearchProxy,
    action_id_for_binding,
    binding_keys_display,
    build_shortcut_editor_targets,
    configure_search_completer,
    first_matching_target_id,
    scroll_row_to_center,
    target_id_from_completer_index,
)
from negpy.desktop.view.shortcut_registry import (
    REGISTRY,
    EditorRowSingle,
    EditorRowSlider,
    ShortcutEntry,
    categories_in_order,
    category_editor_rows,
    default_bindings,
    default_slider_steps,
)
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.key_sequence_edit import KeypadAwareKeySequenceEdit
from negpy.desktop.view.widgets.shortcut_search_line_edit import ShortcutSearchLineEdit
from negpy.desktop.view.styles.theme import THEME


def _format_default_pair(inc_key: str, dec_key: str) -> str:
    inc = inc_key or "—"
    dec = dec_key or "—"
    return f"{inc} / {dec}"


class ShortcutEditorDialog(QDialog):
    def __init__(self, bindings: dict[str, str], slider_steps: dict[str, float] | None = None, parent=None, session=None):
        super().__init__(parent)
        self._initial_bindings = dict(bindings)
        self._initial_slider_steps = dict(slider_steps or default_slider_steps())
        self._session = session
        self._edits: dict[str, KeypadAwareKeySequenceEdit] = {}
        self._step_edits: dict[str, QDoubleSpinBox] = {}
        self._sections: dict[str, CollapsibleSection] = {}
        self._row_widgets: dict[str, QFrame] = {}
        self._row_focus_edits: dict[str, KeypadAwareKeySequenceEdit] = {}
        self._targets: list[ShortcutEditorTarget] = build_shortcut_editor_targets(self._initial_bindings)
        self._highlighted_row: QFrame | None = None
        self._highlight_timer = QTimer(self)
        self._highlight_timer.setSingleShot(True)
        self._highlight_timer.timeout.connect(self._clear_highlight)
        self.setWindowTitle("Customize Shortcuts")
        self.resize(820, 720)
        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        intro = QLabel(
            "Set shortcuts and keyboard step sizes for slider actions. "
            "Search by name or press a shortcut to filter results, then choose or press Enter. "
            "Duplicate bindings are rejected. Reset All restores defaults."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        self._search_edit = ShortcutSearchLineEdit(self._known_bindings)
        self._search_edit.setPlaceholderText("Search actions, press a shortcut, then choose or Enter…")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textEdited.connect(self._on_search_edited)
        self._search_edit.installEventFilter(self)
        self._init_search_completer()
        root.addWidget(self._search_edit)

        self._invert_zoom_chk = QCheckBox("Reverse scroll-to-zoom direction (scroll up zooms out)")
        self._invert_zoom_chk.setToolTip(
            "Flip the mouse-wheel zoom direction on the image viewer: scroll up to zoom out, scroll down to zoom in."
        )
        if self._session is not None:
            self._invert_zoom_chk.setChecked(bool(getattr(self._session.state, "invert_zoom_scroll", False)))
        else:
            self._invert_zoom_chk.setEnabled(False)
        root.addWidget(self._invert_zoom_chk)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {THEME.border_color};")
        root.addWidget(divider)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        sections_layout = QVBoxLayout(container)
        sections_layout.setContentsMargins(0, 0, 0, 0)
        sections_layout.setSpacing(THEME.space_sm)

        for category, items in categories_in_order():
            section = CollapsibleSection(category, expanded=False)
            section.set_content(self._build_category_grid(category, items))
            self._sections[category] = section
            sections_layout.addWidget(section)

        sections_layout.addStretch()
        self._scroll.setWidget(container)
        root.addWidget(self._scroll, stretch=1)

        buttons = QHBoxLayout()
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self._reset_all)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(reset_btn)
        buttons.addStretch()
        buttons.addWidget(cancel_btn)
        buttons.addWidget(save_btn)
        root.addLayout(buttons)

        self._reload_search_model()
        for edit in self._edits.values():
            edit.keySequenceChanged.connect(self._reload_search_model)

    def _init_search_completer(self) -> None:
        self._search_model = QStandardItemModel(self)
        self._search_proxy = ShortcutSearchProxy(self)
        self._search_proxy.setSourceModel(self._search_model)
        self._reload_search_model()

        self._search_completer = QCompleter(self._search_proxy, self)
        configure_search_completer(self._search_completer)
        self._search_completer.setWidget(self._search_edit)
        self._search_completer.activated[QModelIndex].connect(self._on_search_activated)
        self._search_completer.popup().setObjectName("shortcut_editor_search_popup")

    def _current_bindings(self) -> dict[str, str]:
        if self._edits:
            return {action_id: self._portable(edit) for action_id, edit in self._edits.items()}
        return dict(self._initial_bindings)

    def _known_bindings(self) -> frozenset[str]:
        return frozenset(key for key in self._current_bindings().values() if key)

    def _reload_search_model(self) -> None:
        self._search_model.clear()
        bindings = self._current_bindings()
        self._targets = build_shortcut_editor_targets(bindings)
        for target in self._targets:
            item = QStandardItem(f"{target.label}  ·  {target.category}")
            item.setData(target.target_id, TARGET_ROLE)
            item.setData(target.search_text, SEARCH_ROLE)
            item.setData(target.text_search, TEXT_SEARCH_ROLE)
            item.setData(binding_keys_display(target.binding_keys), BINDING_KEYS_ROLE)
            item.setEditable(False)
            self._search_model.appendRow(item)

    def _target_id_from_completer_index(self, index: QModelIndex) -> str:
        return target_id_from_completer_index(self._search_completer, self._search_model, self._search_proxy, index)

    def _first_matching_target_id(self) -> str:
        return first_matching_target_id(self._search_completer, self._search_model, self._search_proxy)

    def _on_search_edited(self, text: str) -> None:
        self._search_proxy.set_query(text)
        if text.strip():
            self._search_completer.complete()

    def _on_search_activated(self, index: QModelIndex) -> None:
        target_id = self._target_id_from_completer_index(index)
        if not target_id:
            return
        query = self._search_edit.text()
        focus_action_id = action_id_for_binding(self._current_bindings(), query) if query else None
        self._search_edit.clear()
        self._search_proxy.set_query("")
        self._search_completer.popup().hide()
        self._navigate_to_target(target_id, focus_action_id=focus_action_id)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self._search_edit and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                popup = self._search_completer.popup()
                if popup.isVisible():
                    idx = popup.currentIndex()
                    if not idx.isValid() and popup.model().rowCount() > 0:
                        idx = popup.model().index(0, 0)
                    if idx.isValid():
                        self._on_search_activated(idx)
                        return True
                target_id = self._first_matching_target_id()
                if target_id:
                    query = self._search_edit.text()
                    focus_action_id = action_id_for_binding(self._current_bindings(), query) if query else None
                    self._search_edit.clear()
                    self._search_proxy.set_query("")
                    self._navigate_to_target(target_id, focus_action_id=focus_action_id)
                    return True
        return super().eventFilter(watched, event)

    def _navigate_to_target(self, target_id: str, focus_action_id: str | None = None) -> None:
        row = self._row_widgets.get(target_id)
        focus_edit = self._row_focus_edits.get(target_id)
        if focus_action_id and focus_action_id in self._edits:
            focus_edit = self._edits[focus_action_id]
        if row is None:
            return

        target = next((t for t in self._targets if t.target_id == target_id), None)
        if target is not None:
            section = self._sections.get(target.category)
            if section is not None:
                section.expand()

        def _reveal() -> None:
            scroll_row_to_center(self._scroll, row)
            self._set_highlight(row)
            if focus_edit is not None:
                focus_edit.setFocus()

        QTimer.singleShot(50, _reveal)

    def _set_highlight(self, row: QFrame) -> None:
        self._clear_highlight()
        row.setProperty("highlighted", True)
        row.style().unpolish(row)
        row.style().polish(row)
        self._highlighted_row = row
        self._highlight_timer.start(HIGHLIGHT_MS)

    def _clear_highlight(self) -> None:
        if self._highlighted_row is None:
            return
        row = self._highlighted_row
        row.setProperty("highlighted", False)
        row.style().unpolish(row)
        row.style().polish(row)
        self._highlighted_row = None

    def _build_category_grid(self, _category: str, items: list[tuple[str, ShortcutEntry]]) -> QWidget:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        header_style = (
            f"color: {THEME.text_muted}; font-size: {THEME.font_size_xs}px; "
            f"font-weight: {THEME.weight_semibold};"
        )
        for col, label in enumerate(("Action", "Default", "Shortcut", "Step")):
            hdr = QLabel(label)
            hdr.setStyleSheet(header_style)
            grid.addWidget(hdr, 0, col)

        mono = f"color: {THEME.text_secondary}; font-family: Consolas, monospace;"
        for row, editor_row in enumerate(category_editor_rows(items), start=1):
            if isinstance(editor_row, EditorRowSlider):
                self._add_slider_row(grid, row, editor_row, mono)
            else:
                self._add_single_row(grid, row, editor_row, mono)

        return body

    def _make_row_frame(self, target_id: str, focus_edit: KeypadAwareKeySequenceEdit) -> QFrame:
        row = QFrame()
        row.setObjectName("shortcut_editor_row")
        self._row_widgets[target_id] = row
        self._row_focus_edits[target_id] = focus_edit
        return row

    def _add_single_row(self, grid: QGridLayout, row: int, editor_row: EditorRowSingle, mono: str) -> None:
        action_id = editor_row.action_id
        entry = editor_row.entry
        edit = self._make_key_edit(action_id, entry.default_key)
        row_frame = self._make_row_frame(action_id, edit)
        inner = QGridLayout(row_frame)
        inner.setContentsMargins(4, 4, 4, 4)
        inner.setHorizontalSpacing(12)
        inner.setVerticalSpacing(0)

        inner.addWidget(QLabel(entry.description), 0, 0)
        default_lbl = QLabel(entry.default_key or "—")
        default_lbl.setStyleSheet(mono)
        inner.addWidget(default_lbl, 0, 1)
        inner.addWidget(edit, 0, 2)
        inner.addWidget(QLabel("—"), 0, 3)
        grid.addWidget(row_frame, row, 0, 1, 4)

    def _add_slider_row(self, grid: QGridLayout, row: int, editor_row: EditorRowSlider, mono: str) -> None:
        group = editor_row.group
        inc_entry = REGISTRY[group.inc_action]
        dec_entry = REGISTRY[group.dec_action]

        inc_edit = self._make_key_edit(group.inc_action, inc_entry.default_key)
        dec_edit = self._make_key_edit(group.dec_action, dec_entry.default_key)
        row_frame = self._make_row_frame(group.id, inc_edit)
        inner = QGridLayout(row_frame)
        inner.setContentsMargins(4, 4, 4, 4)
        inner.setHorizontalSpacing(12)
        inner.setVerticalSpacing(0)

        inner.addWidget(QLabel(group.label), 0, 0)
        default_lbl = QLabel(_format_default_pair(inc_entry.default_key, dec_entry.default_key))
        default_lbl.setStyleSheet(mono)
        inner.addWidget(default_lbl, 0, 1)

        shortcuts = QHBoxLayout()
        shortcuts.setContentsMargins(0, 0, 0, 0)
        shortcuts.setSpacing(6)
        sep = QLabel("/")
        sep.setStyleSheet(f"color: {THEME.text_muted};")
        shortcuts.addWidget(inc_edit, 1)
        shortcuts.addWidget(sep)
        shortcuts.addWidget(dec_edit, 1)
        shortcuts_host = QWidget()
        shortcuts_host.setLayout(shortcuts)
        inner.addWidget(shortcuts_host, 0, 2)
        inner.addWidget(self._make_step_edit(group), 0, 3)
        grid.addWidget(row_frame, row, 0, 1, 4)

    def _make_key_edit(self, action_id: str, default_key: str) -> KeypadAwareKeySequenceEdit:
        edit = KeypadAwareKeySequenceEdit(QKeySequence(self._initial_bindings.get(action_id, default_key)))
        edit.setClearButtonEnabled(True)
        self._edits[action_id] = edit
        return edit

    def _make_step_edit(self, group) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(group.step_decimals)
        spin.setMinimum(10 ** -group.step_decimals)
        spin.setMaximum(10_000.0)
        spin.setSingleStep(group.default_step)
        if group.step_suffix:
            spin.setSuffix(group.step_suffix)
        spin.setValue(self._initial_slider_steps.get(group.id, group.default_step))
        spin.setToolTip("Amount applied per shortcut press")
        self._step_edits[group.id] = spin
        return spin

    def _reset_all(self) -> None:
        for action_id, key in default_bindings().items():
            if action_id in self._edits:
                self._edits[action_id].setKeySequence(QKeySequence(key))
        for group_id, value in default_slider_steps().items():
            if group_id in self._step_edits:
                self._step_edits[group_id].setValue(value)
        self._reload_search_model()

    def _portable(self, edit: KeypadAwareKeySequenceEdit) -> str:
        return edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText)

    def bindings(self) -> dict[str, str]:
        return {action_id: self._portable(edit) for action_id, edit in self._edits.items()}

    def slider_steps(self) -> dict[str, float]:
        return {group_id: float(spin.value()) for group_id, spin in self._step_edits.items()}

    def _save(self) -> None:
        seen: dict[str, str] = {}
        for action_id, edit in self._edits.items():
            key = self._portable(edit)
            if not key:
                continue
            other = seen.get(key)
            if other is not None:
                QMessageBox.warning(
                    self,
                    "Duplicate Shortcut",
                    f'"{key}" is assigned to both "{REGISTRY[other].description}" and "{REGISTRY[action_id].description}".',
                )
                return
            seen[key] = action_id

        for group_id, spin in self._step_edits.items():
            if spin.value() <= 0:
                QMessageBox.warning(self, "Invalid Step", f"Step size for {group_id} must be greater than zero.")
                return

        if self._session is not None:
            invert = self._invert_zoom_chk.isChecked()
            self._session.state.invert_zoom_scroll = invert
            self._session.repo.save_global_setting("invert_zoom_scroll", invert)

        self.accept()
