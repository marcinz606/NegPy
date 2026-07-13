from PyQt6.QtCore import QEvent, QModelIndex, Qt, QTimer
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QCompleter,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
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
    categories_in_order,
    category_editor_rows,
    slider_step_for,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.shortcut_search_line_edit import ShortcutSearchLineEdit


def _format_key_pair(inc_key: str, dec_key: str) -> str:
    inc = inc_key or "—"
    dec = dec_key or "—"
    return f"{inc} / {dec}"


def _format_step_value(group, value: float) -> str:
    if group.step_decimals == 0:
        text = str(int(value)) if value == int(value) else str(value)
    else:
        text = f"{value:.{group.step_decimals}f}".rstrip("0").rstrip(".")
    suffix = group.step_suffix or ""
    return f"{text}{suffix}" if text else "—"


class ShortcutsOverlay(QDialog):
    """Modal keyboard shortcut reference, opened with '?'."""

    def __init__(self, shortcut_manager, parent=None):
        super().__init__(parent)
        self._shortcut_manager = shortcut_manager
        self._bindings = dict(shortcut_manager.bindings)
        self._slider_steps = dict(shortcut_manager.slider_steps)
        self._sections: dict[str, CollapsibleSection] = {}
        self._row_widgets: dict[str, QFrame] = {}
        self._targets: list[ShortcutEditorTarget] = build_shortcut_editor_targets(self._bindings)
        self._highlighted_row: QFrame | None = None
        self._highlight_timer = QTimer(self)
        self._highlight_timer.setSingleShot(True)
        self._highlight_timer.timeout.connect(self._clear_highlight)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(820, 720)
        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        intro = QLabel(
            "Current keyboard shortcuts and slider step sizes. "
            "Search by name or press a shortcut to filter results, then choose or press Enter. "
            "Open Customize to change bindings."
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
            section.set_content(self._build_category_grid(items))
            self._sections[category] = section
            sections_layout.addWidget(section)

        sections_layout.addStretch()
        self._scroll.setWidget(container)
        root.addWidget(self._scroll, stretch=1)

        actions = QHBoxLayout()
        customize_btn = QPushButton("Customize")
        customize_btn.clicked.connect(self._customize)
        actions.addWidget(customize_btn)
        actions.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setProperty("primary", True)
        close_btn.clicked.connect(self.accept)
        actions.addWidget(close_btn)
        root.addLayout(actions)

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

    def _known_bindings(self) -> frozenset[str]:
        return frozenset(key for key in self._bindings.values() if key)

    def _reload_search_model(self) -> None:
        self._search_model.clear()
        self._targets = build_shortcut_editor_targets(self._bindings)
        for target in self._targets:
            item = QStandardItem(f"{target.label}  ·  {target.category}")
            item.setData(target.target_id, TARGET_ROLE)
            item.setData(target.search_text, SEARCH_ROLE)
            item.setData(target.text_search, TEXT_SEARCH_ROLE)
            item.setData(binding_keys_display(target.binding_keys), BINDING_KEYS_ROLE)
            item.setEditable(False)
            self._search_model.appendRow(item)

    def _on_search_edited(self, text: str) -> None:
        self._search_proxy.set_query(text)
        if text.strip():
            self._search_completer.complete()

    def _on_search_activated(self, index: QModelIndex) -> None:
        target_id = target_id_from_completer_index(self._search_completer, self._search_model, self._search_proxy, index)
        if not target_id:
            return
        self._search_edit.clear()
        self._search_proxy.set_query("")
        self._search_completer.popup().hide()
        self._navigate_to_target(target_id)

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
                target_id = first_matching_target_id(self._search_completer, self._search_model, self._search_proxy)
                if target_id:
                    self._search_edit.clear()
                    self._search_proxy.set_query("")
                    self._navigate_to_target(target_id)
                    return True
        return super().eventFilter(watched, event)

    def _navigate_to_target(self, target_id: str) -> None:
        row = self._row_widgets.get(target_id)
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
            row.setFocus()

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

    def _customize(self) -> None:
        if self._shortcut_manager.open_editor(self):
            self.accept()

    def _make_row_frame(self, target_id: str) -> QFrame:
        row = QFrame()
        row.setObjectName("shortcut_editor_row")
        row.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._row_widgets[target_id] = row
        return row

    def _build_category_grid(self, items: list) -> QWidget:
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

    def _keycap(self, text: str) -> QLabel:
        lbl = QLabel(text or "—")
        lbl.setStyleSheet(f"""
            color: {THEME.text_primary};
            background-color: {THEME.bg_header};
            border: 1px solid {THEME.border_primary};
            border-radius: 3px;
            font-family: Consolas, monospace;
            font-size: 11px;
            padding: 2px 6px;
        """)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return lbl

    def _mono_label(self, text: str, mono: str) -> QLabel:
        lbl = QLabel(text or "—")
        lbl.setStyleSheet(mono)
        return lbl

    def _add_single_row(
        self,
        grid: QGridLayout,
        row: int,
        editor_row: EditorRowSingle,
        mono: str,
    ) -> None:
        entry = editor_row.entry
        row_frame = self._make_row_frame(editor_row.action_id)
        inner = QGridLayout(row_frame)
        inner.setContentsMargins(4, 4, 4, 4)
        inner.setHorizontalSpacing(12)
        inner.setVerticalSpacing(0)

        inner.addWidget(QLabel(entry.description), 0, 0)
        inner.addWidget(self._mono_label(entry.default_key, mono), 0, 1)
        inner.addWidget(self._keycap(self._bindings.get(editor_row.action_id, "")), 0, 2)
        inner.addWidget(QLabel("—"), 0, 3)
        grid.addWidget(row_frame, row, 0, 1, 4)

    def _add_slider_row(
        self,
        grid: QGridLayout,
        row: int,
        editor_row: EditorRowSlider,
        mono: str,
    ) -> None:
        group = editor_row.group
        inc_entry = REGISTRY[group.inc_action]
        dec_entry = REGISTRY[group.dec_action]

        row_frame = self._make_row_frame(group.id)
        inner = QGridLayout(row_frame)
        inner.setContentsMargins(4, 4, 4, 4)
        inner.setHorizontalSpacing(12)
        inner.setVerticalSpacing(0)

        inner.addWidget(QLabel(group.label), 0, 0)
        inner.addWidget(
            self._mono_label(_format_key_pair(inc_entry.default_key, dec_entry.default_key), mono),
            0,
            1,
        )

        shortcuts = QHBoxLayout()
        shortcuts.setContentsMargins(0, 0, 0, 0)
        shortcuts.setSpacing(6)
        shortcuts.addWidget(self._keycap(self._bindings.get(group.inc_action, "")), 1)
        sep = QLabel("/")
        sep.setStyleSheet(f"color: {THEME.text_muted};")
        shortcuts.addWidget(sep)
        shortcuts.addWidget(self._keycap(self._bindings.get(group.dec_action, "")), 1)
        shortcuts_host = QWidget()
        shortcuts_host.setLayout(shortcuts)
        inner.addWidget(shortcuts_host, 0, 2)

        step_value = slider_step_for(group.id, self._slider_steps)
        inner.addWidget(QLabel(_format_step_value(group, step_value)), 0, 3)
        grid.addWidget(row_frame, row, 0, 1, 4)
