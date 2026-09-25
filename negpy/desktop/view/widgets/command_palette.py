"""Find: one box that reaches every slider, card and action by name."""

from dataclasses import dataclass
from typing import Callable

import qtawesome as qta
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from negpy.desktop.view.shortcut_editor_search import HIGHLIGHT_MS
from negpy.desktop.view.shortcut_registry import REGISTRY, display_key, key_for, label_with_shortcut
from negpy.desktop.view.slider_shortcut_groups import SLIDER_GROUPS
from negpy.desktop.view.styles.templates import hint_label
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection, hidden_by_gating
from negpy.desktop.view.widgets.sliders import CompactSlider, align_slider_columns, clone_slider
from negpy.desktop.view.widgets.tab_header import TabHeader

# Another editor's word for a job, and the NegPy names that do it, closest first.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "exposure": ("Print Density",),
    "brightness": ("Print Density",),
    "contrast": ("ISO-R Grade", "Contrast Mask"),
    "white balance": ("Filtration", "Temperature"),
    "wb": ("Filtration", "Temperature"),
    "tint": ("Magenta",),
    "kelvin": ("Temperature",),
    "blacks": ("Black Point", "Toe"),
    "whites": ("White Point", "Shoulder"),
    "highlights": ("Highlights Density", "Shoulder"),
    "shadows": ("Shadows Density", "Toe"),
    "levels": ("White Point", "Black Point"),
    "curve": ("Tone",),
    "saturation": ("Chroma", "Dye Separation"),
    "vibrance": ("Chroma", "Skin Protection"),
    "clarity": ("CLAHE", "Contrast Mask"),
    "local contrast": ("CLAHE",),
    "sharpness": ("Sharpening",),
    "noise": ("Chroma Denoise",),
    "orange mask": ("Cast Removal",),
    "invert": ("Film Mode",),
    "straighten": ("Fine Rotation",),
    "perspective": ("Tilt", "Swing"),
    "keystone": ("Tilt", "Swing"),
    "vignette": ("Finishing",),
    "border": ("Finishing",),
    "dust": ("Retouch",),
    "spots": ("Retouch",),
    "split toning": ("Toning",),
    "local adjustments": ("Dodge & Burn",),
    "lens correction": ("Optics",),
    "demosaic": ("Raw Decode",),
    "bloom": ("Glow",),
}

_KIND_ORDER = {"slider": 0, "card": 1, "action": 2}
_MAX_ROWS = 30
_VISIBLE_ROWS = 12


@dataclass(frozen=True, eq=False)
class Entry:
    kind: str  # "slider" | "card" | "action"
    name: str
    where: str
    target: object  # the widget, or the action id
    words: str


def make_entry(kind: str, name: str, where: str, target: object) -> Entry:
    aliases = [word for word, names in SYNONYMS.items() if name in names]
    return Entry(kind, name, where, target, " ".join([name, where, *aliases]).casefold())


def rank(entries: list[Entry], query: str) -> list[Entry]:
    """Entries holding every query word: synonyms first, then exact, prefix, word prefix, substring."""
    q = " ".join(query.casefold().split())
    if not q:
        return []
    synonym_rank: dict[str, int] = {}
    for word, names in SYNONYMS.items():
        if word == q or (len(q) >= 3 and word.startswith(q)):
            for i, name in enumerate(names):
                synonym_rank[name] = min(i, synonym_rank.get(name, i))
    scored = []
    for entry in entries:
        if not all(token in entry.words for token in q.split()):
            continue
        name = entry.name.casefold()
        if entry.name in synonym_rank:
            score = (0, synonym_rank[entry.name])
        elif name == q:
            score = (1, 0)
        elif name.startswith(q):
            score = (2, 0)
        elif any(word.startswith(q) for word in name.split()):
            score = (3, 0)
        elif q in name:
            score = (4, 0)
        else:
            score = (5, 0)
        scored.append((score, _KIND_ORDER[entry.kind], entry.name, entry))
    scored.sort(key=lambda item: item[:3])
    return [item[3] for item in scored]


def _card_of(widget: QWidget) -> CollapsibleSection | None:
    parent = widget.parentWidget()
    while parent is not None and not isinstance(parent, CollapsibleSection):
        parent = parent.parentWidget()
    return parent


def build_index(window) -> list[Entry]:
    """Live cards and sliders in the controls panel, and every action with a handler."""
    panel = window.right_panel
    entries = []
    for section in panel.findChildren(CollapsibleSection):
        if isinstance(section, TabHeader) or section.isHidden():
            continue
        entries.append(make_entry("card", section.title_label.text(), " › ".join(panel.tab_path(section)[-1:]), section))
    for slider in panel.findChildren(CompactSlider):
        if panel.favourites_sidebar.isAncestorOf(slider) or hidden_by_gating(slider):
            continue
        card = _card_of(slider)
        where = panel.tab_path(slider)[-1:] + ([card.title_label.text()] if card is not None else [])
        entries.append(make_entry("slider", slider.label.text(), " › ".join(where), slider))
    nudges = {group.inc_action for group in SLIDER_GROUPS} | {group.dec_action for group in SLIDER_GROUPS}
    for action_id, spec in REGISTRY.items():
        if action_id in nudges or action_id == "command_palette" or window.shortcut_manager.action_for(action_id) is None:
            continue
        entries.append(make_entry("action", spec.description, display_key(key_for(action_id)), action_id))
    return entries


def flash_widget(widget: QWidget) -> None:
    frame = QFrame(widget)
    frame.setObjectName("reveal_flash")
    frame.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    frame.setGeometry(widget.rect())
    frame.show()
    frame.raise_()
    QTimer.singleShot(HIGHLIGHT_MS, frame.deleteLater)


def open_entry(window, entry: Entry) -> None:
    if entry.kind == "action":
        handler = window.shortcut_manager.action_for(entry.target)
        if handler is not None:
            handler()
        return
    if not window.drawer.isVisible():
        window.toggle_controls_dock()
    window.right_panel.reveal_widget(entry.target)
    if entry.kind == "slider":
        entry.target.slider.setFocus()
    flash = entry.target.toggle_button if entry.kind == "card" else entry.target
    # Deferred: a card opened just now lays its widgets out on the next pass.
    QTimer.singleShot(0, lambda: flash_widget(flash))


class FindField(QWidget):
    """The controls panel's title bar: a box that opens Find; its margin still drags the panel."""

    def __init__(self, open_palette: Callable[[], None]):
        super().__init__()
        self._open = open_palette
        layout = QHBoxLayout(self)
        layout.setContentsMargins(THEME.space_lg, THEME.space_sm, THEME.space_lg, THEME.space_sm)
        self.box = QLineEdit()
        self.box.setReadOnly(True)
        self.box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.box.setCursor(Qt.CursorShape.PointingHandCursor)
        search = self.box.addAction(qta.icon("fa5s.search", color=THEME.text_secondary), QLineEdit.ActionPosition.LeadingPosition)
        search.triggered.connect(self._open)
        self.box.installEventFilter(self)
        layout.addWidget(self.box)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.box.setPlaceholderText(label_with_shortcut("Find a control, card or action…", "command_palette"))

    def eventFilter(self, obj, event) -> bool:
        if obj is self.box and event.type() == QEvent.Type.MouseButtonPress:
            self._open()
            return True
        return super().eventFilter(obj, event)


class CommandPalette(QDialog):
    def __init__(self, window):
        super().__init__(window, Qt.WindowType.Popup)
        self.setObjectName("command_palette")
        self._window = window
        self._entries = build_index(window)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_lg, THEME.space_lg, THEME.space_lg, THEME.space_lg)
        root.setSpacing(THEME.space_md)

        self.query = QLineEdit()
        self.query.setPlaceholderText("Find a control, card or action…")
        self.query.setClearButtonEnabled(True)
        self.query.textChanged.connect(self._refresh)
        self.query.installEventFilter(self)
        root.addWidget(self.query)

        self.results = QListWidget()
        self.results.setObjectName("palette_results")
        self.results.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.results.itemClicked.connect(self._activate)
        root.addWidget(self.results, 1)

        self.empty = hint_label("Type a slider, card or action, or another editor's word for it: contrast, white balance.")
        root.addWidget(self.empty)

        self.setFixedWidth(560)
        self._refresh("")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        host = self._window.geometry()
        self.move(host.x() + (host.width() - self.width()) // 2, host.y() + host.height() // 8)
        self.query.setFocus()

    def _refresh(self, text: str) -> None:
        self.results.clear()
        hits = rank(self._entries, text)[:_MAX_ROWS]
        wheres = []
        for entry in hits:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, entry)
            row, where = self._row(entry)
            item.setSizeHint(row.sizeHint())
            self.results.addItem(item)
            self.results.setItemWidget(item, row)
            wheres.append(where)
        align_slider_columns(self.results)
        for where in wheres:
            where.setMinimumWidth(max(w.sizeHint().width() for w in wheres))
        if hits:
            self.results.setCurrentRow(0)
            shown = hits[:_VISIBLE_ROWS]
            self.results.setFixedHeight(sum(self.results.sizeHintForRow(i) for i in range(len(shown))) + 2 * self.results.frameWidth())
        self.results.setVisible(bool(hits))
        self.empty.setVisible(not hits)
        self.adjustSize()

    def _row(self, entry: Entry) -> tuple[QWidget, QLabel]:
        row = QWidget()
        row.setObjectName("palette_row")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(THEME.space_md, THEME.space_xs, THEME.space_md, THEME.space_xs)
        layout.setSpacing(THEME.space_lg)
        if entry.kind == "slider":
            source = entry.target
            mirror = clone_slider(source)
            mirror.setValue(source.value())
            mirror.setEnabled(source.isEnabled())
            mirror.valueChanged.connect(lambda v, s=source: s.mirror_value(v, commit=False))
            mirror.valueCommitted.connect(lambda v, s=source: s.mirror_value(v, commit=True))
            layout.addWidget(mirror, 1)
        else:
            layout.addWidget(QLabel(entry.name), 1)
        where = hint_label(entry.where)
        where.setWordWrap(False)
        layout.addWidget(where)
        return row, where

    def eventFilter(self, obj, event) -> bool:
        if obj is self.query and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            count = self.results.count()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up) and count:
                step = 1 if key == Qt.Key.Key_Down else -1
                self.results.setCurrentRow((self.results.currentRow() + step) % count)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                item = self.results.currentItem()
                if item is not None:
                    self._activate(item)
                return True
        return super().eventFilter(obj, event)

    def _activate(self, item: QListWidgetItem) -> None:
        entry = item.data(Qt.ItemDataRole.UserRole)
        self.accept()
        # Once the popup has closed, so a dialog the action opens is not parented to it.
        QTimer.singleShot(0, lambda: open_entry(self._window, entry))
