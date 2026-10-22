from typing import Sequence

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QScrollArea, QStackedWidget, QVBoxLayout, QWidget

from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.overflow_bar import OverflowBar
from negpy.desktop.view.widgets.panel_tab import PanelTab


def tab_page(*widgets: QWidget) -> QWidget:
    """A sub-tab's page: the widgets in a column, top-aligned."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(THEME.space_lg)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch()
    return page


class SubTabs(QWidget):
    """A tab's own strip of sub-tabs over a stack of scrolling pages, the Frame tab's look: named
    tabs first, then icon-only ones after a divider. The tab in front is kept under setting."""

    current_changed = pyqtSignal(str)

    def __init__(self, repo, setting: str, pages: Sequence[tuple[str, str, str, QWidget]], icon_only: Sequence[str] = ()):
        super().__init__()
        self._repo = repo
        self._setting = setting
        names = [title for key, title, _icon, _page in pages if key not in icon_only]
        self.bar = OverflowBar(height=38, min_item=PanelTab.min_width(names, stacked=False))
        self.stack = QStackedWidget()
        self.stack.setContentsMargins(0, THEME.space_md, 0, 0)
        self.keys: list[str] = []
        self.buttons: list[PanelTab] = []
        for i, (key, title, icon, page) in enumerate(pages):
            btn = PanelTab(icon, title, icon_only=key in icon_only, divider=bool(icon_only) and key == icon_only[0])
            btn.setToolTip(title)
            btn.clicked.connect(lambda _checked=False, idx=i: self.switch(idx))
            self.bar.add_button(btn, title)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
            self.keys.append(key)
            self.buttons.append(btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(THEME.space_sm)
        layout.addWidget(self.bar)
        layout.addWidget(self.stack, 1)

        saved = repo.get_global_setting(setting, self.keys[0])
        self._current = self.keys.index(saved) if saved in self.keys else 0
        self.switch(self._current)

    def switch(self, index: int) -> None:
        self._current = index
        self._repo.save_global_setting(self._setting, self.keys[index])
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)
        self.bar.set_pinned(index)
        self.current_changed.emit(self.keys[index])

    def show_tab(self, key: str) -> None:
        self.switch(self.keys.index(key))

    def current_key(self) -> str:
        return self.keys[self._current]

    def button(self, key: str) -> PanelTab:
        return self.buttons[self.keys.index(key)]
