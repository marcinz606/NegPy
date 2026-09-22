"""One tab's own header: a bar above the cards that reads how many of them hold something
other than their defaults, and acts on all of them at once. A tab holding one card has no
bar, since that card's header covers the same ground."""

from typing import Iterable

import qtawesome as qta
from PyQt6.QtCore import pyqtSignal

from negpy.desktop.view.confirm import confirm_reset_tab
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QPushButton

from negpy.desktop.view.styles.templates import HEADER_BUTTON_SIZE, HEADER_ICON_SIZE, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.kernel.system.text import count_of


class TabHeader(CollapsibleSection):
    apply_requested = pyqtSignal()

    def __init__(self, title: str, parent=None):
        super().__init__(
            title,
            icon=qta.icon("fa5s.layer-group", color=THEME.text_hint),
            collapsible=False,
            parent=parent,
        )
        self.content_area.setVisible(False)
        # A state line, not a name: the tab bar above already says which tab this is.
        self.title_label.setStyleSheet(f"font-size: {THEME.font_size_base}px; color: {THEME.text_secondary}; background: transparent;")
        self._sections: tuple = ()

        self.apply_btn = self._header_button("fa5s.film", f"Apply every {title} setting to the selected frames or the whole roll…")
        self.apply_btn.clicked.connect(self.apply_requested)
        self.cards_btn = self._header_button("fa5s.angle-double-up", "")
        self.cards_btn.clicked.connect(self._toggle_cards)
        index = self._header_row.indexOf(self.reset_btn) + 1
        for widget in (self.apply_btn, self.cards_btn):
            self._header_row.insertWidget(index, widget)
            index += 1

        self.reset_requested.connect(self._on_reset)
        self.refresh()

    def _header_button(self, icon_name: str, tooltip: str) -> QPushButton:
        """The header's own button size and look, the one reset and the scope pair use."""
        btn = QPushButton()
        btn.setIcon(qta.icon(icon_name, color=THEME.text_muted))
        btn.setFixedSize(HEADER_BUTTON_SIZE, HEADER_BUTTON_SIZE)
        btn.setIconSize(QSize(HEADER_ICON_SIZE, HEADER_ICON_SIZE))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setObjectName("collapsible_reset_btn")
        btn.setToolTip(wrap_tooltip(tooltip))
        return btn

    def bind(self, sections: Iterable[CollapsibleSection]) -> None:
        self._sections = tuple(sections)
        self.refresh()

    def refresh(self) -> None:
        live = self._live_sections()
        edited = [s for s in live if s.modified_count]
        self.set_modified(sum(s.modified_count for s in edited))
        self.title_label.setText(f"{len(edited)} of {count_of(len(live), 'card')} edited" if edited else "No cards edited")
        self._sync_cards_button()

    def _live_sections(self) -> list:
        """A card the mode has retired (Filtration on a B&W frame) is not this tab's to
        count, reset or carry."""
        return [s for s in self._sections if not s.isHidden()]

    def _on_reset(self) -> None:
        touched = [s for s in self._live_sections() if s.modified_count]
        if not touched or not confirm_reset_tab(self, self._title_text, len(touched)):
            return
        for section in touched:
            section.reset_requested.emit()

    def _toggle_cards(self) -> None:
        expand = not self._any_expanded()
        for section in self._live_sections():
            section.set_expanded(expand)
        self._sync_cards_button()

    def _any_expanded(self) -> bool:
        return any(s.collapsible and s.toggle_button.isChecked() for s in self._live_sections())

    def _sync_cards_button(self) -> None:
        collapse = self._any_expanded()
        self.cards_btn.setIcon(qta.icon("fa5s.angle-double-up" if collapse else "fa5s.angle-double-down", color=THEME.text_muted))
        self.cards_btn.setToolTip(wrap_tooltip(f"{'Collapse' if collapse else 'Expand'} every card on this tab"))
