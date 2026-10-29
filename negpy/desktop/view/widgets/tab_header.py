"""One tab's own header: a bar above the cards that reads how many of them hold something
other than their defaults, and acts on all of them at once. A tab holding one card has no
bar, since that card's header covers the same ground."""

from negpy.kernel.system.i18n import tr

from typing import Iterable

import qtawesome as qta
from PyQt6.QtCore import QEvent, QObject, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QAbstractButton, QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from negpy.desktop.view.confirm import confirm_reset_tab
from negpy.desktop.view.styles.templates import header_button, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection, roll_revert_icon
from negpy.kernel.system.text import count_of


class TabHeader(CollapsibleSection):
    apply_requested = pyqtSignal()

    def __init__(self, title: str, parent=None):
        super().__init__(title, collapsible=False, parent=parent)
        self.content_area.setVisible(False)
        # A flat strip on the panel ground (QSS [bar] rule), so it never reads as a card.
        self.toggle_button.setProperty("bar", "true")
        self._header_row.setContentsMargins(0, THEME.space_sm, 0, THEME.space_sm)
        self._header_row.setSpacing(THEME.space_lg)
        # A state line, not a name: the tab bar above already says which tab this is.
        self.title_label.setStyleSheet(f"font-size: {THEME.font_size_base}px; color: {THEME.text_secondary}; background: transparent;")
        self._sections: tuple = ()

        self.apply_btn = header_button(
            qta.icon("fa5s.film", color=THEME.text_hint, color_disabled=THEME.text_muted),
            tr("Apply every {title} setting to the selected frames or the whole roll…", title=title),
        )
        self.apply_btn.clicked.connect(self.apply_requested)
        self.cards_btn = header_button(qta.icon("fa5s.angle-double-up", color=THEME.text_hint), "")
        self.cards_btn.clicked.connect(self._toggle_cards)
        self.roll_revert_btn.setIcon(roll_revert_icon(THEME.text_hint, cutout=THEME.bg_input))
        for widget in (self.reset_btn, self.roll_revert_btn):
            self._header_row.removeWidget(widget)
        self._lead_track = _Track()
        self._set_lead_visible(False)
        self.edited_dot = QLabel()
        self.edited_dot.setPixmap(qta.icon("fa5s.circle", color=THEME.mark_edited).pixmap(6, 6))
        self._header_row.insertWidget(0, self._lead_track)
        self._header_row.insertWidget(1, self.edited_dot)
        self._header_row.addWidget(_Track(self.reset_btn, self.roll_revert_btn, self.apply_btn, self.cards_btn))

        self.reset_requested.connect(self._on_reset)
        self.refresh()

    def add_lead_toggle(self, icon: QIcon, tooltip: str) -> QPushButton:
        """A tool that works across the tab's cards, in the track before the state line."""
        btn = header_button(icon, tooltip)
        btn.setCheckable(True)
        self._lead_track.row.addWidget(btn)
        self._set_lead_visible(True)
        return btn

    def add_mirror(self, source: QAbstractButton, icon: QIcon | None = None) -> QPushButton:
        """A lead button that clicks `source`, a card's own button, and follows its check,
        its enabled state and whether the card hides it. `icon` replaces an icon drawn for a
        labeled button, which is always lit."""
        btn = self.add_lead_toggle(icon or source.icon(), source.toolTip())
        btn.plain_tooltip = getattr(source, "plain_tooltip", "")
        btn.setCheckable(source.isCheckable())
        btn.clicked.connect(lambda: (source.click(), btn.setChecked(source.isChecked())))
        source.toggled.connect(btn.setChecked)
        _MirrorSync(source, btn, self._sync_lead_track)
        return btn

    def _sync_lead_track(self) -> None:
        self._set_lead_visible(any(not b.isHidden() for b in self._lead_track.findChildren(QPushButton)))

    def _set_lead_visible(self, visible: bool) -> None:
        """The lead track sits on the panel edge like the cards' frames; without it the state line takes the cards' inset."""
        self._lead_track.setVisible(visible)
        self._header_row.setContentsMargins(0 if visible else THEME.space_xl, THEME.space_sm, 0, THEME.space_sm)

    def bind(self, sections: Iterable[CollapsibleSection]) -> None:
        self._sections = tuple(sections)
        self.refresh()

    def refresh(self) -> None:
        live = self._live_sections()
        edited = [s for s in live if s.modified_count]
        self.set_modified(sum(s.modified_count for s in edited))
        self.set_roll_revert(any(s.roll_revert_available for s in live))
        self.edited_dot.setVisible(bool(edited))
        self.title_label.setText(
            tr("{value} of {value1} edited", value=len(edited), value1=count_of(len(live), "card")) if edited else tr("No cards edited")
        )
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
        self.cards_btn.setIcon(qta.icon("fa5s.angle-double-up" if collapse else "fa5s.angle-double-down", color=THEME.text_hint))
        self.cards_btn.setToolTip(wrap_tooltip(tr("Collapse every card on this tab") if collapse else tr("Expand every card on this tab")))


class _MirrorSync(QObject):
    """Copies a source button's explicit hide and enabled state to its mirror. ShowToParent and
    HideToParent arrive even while a collapsed card keeps the source off screen."""

    _EVENTS = (QEvent.Type.ShowToParent, QEvent.Type.HideToParent, QEvent.Type.EnabledChange)

    def __init__(self, source: QAbstractButton, mirror: QPushButton, on_sync):
        super().__init__(mirror)
        self._source, self._mirror, self._on_sync = source, mirror, on_sync
        source.installEventFilter(self)
        self._sync()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() in self._EVENTS:
            self._sync()
        return False

    def _sync(self) -> None:
        self._mirror.setVisible(not self._source.isHidden())
        self._mirror.setEnabled(self._source.isEnabled())
        self._mirror.setChecked(self._source.isChecked())
        self._on_sync()


class _Track(QFrame):
    """A group of header buttons on one boxed track, as the canvas bar groups its buttons."""

    def __init__(self, *buttons: QWidget):
        super().__init__()
        self.setObjectName("header_track")
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(1, 1, 1, 1)
        self.row.setSpacing(0)
        for button in buttons:
            self.row.addWidget(button)
