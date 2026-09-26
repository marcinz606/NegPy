import qtawesome as qta
from PyQt6.QtCore import QPoint, pyqtSignal
from PyQt6.QtGui import QActionGroup, QPainter
from PyQt6.QtWidgets import QMenu, QPushButton

from negpy.desktop.view.styles.templates import EditedDot, default_button_height, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME


class ChoiceButton(QPushButton):
    """One choice out of a few, as a button that opens a menu of them. A choice is
    (icon, label) or (icon, label, icon color); an empty icon name shows none. The button's dot marks the current choice as
    edited; the menu marks every edited choice."""

    currentChanged = pyqtSignal(int)

    def __init__(self, choices: tuple[tuple[str, ...], ...], tooltip: str, parent=None):
        super().__init__(parent)
        self._choices = choices
        self._edited = [False] * len(choices)
        self._index = 0
        self._chevron = qta.icon("fa5s.chevron-down", color=THEME.text_secondary, color_disabled=THEME.text_muted)
        self._chevron_size = THEME.font_size_small
        # Not setMenu: any ::menu-indicator rule then drops the button's padding.
        self.choice_menu = menu = QMenu(self)
        menu.setToolTipsVisible(True)
        group = QActionGroup(self)
        group.setExclusive(True)
        self._actions = []
        # No icons on the items: a checkable item with an icon draws no check mark.
        for i, (_icon, label, *_color) in enumerate(choices):
            action = menu.addAction(label)
            action.setCheckable(True)
            group.addAction(action)
            action.triggered.connect(lambda _checked=False, i=i: self.setCurrentIndex(i))
            self._actions.append(action)
        self.clicked.connect(lambda: menu.exec(self.mapToGlobal(self.rect().bottomLeft())))
        # The chevron sits clear of the edited dot in the top-right corner.
        self._chevron_inset = THEME.space_2xl
        self.setStyleSheet(
            f"QPushButton {{font-size: {THEME.font_size_base}px;"
            f" padding: 6px {self._chevron_inset + self._chevron_size + THEME.space_md}px 6px {THEME.space_xl}px;"
            " text-align: left;}"
        )
        self.setFixedHeight(default_button_height())
        self.setToolTip(wrap_tooltip(tooltip))
        self.plain_tooltip = tooltip
        self.edited_dot = EditedDot(self)
        self._show(0)

    def currentIndex(self) -> int:  # noqa: N802
        return self._index

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        if index == self._index:
            return
        self._show(index)
        self.currentChanged.emit(index)

    def set_edited(self, index: int, edited: bool) -> None:
        self._edited[index] = edited
        label = self._choices[index][1]
        # Text after a tab lands in the menu's right-aligned shortcut column.
        self._actions[index].setText(f"{label}\t•" if edited else label)
        self.edited_dot.set_active(self._edited[self._index])

    def _show(self, index: int) -> None:
        self._index = index
        icon_name, label, *color = self._choices[index]
        if icon_name:
            self.setIcon(qta.icon(icon_name, color=color[0] if color else THEME.text_primary, color_disabled=THEME.text_muted))
        self.setText(f" {label}" if icon_name else label)
        self._actions[index].setChecked(True)
        self.edited_dot.set_active(self._edited[index])

    def wheelEvent(self, event) -> None:  # noqa: N802
        step = -1 if event.angleDelta().y() > 0 else 1
        i = self._index + step
        while 0 <= i < len(self._actions) and not self._actions[i].isEnabled():
            i += step
        if 0 <= i < len(self._actions):
            self.setCurrentIndex(i)
        event.accept()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        s = self._chevron_size
        mode = self._chevron.Mode.Normal if self.isEnabled() else self._chevron.Mode.Disabled
        pix = self._chevron.pixmap(s, s, mode)
        QPainter(self).drawPixmap(QPoint(self.width() - self._chevron_inset - s, (self.height() - s) // 2), pix)
