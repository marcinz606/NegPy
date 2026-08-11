import qtawesome as qta
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QMenu, QToolButton, QWidget

from negpy.desktop.view.styles.theme import THEME


class OverflowBar(QWidget):
    """A row of buttons that spills what doesn't fit into a » menu, so its minimum width is
    one button rather than the sum of them. Without this, every button added to a panel
    widened that panel permanently.

    Geometry is placed by hand: a QHBoxLayout's minimum is the sum of its children's, which is
    the floor being removed here. QToolBar has a native extension menu but it is unusable for
    widget-based bars — items added with addWidget() become QWidgetActions that its extension
    popup cannot host, leaving the » button with an empty menu.

    Two shapes, per `tile`:
      tile=True  — equal-width tabs filling the bar (the right panel's tab strip)
      tile=False — natural widths packed left, separators allowed (the file browser toolbar)
    """

    OVERFLOW_W = 30

    def __init__(self, *, tile: bool = False, height: int = 38, spacing: int = 0, min_item: int = 36, parent=None):
        super().__init__(parent)
        self._tile = tile
        self._height = height
        self._spacing = spacing
        self._min_item = min_item
        self._items: list[tuple[QWidget, str | None]] = []
        self._button_items: list[int] = []
        self._pinned_item = -1
        self._hidden: list[int] = []
        self.setFixedHeight(height)

        self.overflow_btn = QToolButton(self)
        self.overflow_btn.setIcon(qta.icon("fa5s.angle-double-right", color=THEME.text_secondary))
        self.overflow_btn.setIconSize(QSize(16, 16))
        self.overflow_btn.setToolTip("More")
        self.overflow_btn.setFixedHeight(height)
        self.overflow_btn.clicked.connect(self._show_overflow_menu)
        self.overflow_btn.hide()

    def add_button(self, btn: QWidget, label: str) -> None:
        btn.setParent(self)
        self._button_items.append(len(self._items))
        self._items.append((btn, label))

    def add_separator(self, sep: QWidget) -> None:
        """Decoration: never listed in the overflow menu, and dropped when it would trail."""
        sep.setParent(self)
        self._items.append((sep, None))

    @property
    def buttons(self) -> list[QWidget]:
        return [self._items[i][0] for i in self._button_items]

    def set_pinned(self, button_index: int) -> None:
        """The button that must stay visible (the active tab). -1 for none."""
        self._pinned_item = self._button_items[button_index] if button_index >= 0 else -1
        self._relayout()

    @staticmethod
    def _item_width(widget) -> int:
        """The width the item will actually occupy.

        Not `sizeHint().width()` alone: a hint is only a preference, `setFixedWidth` does
        not change it, and a bare QFrame has no hint at all and reports -1. Measuring by
        the hint therefore over-reserves for anything constrained -- the toolbar's 1 px
        separators report their frame's natural width -- while `setGeometry` clamps the
        widget back to its real size, which both leaves a gap and overflows the bar
        sooner than it needs to.
        """
        return min(max(widget.sizeHint().width(), widget.minimumWidth()), widget.maximumWidth())

    def minimumSizeHint(self) -> QSize:
        return QSize(self._min_item + self.OVERFLOW_W, self._height)

    def sizeHint(self) -> QSize:
        if self._tile:
            return QSize(max(1, len(self._items)) * self._min_item, self._height)
        widths = [self._item_width(w) for w, _ in self._items]
        return QSize(sum(widths) + self._spacing * max(0, len(widths) - 1), self._height)

    def resizeEvent(self, event) -> None:
        self._relayout()
        super().resizeEvent(event)

    def _visible_items(self, avail: int) -> list[int]:
        count = len(self._items)
        if self._tile:
            if avail >= count * self._min_item:
                return list(range(count))
            fits = max(1, (avail - self.OVERFLOW_W) // self._min_item)
            if 0 <= self._pinned_item and self._pinned_item >= fits:
                return list(range(fits - 1)) + [self._pinned_item]
            return list(range(fits))

        widths = [self._item_width(w) for w, _ in self._items]
        if sum(widths) + self._spacing * max(0, count - 1) <= avail:
            return list(range(count))

        def fit(budget: int) -> list[int]:
            visible: list[int] = []
            used = 0
            for i, width in enumerate(widths):
                step = width + (self._spacing if visible else 0)
                if used + step > budget:
                    break
                used += step
                visible.append(i)
            # A trailing separator is a divider with nothing after it to divide.
            while visible and self._items[visible[-1]][1] is None:
                visible.pop()
            return visible

        # Try the full width first. Only labelled items reach the menu, so if nothing but
        # separators spills there is no chevron to show — and reserving room for one would
        # push a button out and *create* the chevron it was making room for.
        full = fit(avail)
        if all(self._items[i][1] is None for i in range(count) if i not in full):
            return full
        return fit(avail - self.OVERFLOW_W)

    def _relayout(self) -> None:
        if not self._items:
            return
        avail = self.width()
        visible = self._visible_items(avail)
        self._hidden = [i for i in range(len(self._items)) if i not in visible]

        if self._tile:
            strip = avail - (self.OVERFLOW_W if self._hidden else 0)
            share = max(1, len(visible))
            x = 0
            for slot, i in enumerate(visible):
                width = strip // share + (1 if slot < strip % share else 0)
                self._items[i][0].setGeometry(x, 0, width, self._height)
                x += width
        else:
            x = 0
            for i in visible:
                width = self._item_width(self._items[i][0])
                self._items[i][0].setGeometry(x, 0, width, self._height)
                x += width + self._spacing

        for i in visible:
            self._items[i][0].show()
        for i in self._hidden:
            self._items[i][0].hide()

        if any(self._items[i][1] is not None for i in self._hidden):
            self.overflow_btn.setGeometry(max(0, avail - self.OVERFLOW_W), 0, self.OVERFLOW_W, self._height)
            self.overflow_btn.show()
        else:
            self.overflow_btn.hide()

    def build_overflow_menu(self) -> QMenu:
        menu = QMenu(self)
        for i in self._hidden:
            widget, label = self._items[i]
            if label is None:
                continue
            action = menu.addAction(widget.icon(), label)
            action.setEnabled(widget.isEnabled())
            if widget.isCheckable():
                action.setCheckable(True)
                action.setChecked(widget.isChecked())
            action.triggered.connect(lambda _checked=False, w=widget: w.click())
        return menu

    def _show_overflow_menu(self) -> None:
        menu = self.build_overflow_menu()
        menu.exec(self.overflow_btn.mapToGlobal(self.overflow_btn.rect().bottomLeft()))
