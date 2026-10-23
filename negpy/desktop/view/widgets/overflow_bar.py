from negpy.kernel.system.i18n import tr
import qtawesome as qta
from PyQt6.QtCore import QRectF, QSize
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QMenu, QToolButton, QWidget

from negpy.desktop.view.styles.theme import THEME


class OverflowBar(QWidget):
    """A row of equal-width tabs filling the bar (the right panel's tab strips) that spills what
    doesn't fit into a » menu, so its minimum width is one tab rather than the sum of them. A
    tab strip is a choice of one, so it sits on a segmented control's track. A
    fixed-width tab keeps its width; the rest share the room. Without this, every tab added to a
    panel widened that panel permanently.

    Geometry is placed by hand: a QHBoxLayout's minimum is the sum of its children's, which is
    the floor being removed here. QToolBar has a native extension menu but it is unusable for
    widget-based bars — items added with addWidget() become QWidgetActions that its extension
    popup cannot host, leaving the » button with an empty menu.
    """

    OVERFLOW_W = 30

    def __init__(self, *, height: int = 38, min_item: int = 36, parent=None):
        super().__init__(parent)
        self._height = height
        self._min_item = min_item
        self._items: list[tuple[QWidget, str]] = []
        self._pinned = -1
        self._hidden: list[int] = []
        self.setFixedHeight(height)

        self.overflow_btn = QToolButton(self)
        self.overflow_btn.setIcon(qta.icon("fa5s.angle-double-right", color=THEME.text_secondary))
        self.overflow_btn.setIconSize(QSize(16, 16))
        self.overflow_btn.setToolTip(tr("More"))
        self.overflow_btn.setFixedHeight(height)
        self.overflow_btn.clicked.connect(self._show_overflow_menu)
        self.overflow_btn.hide()

    def add_button(self, btn: QWidget, label: str) -> None:
        btn.setParent(self)
        self._items.append((btn, label))

    @property
    def buttons(self) -> list[QWidget]:
        return [widget for widget, _label in self._items]

    def set_pinned(self, button_index: int) -> None:
        """The button that must stay visible (the active tab). -1 for none."""
        self._pinned = button_index
        self._relayout()

    @staticmethod
    def _own_width(widget) -> int:
        """A fixed-width item's width; 0 for an item that shares the room."""
        return widget.maximumWidth() if widget.maximumWidth() == widget.minimumWidth() else 0

    def _room(self, items) -> int:
        return sum(self._own_width(self._items[i][0]) or self._min_item for i in items)

    def minimumSizeHint(self) -> QSize:
        return QSize(self._min_item + self.OVERFLOW_W, self._height)

    def sizeHint(self) -> QSize:
        return QSize(self._room(range(len(self._items))) or self._min_item, self._height)

    def resizeEvent(self, event) -> None:
        self._relayout()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(THEME.border_primary), 1))
        p.setBrush(QColor(THEME.bg_input))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), THEME.radius_lg, THEME.radius_lg)

    def _visible_items(self, avail: int) -> list[int]:
        count = len(self._items)
        if avail >= self._room(range(count)):
            return list(range(count))
        fits = 1
        while fits < count and self._room(range(fits + 1)) <= avail - self.OVERFLOW_W:
            fits += 1
        if 0 <= self._pinned and self._pinned >= fits:
            return list(range(fits - 1)) + [self._pinned]
        return list(range(fits))

    def _relayout(self) -> None:
        if not self._items:
            return
        avail = self.width()
        visible = self._visible_items(avail)
        self._hidden = [i for i in range(len(self._items)) if i not in visible]

        own = [self._own_width(self._items[i][0]) for i in visible]
        room = avail - (self.OVERFLOW_W if self._hidden else 0) - sum(own)
        share = max(1, own.count(0))
        x = 0
        slot = 0
        for i, width in zip(visible, own):
            if not width:
                width = room // share + (1 if slot < room % share else 0)
                slot += 1
            self._items[i][0].setGeometry(x, 0, width, self._height)
            x += width

        for i in visible:
            self._items[i][0].show()
        for i in self._hidden:
            self._items[i][0].hide()

        if self._hidden:
            self.overflow_btn.setGeometry(max(0, avail - self.OVERFLOW_W), 0, self.OVERFLOW_W, self._height)
            self.overflow_btn.show()
        else:
            self.overflow_btn.hide()

    def build_overflow_menu(self) -> QMenu:
        menu = QMenu(self)
        for i in self._hidden:
            widget, label = self._items[i]
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
