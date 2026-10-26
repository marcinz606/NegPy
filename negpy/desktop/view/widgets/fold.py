"""Fine-tune controls folded behind a chevron: a rail under its parent control, or a group under its sub-header."""

from negpy.kernel.system.i18n import tr

from typing import Callable

from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import QAbstractButton, QHBoxLayout, QLayout, QVBoxLayout, QWidget
import qtawesome as qta

from negpy.desktop.view.styles.templates import EditedDot, FieldLabel, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider, SliderGroup

_CHEVRON = 10
_DOT = 6
_SIGN = 10


class FoldButton(QAbstractButton):
    """A chevron, and with text a sub-header that folds the group under it. A dot marks an
    edited control behind a closed fold, and a warning sign a warning its owner reports."""

    GUTTER = THEME.space_xl

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(text.upper())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._marked = False
        self._warning = False
        self._font = QFont(self.font())
        self._font.setPixelSize(THEME.font_size_small)
        self._font.setWeight(QFont.Weight(THEME.weight_semibold))
        if not text:
            self.setFixedWidth(self.GUTTER)

    def set_marked(self, marked: bool, warning: bool = False) -> None:
        if (marked, warning) != (self._marked, self._warning):
            self._marked = marked
            self._warning = warning
            self.update()

    def is_marked(self) -> bool:
        return self._marked

    def is_warning(self) -> bool:
        return self._marked and self._warning

    def sizeHint(self) -> QSize:  # noqa: N802
        # A sub-header's mark follows its name.
        text = QFontMetrics(self._font).horizontalAdvance(self.text()) + THEME.space_md + _SIGN if self.text() else 0
        return QSize(self.GUTTER + text, max(QFontMetrics(self._font).height(), _CHEVRON + 2 * THEME.space_xs))

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = THEME.text_primary if self.underMouse() else THEME.text_hint
        icon = qta.icon("fa5s.chevron-down" if self.isChecked() else "fa5s.chevron-right", color=color)
        top = (self.height() - _CHEVRON) // 2
        p.drawPixmap(0, top, icon.pixmap(_CHEVRON, _CHEVRON))
        if self.text():
            p.setFont(self._font)
            p.setPen(QColor(color))
            p.drawText(QRectF(self.GUTTER, 0, self.width() - self.GUTTER, self.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        if self._marked:
            if self.text():
                center = QPointF(self.GUTTER + QFontMetrics(self._font).horizontalAdvance(self.text()) + THEME.space_md, self.height() / 2)
                center += QPointF((_SIGN if self._warning else _DOT) / 2, 0)
            else:
                center = QPointF(_CHEVRON, top)
            if self._warning:
                sign = qta.icon("mdi6.alert", color=THEME.warn_amber).pixmap(_SIGN, _SIGN)
                p.drawPixmap(int(center.x() - _SIGN / 2), int(center.y() - _SIGN / 2), sign)
            else:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(THEME.mark_edited))
                p.drawEllipse(center, _DOT / 2, _DOT / 2)


def _names(content: QWidget) -> str:
    names = [s.label.text() for s in content.findChildren(CompactSlider)] + [f.text() for f in content.findChildren(FieldLabel)]
    return ", ".join(names)


class Fold(QWidget):
    """Holds content behind FoldButtons. Open or closed is the user's choice, saved per key
    once bound; mode gating hides the content, never the fold, so the buttons hide with it."""

    def __init__(self, key: str, content: QWidget, button: FoldButton, parent=None, *, default_open: bool = False) -> None:
        super().__init__(parent)
        self.key = key
        self.content = content
        self.button = button
        self.buttons: list[FoldButton] = []
        self._open = False
        self._warning = False
        self._default_open = default_open
        self._repo = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(content)
        content.installEventFilter(self)
        for dot in content.findChildren(EditedDot):
            dot.installEventFilter(self)
        self.setVisible(False)
        self._add(button)
        self.set_open(default_open)

    def _add(self, button: FoldButton) -> None:
        self.buttons.append(button)
        button.setChecked(self._open)
        button.toggled.connect(self.set_open)
        button.setVisible(not self.content.isHidden())

    def attach(self, parent: CompactSlider | QHBoxLayout) -> None:
        """A second chevron, for a parent row that stands in for the first (a channel trim)."""
        self._add(_gutter_button(parent))

    def bind(self, repo) -> None:
        """Restore the saved state, then save each change."""
        saved = repo.get_global_setting(f"fold_{self.key}")
        self.set_open(self._default_open if saved is None else bool(saved))
        self._repo = repo
        self._sync_tooltip()

    def is_open(self) -> bool:
        return self._open

    def set_open(self, open_: bool) -> None:
        if open_ == self._open:
            return
        self._open = open_
        for button in self.buttons:
            button.setChecked(open_)
        self.setVisible(open_)
        self._sync_mark()
        self._sync_tooltip()
        if self._repo is not None:
            self._repo.save_global_setting(f"fold_{self.key}", open_)

    def _sync_tooltip(self) -> None:
        names = _names(self.content)
        text = (tr("Hide {names}", names=names) if self._open else tr("Show {names}", names=names)) if names else ""
        for button in self.buttons:
            button.setToolTip(wrap_tooltip(text) if text else "")

    def set_warning(self, warning: bool) -> None:
        """Mark the closed fold with a warning sign: something behind it needs a look (a clipping readout)."""
        self._warning = warning
        self._sync_mark()

    def _sync_mark(self) -> None:
        edited = not self._open and any(not dot.isHidden() for dot in self.content.findChildren(EditedDot))
        warning = not self._open and self._warning and not edited
        for button in self.buttons:
            button.set_marked(edited or warning, warning)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() in (QEvent.Type.ShowToParent, QEvent.Type.HideToParent):
            if obj is self.content:
                for button in self.buttons:
                    button.setVisible(not self.content.isHidden())
            self._sync_mark()
        return False


def _gutter_button(parent: CompactSlider | QHBoxLayout) -> FoldButton:
    button = FoldButton()
    row = QWidget.layout(parent) if isinstance(parent, CompactSlider) else parent
    anchor = parent if isinstance(parent, CompactSlider) else parent.itemAt(0).widget()
    # An unparented row has no spacing yet; the style's default is space_md.
    if row.spacing() < 0:
        row.setSpacing(THEME.space_md)
    row.insertWidget(0, button)
    # align_slider_columns gives the anchor's label the gutter less.
    anchor.setProperty("fold_gutter", FoldButton.GUTTER + row.spacing())
    return button


def rail_fold(key: str, parent: CompactSlider | QHBoxLayout, *items: QWidget | QLayout) -> Fold:
    """The rail of items under parent, folded behind a chevron in the gutter of parent's row.
    parent is a CompactSlider or a field_row. Only the chevron toggles the fold: a click on the
    name is half of a slider's reset double-click."""
    return Fold(key, SliderGroup(*items), _gutter_button(parent))


def group_fold(
    key: str, title: str, *items: QWidget | QLayout, buttons: tuple[QWidget, ...] = (), default_open: bool = False
) -> tuple[QHBoxLayout, Fold]:
    """A sub-header that folds the items under it: (header row, fold). buttons sit at the row's
    right end, as in header_row."""
    button = FoldButton(tr(title))
    row = QHBoxLayout()
    row.setContentsMargins(0, THEME.space_xl, 0, 0)
    row.addWidget(button, 1)
    for btn in buttons:
        row.addWidget(btn)
    body = QWidget()
    body.setObjectName("collapsible_content_body")
    layout = QVBoxLayout(body)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(THEME.space_xs)
    for item in items:
        if isinstance(item, QLayout):
            layout.addLayout(item)
        else:
            layout.addWidget(item)
    return row, Fold(key, body, button, default_open=default_open)


def show_while(parent: CompactSlider, rail: QWidget, active: Callable[[float], bool] = lambda v: v != 0.0) -> Callable[[], None]:
    """Show rail only while parent's value lets its members act. Returns the refresh a sidebar's
    sync_ui calls after it sets parent, since that path emits nothing."""

    def refresh(*_) -> None:
        rail.setVisible(active(parent.value()))

    parent.valueChanged.connect(refresh)
    refresh()
    return refresh
