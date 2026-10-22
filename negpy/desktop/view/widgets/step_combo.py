from collections.abc import Sequence
from typing import Any, Optional

import qtawesome as qta
from PyQt6.QtCore import QPoint, QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QComboBox, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from negpy.desktop.view.styles.templates import HEADER_BUTTON_SIZE, default_button_height, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME

_HEADER_ROLE = Qt.ItemDataRole.UserRole + 1
_ICON = 14
_CHEVRON = 11
_CHECK = 14


def _is_separator(index) -> bool:
    return index.data(Qt.ItemDataRole.AccessibleDescriptionRole) == "separator"


class _ChoiceDelegate(QStyledItemDelegate):
    """List rows: a chip, a check and a semibold label on the current choice, a caption for a header,
    a rule for a separator."""

    def __init__(self, combo: QComboBox):
        super().__init__(combo)
        self._combo = combo

    def sizeHint(self, option, index):  # noqa: N802
        size = super().sizeHint(option, index)
        if _is_separator(index):
            return QSize(size.width(), 2 * THEME.space_sm + 1)
        return QSize(size.width() + _CHECK + THEME.space_xl, size.height())

    def paint(self, painter, option, index) -> None:
        if index.data(_HEADER_ROLE):
            font = QFont(option.font)
            font.setPixelSize(THEME.font_size_small)
            font.setWeight(QFont.Weight.DemiBold)
            painter.save()
            painter.setFont(font)
            painter.setPen(QColor(THEME.text_hint))
            painter.drawText(QRectF(option.rect).adjusted(THEME.space_xl, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, index.data())
            painter.restore()
            return
        if _is_separator(index):
            y = option.rect.center().y()
            painter.save()
            painter.setPen(QPen(QColor(THEME.border_color), 1))
            painter.drawLine(option.rect.left() + THEME.space_lg, y, option.rect.right() - THEME.space_lg, y)
            painter.restore()
            return
        current = index.row() == self._combo.currentIndex()
        opt = QStyleOptionViewItem(option)
        if current:
            opt.font.setWeight(QFont.Weight.DemiBold)
            # The chip of a checked toggle; hover lightens it instead of the gray hover fill.
            hover = bool(option.state & (QStyle.StateFlag.State_Selected | QStyle.StateFlag.State_MouseOver))
            opt.state &= ~(QStyle.StateFlag.State_Selected | QStyle.StateFlag.State_MouseOver)
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor(THEME.border_checked_hover if hover else THEME.border_checked), 1))
            painter.setBrush(QColor(THEME.bg_checked_hover if hover else THEME.bg_checked))
            painter.drawRoundedRect(QRectF(option.rect).adjusted(0.5, 0.5, -0.5, -0.5), THEME.radius_sm, THEME.radius_sm)
            painter.restore()
        super().paint(painter, opt, index)
        if current:
            r = option.rect
            check = qta.icon("mdi6.check", color=THEME.text_primary).pixmap(_CHECK, _CHECK)
            painter.drawPixmap(r.right() - THEME.space_xl - _CHECK, r.center().y() - _CHECK // 2, check)


class StepCombo(QComboBox):
    """A dropdown drawn as one outlined button split into ‹ | choice | ›. The arrows step to the
    neighbor choice, past headers, separators and grayed rows, and wrap past the ends; the face
    opens the list. The QComboBox API is unchanged."""

    _ARROW_W = HEADER_BUTTON_SIZE

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover: str | None = None
        self.setMouseTracking(True)
        self.setFixedHeight(default_button_height())
        # Tab focus only: a click must not leave the focus border on the control.
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setIconSize(QSize(_ICON, _ICON))
        self.setItemDelegate(_ChoiceDelegate(self))

    @classmethod
    def of(cls, labels: Sequence[str], tooltip: str, data: Optional[Sequence[Any]] = None) -> "StepCombo":
        """A StepCombo holding `labels`, the first current; `data` is each choice's stored value."""
        combo = cls()
        combo.set_choices(labels, data)
        combo.setToolTip(wrap_tooltip(tooltip))
        return combo

    def set_choices(self, labels: Sequence[str], data: Optional[Sequence[Any]] = None) -> None:
        """Replace the choices without emitting; the first becomes current."""
        blocked = self.blockSignals(True)
        self.clear()
        for i, label in enumerate(labels):
            self.addItem(label, None if data is None else data[i])
        self.blockSignals(blocked)

    def set_item_enabled(self, index: int, enabled: bool) -> None:
        """A grayed choice: the list will not pick it and the arrows step past it."""
        item = self.model().item(index)
        if item is not None:
            item.setEnabled(enabled)

    def is_item_enabled(self, index: int) -> bool:
        item = self.model().item(index)
        return item is not None and item.isEnabled()

    def set_item_tooltip(self, index: int, tooltip: str) -> None:
        self.setItemData(index, tooltip, Qt.ItemDataRole.ToolTipRole)

    def add_header(self, text: str) -> None:
        """A caption row in the list. It never becomes current, so the caller sets the current choice."""
        blocked = self.blockSignals(True)
        self.addItem(text)
        item = self.model().item(self.count() - 1)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setData(True, _HEADER_ROLE)
        if self.currentIndex() == self.count() - 1:
            self.setCurrentIndex(-1)
        self.blockSignals(blocked)

    def _step_target(self, step: int) -> int:
        """The neighbor enabled choice, wrapping past the ends; -1 when there is no other choice."""
        model, n, cur = self.model(), self.count(), self.currentIndex()
        start = cur if cur >= 0 else (-1 if step > 0 else n)
        for k in range(1, n + 1):
            i = (start + k * step) % n
            if i != cur and model.flags(model.index(i, 0)) & Qt.ItemFlag.ItemIsEnabled:
                return i
        return -1

    def _cells(self) -> dict[str, QRectF]:
        w, h, a = float(self.width()), float(self.height()), float(self._ARROW_W)
        return {"prev": QRectF(0, 0, a, h), "face": QRectF(a, 0, w - 2 * a, h), "next": QRectF(w - a, 0, a, h)}

    def _zone_at(self, pos: QPointF) -> str | None:
        return next((zone for zone, r in self._cells().items() if r.contains(pos)), None)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(super().sizeHint().width() + 2 * self._ARROW_W, default_button_height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(super().minimumSizeHint().width() + 2 * self._ARROW_W, default_button_height())

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        self._hover = self._zone_at(e.position())
        self.update()
        super().mouseMoveEvent(e)

    def leaveEvent(self, e) -> None:  # noqa: N802
        self._hover = None
        self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        zone = self._zone_at(e.position())
        if e.button() != Qt.MouseButton.LeftButton or zone == "face":
            super().mousePressEvent(e)
            return
        self._step(-1 if zone == "prev" else 1)
        e.accept()

    def _step(self, step: int) -> None:
        target = self._step_target(step)
        if target >= 0:
            self.setCurrentIndex(target)
            self.activated.emit(target)
            self.textActivated.emit(self.itemText(target))

    def _draw_arrow(self, p: QPainter, rect: QRectF, name: str, usable: bool) -> None:
        pix = qta.icon(name, color=THEME.text_secondary if usable else THEME.text_muted).pixmap(_CHEVRON, _CHEVRON)
        p.drawPixmap(QPoint(int(rect.center().x() - _CHEVRON / 2), int(rect.center().y() - _CHEVRON / 2)), pix)

    def _draw_face(self, p: QPainter, rect: QRectF, live: bool) -> None:
        i = self.currentIndex()
        x, right = rect.left() + THEME.space_lg, rect.right() - THEME.space_md
        icon = self.itemIcon(i) if i >= 0 else None
        if icon is not None and not icon.isNull():
            mode = icon.Mode.Normal if live else icon.Mode.Disabled
            p.drawPixmap(int(x), int(rect.center().y() - _ICON / 2), icon.pixmap(_ICON, _ICON, mode))
            x += _ICON + THEME.space_md
        font = QFont(self.font())
        font.setPixelSize(THEME.font_size_base)
        p.setFont(font)
        text, color = (self.currentText(), THEME.text_primary) if i >= 0 else (self.placeholderText(), THEME.text_hint)
        p.setPen(QColor(color if live else THEME.text_muted))
        text = QFontMetrics(font).elidedText(text, Qt.TextElideMode.ElideRight, int(right - x))
        p.drawText(QRectF(x, rect.top(), right - x, rect.height()), Qt.AlignmentFlag.AlignVCenter, text)

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        live = self.isEnabled()
        hover = self._hover if live else None
        cells = self._cells()
        outline = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        if hover in ("prev", "next") and self._step_target(-1 if hover == "prev" else 1) < 0:
            hover = None
        clip = QPainterPath()
        clip.addRoundedRect(outline, THEME.radius_md, THEME.radius_md)
        p.save()
        p.setClipPath(clip)
        p.fillRect(outline, QColor(THEME.bg_input))
        if hover:
            p.fillRect(cells[hover], QColor(THEME.border_primary))
        p.restore()

        border = THEME.border_focus if self.hasFocus() else (THEME.border_hover if hover else THEME.border_primary)
        p.setPen(QPen(QColor(border), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(outline, THEME.radius_md, THEME.radius_md)
        # Short dividers, as between a segmented control's choices; one beside a hovered cell goes.
        p.setPen(QPen(QColor(THEME.border_primary), 1))
        inset = THEME.space_md + 1
        for cell, x in (("prev", cells["prev"].right()), ("next", cells["next"].left())):
            if hover not in (cell, "face"):
                p.drawLine(QPointF(x, inset), QPointF(x, self.height() - inset))

        self._draw_face(p, cells["face"], live)
        self._draw_arrow(p, cells["prev"], "fa5s.chevron-left", live and self._step_target(-1) >= 0)
        self._draw_arrow(p, cells["next"], "fa5s.chevron-right", live and self._step_target(1) >= 0)
