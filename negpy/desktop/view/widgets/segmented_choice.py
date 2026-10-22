import qtawesome as qta
from PyQt6.QtCore import QEvent, QPoint, QRectF, QSize, Qt, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QCursor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QMenu, QPushButton, QToolTip, QWidget

from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, EditedDot, default_button_height, tool_toggle_qss, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME


class _MenuButton(QPushButton):
    """A button that opens its menu on click."""

    def __init__(self, tooltip: str, parent=None):
        super().__init__(parent)
        # Not setMenu: any ::menu-indicator rule then drops the button's padding.
        self.choice_menu = menu = QMenu(self)
        menu.setToolTipsVisible(True)
        # Qt hides the popup on the press, then emits clicked on the release; that click is swallowed.
        self._swallow_next_click = False
        menu.aboutToHide.connect(self._note_menu_hidden)
        self.clicked.connect(self._open_menu)
        self.setFixedHeight(default_button_height())
        self.setToolTip(wrap_tooltip(tooltip))
        self.plain_tooltip = tooltip
        self.edited_dot = EditedDot(self)

    def _dismissed_by_press_on_button(self) -> bool:
        pressed = bool(QApplication.mouseButtons() & Qt.MouseButton.LeftButton)
        return pressed and self.rect().contains(self.mapFromGlobal(QCursor.pos()))

    def _note_menu_hidden(self) -> None:
        self._swallow_next_click = self._dismissed_by_press_on_button()

    def _open_menu(self) -> None:
        if self._swallow_next_click:
            self._swallow_next_click = False
            return
        self.choice_menu.exec(self.mapToGlobal(self.rect().bottomLeft()))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        # clicked fires inside super(); clearing after it also disarms a release off the button.
        super().mouseReleaseEvent(event)
        self._swallow_next_click = False


class MenuButton(_MenuButton):
    """A button with a chevron that opens a menu of actions: the owner sets the text and fills
    `choice_menu`."""

    def __init__(self, tooltip: str, parent=None):
        super().__init__(tooltip, parent)
        self._chevron = qta.icon("fa5s.chevron-down", color=THEME.text_secondary, color_disabled=THEME.text_muted)
        self._chevron_size = THEME.font_size_small
        # The chevron sits clear of the edited dot in the top-right corner.
        self._chevron_inset = THEME.space_2xl
        self.setStyleSheet(
            f"QPushButton {{font-size: {THEME.font_size_base}px;"
            f" padding: 6px {self._chevron_inset + self._chevron_size + THEME.space_md}px 6px {THEME.space_xl}px;"
            " text-align: left;}"
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        s = self._chevron_size
        mode = self._chevron.Mode.Normal if self.isEnabled() else self._chevron.Mode.Disabled
        pix = self._chevron.pixmap(s, s, mode)
        QPainter(self).drawPixmap(QPoint(self.width() - self._chevron_inset - s, (self.height() - s) // 2), pix)


class ToggleMenuButton(_MenuButton):
    """Several independent on/off options behind one button, in the tool-toggle look with no
    chevron. The button takes the checked look while any option is on; an empty label leaves
    it icon-only."""

    def __init__(self, icon_name: str, label: str, tooltip: str, parent=None):
        super().__init__(tooltip, parent)
        self.setStyleSheet(tool_toggle_qss(icon_only=not label))
        self.setCheckable(True)
        self.setIcon(qta.icon(icon_name, color=THEME.text_hint, color_on=THEME.text_primary, color_disabled=THEME.text_muted))
        self.setText(f" {label}" if label else "")
        self._toggles: list[QAction] = []

    def add_toggle(self, label: str, tooltip: str) -> QAction:
        action = QAction(label, self)
        self.choice_menu.addAction(action)
        action.setCheckable(True)
        action.setToolTip(tooltip)
        action.plain_tooltip = tooltip
        action.toggled.connect(self.refresh)
        self._toggles.append(action)
        return action

    def refresh(self) -> None:
        """Re-read the options; call after setting them with signals blocked."""
        self.setChecked(any(a.isChecked() for a in self._toggles))

    def nextCheckState(self) -> None:  # noqa: N802
        # A click opens the menu; the checked look follows the options, not the click.
        pass


class SegmentedChoice(QWidget):
    """One choice out of two to four, all in view: a track whose current segment is filled. A choice
    is (icon, label) or (icon, label, icon color); an icon_only segment shows its label as a
    tooltip. The colored icon of a choice dims while another one is current."""

    currentChanged = pyqtSignal(int)

    _ICON = 14
    _DOT = 5
    _SLIDE_MS = 120

    def __init__(
        self,
        choices: tuple[tuple[str, ...], ...],
        tooltip: str,
        parent=None,
        data: tuple | None = None,
        icon_only: tuple[int, ...] = (),
    ):
        super().__init__(parent)
        self._choices = tuple(choices)
        self._data = tuple(data) if data is not None else tuple(range(len(self._choices)))
        self._icon_only = frozenset(icon_only)
        self._edited = [False] * len(self._choices)
        self._enabled = [True] * len(self._choices)
        self._tips = [""] * len(self._choices)
        self._index = 0
        self._hover = -1
        self._chip_x = 0.0
        self._slide = QVariantAnimation(self)
        self._slide.setDuration(self._SLIDE_MS)
        self._slide.valueChanged.connect(self._on_slide)
        self.setFixedHeight(default_button_height())
        self.setMouseTracking(True)
        # Tab focus only: a click must not leave the focus ring on the track.
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._tab_focus = False
        self.setToolTip(wrap_tooltip(tooltip))
        self.plain_tooltip = tooltip

    def count(self) -> int:
        return len(self._choices)

    def currentIndex(self) -> int:  # noqa: N802
        return self._index

    def currentData(self):  # noqa: N802
        return self._data[self._index] if self._choices else None

    def findData(self, value) -> int:  # noqa: N802
        return self._data.index(value) if value in self._data else -1

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        if index == self._index or not 0 <= index < len(self._choices):
            return
        if self.isVisible():
            segments = self._segments()
            self._slide.setStartValue(segments[self._index].x() if self._slide.state() != QVariantAnimation.State.Running else self._chip_x)
            self._slide.setEndValue(segments[index].x())
            self._slide.start()
        self._index = index
        self.update()
        self.currentChanged.emit(index)

    def set_edited(self, index: int, edited: bool) -> None:
        self._edited[index] = edited
        self.update()

    def is_edited(self, index: int) -> bool:
        return self._edited[index]

    def set_choice_enabled(self, index: int, enabled: bool) -> None:
        self._enabled[index] = enabled
        self.update()

    def is_choice_enabled(self, index: int) -> bool:
        return self._enabled[index]

    def set_choice_tooltip(self, index: int, tooltip: str) -> None:
        self._tips[index] = tooltip

    def _font(self, current: bool) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(THEME.font_size_base)
        font.setWeight(QFont.Weight.DemiBold if current else QFont.Weight.Normal)
        return font

    def _natural(self, i: int, with_icon: bool) -> int:
        if i in self._icon_only:
            return ICON_BUTTON_WIDTH
        icon_name, label, *_ = self._choices[i]
        icon = self._ICON + THEME.space_md if (icon_name and with_icon) else 0
        return 2 * THEME.space_lg + icon + QFontMetrics(self._font(True)).horizontalAdvance(label)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(sum(self._natural(i, True) for i in range(len(self._choices))), default_button_height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(sum(self._natural(i, False) for i in range(len(self._choices))), default_button_height())

    def _show_icons(self) -> bool:
        return self.width() >= self.sizeHint().width()

    def _segments(self) -> list[QRectF]:
        """Labeled segments share the width equally when the widest fits, else each takes
        its own width plus an equal share of the rest."""
        icons = self._show_icons()
        natural = [self._natural(i, icons) for i in range(len(self._choices))]
        labeled = [i for i in range(len(self._choices)) if i not in self._icon_only]
        room = self.width() - sum(natural[i] for i in self._icon_only)
        if labeled and max(natural[i] for i in labeled) * len(labeled) <= room:
            widths = [natural[i] if i in self._icon_only else room / len(labeled) for i in range(len(natural))]
        else:
            share = max(0.0, room - sum(natural[i] for i in labeled)) / max(1, len(labeled))
            widths = [natural[i] + (0 if i in self._icon_only else share) for i in range(len(natural))]
        x, out = 0.0, []
        for w in widths:
            out.append(QRectF(x, 0, w, self.height()))
            x += w
        return out

    def _segment_at(self, pos) -> int:
        return next((i for i, r in enumerate(self._segments()) if r.contains(pos)), -1)

    def _on_slide(self, x) -> None:
        self._chip_x = x
        self.update()

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.ToolTip:
            i = self._segment_at(e.pos().toPointF())
            if i >= 0 and self._tips[i]:
                QToolTip.showText(e.globalPos(), wrap_tooltip(self._tips[i]), self, self._segments()[i].toRect())
                return True
            if i in self._icon_only:
                QToolTip.showText(e.globalPos(), self._choices[i][1], self, self._segments()[i].toRect())
                return True
        return super().event(e)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        self._hover = self._segment_at(e.position())
        self.update()

    def leaveEvent(self, e) -> None:  # noqa: N802
        self._hover = -1
        self.update()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        i = self._segment_at(e.position())
        if e.button() == Qt.MouseButton.LeftButton and i >= 0 and self._enabled[i]:
            self.setCurrentIndex(i)

    def keyPressEvent(self, e) -> None:  # noqa: N802
        step = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1}.get(Qt.Key(e.key()))
        if step is None:
            super().keyPressEvent(e)
            return
        i = self._index + step
        while 0 <= i < len(self._choices) and not self._enabled[i]:
            i += step
        if 0 <= i < len(self._choices):
            self.setCurrentIndex(i)

    def wheelEvent(self, e) -> None:  # noqa: N802
        e.ignore()

    def focusInEvent(self, e) -> None:  # noqa: N802
        # A dialog focuses its first widget when it opens; only Tab earns the ring, and
        # a window regaining focus keeps the one it had.
        if e.reason() != Qt.FocusReason.ActiveWindowFocusReason:
            self._tab_focus = e.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(e)

    def paintEvent(self, e) -> None:  # noqa: N802
        if not self._choices:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        segments = self._segments()
        live = self.isEnabled()

        border = THEME.border_focus if self.hasFocus() and self._tab_focus else THEME.border_primary
        p.setPen(QPen(QColor(border), 1))
        p.setBrush(QColor(THEME.bg_input))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), THEME.radius_lg, THEME.radius_lg)

        if live and 0 <= self._hover != self._index and self._enabled[self._hover]:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(THEME.surface_hover_faint))
            p.drawRoundedRect(segments[self._hover], THEME.radius_lg, THEME.radius_lg)

        chip = QRectF(segments[self._index])
        if self._slide.state() == QVariantAnimation.State.Running:
            chip.moveLeft(self._chip_x)
        # A disabled track keeps a gray chip, so the warm chip always means a live choice.
        p.setPen(QPen(QColor(THEME.border_checked), 1) if live else Qt.PenStyle.NoPen)
        p.setBrush(QColor(THEME.bg_checked if live else THEME.bg_selected))
        pad = THEME.space_xs + 0.5
        p.drawRoundedRect(chip.adjusted(pad, pad, -pad, -pad), THEME.radius_lg - pad / 2, THEME.radius_lg - pad / 2)

        p.setPen(QPen(QColor(THEME.border_primary), 1))
        inset = THEME.space_md + 1
        for i in range(len(segments) - 1):
            if self._index not in (i, i + 1):
                x = int(segments[i].right())
                p.drawLine(x, inset, x, self.height() - inset)

        icons = self._show_icons()
        for i, (r, (icon_name, label, *color)) in enumerate(zip(segments, self._choices)):
            current = i == self._index
            usable = live and self._enabled[i]
            font = self._font(current)
            p.setFont(font)
            show_label = i not in self._icon_only
            show_icon = bool(icon_name) and (icons or not show_label)
            text_w = QFontMetrics(font).horizontalAdvance(label) if show_label else 0
            icon_w = self._ICON if show_icon else 0
            gap = THEME.space_md if (show_icon and show_label) else 0
            x = r.center().x() - (icon_w + gap + text_w) / 2
            if show_icon:
                tint = color[0] if color else (THEME.text_primary if current else THEME.text_hint)
                p.setOpacity(1.0 if (current or not color or not usable) else 0.55)
                pix = qta.icon(icon_name, color=tint if usable else THEME.text_muted).pixmap(self._ICON, self._ICON)
                p.drawPixmap(int(x), int(r.center().y() - self._ICON / 2), pix)
                p.setOpacity(1.0)
            if show_label:
                fg = THEME.text_muted if not usable else (THEME.text_primary if current else THEME.text_secondary)
                p.setPen(QColor(fg))
                p.drawText(QRectF(x + icon_w + gap, r.top(), text_w + 1, r.height()), Qt.AlignmentFlag.AlignVCenter, label)
            if self._edited[i]:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(THEME.mark_edited))
                p.drawEllipse(QRectF(r.right() - self._DOT - THEME.space_sm, r.top() + THEME.space_sm, self._DOT, self._DOT))
