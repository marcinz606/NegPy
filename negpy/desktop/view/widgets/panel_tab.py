"""The right panel's tabs."""

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPen
from PyQt6.QtWidgets import QAbstractButton
import qtawesome as qta

from negpy.desktop.view.styles.fonts import ui_font_family
from negpy.desktop.view.styles.theme import THEME

_STACKED_ICON = 21
_ICON = 17
_DOT = 6


def _font(checked: bool) -> QFont:
    font = QFont(ui_font_family())
    font.setPixelSize(THEME.font_size_base)
    font.setWeight(QFont.Weight(THEME.weight_semibold if checked else THEME.weight_medium))
    return font


class PanelTab(QAbstractButton):
    """A right-panel tab. A main tab has its name under a large icon. A
    page's tab has its name beside the icon, or shows only the icon and keeps the name for the
    tooltip and the » menu. A dot marks a tab with edits."""

    ICON_ONLY_WIDTH = 38

    def __init__(self, icon: str, text: str, *, stacked: bool = False, icon_only: bool = False, divider: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(text)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self._stacked = stacked
        self._icon_only = icon_only
        self._divider = divider
        self._marked = False
        self.setIcon(
            qta.icon(
                icon,
                color=THEME.text_hint,
                color_active=THEME.text_primary,
                color_on=THEME.text_primary,
                color_on_active=THEME.text_primary,
            )
        )
        if icon_only:
            self.setFixedWidth(self.ICON_ONLY_WIDTH)

    @staticmethod
    def min_width(labels: list[str], *, stacked: bool) -> int:
        """Widest name at the checked weight, so a tab spills into » before its name or edit dot clips.
        The dot overhangs the centered name by space_xs + _DOT / 2, so each side keeps that room."""
        text = max(QFontMetrics(_font(True)).horizontalAdvance(label) for label in labels)
        return text + 2 * THEME.space_md if stacked else text + _ICON + THEME.space_md + 2 * (THEME.space_sm + THEME.space_xs + _DOT // 2)

    def set_marked(self, marked: bool) -> None:
        if marked != self._marked:
            self._marked = marked
            self.update()

    def is_marked(self) -> bool:
        return self._marked

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        checked, hover = self.isChecked(), self.underMouse()
        if self._divider:
            p.fillRect(0, THEME.space_xl, 1, h - 2 * THEME.space_xl, QColor(THEME.border_primary))
        # The chip of a segmented control's current choice; hover takes its hover tint.
        pad_x, pad_y = THEME.space_xs + 0.5, THEME.space_sm + 0.5
        chip = QRectF(0, 0, w, h).adjusted(pad_x, pad_y, -pad_x, -pad_y)
        radius = THEME.radius_lg - pad_x / 2
        if checked:
            p.setPen(QPen(QColor(THEME.border_checked), 1))
            p.setBrush(QColor(THEME.bg_checked))
            p.drawRoundedRect(chip, radius, radius)
        elif hover:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(THEME.surface_hover_faint))
            p.drawRoundedRect(chip, radius, radius)

        font = _font(checked)
        fm = QFontMetrics(font)
        text_w = 0 if self._icon_only else fm.horizontalAdvance(self.text())
        if self._stacked:
            size = _STACKED_ICON
            top = (h - size - THEME.space_xs - fm.height()) // 2
            icon_rect = QRect((w - size) // 2, top, size, size)
            text_rect = QRect((w - text_w) // 2, top + size + THEME.space_xs, text_w, fm.height())
        else:
            size = _ICON
            left = (w - size - (THEME.space_md + text_w if text_w else 0)) // 2
            icon_rect = QRect(left, (h - size) // 2, size, size)
            text_rect = QRect(left + size + THEME.space_md, (h - fm.height()) // 2, text_w, fm.height())
        mode = QIcon.Mode.Active if hover else QIcon.Mode.Normal
        self.icon().paint(p, icon_rect, Qt.AlignmentFlag.AlignCenter, mode, QIcon.State.On if checked else QIcon.State.Off)
        if text_w:
            p.setFont(font)
            p.setPen(QColor(THEME.text_primary if checked or hover else THEME.text_hint))
            p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.text())

        p.setPen(Qt.PenStyle.NoPen)
        if self._marked:
            anchor = text_rect if text_w else icon_rect
            p.setBrush(QColor(THEME.mark_edited))
            # Close to the name, so the dot stays inside the chip of the narrowest tab.
            p.drawEllipse(QPointF(anchor.right() + THEME.space_xs, anchor.top() + THEME.space_xs), _DOT / 2, _DOT / 2)
        if self.hasFocus():
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QColor(THEME.border_focus))
            p.drawRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5))
