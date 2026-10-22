from negpy.kernel.system.i18n import tr
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.styles.templates import BAR_BUTTON_HEIGHT
from negpy.desktop.view.styles.theme import THEME
from negpy.features.exposure.densitometer import zone_roman

_CELL_W = 26
_PAD = THEME.space_xs + 1
_WEDGE_W = 11 * _CELL_W + 2 * _PAD


class ZoneLegend(QWidget):
    """The Zone Overlay's key on the toolbar: a step wedge in the overlay's colors, then a readout
    of the hovered cell's zone, else the pointer's zone on the canvas, and its share of the frame.
    Hovering a cell asks for that zone alone."""

    zone_hovered = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._readout_font = QFont(self.font())
        self._readout_font.setPixelSize(THEME.font_size_base)
        widest = max(
            QFontMetrics(self._readout_font).horizontalAdvance(f"{tr('Zone {zone}', zone=zone_roman(float(z)))}  100%") for z in range(11)
        )
        self._readout_w = THEME.space_md + widest
        self._colors: List[Tuple[QColor, int]] = []
        self._shares: Optional[np.ndarray] = None
        self._ringed: Optional[int] = None
        self._hovered: Optional[int] = None
        self.show_readout(True)

    def show_readout(self, shown: bool) -> None:
        """A narrow bar drops the readout; the cell tooltips still give each share."""
        self.setFixedSize(QSize(_WEDGE_W + (self._readout_w if shown else 0), BAR_BUTTON_HEIGHT))

    def set_colors(self, colors: Sequence[Tuple[QColor, int]]) -> None:
        self._colors = list(colors)
        self.update()

    def set_shares(self, shares: Optional[np.ndarray]) -> None:
        self._shares = shares
        self.update()

    def set_ringed(self, zone: Optional[int]) -> None:
        if zone != self._ringed:
            self._ringed = zone
            self.update()

    def _cell(self, zone: int) -> QRectF:
        return QRectF(_PAD + zone * _CELL_W, _PAD, _CELL_W, self.height() - 2 * _PAD)

    def _share_text(self, zone: int) -> str:
        if self._shares is None:
            return ""
        share = float(self._shares[zone])
        return f"{share * 100:.0f}%" if share >= 0.005 else "–"

    def readout(self) -> Tuple[str, str]:
        zone = self._hovered if self._hovered is not None else self._ringed
        if zone is None:
            return tr("Zone {zone}", zone="–"), ""
        return tr("Zone {zone}", zone=zone_roman(float(zone))), self._share_text(zone)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(THEME.border_primary), 1))
        painter.setBrush(QColor(THEME.bg_input))
        painter.drawRoundedRect(QRectF(0.5, 0.5, _WEDGE_W - 1.0, self.height() - 1.0), THEME.radius_lg, THEME.radius_lg)
        font = QFont(self.font())
        font.setPixelSize(THEME.font_size_small)
        for zone, (color, _alpha) in enumerate(self._colors):
            cell = self._cell(zone)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRect(cell)
            lit = zone in (self._ringed, self._hovered)
            font.setBold(lit)
            painter.setFont(font)
            painter.setPen(QColor(THEME.mark_shadow) if color.lightnessF() > 0.55 else QColor(THEME.mark_photo))
            painter.drawText(cell, Qt.AlignmentFlag.AlignCenter, zone_roman(float(zone)))
            if lit:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(THEME.text_primary), 1.5))
                painter.drawRect(cell.adjusted(0.75, 0.75, -0.75, -0.75))

        name, share = self.readout()
        painter.setFont(self._readout_font)
        text = QRectF(_WEDGE_W + THEME.space_md, 0, self.width() - _WEDGE_W - THEME.space_md, self.height())
        align = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        painter.setPen(QColor(THEME.text_primary if share else THEME.text_hint))
        painter.drawText(text, align, name)
        painter.setPen(QColor(THEME.text_secondary))
        painter.drawText(text.adjusted(painter.fontMetrics().horizontalAdvance(name + "  "), 0, 0, 0), align, share)

    def _set_hovered(self, zone: Optional[int]) -> None:
        if zone != self._hovered:
            self._hovered = zone
            self.update()
            self.zone_hovered.emit(zone)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        x = event.position().x() - _PAD
        if not 0 <= x < 11 * _CELL_W:
            self._set_hovered(None)
            self.setToolTip("")
        else:
            zone = int(x // _CELL_W)
            self._set_hovered(zone)
            share = self._share_text(zone)
            roman = zone_roman(float(zone))
            self.setToolTip(tr("Zone {zone}: {share} of the frame", zone=roman, share=share) if share else tr("Zone {zone}", zone=roman))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._set_hovered(None)
        super().leaveEvent(event)
