from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.styles.templates import default_button_height
from negpy.desktop.view.styles.theme import THEME
from negpy.features.exposure.densitometer import zone_roman

_CELL_W = 32


class ZoneLegend(QWidget):
    """The Zone Overlay's key on the toolbar: one cell per zone in the overlay's colors, with its
    share of the frame. The pointer's zone on the canvas is ringed; hovering a cell asks for
    that zone alone."""

    zone_hovered = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFixedSize(QSize(11 * _CELL_W, default_button_height()))
        self._colors: List[Tuple[QColor, int]] = []
        self._shares: Optional[np.ndarray] = None
        self._ringed: Optional[int] = None
        self._hovered: Optional[int] = None

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
        return QRectF(zone * _CELL_W + 1.0, 0.0, _CELL_W - 2.0, float(self.height()))

    def _share_text(self, zone: int) -> str:
        if self._shares is None:
            return ""
        share = float(self._shares[zone])
        return f"{share * 100:.0f}%" if share >= 0.005 else "–"

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = painter.font()
        font.setPixelSize(THEME.font_size_small)
        for zone, (color, _alpha) in enumerate(self._colors):
            cell = self._cell(zone)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(cell, THEME.radius_md, THEME.radius_md)
            if zone in (self._ringed, self._hovered):
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(THEME.text_primary), 2.0))
                painter.drawRoundedRect(cell.adjusted(1.0, 1.0, -1.0, -1.0), THEME.radius_md, THEME.radius_md)
            painter.setPen(QColor(0, 0, 0) if color.lightnessF() > 0.55 else QColor(255, 255, 255))
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(cell.adjusted(0, 1, 0, -cell.height() / 2), Qt.AlignmentFlag.AlignCenter, zone_roman(float(zone)))
            font.setBold(False)
            painter.setFont(font)
            painter.drawText(cell.adjusted(0, cell.height() / 2 - 2, 0, -1), Qt.AlignmentFlag.AlignCenter, self._share_text(zone))

    def _set_hovered(self, zone: Optional[int]) -> None:
        if zone != self._hovered:
            self._hovered = zone
            self.update()
            self.zone_hovered.emit(zone)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        zone = min(10, max(0, int(event.position().x() // _CELL_W)))
        self._set_hovered(zone)
        share = self._share_text(zone)
        self.setToolTip(f"Zone {zone_roman(float(zone))}" + (f" — {share} of the frame" if share else ""))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._set_hovered(None)
        super().leaveEvent(event)
