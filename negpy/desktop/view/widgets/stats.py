from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QWidget

from negpy.desktop.view.styles.theme import THEME
from negpy.features.exposure.stats import StatRow

_TOOLTIPS = {
    "Negative": (
        "The negative itself: relative density range (luminance) and its development character vs a "
        "nominal frame — flat (≈N−1), normal, contrasty (≈N+1). Relative scale, comparable across a "
        "roll; a heuristic from this scan's normalized bounds, not a calibrated densitometer reading."
    ),
    "Exposure": (
        "Where the frame's midtone sits, in stops from neutral: positive = brighter (high-key), "
        "negative = darker (low-key). Approximate — read off the metered midtone, not a precise meter."
    ),
    "Clipping": ("Share of pixels crushed to black (shadows) or blown to white (highlights), worst channel. Turns red above 1%."),
    "Scan clip": (
        "Share of source-scan pixels at/above sensor white, per channel. In a negative scan the film base and "
        "scene shadows sit near sensor white — clipping there destroys base/shadow separation. Fix at capture: "
        "expose the scan lower. Turns red above 1%."
    ),
}


_PROBE_EMPTY = "—"


class DensitometerRow(QWidget):
    """Hover spot-densitometer read-out shown between the H&D curve and the stats."""

    _TOOLTIP = (
        "Spot densitometer — hover the image to read the pixel: per-channel density above film base "
        "(ΔD, relative to this scan's normalization, not absolute), the displayed tone's reflection "
        "print density, and its print zone (0 = paper black, V = 18% mid-gray, X = paper white). "
        "In B&W mode the ΔD channels read the pre-conversion colour record."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(4, 0, 4, 0)
        grid.setHorizontalSpacing(8)
        grid.setColumnStretch(1, 1)
        name = QLabel("Probe")
        name.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_xs}px;")
        self._value = QLabel(_PROBE_EMPTY)
        self._value.setStyleSheet(f"color: {THEME.text_primary}; font-size: {THEME.font_size_xs}px;")
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(name, 0, 0)
        grid.addWidget(self._value, 0, 1)
        name.setToolTip(self._TOOLTIP)
        self._value.setToolTip(self._TOOLTIP)

    def set_reading(self, reading) -> None:
        from negpy.features.exposure.densitometer import format_reading

        self._value.setText(_PROBE_EMPTY if reading is None else format_reading(reading))


class NegativeStatsWidget(QWidget):
    """Compact numerical read-out of the negative under the Analysis charts."""

    _ROWS = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(4, 4, 4, 2)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(1, 1)

        name_css = f"color: {THEME.text_secondary}; font-size: {THEME.font_size_xs}px;"
        self._value_css = f"color: {THEME.text_primary}; font-size: {THEME.font_size_xs}px;"
        self._warn_css = f"color: {THEME.accent_secondary}; font-size: {THEME.font_size_xs}px;"

        self._names: List[QLabel] = []
        self._values: List[QLabel] = []
        for r in range(self._ROWS):
            name = QLabel("")
            name.setStyleSheet(name_css)
            value = QLabel("")
            value.setStyleSheet(self._value_css)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(name, r, 0)
            grid.addWidget(value, r, 1)
            self._names.append(name)
            self._values.append(value)

    def update_stats(self, rows: List[StatRow]) -> None:
        for i in range(self._ROWS):
            if i < len(rows):
                row = rows[i]
                tip = _TOOLTIPS.get(row.name, "")
                self._names[i].setText(row.name)
                self._values[i].setText(row.value)
                self._values[i].setStyleSheet(self._warn_css if row.warn else self._value_css)
                # Tooltip on the whole row (hover anywhere shows it).
                self._names[i].setToolTip(tip)
                self._values[i].setToolTip(tip)
            else:
                self._names[i].setText("")
                self._values[i].setText("")
