from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QWidget

from negpy.desktop.view.styles.theme import THEME
from negpy.features.exposure.stats import StatRow

_TOOLTIPS = {
    "Density range": (
        "Relative density range of the negative (luminance). Higher = more contrast — a product of "
        "scene contrast and development. Relative scale, comparable across a roll, not absolute scanner density."
    ),
    "Exposure": (
        "Where the frame's midtone sits, in stops from neutral: positive = brighter (high-key), "
        "negative = darker (low-key). Approximate — read off the metered midtone, not a precise meter."
    ),
    "Contrast": (
        "Contrast the conversion is applying, on the ISO R paper scale (R50 = hard … R180 = soft, "
        "R110 ≈ grade 2). Reflects the effective grade including Auto Grade, not just the Grade slider."
    ),
    "Print": (
        "Current print exposure in darkroom units: stops of print exposure (positive = darker print, "
        "like f-stop printing) and the CMY white balance as dichroic-head CC filtration (slider ±1.0 = ±20cc)."
    ),
    "Clipping": ("Share of pixels crushed to black (shadows) or blown to white (highlights), worst channel. Turns red above 1%."),
    "Scan clip": (
        "Share of source-scan pixels at/above sensor white, per channel. In a negative scan the film base and "
        "scene shadows sit near sensor white — clipping there destroys base/shadow separation. Fix at capture: "
        "expose the scan lower. Turns red above 1%."
    ),
}


class NegativeStatsWidget(QWidget):
    """Compact numerical read-out of the negative under the Analysis charts."""

    _ROWS = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(4, 4, 4, 2)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(1, 1)

        name_css = f"color: {THEME.text_secondary}; font-size: {THEME.font_size_xs}px;"
        value_css = f"color: {THEME.text_primary}; font-size: {THEME.font_size_xs}px;"
        tag_css = f"color: {THEME.accent_edited}; font-size: {THEME.font_size_xs}px;"

        self._names: List[QLabel] = []
        self._values: List[QLabel] = []
        self._tags: List[QLabel] = []
        for r in range(self._ROWS):
            name = QLabel("")
            name.setStyleSheet(name_css)
            value = QLabel("")
            value.setStyleSheet(value_css)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            tag = QLabel("")
            tag.setStyleSheet(tag_css)
            tag.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(name, r, 0)
            grid.addWidget(value, r, 1)
            grid.addWidget(tag, r, 2)
            self._names.append(name)
            self._values.append(value)
            self._tags.append(tag)

        self._tag_css = tag_css
        self._warn_css = f"color: {THEME.accent_secondary}; font-size: {THEME.font_size_xs}px;"

    def update_stats(self, rows: List[StatRow]) -> None:
        for i in range(self._ROWS):
            if i < len(rows):
                row = rows[i]
                tip = _TOOLTIPS.get(row.name, "")
                self._names[i].setText(row.name)
                self._values[i].setText(row.value)
                self._tags[i].setText(row.tag)
                self._tags[i].setStyleSheet(self._warn_css if row.warn else self._tag_css)
                # Tooltip on the whole row (hover anywhere shows it).
                self._names[i].setToolTip(tip)
                self._values[i].setToolTip(tip)
                self._tags[i].setToolTip(tip)
            else:
                self._names[i].setText("")
                self._values[i].setText("")
                self._tags[i].setText("")
