from typing import List, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget

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

# One per zone-placement pin, in pin order; the canvas overlay draws its rings in these
# same colors, so the sidebar row and the pin on the photo can be matched by eye.
PIN_COLORS = (THEME.accent_primary, "#FFFFFF", THEME.channel_blue)


class DensitometerRow(QWidget):
    """Hover spot-densitometer read-out shown between the H&D curve and the stats."""

    _TOOLTIP = (
        "Spot densitometer — hover the image to read the pixel: per-channel density above film base "
        "(ΔD, relative to this scan's normalization, not absolute), the displayed tone's reflection "
        "print density, and its print zone (0 = paper black, V = 18% mid-gray, X = paper white). "
        "In B&W Negative mode the ΔD channels read the pre-conversion color record."
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


class ZonePlacementRows(QWidget):
    """Zone-placement rows under the probe: one per pinned spot with a ⅓-step target
    stepper and its own remove button, plus Place zones. Dumb widget — the controller
    owns the pins and the solve; rows arrive via refresh() as (index, measured roman,
    target zone, achieved roman when the target is out of the paper's scale, solvable)."""

    target_changed = pyqtSignal(int, float)
    apply_clicked = pyqtSignal()
    remove_clicked = pyqtSignal(int)

    _TOOLTIP = (
        "Zone placement — pick a zone on the strip above, then click the photo: that spot is asked "
        "to print there. The steppers trim a pin by thirds, and Place zones solves Print Density "
        "(one pin), Print Density + Grade (two pins), or those plus one knee control for the middle "
        "tone (three pins) so the pinned tones land. Applying turns the matching autos off for this frame."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        col = QVBoxLayout(self)
        col.setContentsMargins(4, 0, 4, 0)
        col.setSpacing(2)
        name_css = f"color: {THEME.text_secondary}; font-size: {THEME.font_size_xs}px;"
        value_css = f"color: {THEME.text_primary}; font-size: {THEME.font_size_xs}px;"
        warn_css = f"color: {THEME.accent_secondary}; font-size: {THEME.font_size_xs}px;"

        self._targets: dict = {}
        self._rows: List[QWidget] = []
        self._names: List[QLabel] = []
        self._target_labels: List[QLabel] = []
        self._lands: List[QLabel] = []
        self._steppers: List[Tuple[QToolButton, QToolButton]] = []
        self._removes: List[QToolButton] = []
        for i, color in enumerate(PIN_COLORS):
            row = QWidget()
            grid = QGridLayout(row)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(6)
            grid.setColumnStretch(1, 1)
            swatch = QLabel()
            swatch.setFixedSize(8, 8)
            swatch.setStyleSheet(f"background: {color}; border-radius: 4px;")
            name = QLabel("")
            name.setStyleSheet(name_css)
            minus = QToolButton()
            minus.setText("−")
            plus = QToolButton()
            plus.setText("+")
            target = QLabel("")
            target.setStyleSheet(value_css)
            target.setAlignment(Qt.AlignmentFlag.AlignCenter)
            target.setMinimumWidth(34)
            lands = QLabel("")
            lands.setStyleSheet(warn_css)
            remove = QToolButton()
            remove.setText("✕")
            remove.setToolTip("Remove this pin")
            minus.clicked.connect(lambda _=False, idx=i: self._step(idx, -1.0 / 3.0))
            plus.clicked.connect(lambda _=False, idx=i: self._step(idx, 1.0 / 3.0))
            remove.clicked.connect(lambda _=False, idx=i: self.remove_clicked.emit(idx))
            grid.addWidget(swatch, 0, 0)
            grid.addWidget(name, 0, 1)
            grid.addWidget(minus, 0, 2)
            grid.addWidget(target, 0, 3)
            grid.addWidget(plus, 0, 4)
            grid.addWidget(remove, 0, 5)
            grid.addWidget(lands, 1, 1, 1, 5)
            col.addWidget(row)
            self._rows.append(row)
            self._names.append(name)
            self._target_labels.append(target)
            self._lands.append(lands)
            self._steppers.append((minus, plus))
            self._removes.append(remove)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 2, 0, 2)
        self.apply_btn = QPushButton("Place zones")
        self.apply_btn.setProperty("primary", True)
        self.apply_btn.setToolTip("Commit the solved print (Enter)")
        self.apply_btn.clicked.connect(self.apply_clicked.emit)
        buttons.addWidget(self.apply_btn)
        col.addLayout(buttons)

        self.solving = QLabel("")
        self.solving.setStyleSheet(name_css)
        self.solving.setVisible(False)
        col.addWidget(self.solving)

        self.setToolTip(self._TOOLTIP)
        self.setVisible(False)

    def _step(self, index: int, delta: float) -> None:
        if index in self._targets:
            self.target_changed.emit(index, self._targets[index] + delta)

    def refresh(self, readouts: Sequence[Tuple[int, str, float, Optional[str], bool]], solving: str = "") -> None:
        from negpy.features.exposure.densitometer import zone_roman

        self._targets = {}
        self.setVisible(bool(readouts))
        for row in self._rows:
            row.setVisible(False)
        solvable = False
        for index, measured, target, achieved, row_solvable in readouts:
            if not 0 <= index < len(self._rows):
                continue
            self._targets[index] = target
            self._rows[index].setVisible(True)
            self._names[index].setText(f"Pin {index + 1} · reads {measured}")
            self._target_labels[index].setText(zone_roman(target))
            if achieved is not None:
                self._lands[index].setText(f"→ lands {achieved}")
                self._lands[index].setToolTip("Outside the paper's scale at this grade and exposure — closest print shown.")
            else:
                self._lands[index].setText("")
                self._lands[index].setToolTip("")
            self._lands[index].setVisible(achieved is not None)
            solvable = row_solvable
        self.apply_btn.setEnabled(solvable)
        self.apply_btn.setToolTip("The outer pins read the same tone — grade needs two different tones." if not solvable else "")
        self.solving.setText(solving)
        self.solving.setVisible(bool(solving))


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
