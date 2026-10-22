from negpy.kernel.system.i18n import mark, tr
from typing import List, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
import qtawesome as qta
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from negpy.desktop.view.styles.templates import header_button, labeled_action
from negpy.desktop.view.styles.theme import THEME
from negpy.features.exposure.stats import StatRow

CLIPPING_TOOLTIP = tr("Share of the picture that prints pure black in the shadows or pure white in the highlights.")

_TOOLTIPS = {
    mark("Negative"): (tr("How contrasty the negative is: its estimated density range (Δ), read as flat, normal or contrasty.")),
    mark("Exposure"): (tr("Where the midtones sit, in stops: positive is bright (high-key), negative is dark (low-key).")),
    mark("Clipping"): CLIPPING_TOOLTIP,
    mark("Scan clip"): (tr("Share of each channel at the scanner's maximum, which no edit can bring back.")),
    mark("Gamut"): (tr("Share of the picture that the proof printer or paper cannot print, shown while soft proofing.")),
    mark("Repair"): (tr("Share of the scan that dust removal rewrote, which can change the picture itself above 5%.")),
}


def _channel_markup(value: str) -> str:
    """R · G · B in the channel colors; a warning stays all amber, so the amber reads."""
    colors = (THEME.channel_red_text, THEME.channel_green_text, THEME.channel_blue_text)
    return " · ".join(f'<span style="color:{color}">{part}</span>' for color, part in zip(colors, value.split(" · ")))


# One per zone-placement pin, in pin order. The canvas overlay draws its rings in these
# same colors, so the sidebar row and the pin on the photo match by eye.
PIN_COLORS = (THEME.mark_photo, THEME.channel_blue, THEME.status_success)


def _lock_height(label: QLabel, sample: str) -> None:
    """Freeze a label at the height of *sample*. Zone fractions can come from a taller
    fallback font, and the Analysis chart above absorbs every pixel a row gains."""
    text = label.text()
    label.setText(sample)
    label.ensurePolished()
    label.setFixedHeight(label.sizeHint().height())
    label.setText(text)


class ZonePlacementRows(QWidget):
    """Zone-placement rows under the zone strip: one per pinned spot with a ⅓-step target
    stepper and its own remove button, plus Place zones. Dumb widget — the controller
    owns the pins and the solve; rows arrive via refresh() as (index, measured roman,
    target zone, achieved roman when the target is out of the paper's scale, solvable)."""

    target_changed = pyqtSignal(int, float)
    apply_clicked = pyqtSignal()
    remove_clicked = pyqtSignal(int)

    _TOOLTIP = tr("Zone placement: pick a zone on the strip, then click the picture, and that spot prints in that zone.")

    def __init__(self, parent=None):
        super().__init__(parent)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_xs)
        name_css = f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px;"
        value_css = f"color: {THEME.text_primary}; font-size: {THEME.font_size_small}px;"
        warn_css = f"color: {THEME.warn_amber}; font-size: {THEME.font_size_small}px;"

        self._targets: dict = {}
        self._rows: List[QWidget] = []
        self._names: List[QLabel] = []
        self._target_labels: List[QLabel] = []
        self._lands: List[QLabel] = []
        self._steppers: List[Tuple[QPushButton, QPushButton]] = []
        self._removes: List[QPushButton] = []
        for i, color in enumerate(PIN_COLORS):
            row = QWidget()
            grid = QGridLayout(row)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(THEME.space_md)
            grid.setColumnStretch(1, 1)
            swatch = QLabel()
            swatch.setFixedSize(2 * THEME.radius_md, 2 * THEME.radius_md)
            swatch.setStyleSheet(f"background: {color}; border-radius: {THEME.radius_md}px;")
            name = QLabel("")
            name.setStyleSheet(name_css)
            minus = header_button(qta.icon("fa5s.minus", color=THEME.text_secondary), tr("One third of a zone darker"))
            plus = header_button(qta.icon("fa5s.plus", color=THEME.text_secondary), tr("One third of a zone lighter"))
            target = QLabel("")
            target.setStyleSheet(value_css)
            target.setAlignment(Qt.AlignmentFlag.AlignCenter)
            target.setMinimumWidth(34)
            lands = QLabel("")
            lands.setStyleSheet(warn_css)
            remove = header_button(qta.icon("fa5s.times", color=THEME.text_secondary), tr("Remove this pin"))
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
            _lock_height(name, "Pin 1 · reads VIII⅔")
            _lock_height(target, "VIII⅔")
            _lock_height(lands, "→ lands VIII⅔")
            col.addWidget(row)
            self._rows.append(row)
            self._names.append(name)
            self._target_labels.append(target)
            self._lands.append(lands)
            self._steppers.append((minus, plus))
            self._removes.append(remove)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, THEME.space_xs, 0, THEME.space_xs)
        self.apply_btn = labeled_action("", tr("Place Zones"), tr("Commit the solved print (Enter)"), primary=True)
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
            self._names[index].setText(tr("Pin {value} · reads {measured}", value=index + 1, measured=measured))
            self._target_labels[index].setText(zone_roman(target))
            if achieved is not None:
                self._lands[index].setText(tr("→ lands {achieved}", achieved=achieved))
                self._lands[index].setToolTip(tr("Outside the paper's range at this grade and exposure, so the closest print is shown."))
            else:
                self._lands[index].setText("")
                self._lands[index].setToolTip("")
            self._lands[index].setVisible(achieved is not None)
            solvable = row_solvable
        self.apply_btn.setEnabled(solvable)
        self.apply_btn.setToolTip(tr("Grade needs two different tones, but the outer pins read the same tone.") if not solvable else "")
        self.solving.setText(solving)
        self.solving.setVisible(bool(solving))


# Grid columns for a stat's (name, value) pair: even slots sit left of the divider
# (column 2), odd slots to its right.
_PAIR_COLUMNS = ((0, 1), (3, 4))


class NegativeStatsWidget(QWidget):
    """Compact numerical read-out of the negative under the Analysis charts.

    Two stats share each grid row (name|value|name|value) so the read-out
    claims about half the vertical height of one-stat-per-row.
    """

    _ROWS = 6

    # True while any stat shows amber.
    warning_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, THEME.space_sm, 0, THEME.space_xs)
        grid.setHorizontalSpacing(THEME.space_lg)
        grid.setVerticalSpacing(THEME.space_xs)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(4, 1)

        name_css = f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px;"
        self._value_css = f"color: {THEME.text_primary}; font-size: {THEME.font_size_small}px;"
        self._warn_css = f"color: {THEME.warn_amber}; font-size: {THEME.font_size_small}px;"

        self._names: List[QLabel] = []
        self._values: List[QLabel] = []
        self._dividers: List[QFrame] = []
        for i in range(self._ROWS):
            grid_row = i // 2
            name_col, value_col = _PAIR_COLUMNS[i % 2]
            name = QLabel("")
            name.setStyleSheet(name_css)
            value = QLabel("")
            value.setStyleSheet(self._value_css)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(name, grid_row, name_col)
            grid.addWidget(value, grid_row, value_col)
            self._names.append(name)
            self._values.append(value)
            if i % 2 == 0:
                divider = QFrame()
                divider.setFrameShape(QFrame.Shape.VLine)
                divider.setFrameShadow(QFrame.Shadow.Plain)
                divider.setStyleSheet(f"color: {THEME.border_color};")
                grid.addWidget(divider, grid_row, 2)
                self._dividers.append(divider)

    def update_stats(self, rows: List[StatRow]) -> None:
        for i in range(self._ROWS):
            # Unused slots hide rather than blank: an empty QLabel still claims its font
            # height, and the Analysis chart above absorbs every pixel a row takes.
            used = i < len(rows)
            self._names[i].setVisible(used)
            self._values[i].setVisible(used)
            if not used:
                self._names[i].setText("")
                self._values[i].setText("")
                self._names[i].setToolTip("")
                self._values[i].setToolTip("")
                continue
            row = rows[i]
            tip = _TOOLTIPS.get(row.name, "")
            if row.detail:
                tip = f"{tip}\n\n{row.detail}" if tip else row.detail
            self._names[i].setText(tr(row.name))
            self._values[i].setText(_channel_markup(row.value) if row.per_channel and not row.warn else row.value)
            self._values[i].setStyleSheet(self._warn_css if row.warn else self._value_css)
            # Tooltip per stat (hover its name or value), not merged across the pair.
            self._names[i].setToolTip(tip)
            self._values[i].setToolTip(tip)
        for pair, divider in enumerate(self._dividers):
            # A divider separates two stats; hide it when the pair's second slot is empty.
            divider.setVisible(2 * pair + 1 < len(rows))
        self.warning_changed.emit(any(row.warn for row in rows[: self._ROWS]))
