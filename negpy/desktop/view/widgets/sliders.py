from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
    QDoubleSpinBox,
)
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QRect
from negpy.desktop.view.styles.theme import THEME


class BaseSlider(QWidget):
    """
    Base class for sliders with value synchronization, debouncing, and reset functionality.
    """

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        min_val: float,
        max_val: float,
        default_val: float,
        precision: int = 100,
        has_neutral: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._default = default_val
        self._precision = precision

        self.slider = QSlider(Qt.Orientation.Horizontal)
        if has_neutral:
            self.slider.setObjectName("neutral_slider")
        self.slider.setRange(int(min_val * self._precision), int(max_val * self._precision))
        self.slider.setValue(int(default_val * self._precision))

        self.spin = QDoubleSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setValue(default_val)

        # Debounce timer
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.setInterval(100)

        self._connect_base_signals()

    def _connect_base_signals(self) -> None:
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)
        self.timer.timeout.connect(self._emit_value)

    def _on_slider_changed(self, value: int) -> None:
        f_val = value / self._precision
        self.spin.blockSignals(True)
        self.spin.setValue(f_val)
        self.spin.blockSignals(False)
        self.timer.start()

    def _on_spin_changed(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(value * self._precision))
        self.slider.blockSignals(False)
        self.timer.start()

    def _emit_value(self) -> None:
        self.valueChanged.emit(self.spin.value())

    def setValue(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(int(value * self._precision))
        self.spin.setValue(value)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def value(self) -> float:
        return self.spin.value()

    def mouseDoubleClickEvent(self, event) -> None:
        """Resets to default value."""
        self.setValue(self._default)
        self._emit_value()


class SignalSlider(BaseSlider):
    """
    Standard slider with side-by-side label, slider, and spinbox.
    """

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        default_val: float,
        step: float = 0.01,
        precision: int = 100,
        color: str = None,
        has_neutral: bool = False,
        parent=None,
    ):
        super().__init__(min_val, max_val, default_val, precision=precision, has_neutral=has_neutral, parent=parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.label = QLabel(label)
        self.label.setMinimumWidth(80)
        self.label.setStyleSheet(f"font-size: {THEME.font_size_base}px; color: {color if color else THEME.text_primary};")

        self.spin.setSingleStep(step)
        if step >= 1.0:
            self.spin.setDecimals(0)

        self.spin.setFixedWidth(70)
        self.spin.setStyleSheet(f"font-size: {THEME.font_size_base}px;")

        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)


class CompactSlider(BaseSlider):
    """
    Compact slider with label and value in a header row, slider below.
    """

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        default_val: float,
        step: float = 0.01,
        precision: int = 100,
        color: str = None,
        has_neutral: bool = False,
        parent=None,
    ):
        super().__init__(min_val, max_val, default_val, precision=precision, has_neutral=has_neutral, parent=parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.label = QLabel(label)
        self.label.setStyleSheet(f"font-size: {THEME.font_size_base}px; color: {color if color else THEME.text_secondary};")

        self.spin.setSingleStep(step)
        if step >= 1.0:
            self.spin.setDecimals(0)
            self.slider.setTickInterval(int(step))
            self.slider.setSingleStep(int(step))

        self.spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.spin.setFixedWidth(50)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.spin.setStyleSheet(f"font-size: {THEME.font_size_base}px; background: transparent; border: none; font-weight: bold;")

        header.addWidget(self.label)
        header.addStretch()
        header.addWidget(self.spin)

        layout.addLayout(header)
        layout.addWidget(self.slider)


class RangeSlider(QWidget):
    """
    Dual-handle slider for selecting a range (0.0 to 1.0).
    """

    rangeChanged = pyqtSignal(float, float)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)
        self._label = label
        self._min_val = 0.0
        self._max_val = 1.0
        self._active_handle = None

        self._margin = 10
        self._handle_r = 6

        # Debounce
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.setInterval(50)
        self.timer.timeout.connect(lambda: self.rangeChanged.emit(self._min_val, self._max_val))

    def setRange(self, low: float, high: float) -> None:
        self._min_val = low
        self._max_val = high
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw Label
        painter.setPen(QColor(THEME.text_secondary))
        painter.setFont(painter.font())
        painter.drawText(QRect(0, 0, self.width(), 15), Qt.AlignmentFlag.AlignLeft, self._label)

        # Track math
        w = self.width() - 2 * self._margin
        y = 35

        # Draw Groove
        painter.setPen(QPen(QColor("#444"), 4))
        painter.drawLine(self._margin, y, self.width() - self._margin, y)

        # Draw Active Part
        x1 = self._margin + int(self._min_val * w)
        x2 = self._margin + int(self._max_val * w)
        painter.setPen(QPen(QColor(THEME.accent_primary), 4))
        painter.drawLine(x1, y, x2, y)

        # Draw Handles
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(THEME.accent_primary))
        painter.drawEllipse(
            x1 - self._handle_r,
            y - self._handle_r,
            self._handle_r * 2,
            self._handle_r * 2,
        )
        painter.drawEllipse(
            x2 - self._handle_r,
            y - self._handle_r,
            self._handle_r * 2,
            self._handle_r * 2,
        )

    def _get_val(self, x: int) -> float:
        w = self.width() - 2 * self._margin
        val = (x - self._margin) / max(1, w)
        return float(max(0.0, min(1.0, val)))

    def mousePressEvent(self, event) -> None:
        x = int(event.position().x())
        w = self.width() - 2 * self._margin
        x1 = self._margin + int(self._min_val * w)
        x2 = self._margin + int(self._max_val * w)

        if abs(x - x1) < 15:
            self._active_handle = "min"
        elif abs(x - x2) < 15:
            self._active_handle = "max"
        else:
            self._active_handle = None

    def mouseMoveEvent(self, event) -> None:
        if not self._active_handle:
            return

        val = self._get_val(int(event.position().x()))
        if self._active_handle == "min":
            self._min_val = min(val, self._max_val - 0.05)
        else:
            self._max_val = max(val, self._min_val + 0.05)

        self.update()
        self.timer.start()

    def mouseReleaseEvent(self, event) -> None:
        self._active_handle = None

    def mouseDoubleClickEvent(self, event) -> None:
        """Reset for the entire range."""
        self.setRange(0.0, 1.0)
        self.rangeChanged.emit(0.0, 1.0)
