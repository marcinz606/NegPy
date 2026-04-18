from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
    QDoubleSpinBox,
)
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QRect, QEvent
from negpy.desktop.view.styles.theme import THEME


class _NoScrollSlider(QSlider):
    def __init__(self, *args, default_pos: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_pos = default_pos

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._default_pos is None:
            return
        p = QPainter(self)
        groove_y = self.height() // 2
        handle_w = 12  # matches QSS handle width
        usable = self.width() - handle_w
        x = handle_w // 2 + int(self._default_pos * usable)
        pen = QPen(QColor(THEME.text_muted), 1)
        p.setPen(pen)
        p.drawLine(x, groove_y - 4, x, groove_y + 4)


class _NoScrollSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class BaseSlider(QWidget):
    """
    Base class for sliders with value synchronization, debouncing, and reset functionality.
    """

    valueChanged = pyqtSignal(float)
    valueCommitted = pyqtSignal(float)

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
        self._last_committed_value = default_val

        default_pos = (default_val - min_val) / (max_val - min_val) if max_val > min_val else None
        self.slider = _NoScrollSlider(Qt.Orientation.Horizontal, default_pos=default_pos)
        if has_neutral:
            self.slider.setObjectName("neutral_slider")
        self.slider.setRange(int(min_val * self._precision), int(max_val * self._precision))
        self.slider.setValue(int(default_val * self._precision))

        self.spin = _NoScrollSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setValue(default_val)

        # Debounce timer
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.setInterval(100)

        self._connect_base_signals()

        self.slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.slider.installEventFilter(self)
        self.spin.installEventFilter(self)

    def _connect_base_signals(self) -> None:
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)
        self.timer.timeout.connect(self._emit_value)
        self.slider.sliderReleased.connect(self._on_committed)
        self.spin.editingFinished.connect(self._on_committed)

    def _on_committed(self) -> None:
        current_val = self.spin.value()
        if current_val != self._last_committed_value:
            self._last_committed_value = current_val
            self.valueCommitted.emit(current_val)

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
        self._on_committed()

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.MouseButtonDblClick:
            self.mouseDoubleClickEvent(event)
            return True
        return super().eventFilter(obj, event)


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
        unit: str = "",
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

        if unit:
            self.spin.setSuffix(unit)

        self.spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.spin.setFixedWidth(60 if unit else 50)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.spin.setStyleSheet(f"font-size: {THEME.font_size_base}px; background: transparent; border: none; font-weight: bold;")

        # Label-scrub: drag the label horizontally to change value
        self.label.setCursor(Qt.CursorShape.SizeHorCursor)
        self.label.installEventFilter(self)
        self._scrub_active = False
        self._scrub_start_x = 0.0
        self._scrub_start_val = 0.0

        header.addWidget(self.label)
        header.addStretch()
        header.addWidget(self.spin)

        layout.addLayout(header)
        layout.addWidget(self.slider)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.label:
            et = event.type()
            if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._scrub_active = True
                self._scrub_start_x = event.position().x()
                self._scrub_start_val = self.spin.value()
                return True
            if et == QEvent.Type.MouseMove and self._scrub_active:
                dx = event.position().x() - self._scrub_start_x
                span = self._max - self._min
                sensitivity = span / 400.0
                mods = event.modifiers()
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    sensitivity *= 0.1
                elif mods & Qt.KeyboardModifier.ControlModifier:
                    sensitivity *= 10.0
                new_val = max(self._min, min(self._max, self._scrub_start_val + dx * sensitivity))
                self.setValue(new_val)
                self._emit_value()
                return True
            if et == QEvent.Type.MouseButtonRelease and self._scrub_active:
                self._scrub_active = False
                self._on_committed()
                return True
        return super().eventFilter(obj, event)


class HueSlider(CompactSlider):
    """
    CompactSlider variant for 0–360° hue selection.
    The groove shows a full rainbow gradient; the label color tracks the current hue.
    """

    def __init__(self, label: str, default_val: float = 0.0, parent=None):
        super().__init__(label, 0.0, 360.0, default_val, step=1.0, precision=1, unit="°", parent=parent)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0.000 hsl(0,80%,50%),
                    stop:0.167 hsl(60,80%,50%),
                    stop:0.333 hsl(120,80%,50%),
                    stop:0.500 hsl(180,80%,50%),
                    stop:0.667 hsl(240,80%,50%),
                    stop:0.833 hsl(300,80%,50%),
                    stop:1.000 hsl(360,80%,50%));
                height: 6px; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: white;
                width: 12px; height: 12px;
                margin: -3px 0;
                border-radius: 6px;
                border: 2px solid rgba(0,0,0,0.5);
            }
        """)
        self._update_label_color(default_val)

    def _update_label_color(self, hue_deg: float) -> None:
        color = QColor.fromHsv(int(hue_deg) % 360, 200, 210)
        self.label.setStyleSheet(f"font-size: {THEME.font_size_base}px; color: {color.name()};")

    def _on_slider_changed(self, value: int) -> None:
        super()._on_slider_changed(value)
        self._update_label_color(value / self._precision)

    def setValue(self, value: float) -> None:
        super().setValue(value)
        self._update_label_color(value)


class RangeSlider(QWidget):
    """
    Dual-handle slider for selecting a range (0.0 to 1.0).
    """

    rangeChanged = pyqtSignal(float, float)
    rangeCommitted = pyqtSignal(float, float)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)
        self._label = label
        self._min_val = 0.0
        self._max_val = 1.0
        self._last_min = 0.0
        self._last_max = 1.0
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
        self._last_min = low
        self._last_max = high
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
        if self._active_handle:
            if self._min_val != self._last_min or self._max_val != self._last_max:
                self._last_min = self._min_val
                self._last_max = self._max_val
                self.rangeCommitted.emit(self._min_val, self._max_val)
        self._active_handle = None

    def mouseDoubleClickEvent(self, event) -> None:
        """Reset for the entire range."""
        self.setRange(0.0, 1.0)
        self.rangeChanged.emit(0.0, 1.0)
        self.rangeCommitted.emit(0.0, 1.0)
