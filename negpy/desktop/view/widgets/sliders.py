import math
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
from negpy.desktop.view.styles.templates import EditedDot, slider_label_qss, slider_handle_qss, wrap_tooltip


# text_secondary, not text_muted: #555 on the #161616 tooltip background is ~2.4:1.
_RESET_HINT = f'<div style="color:{THEME.text_secondary};">Double-click to reset</div>'

# Quiet period after the last slider step before a render is asked for.
_EMIT_INTERVAL_MS = 33


class _NoScrollSlider(QSlider):
    def __init__(self, *args, default_pos: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_pos = default_pos
        self._default_slider_value: Optional[int] = None
        self._drag_anchor_px: Optional[float] = None
        self._drag_anchor_value: int = 0

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

    def _value_from_x(self, x: float) -> int:
        # Mirrors paintEvent's handle mapping (handle_w=12) so click-to-position
        # and the default-marker tick agree. Any y within the widget maps here.
        handle_w = 12
        usable = max(1, self.width() - handle_w)
        pos = max(0.0, min(1.0, (x - handle_w / 2) / usable))
        if self.invertedAppearance():
            pos = 1.0 - pos
        return self.minimum() + round(pos * (self.maximum() - self.minimum()))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier and self._default_slider_value is not None:
                self.setValue(self._default_slider_value)
                self.sliderReleased.emit()
                event.accept()
                return
            # Jump the handle under the cursor (ignoring y), then let Qt grab it
            # so the drag continues from there. Shift keeps the handle put for a
            # relative fine-drag from wherever it already is.
            if not mods & Qt.KeyboardModifier.ShiftModifier:
                self.setValue(self._value_from_x(event.position().x()))
            self._drag_anchor_px = event.position().x()
            self._drag_anchor_value = self.value()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.isSliderDown() and self._drag_anchor_px is not None and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            usable = max(1, self.width() - 12)
            rng = self.maximum() - self.minimum()
            delta_px = event.position().x() - self._drag_anchor_px
            if self.invertedAppearance():
                delta_px = -delta_px
            value_change = int(round(delta_px * 0.1 * rng / usable))
            new_value = max(self.minimum(), min(self.maximum(), self._drag_anchor_value + value_change))
            self.setValue(new_value)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_anchor_px = None
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._default_pos is None:
            return
        p = QPainter(self)
        groove_y = self.height() // 2
        handle_w = 12
        usable = self.width() - handle_w
        x = handle_w // 2 + int(self._default_pos * usable)
        pen = QPen(QColor(THEME.text_unit), 1)
        p.setPen(pen)
        p.drawLine(x, groove_y - 1, x, groove_y + 2)


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
    # Handle grab/release, independent of whether the value actually changed.
    dragStarted = pyqtSignal()
    dragEnded = pyqtSignal()

    def setToolTip(self, text: str) -> None:
        super().setToolTip(wrap_tooltip(text, _RESET_HINT))

    def __init__(
        self,
        min_val: float,
        max_val: float,
        default_val: float,
        precision: int = 100,
        has_neutral: bool = False,
        inverted: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._default = default_val
        self._precision = precision
        self._last_committed_value = default_val

        # sorted() keeps a decreasing value->int mapping (e.g. Kelvin->mired) legal.
        lo, hi = sorted((self._to_int(min_val), self._to_int(max_val)))
        default_pos = (self._to_int(default_val) - lo) / (hi - lo) if hi > lo else None
        if inverted and default_pos is not None:
            default_pos = 1.0 - default_pos
        self.slider = _NoScrollSlider(Qt.Orientation.Horizontal, default_pos=default_pos)
        if inverted:
            self.slider.setInvertedAppearance(True)
            self.slider.setInvertedControls(True)
        if has_neutral:
            self.slider.setObjectName("neutral_slider")
        self.slider.setRange(lo, hi)
        self.slider.setValue(self._to_int(default_val))
        self.slider._default_slider_value = self._to_int(default_val)

        self.spin = _NoScrollSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setValue(default_val)

        # Trailing debounce — see _schedule_emit for why it stays trailing.
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.setInterval(_EMIT_INTERVAL_MS)

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
        self.slider.sliderPressed.connect(self.dragStarted.emit)
        self.slider.sliderReleased.connect(self.dragEnded.emit)

    def _on_committed(self) -> None:
        # The commit below renders this value; a trailing frame would repeat it.
        pending = self.timer.isActive()
        self.timer.stop()
        current_val = self.spin.value()
        if current_val != self._last_committed_value:
            self._last_committed_value = current_val
            self.valueCommitted.emit(current_val)
        elif pending:
            # Dragged away and back: no commit fires, so that trailing frame is the
            # only thing that would put the preview back on the committed value.
            self._emit_value()

    def _to_int(self, value: float) -> int:
        """Value -> slider int; subclasses override the pair for nonlinear
        travel. Must stay stateless: called during __init__.

        round(), not int(): truncation drops the handle a step on the
        commit->sync round-trip (e.g. 0.29*100 == 28.9999) — a visible jump-back."""
        return round(value * self._precision)

    def _from_int(self, i: int) -> float:
        return i / self._precision

    def _on_slider_changed(self, value: int) -> None:
        f_val = self._from_int(value)
        self.spin.blockSignals(True)
        self.spin.setValue(f_val)
        self.spin.blockSignals(False)
        self._schedule_emit()

    def _on_spin_changed(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_int(value))
        self.slider.blockSignals(False)
        self._schedule_emit()

    def _schedule_emit(self) -> None:
        """Trailing debounce: the handle must never wait on a render.

        Emitting mid-drag pulls the whole per-render UI fan-out onto the handle's own
        thread. A leading edge was tried here and reverted for that reason.
        """
        self.timer.start(_EMIT_INTERVAL_MS)

    def _emit_value(self) -> None:
        self.valueChanged.emit(self.spin.value())

    def setValue(self, value: float, _rebase_commit: bool = True) -> None:
        """_rebase_commit=True is the external-sync path (a sidebar reflecting a newly
        loaded config onto this reused widget) and rebases the commit baseline — or a
        later drag landing on a value some *other* image last committed here would
        silently no-op in _on_committed. Internal gesture steps (adjust_by, dbl-click
        reset, label-scrub) pass False so their own trailing _on_committed() still sees
        the pre-gesture baseline and fires correctly."""
        if self.slider.isSliderDown() or self.spin.hasFocus():
            return
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(self._to_int(value))
        self.spin.setValue(value)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)
        if _rebase_commit:
            self._last_committed_value = value

    def value(self) -> float:
        return self.spin.value()

    def adjust_by(self, delta: float) -> None:
        new_value = max(self._min, min(self._max, self.value() + delta))
        self.setValue(new_value, _rebase_commit=False)
        self._emit_value()
        self._on_committed()

    def mirror_value(self, value: float, commit: bool) -> None:
        """Drive this slider from a mirror copy of it (see clone_slider), so the mirror's
        gesture runs the original's own binding chain instead of writing config itself.

        _rebase_commit=False for the same reason as adjust_by: it leaves the commit baseline
        pre-gesture, or the trailing _on_committed() would see no change and never fire."""
        self.setValue(value, _rebase_commit=False)
        self._emit_value()
        if commit:
            self._on_committed()

    def mouseDoubleClickEvent(self, event) -> None:
        """Resets to default value."""
        self.setValue(self._default, _rebase_commit=False)
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
    Spin-box is hidden at rest; revealed on hover or keyboard focus.
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
        inverted: bool = False,
        parent=None,
    ):
        super().__init__(min_val, max_val, default_val, precision=precision, has_neutral=has_neutral, inverted=inverted, parent=parent)

        self._label_color = color if color else THEME.text_secondary

        layout = QVBoxLayout(self)
        # Bias the row's dead space below the groove so the near-miss grab band
        # (see _GRAB_PAD) has room underneath: drop the header<->slider gap and
        # move it below the slider. Net row height is unchanged; the groove sits
        # ~2px closer to its label than the default.
        layout.setContentsMargins(2, 2, 2, 4)
        layout.setSpacing(0)

        header = QHBoxLayout()
        header.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        header.setSpacing(2)  # explicit: the VBox spacing (0) would otherwise collapse the label<->edited-dot gap
        self.label = QLabel(label)
        self.label.setStyleSheet(slider_label_qss(self._label_color))
        self.setToolTip(label)

        self._edited_dot = EditedDot()

        self.spin.setSingleStep(step)
        if step >= 1.0:
            self.spin.setDecimals(0)
            self.slider.setTickInterval(int(step))
            self.slider.setSingleStep(int(step))

        if unit:
            self.spin.setSuffix(unit)

        self._value_pinned = False
        self._spin_full_width = 60 if unit else 50
        self.spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.spin.setMinimumWidth(0)
        self.spin.setMaximumWidth(0)  # collapsed at rest; expanded on hover/focus
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.spin.setStyleSheet(f"font-size: {THEME.font_size_base}px; background: transparent; border: none; font-weight: bold;")

        # Label-scrub: drag the label horizontally to change value
        self.label.setCursor(Qt.CursorShape.SizeHorCursor)
        self.label.installEventFilter(self)
        self._scrub_active = False
        self._scrub_start_x = 0.0
        self._scrub_start_val = 0.0
        self._band_drag = False

        header.addWidget(self.label)
        header.addWidget(self._edited_dot)
        header.addStretch()
        header.addWidget(self.spin)

        layout.addLayout(header)
        layout.addWidget(self.slider)

    def setToolTip(self, text: str) -> None:
        """Mirror onto the label: a child with its own tooltip shadows the parent's, and
        without one here the label would show nothing on hover. toolTip() is the already
        wrapped string, so the reset footer is never appended twice."""
        super().setToolTip(text)
        self.label.setToolTip(self.toolTip())

    def set_value_pinned(self, pinned: bool) -> None:
        """Keep the value box open at rest. Off by default — hover reveals it — but a
        panel of hidden numbers is unreadable if you work by the numbers."""
        self._value_pinned = pinned
        self.spin.setMaximumWidth(self._spin_full_width if pinned else 0)

    def enterEvent(self, event) -> None:
        self.spin.setMaximumWidth(self._spin_full_width)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if not (self._value_pinned or self.spin.hasFocus()):
            self.spin.setMaximumWidth(0)
        super().leaveEvent(event)

    # Extra clickable px above/below the thin slider so a near-miss on the
    # handle still grabs it. The slider widget's own rect is only ~handle-tall,
    # so clicks land here on the container's padding and get forwarded as a
    # real grab-drag — no layout/height change to the visible row.
    _GRAB_PAD = 8

    def _band_hit(self, pos) -> bool:
        if not self.isEnabled():
            return False
        r = self.slider.geometry()
        return r.left() <= pos.x() <= r.right() and (r.top() - self._GRAB_PAD) <= pos.y() <= (r.bottom() + self._GRAB_PAD)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._band_hit(event.position()):
            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier and self.slider._default_slider_value is not None:
                self.slider.setValue(self.slider._default_slider_value)
                self.slider.sliderReleased.emit()
            else:
                self._band_drag = True
                self.slider.setSliderDown(True)  # emits sliderPressed -> dragStarted
                self.slider.setValue(self.slider._value_from_x(event.position().x() - self.slider.x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._band_drag:
            self.slider.setValue(self.slider._value_from_x(event.position().x() - self.slider.x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._band_drag:
            self._band_drag = False
            self.slider.setSliderDown(False)  # emits sliderReleased -> commit + dragEnded
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _on_slider_changed(self, value: int) -> None:
        super()._on_slider_changed(value)
        self._update_edited_state()

    def setValue(self, value: float, _rebase_commit: bool = True) -> None:
        super().setValue(value, _rebase_commit=_rebase_commit)
        self._update_edited_state()

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self._update_edited_state()

    def _update_edited_state(self) -> None:
        if not self.isEnabled():
            self.label.setStyleSheet(slider_label_qss(THEME.text_muted))
            self._edited_dot.setVisible(False)
            return
        self.label.setStyleSheet(slider_label_qss(self._label_color))
        self._edited_dot.setVisible(abs(self.spin.value() - self._default) > 1e-6)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.spin:
            et = event.type()
            if et == QEvent.Type.FocusIn:
                self.spin.setMaximumWidth(self._spin_full_width)
            elif et == QEvent.Type.FocusOut:
                if not self.underMouse():
                    self.spin.setMaximumWidth(0)

        if obj is self.label:
            et = event.type()
            if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if not self.isEnabled():
                    return True
                self._scrub_active = True
                self._scrub_start_x = event.position().x()
                self._scrub_start_val = self.spin.value()
                return True
            if et == QEvent.Type.MouseMove and self._scrub_active and self.isEnabled():
                dx = event.position().x() - self._scrub_start_x
                span = self._max - self._min
                sensitivity = span / 400.0
                mods = event.modifiers()
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    sensitivity *= 0.1
                elif mods & Qt.KeyboardModifier.ControlModifier:
                    sensitivity *= 10.0
                new_val = max(self._min, min(self._max, self._scrub_start_val + dx * sensitivity))
                self.setValue(new_val, _rebase_commit=False)
                # throttle like groove drags; a direct emit renders per mouse-move
                self._schedule_emit()
                return True
            if et == QEvent.Type.MouseButtonRelease and self._scrub_active:
                self._scrub_active = False
                self._on_committed()
                return True
        return super().eventFilter(obj, event)


class PowerWarpSlider(CompactSlider):
    """
    CompactSlider variant with travel concentrated around `center` via a
    power-law warp: physical drag near center moves the value slowly (fine
    control), drag near the min/max extremes moves it quickly (coarse). Value
    and slider-int conversion stay exact inverses of each other so committed
    values still round-trip precisely; only the *spacing* is nonlinear.
    """

    def __init__(self, label: str, min_val: float, max_val: float, default_val: float, center: float, gamma: float = 2.2, **kwargs):
        self._warp_center = center
        self._warp_gamma = gamma
        self._warp_half_lo = center - min_val
        self._warp_half_hi = max_val - center
        super().__init__(label, min_val, max_val, default_val, **kwargs)

    def _to_int(self, value: float) -> int:
        if value >= self._warp_center:
            half = self._warp_half_hi
            frac = 0.0 if half <= 0 else (value - self._warp_center) / half
            sign = 1.0
        else:
            half = self._warp_half_lo
            frac = 0.0 if half <= 0 else (self._warp_center - value) / half
            sign = -1.0
        s = sign * max(0.0, frac) ** (1.0 / self._warp_gamma)
        return round(s * self._precision)

    def _from_int(self, i: int) -> float:
        s = i / self._precision
        frac = abs(s) ** self._warp_gamma
        half = self._warp_half_hi if s >= 0 else self._warp_half_lo
        return self._warp_center + math.copysign(frac * half, s)


class HueSlider(CompactSlider):
    """
    CompactSlider variant for 0–360° hue selection.
    The label color and slider handle track the current hue.
    """

    def __init__(self, label: str, default_val: float = 0.0, parent=None):
        super().__init__(label, 0.0, 360.0, default_val, step=1.0, precision=1, unit="°", parent=parent)
        self._apply_hue(default_val)

    def _apply_hue(self, hue_deg: float) -> None:
        """Update slider handle to match the current hue; label stays grey (yellow when edited)."""
        if self.isEnabled():
            h = int(hue_deg) % 360
            color = QColor.fromHsv(h, 200, 210).name()
        else:
            color = THEME.text_muted
        self._update_edited_state()
        self.slider.setStyleSheet(slider_handle_qss(color))

    def _on_slider_changed(self, value: int) -> None:
        super()._on_slider_changed(value)
        self._apply_hue(value / self._precision)

    def setValue(self, value: float, _rebase_commit: bool = True) -> None:
        super().setValue(value, _rebase_commit=_rebase_commit)
        self._apply_hue(value)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self._apply_hue(self.value())


def _kelvin_handle_color(kelvin: float) -> QColor:
    """Blackbody colour (Tanner Helland approximation), softened to the same
    saturation/brightness as the HueSlider handles."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    c = QColor(*(int(min(255.0, max(0.0, v))) for v in (r, g, b)))
    if c.hue() < 0:
        return QColor(210, 210, 210)
    return QColor.fromHsv(c.hue(), min(c.saturation(), 200), 210)


class KelvinSlider(CompactSlider):
    """
    Kelvin readout with mired-linear travel: slider ints are mired*10, so warm
    (low K) sits on the right and equal drag distance = equal perceived shift.
    The handle tints to the blackbody colour of the current temperature.
    """

    def __init__(self, label: str, parent=None):
        super().__init__(label, 3000.0, 12000.0, 5500.0, step=50.0, precision=1, unit="K", parent=parent)
        self._spin_full_width = 68  # room for "12000K"
        self._apply_temp(5500.0)

    def _to_int(self, value: float) -> int:
        return round(1e7 / max(value, 1.0))

    def _from_int(self, i: int) -> float:
        # Snap to 10K so the 5500 default round-trips exactly (edited-state check).
        return round(1e6 / (i / 10.0) / 10.0) * 10.0

    def _apply_temp(self, kelvin: float) -> None:
        color = _kelvin_handle_color(kelvin).name() if self.isEnabled() else THEME.text_muted
        self.slider.setStyleSheet(slider_handle_qss(color))

    def _on_slider_changed(self, value: int) -> None:
        super()._on_slider_changed(value)
        self._apply_temp(self._from_int(value))

    def setValue(self, value: float, _rebase_commit: bool = True) -> None:
        super().setValue(value, _rebase_commit=_rebase_commit)
        self._apply_temp(self.value())

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self._apply_temp(self.value())


def clone_slider(src: CompactSlider) -> CompactSlider:
    """A second widget onto the same control, for the Favourites panel: a QWidget has one
    parent, so a favourite mirrors its slider rather than re-parenting it out of its section.

    Exact-type dispatch, and unhandled classes raise: a PowerWarpSlider rebuilt as a plain
    CompactSlider would keep its range but silently lose its nonlinear travel."""
    label = src.label.text()
    cls = type(src)
    if cls is HueSlider:
        return HueSlider(label, src._default)
    if cls is KelvinSlider:
        return KelvinSlider(label)
    if cls is CompactSlider:
        return CompactSlider(
            label,
            src._min,
            src._max,
            src._default,
            step=src.spin.singleStep(),
            precision=src._precision,
            color=src._label_color,
            has_neutral=src.slider.objectName() == "neutral_slider",
            unit=src.spin.suffix(),
            inverted=src.slider.invertedAppearance(),
        )
    raise TypeError(f"clone_slider: unhandled slider class {cls.__name__}")


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

        # Draw Handles — filled with accent + 1px dark stroke ring for visibility
        r = self._handle_r
        for cx in (x1, x2):
            painter.setBrush(QColor(THEME.accent_primary))
            painter.setPen(QPen(QColor("#050505"), 1))
            painter.drawEllipse(cx - r, y - r, r * 2, r * 2)

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


def apply_slider_value_visibility(root: QWidget, pinned: bool) -> None:
    """Pin or unpin every CompactSlider under a widget tree, so the preference reaches
    panels built long before it was toggled."""
    for slider in root.findChildren(CompactSlider):
        slider.set_value_pinned(pinned)
