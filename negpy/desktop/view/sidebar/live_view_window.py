"""Pop-out window for the Scanlight live view: frame, focus, meter and scan a whole roll here.

A click aims the camera focus magnifier at that spot, a double-click returns to full frame.
The buttons emit signals; `ScanlightSidebar` wires them and mirrors scanning state and status.
"""

from negpy.kernel.system.i18n import mark, tr

import time
from typing import Optional

import numpy as np
import qtawesome as qta
from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor, QKeySequence
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QProgressBar, QPushButton, QVBoxLayout, QWidget

from negpy.desktop.view.shortcut_registry import key_for, tooltip_with_shortcut
from negpy.desktop.view.sidebar.roi_image import RoiImageLabel
from negpy.desktop.view.styles.templates import (
    SCAN_BUTTON_HEIGHT,
    default_button_height,
    hint_label,
    icon_text,
    labeled_action,
    pin_dialog_default,
    set_hint_kind,
    tool_toggle,
    wrap_tooltip,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.charts import MiniRGBHistogramWidget
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.desktop.view.widgets.floating_panel import float_over_app
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.services.capture.calibration import shutter_displays

#: Progress-bar chunk color per triplet channel. The live view freezes during a triplet,
#: because the capture holds the camera without gaps, so the bar shows the R to G to B switch.
_CHANNEL_COLORS = {"R": THEME.channel_red_text, "G": THEME.channel_green_text, "B": THEME.channel_blue_text}
_DONE_COLOR = THEME.status_success
_FLASH_MS = 1500
_FOCUS_AT_PEAK = 0.99


# The camera's own label of an item, which a write sends; the item text is only its display.
_LABEL_ROLE = Qt.ItemDataRole.UserRole + 1


class SettingStepper(StepCombo):
    """StepCombo for a camera setting, with no list: every ISO, shutter or aperture step would
    fill the screen. The arrows stop at the ends, so a step never wraps from the slowest
    shutter to the fastest. `shutter=True` shows each label as a camera shows it ('1/30')."""

    def __init__(self, parent=None, *, shutter: bool = False) -> None:
        super().__init__(parent)
        self.setPlaceholderText(tr("—"))
        self._shutter = shutter
        self._last_step = 0.0  # monotonic time of the last step (see stepping)
        self.activated.connect(self._on_stepped)

    def _on_stepped(self, _index: int) -> None:
        self._last_step = time.monotonic()

    def addItem(self, label: str, raw=None) -> None:  # noqa: N802
        """A placeholder keeps QComboBox at no choice; the first option is the shown value."""
        super().addItem(label, raw)
        self.setItemData(self.count() - 1, label, _LABEL_ROLE)
        if self._shutter:
            # Every item, because the whole ladder decides third or half stops.
            ladder = [self.label_at(i) for i in range(self.count())]
            for i, text in enumerate(shutter_displays(ladder)):
                self.setItemText(i, text)
        if self.currentIndex() < 0:
            self.setCurrentIndex(0)

    def label_at(self, index: int) -> str:
        return str(self.itemData(index, _LABEL_ROLE) or "")

    def current_label(self) -> str:
        return self.label_at(self.currentIndex()) if self.currentIndex() >= 0 else ""

    def find_label(self, label: str) -> int:
        return next((i for i in range(self.count()) if self.label_at(i) == label), -1)

    def _step_target(self, step: int) -> int:
        target = self.currentIndex() + step
        return target if 0 <= target < self.count() else -1

    def showPopup(self) -> None:  # noqa: N802
        pass

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        super().mouseMoveEvent(e)
        if self._hover == "face":  # the face opens nothing, so it does not light up
            self._hover = None

    def stepping(self) -> bool:
        """True just after a step, so the settings refresh does not snap the value back while the
        camera's reported `cur` catches up. The window must outlast a debounced and verified
        camera write, or the stepper flickers back to the old value mid-write."""
        return (time.monotonic() - self._last_step) < 2.5 or self.hasFocus()


class LiveViewWindow(QDialog):
    """Resizable, non-modal live-view window: magnifiable image + capture toolbar."""

    closed = pyqtSignal()
    scanRequested = pyqtSignal()
    retakeRequested = pyqtSignal()
    meterRequested = pyqtSignal()
    channelsChanged = pyqtSignal(str)  # the letter whose toggle moved: R, G, B or IR
    captureDelayChanged = pyqtSignal(float)  # seconds, stepped on the toolbar's delay stepper

    def __init__(self, parent=None, *, repo=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Scanlight — Live View"))
        self.setModal(False)
        float_over_app(self)
        self.resize(900, 720)
        layout = QVBoxLayout(self)

        # The capture toolbar mirrors the panel, so the operator does not switch tabs.
        bar = QHBoxLayout()
        self.scan_btn = labeled_action("fa5s.camera-retro", tr("Scan"), tr("Capture this frame, or stop the capture"))
        self.scan_btn.setObjectName("scan_btn")
        self.scan_btn.setProperty("scanning", "false")
        self.scan_btn.setIcon(qta.icon("fa5s.camera-retro", color=THEME.text_on_accent))
        self.scan_btn.setFixedHeight(SCAN_BUTTON_HEIGHT)
        self.retake_btn = labeled_action("fa5s.redo", tr("Retake"), tr("Re-capture the current frame without advancing the counter"))
        bar.addWidget(self.scan_btn, 2)
        # The card's Capture Delay, where Scan is pressed. Mirrored both ways by the panel.
        self.delay_stepper = SettingStepper()
        self.set_capture_delay(0.0)
        self.delay_stepper.setToolTip(
            tr("Capture Delay: wait this long after Scan or Retake before the shot, so a shaky table can settle.")
        )
        self.delay_stepper.activated.connect(lambda _i: self.captureDelayChanged.emit(float(self.delay_stepper.currentData())))
        bar.addWidget(self.delay_stepper)
        bar.addWidget(self.retake_btn, 1)
        # Which exposures Retake shoots, in the channel colors: R, G and B for a triplet preset.
        # IR is the Infrared setting, which Scan shoots too.
        self.channel_widget = QWidget()
        channels = QHBoxLayout(self.channel_widget)
        channels.setContentsMargins(0, 0, 0, 0)
        self.channel_btns: dict[str, QPushButton] = {}
        for letter, name, text in (
            ("R", mark("red"), THEME.channel_red_text),
            ("G", mark("green"), THEME.channel_green_text),
            ("B", mark("blue"), THEME.channel_blue_text),
            ("IR", "", ""),
        ):
            tip = (
                tr("Shoot the infrared exposure with each frame")
                if letter == "IR"
                else tr("Retake shoots the {color} exposure again over this frame's.", color=tr(name))
            )
            btn = tool_toggle("", letter, tip)
            if text:
                btn.setStyleSheet(btn.styleSheet() + f" QPushButton:checked {{ color: {text}; }}")
            btn.setChecked(letter != "IR")
            btn.toggled.connect(lambda _on, letter=letter: self.channelsChanged.emit(letter))
            self.channel_btns[letter] = btn
            channels.addWidget(btn)
        bar.addWidget(self.channel_widget)
        # One RAW probe sets the shutter for the light as lit.
        self.meter_btn = labeled_action(
            "fa5s.tachometer-alt",
            tr("Meter"),
            tr("Set the shutter from one RAW probe: a negative's base or a slide's highlights, just under clipping"),
        )
        bar.addWidget(self.meter_btn, 1)
        # Exposure readouts of the preview frame.
        self.histogram_btn = tool_toggle("fa5s.chart-area", "", tr("Show the frame's R, G and B histogram under the image"))
        self.zebra_btn = tool_toggle("fa5s.grip-lines", "", tr("Hatch the clipped pixels in the Clipping Overlay colors"))
        bar.addWidget(self.histogram_btn)
        bar.addWidget(self.zebra_btn)
        layout.addLayout(bar)

        self.image = RoiImageLabel()
        self.image.roi_mode = False  # clicks aim the magnifier here, not a calibration ROI
        # Magnifier cursor over the live image → signals "click to magnify here".
        _loupe = qta.icon("fa5s.search-plus", color=THEME.text_primary).pixmap(22, 22)
        self.image.setCursor(QCursor(_loupe, 9, 9))  # hotspot ≈ the lens centre
        layout.addWidget(self.image, 1)

        self.histogram = MiniRGBHistogramWidget()
        self.histogram.setFixedHeight(2 * default_button_height())
        layout.addWidget(self.histogram)
        self._repo = repo
        for btn, key in ((self.histogram_btn, "live_view_histogram"), (self.zebra_btn, "live_view_zebra")):
            btn.setChecked(bool(repo.get_global_setting(key, False)) if repo is not None else False)
            btn.toggled.connect(lambda on, key=key: self._remember(key, on))
        self.histogram.setVisible(self.histogram_btn.isChecked())
        self.histogram_btn.toggled.connect(self.histogram.setVisible)
        self.zebra_btn.toggled.connect(self._on_zebra_toggled)

        self.focus_label = hint_label("")
        self.focus_label.setToolTip(
            wrap_tooltip(tr("Sharpness compared with the best seen: turn the focus ring past the peak and back to find it."))
        )
        self._focus_kind = "muted"
        layout.addWidget(self.focus_label)
        self.set_focus(None)

        # Shown in the image's place on bodies that advertise no live view (issue #621). The
        # window still scans: only the preview pane is replaced, so the toolbar, the settings row
        # and the frame counter all keep working.
        self.no_preview = hint_label()
        self.no_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_preview.setVisible(False)
        layout.addWidget(self.no_preview, 1)

        # Live camera settings, populated from the stream's settings JSON.
        # Compact steppers instead of dropdowns: shutter and ISO span dozens of steps, so a full
        # popup would fill the screen. The arrows nudge one stop at a time.
        # Wrapped in a widget so the panel can hide the whole row as a unit. A calibrated RGB
        # preset locks ISO, shutter and aperture, because changing them would falsify the scan,
        # so the steppers show only for white-light presets and normal camera-only scanning.
        self.settings_widget = QWidget()
        settings_row = QHBoxLayout(self.settings_widget)
        settings_row.setContentsMargins(0, 0, 0, 0)
        self.iso_stepper = SettingStepper()
        self.shutter_stepper = SettingStepper(shutter=True)
        self.aperture_stepper = SettingStepper()
        # No white-balance control: the scan decodes RAW with a fixed neutral WB, so the camera's
        # WB only tints the preview, never the result.
        for tag_text, stepper, tip in (
            ("ISO", self.iso_stepper, tr("ISO sensitivity")),
            (tr("Shutter"), self.shutter_stepper, tr("Shutter speed")),
            (tr("Aperture"), self.aperture_stepper, tr("Aperture (needs an electronically controlled lens)")),
        ):
            tag = hint_label(tag_text)
            tag.setAlignment(Qt.AlignmentFlag.AlignHCenter)  # label sits centred above its value
            stepper.setToolTip(tip)
            col = QVBoxLayout()  # label stacked over the ‹ value › stepper (clearer than side-by-side)
            col.setSpacing(THEME.space_xs)
            col.addWidget(tag)
            col.addWidget(stepper)
            settings_row.addLayout(col, 1)
        layout.addWidget(self.settings_widget)

        # Capture progress lives here as well as on the side panel: while scanning a roll the
        # operator watches this window, and the bar reaching 100% is the signal that the film may
        # be advanced. Below the view, mirroring the calibration window's bar placement.
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFormat(tr("Capturing… %p%"))
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        # Invalidates a pending post-capture flash when a new capture starts underneath it.
        self._flash_token = 0

        self.status = hint_label("")
        layout.addWidget(self.status)

        self.scan_btn.clicked.connect(lambda: self.scanRequested.emit())
        self.retake_btn.clicked.connect(lambda: self.retakeRequested.emit())
        self.meter_btn.clicked.connect(lambda: self.meterRequested.emit())

        # Pin Scan as the dialog's permanent default button. Without this, Qt hands "default"
        # status to whichever autoDefault button was clicked most recently, so pressing Retake
        # once made Enter keep retaking until Scan was clicked again to reclaim it (issue #997).
        pin_dialog_default(self.scan_btn, scope=self)

        # Plain letter keys are safe: the pop-up has no text fields.
        self._key_buttons = {
            "live_view_scan": self.scan_btn,
            "live_view_scan_space": self.scan_btn,
            "live_view_retake": self.retake_btn,
            "live_view_meter": self.meter_btn,
            "live_view_histogram": self.histogram_btn,
            "live_view_zebra": self.zebra_btn,
            "live_view_capture_red": self.channel_btns["R"],
            "live_view_capture_green": self.channel_btns["G"],
            "live_view_capture_blue": self.channel_btns["B"],
            "live_view_capture_ir": self.channel_btns["IR"],
        }
        # A focused button takes Space for itself, so a click on Retake or a toggle would leave
        # Space working that control instead of its binding. The letters already reach the
        # window; the stepper arrows keep Space, their only key.
        for btn in self._key_buttons.values():
            btn.installEventFilter(self)
        remember_dialog_geometry(self, repo, "live_view")

    def _remember(self, key: str, on: bool) -> None:
        if self._repo is not None:
            self._repo.save_global_setting(key, on)

    def _on_zebra_toggled(self, on: bool) -> None:
        if not on:
            self.image.set_zebra(None)

    def set_histogram(self, bins: Optional[np.ndarray]) -> None:
        """Show one frame's `(4, 256)` R, G, B, luma counts, or clear the row with None."""
        self.histogram.update_data(bins)

    def selected_rgb(self) -> tuple[str, ...]:
        """The visible exposures Retake shoots, in capture order."""
        return tuple(letter for letter in ("R", "G", "B") if self.channel_btns[letter].isChecked())

    def apply_shortcut_tooltips(self) -> None:
        for btn in dict.fromkeys(self._key_buttons.values()):
            ids = [action_id for action_id, b in self._key_buttons.items() if b is btn]
            btn.setToolTip(wrap_tooltip(tooltip_with_shortcut(btn.plain_tooltip, ids)))

    def _key_button(self, ev):
        modifiers = ev.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        # "?" arrives as Key_Question with Shift held: drop Shift only when the exact chord has no match.
        for mods in (modifiers, modifiers & ~Qt.KeyboardModifier.ShiftModifier):
            pressed = QKeySequence(ev.key() | mods.value)
            for action_id, btn in self._key_buttons.items():
                key = key_for(action_id)
                if key and QKeySequence(key) == pressed:
                    return btn
        return None

    def keyPressEvent(self, ev) -> None:
        btn = self._key_button(ev)
        if btn is None:
            super().keyPressEvent(ev)
        elif not ev.isAutoRepeat():
            btn.click()

    def eventFilter(self, watched, ev) -> bool:
        if ev.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and ev.key() == Qt.Key.Key_Space and self._key_button(ev):
            if ev.type() == QEvent.Type.KeyPress:
                self.keyPressEvent(ev)
            return True
        return super().eventFilter(watched, ev)

    def set_preview_available(self, available: bool, reason: str = "") -> None:
        """Swap the preview pane for an explanation on bodies that cannot stream.

        Scanning is unaffected, so the window stays fully usable — hiding it entirely would
        strip the only Scan button in the app and lock these cameras out (issue #621).
        """
        self.image.setVisible(available)
        self.focus_label.setVisible(available)
        self.histogram.setVisible(available and self.histogram_btn.isChecked())
        self.histogram_btn.setEnabled(available)
        self.zebra_btn.setEnabled(available)
        self.no_preview.setVisible(not available)
        if not available:
            self.no_preview.setText(
                tr("{reason}\n\nFraming and focus have to be set on the camera itself. Scanning works as usual.", reason=reason)
            )
        self.setWindowTitle(tr("Scanlight — Live View") if available else tr("Scanlight — Scan (no live view)"))

    def set_focus(self, fraction: Optional[float]) -> None:
        """Show the focus meter reading: 0..1 of the peak, or None while there is none."""
        if fraction is None:
            text, kind = tr("Focus meter: no reading"), "muted"
        else:
            at_peak = fraction >= _FOCUS_AT_PEAK
            text = tr("Focus meter: at peak") if at_peak else tr("Focus meter: {value}% of peak", value=round(fraction * 100))
            kind = "success" if at_peak else "muted"
        self.focus_label.setText(text)
        if kind != self._focus_kind:
            self._focus_kind = kind
            set_hint_kind(self.focus_label, kind)

    def set_progress(self, frac: float) -> None:
        self._flash_token += 1
        self.progress.setVisible(True)
        self.progress.setValue(int(frac * 100))

    def set_channel(self, letter: str) -> None:
        """Tint the bar in the channel being exposed and name it in the bar text."""
        self._flash_token += 1
        color = _CHANNEL_COLORS.get(letter)
        if color:
            self.progress.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
            self.progress.setFormat(tr("Capturing {letter}… %p%", letter=letter))

    def flash_captured(self, frame: str) -> None:
        """Fill the bar green with a checkmark for a beat, then hide it — the 'frame is
        in the can, film may be advanced' moment the frozen live view can't show."""
        self._flash_token += 1
        token = self._flash_token
        self.progress.setStyleSheet(f"QProgressBar::chunk {{ background-color: {_DONE_COLOR}; }}")
        self.progress.setFormat(tr("✓ Frame {frame} captured", frame=frame) if frame else tr("✓ Captured"))
        self.progress.setValue(100)
        self.progress.setVisible(True)
        QTimer.singleShot(_FLASH_MS, lambda: self._end_flash(token))

    def _end_flash(self, token: int) -> None:
        if token == self._flash_token:  # stale flashes must not hide a newer capture's bar
            self.clear_progress()

    def clear_progress(self) -> None:
        self._flash_token += 1
        self.progress.setVisible(False)
        self.progress.setStyleSheet("")
        self.progress.setFormat(tr("Capturing… %p%"))

    #: The delay stepper's rungs; the card's slider sets any value, which joins the ladder in place.
    DELAY_LADDER_S = (0.0, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def set_capture_delay(self, seconds: float) -> None:
        seconds = round(float(seconds), 2)
        self.delay_stepper.blockSignals(True)
        self.delay_stepper.clear()
        for rung in sorted(set(self.DELAY_LADDER_S) | {seconds}):
            self.delay_stepper.addItem(tr("{seconds:g} s", seconds=rung), rung)
        self.delay_stepper.setCurrentIndex(self.delay_stepper.findData(seconds))
        self.delay_stepper.blockSignals(False)

    def set_scanning(self, active: bool) -> None:
        """Mirror the panel's Scan/Stop toggle on the pop-up button."""
        if active:
            self.scan_btn.setText(icon_text(tr("Stop")))
            self.scan_btn.setIcon(qta.icon("fa5s.stop", color=THEME.text_primary))
        else:
            self.scan_btn.setText(icon_text(tr("Scan")))
            self.scan_btn.setIcon(qta.icon("fa5s.camera-retro", color=THEME.text_on_accent))
        # Qt re-reads a QSS property selector only on a repolish.
        self.scan_btn.setProperty("scanning", "true" if active else "false")
        style = self.scan_btn.style()
        style.unpolish(self.scan_btn)
        style.polish(self.scan_btn)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def done(self, result: int) -> None:
        # Esc reaches here without a closeEvent, so the session cleanup hangs off done().
        self.closed.emit()
        super().done(result)
