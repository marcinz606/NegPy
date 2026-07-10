"""Large pop-out window for the Scanlight live view.

Hosts a `RoiImageLabel` plus an inline toolbar (Scan / Retake) and a status line,
so a whole roll can be framed, focused, and scanned without switching back to the
side panel. The live image carries a magnifier cursor: a click aims the camera
focus magnifier at that spot, a double-click returns to full frame. The buttons
emit signals; `ScanlightSidebar` wires them and mirrors scanning state + status.
"""

import time

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QKeySequence, QShortcut
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget

from negpy.desktop.view.sidebar.roi_image import RoiImageLabel
from negpy.desktop.view.styles.theme import THEME


class SettingStepper(QWidget):
    """Compact ‹ value › stepper for a camera setting, in place of a dropdown that would
    otherwise list every ISO/shutter/aperture step and fill the screen.

    Exposes the slice of the QComboBox API the settings refresh already uses
    (clear/addItem/count/findData/currentData/currentIndex/setCurrentIndex/setEnabled), so
    populating it is unchanged; the arrows step one option and emit `activated(index)`.
    """

    activated = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._items: list[tuple[str, object]] = []  # (label shown, raw value sent to the camera)
        self._index = -1
        self._last_step = 0.0  # monotonic time of the last arrow press (see hasFocus)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self._prev = QToolButton()
        self._prev.setText("‹")
        self._next = QToolButton()
        self._next.setText("›")
        self._value = QLabel("—")
        self._value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value.setMinimumWidth(64)
        self._value.setStyleSheet(f"color: {THEME.text_primary};")
        row.addWidget(self._prev)
        row.addWidget(self._value, 1)
        row.addWidget(self._next)
        self._prev.clicked.connect(lambda: self._step(-1))
        self._next.clicked.connect(lambda: self._step(1))
        self._sync()

    # ── stepping ──────────────────────────────────────────────────────
    def _step(self, delta: int) -> None:
        if not self._items:
            return
        new = min(max(self._index + delta, 0), len(self._items) - 1)
        if new != self._index:
            self._index = new
            self._last_step = time.monotonic()
            self._sync()
            self.activated.emit(new)  # one SET per click (no auto-repeat → no flooding)

    def _sync(self) -> None:
        on = self.isEnabled()
        self._value.setText(self._items[self._index][0] if 0 <= self._index < len(self._items) else "—")
        self._prev.setEnabled(on and self._index > 0)
        self._next.setEnabled(on and self._index < len(self._items) - 1)

    # ── QComboBox-ish API used by ScanlightSidebar._refresh_camera_settings ──
    def clear(self) -> None:
        self._items.clear()
        self._index = -1
        self._sync()

    def addItem(self, label: str, raw) -> None:
        self._items.append((label, raw))
        if self._index < 0:
            self._index = 0
        self._sync()

    def count(self) -> int:
        return len(self._items)

    def findData(self, raw) -> int:
        return next((i for i, (_, r) in enumerate(self._items) if r == raw), -1)

    def currentData(self):
        return self._items[self._index][1] if 0 <= self._index < len(self._items) else None

    def currentIndex(self) -> int:
        return self._index

    def setCurrentIndex(self, idx: int) -> None:
        if 0 <= idx < len(self._items):
            self._index = idx
            self._sync()

    def setEnabled(self, on: bool) -> None:
        super().setEnabled(on)
        self._sync()

    def hasFocus(self) -> bool:
        # Treat a just-pressed arrow as "busy" so the ~1 Hz settings refresh doesn't snap the
        # value back while the camera's reported `cur` catches up to the step the user made.
        return (time.monotonic() - self._last_step) < 1.2 or super().hasFocus()


class LiveViewWindow(QDialog):
    """Resizable, non-modal live-view window: magnifiable image + capture toolbar."""

    closed = pyqtSignal()
    scanRequested = pyqtSignal()
    retakeRequested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scanlight — Live View")
        self.setModal(False)
        self.resize(900, 720)
        layout = QVBoxLayout(self)

        # ── capture toolbar (mirrors the panel so you needn't switch tabs) ──
        bar = QHBoxLayout()
        self.scan_btn = QPushButton(qta.icon("fa5s.camera-retro", color=THEME.text_primary), " Scan")
        self.scan_btn.setFixedHeight(36)
        self.retake_btn = QPushButton(qta.icon("fa5s.redo", color=THEME.text_primary), " Retake")
        self.retake_btn.setToolTip("Re-capture the current frame without advancing the counter")
        bar.addWidget(self.scan_btn, 2)
        bar.addWidget(self.retake_btn, 1)
        layout.addLayout(bar)

        self.image = RoiImageLabel()
        self.image.roi_mode = False  # clicks aim the magnifier here, not a calibration ROI
        # Magnifier cursor over the live image → signals "click to magnify here".
        _loupe = qta.icon("fa5s.search-plus", color="#EDEBE4").pixmap(22, 22)
        self.image.setCursor(QCursor(_loupe, 9, 9))  # hotspot ≈ the lens centre
        layout.addWidget(self.image, 1)

        # ── live camera settings (populated from the stream's settings JSON) ──
        # Compact ‹ value › steppers instead of dropdowns: shutter/ISO span dozens of steps,
        # so a full popup would fill the screen. Arrows nudge one stop at a time.
        settings_row = QHBoxLayout()
        self.iso_stepper = SettingStepper()
        self.shutter_stepper = SettingStepper()
        self.aperture_stepper = SettingStepper()
        # No white-balance control: the scan decodes RAW with a fixed neutral WB
        # (use_camera_wb=False), so the camera's WB only tints the preview, never the result.
        for tag_text, stepper, tip in (
            ("ISO", self.iso_stepper, "ISO sensitivity"),
            ("Shutter", self.shutter_stepper, "Shutter speed"),
            ("f", self.aperture_stepper, "Aperture (needs an electronically controlled lens)"),
        ):
            tag = QLabel(tag_text)
            tag.setAlignment(Qt.AlignmentFlag.AlignHCenter)  # label sits centred above its value
            tag.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
            stepper.setToolTip(tip)
            col = QVBoxLayout()  # label stacked over the ‹ value › stepper (clearer than side-by-side)
            col.setSpacing(2)
            col.addWidget(tag)
            col.addWidget(stepper)
            settings_row.addLayout(col, 1)
        layout.addLayout(settings_row)

        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.scan_btn.clicked.connect(lambda: self.scanRequested.emit())
        self.retake_btn.clicked.connect(lambda: self.retakeRequested.emit())

        # Keyboard shortcuts while the pop-up is focused (no text fields here, so
        # letter keys are safe). The buttons respect their disabled/gated state.
        for key, btn in (("S", self.scan_btn), ("R", self.retake_btn)):
            QShortcut(QKeySequence(key), self, btn.click)
        self.scan_btn.setToolTip("Scan / Stop  (shortcut: S)")
        self.retake_btn.setToolTip("Re-capture the current frame without advancing the counter  (shortcut: R)")

    def set_scanning(self, active: bool) -> None:
        """Mirror the panel's Scan/Stop toggle on the pop-up button."""
        if active:
            self.scan_btn.setText(" Stop")
            self.scan_btn.setIcon(qta.icon("fa5s.stop", color=THEME.text_primary))
        else:
            self.scan_btn.setText(" Scan")
            self.scan_btn.setIcon(qta.icon("fa5s.camera-retro", color=THEME.text_primary))

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def closeEvent(self, ev) -> None:
        self.closed.emit()
        super().closeEvent(ev)
