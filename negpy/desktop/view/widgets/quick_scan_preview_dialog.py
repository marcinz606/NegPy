"""Modal pop-up: a single low-res preview scan and a crop window, for devices with
no addressable frame adapter (Plustek: one manual holder, not a strip/roll feeder —
see StripPreviewDialog for that case).

Read after ``exec()`` via ``window()``.
"""

import qtawesome as qta
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from negpy.desktop.converters import ImageConverter
from negpy.desktop.view.styles.templates import StatusStrip, pin_dialog_default
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.scan_preview_common import RollPreviewSignalsMixin, preview_positive
from negpy.desktop.view.widgets.scan_window_label import ScanWindowLabel
from negpy.desktop.workers.scan_worker import RollPreviewRequest
from negpy.infrastructure.scanners.base import ScannerDevice

_PREVIEW_FALLBACK_DPI = 500  # only when the device reports no DPI list at all
_PREVIEW_SLOT = 1  # PerFrameRollSession's only slot on a frame-less device


class QuickScanPreviewDialog(RollPreviewSignalsMixin, QDialog):
    """Preview the current holder position at low res; set a crop window for the real scan."""

    def __init__(self, controller, device: ScannerDevice, initial_window=None, film_type: str = "negative", parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._device = device
        self._film_type = film_type
        self._caps = device.capabilities
        self._previewing = False
        self._scan_now = False  # set when the user chooses "Scan" over "Use"
        self.setWindowTitle("Preview — set the scan window")
        self.setModal(True)
        self.resize(560, 480)

        layout = QVBoxLayout(self)

        help_lbl = QLabel(
            "Preview the current holder position, then drag to crop — a corner to resize, "
            "inside to move. Use (apply and return) or Scan (start scanning now)."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet(
            f"color: {THEME.text_secondary}; font-size: {THEME.font_size_small}px;"
            f" background: rgba(255,255,255,0.04); border-radius: 6px; padding: 6px 8px;"
        )
        layout.addWidget(help_lbl)

        top = QHBoxLayout()
        top.addWidget(QLabel("Preview DPI"))
        self.preview_dpi_combo = QComboBox()
        for dpi in sorted(self._caps.supported_dpi) or [_PREVIEW_FALLBACK_DPI]:
            self.preview_dpi_combo.addItem(str(dpi), dpi)
        self.preview_dpi_combo.setCurrentIndex(0)  # lowest: fastest, framing only
        self.preview_dpi_combo.setToolTip("Resolution used for the preview scan")
        top.addWidget(self.preview_dpi_combo)
        top.addStretch()
        self.preview_btn = QPushButton(qta.icon("fa5s.eye", color=THEME.text_primary), " Preview")
        self.preview_btn.clicked.connect(self._on_preview)
        top.addWidget(self.preview_btn)
        layout.addLayout(top)

        self.label = ScanWindowLabel()
        self.label.set_window(tuple(initial_window) if initial_window else None)
        layout.addWidget(self.label, 1)

        # One reserved row: the pass that is running, or the message it left behind.
        self.status_strip = StatusStrip(lines=1)
        layout.addWidget(self.status_strip)

        btns = QHBoxLayout()
        self.clear_btn = QPushButton("Clear crop")
        self.clear_btn.setToolTip("Scan the whole frame instead")
        self.clear_btn.clicked.connect(self.label.clear_window)
        btns.addWidget(self.clear_btn)
        btns.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        btns.addWidget(self.cancel_btn)
        self.ok_btn = QPushButton("Apply window")
        self.ok_btn.setToolTip("Keep this window and return to the Scan panel")
        self.ok_btn.clicked.connect(self.accept)
        btns.addWidget(self.ok_btn)
        self.scan_btn = QPushButton(qta.icon("fa5s.play", color=THEME.text_primary), " Scan frame")
        self.scan_btn.setToolTip("Scan now with the current settings")
        self.scan_btn.clicked.connect(self._on_scan_clicked)
        btns.addWidget(self.scan_btn)
        pin_dialog_default(self.scan_btn, self.clear_btn, self.cancel_btn, self.ok_btn)
        layout.addLayout(btns)

        self._connect_preview_signals()

    # ── result getters ────────────────────────────────────────────────

    def window(self):
        return self.label.window()

    def scan_requested(self) -> bool:
        """True when the dialog was accepted via Scan (start now), not Use."""
        return self._scan_now

    # ── ui state ──────────────────────────────────────────────────────

    def _on_scan_clicked(self) -> None:
        self._scan_now = True
        self.accept()

    def _on_cancel_clicked(self) -> None:
        """Stop the pass in flight, or leave when there is none."""
        if self._previewing:
            self.stop_preview()
            return
        self.reject()

    def _set_previewing(self, busy: bool) -> None:
        self.preview_btn.setEnabled(not busy)
        # Committing mid-pass would hand the scan a unit the preview still holds.
        self.ok_btn.setEnabled(not busy)
        self.scan_btn.setEnabled(not busy)
        self.cancel_btn.setText("Stop preview" if busy else "Cancel")
        if busy:
            self.status_strip.start_progress("Previewing… %p%")
        else:
            self.status_strip.stop_progress()

    def _preview_dpi(self) -> int:
        return int(self.preview_dpi_combo.currentData() or _PREVIEW_FALLBACK_DPI)

    def _on_preview(self) -> None:
        if self._previewing:
            return
        req = RollPreviewRequest(
            device=self._device,
            slots=(_PREVIEW_SLOT,),
            dpi=self._preview_dpi(),
            offsets={},
        )
        try:
            self._controller.start_roll_preview(req)
        except Exception as e:
            self.status_strip.set_message(f"Scanner busy — {e}")
            return
        self._previewing = True
        self._set_previewing(True)
        self.status_strip.set_message("Previewing…")

    @pyqtSlot(object)
    def _on_preview_ready(self, preview) -> None:
        if preview.slot != _PREVIEW_SLOT:
            return
        if preview.error is not None:
            self.status_strip.set_message(f"Preview failed: {preview.error}")
            return
        try:
            positive = preview_positive(preview.rgb, self._film_type)
            pixmap = QPixmap.fromImage(ImageConverter.to_qimage(positive))
        except Exception as e:
            self.status_strip.set_message(f"Could not display preview: {e}")
            return
        self.label.set_frame(pixmap)

    @pyqtSlot()
    def _on_preview_finished(self) -> None:
        self._previewing = False
        self._set_previewing(False)
        if not self.status_strip.message().startswith("Preview failed") and not self.status_strip.message().startswith("Could not display"):
            self.status_strip.set_message("")

    @pyqtSlot(str)
    def _on_error(self, msg) -> None:
        if not self._previewing:
            return
        self._previewing = False
        self._set_previewing(False)
        self.status_strip.set_message(f"Preview failed: {msg}")

    @pyqtSlot()
    def _on_cancelled(self) -> None:
        if not self._previewing:
            return
        self._previewing = False
        self._set_previewing(False)
        self.status_strip.set_message("Preview cancelled.")
