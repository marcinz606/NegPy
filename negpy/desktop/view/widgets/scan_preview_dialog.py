"""Modal pop-up: preview a frame, set one scan window + offset reused for every frame.

Result read via ``window_rect()`` / ``frame_offset()`` after ``exec()``.
"""

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from negpy.desktop.converters import ImageConverter
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.scan_window_label import ScanWindowLabel
from negpy.desktop.workers.scan_worker import ScanRequest
from negpy.infrastructure.scanners.base import ScannerDevice
from negpy.infrastructure.scanners.params import ScanParams, scan_window_to_area

_PREVIEW_FALLBACK_DPI = 500


class ScanPreviewDialog(QDialog):
    """Preview a frame and set one scan window reused for every frame."""

    def __init__(
        self, controller, device: ScannerDevice, initial_rect=None, default_frame: int = 1, initial_offset: float = 0.0, parent=None
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._device = device
        self._caps = device.capabilities
        self._max_area_mm = self._caps.max_area_mm
        self._previewed_offset: float | None = None  # offset (mm) the shown preview was scanned at
        self._preview_offset_pending = 0.0
        self.setWindowTitle("Scan window — preview and set")
        self.setModal(True)
        self.resize(820, 680)

        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Preview frame"))
        self.frame_spin = QSpinBox()
        capacity = max(1, self._caps.adapter_frame_capacity or 1)
        self.frame_spin.setRange(1, capacity)
        self.frame_spin.setValue(min(max(default_frame, 1), capacity))
        self.frame_spin.setToolTip("Which frame to preview — the window applies to every frame")
        top.addWidget(self.frame_spin)
        top.addSpacing(12)
        top.addWidget(QLabel("Offset"))
        self.offset_slider = QSlider(Qt.Orientation.Horizontal)
        self.offset_slider.setRange(0, 40)  # tenths of a mm → 0..4.0 mm
        self.offset_slider.setSingleStep(1)
        self.offset_slider.setPageStep(5)
        self.offset_slider.setFixedWidth(160)
        self.offset_slider.setValue(int(round(max(0.0, float(initial_offset)) * 10)))
        self.offset_slider.setToolTip("Feed-axis offset applied to every frame — drag to see where it cuts, then Preview to apply")
        top.addWidget(self.offset_slider)
        self.offset_label = QLabel()
        top.addWidget(self.offset_label)
        top.addStretch()
        self.preview_btn = QPushButton(qta.icon("fa5s.eye", color=THEME.text_primary), " Preview")
        self.preview_btn.clicked.connect(self._on_preview)
        top.addWidget(self.preview_btn)
        layout.addLayout(top)

        self.image = ScanWindowLabel()
        self.image.set_window(initial_rect)
        self.image.windowChanged.connect(self._update_readout)
        layout.addWidget(self.image, 1)

        # Connect after the image exists — setValue above would otherwise fire the
        # slot before self.image is built.
        self.offset_slider.valueChanged.connect(self._on_offset_changed)
        self._on_offset_changed(self.offset_slider.value())

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("Click Preview to scan a frame, then drag a window over it.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        layout.addWidget(self.status)

        self.readout = QLabel("")
        self.readout.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        layout.addWidget(self.readout)
        self._update_readout(initial_rect)

        btns = QHBoxLayout()
        self.clear_btn = QPushButton("Clear window")
        self.clear_btn.setToolTip("Scan the whole default frame (no window)")
        self.clear_btn.clicked.connect(self.image.clear_window)
        btns.addWidget(self.clear_btn)
        btns.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        self.ok_btn = QPushButton("Use window")
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self.accept)
        btns.addWidget(self.ok_btn)
        layout.addLayout(btns)

        controller.scan_preview_ready.connect(self._on_preview_ready)
        controller.scan_error.connect(self._on_error)
        controller.scan_cancelled.connect(self._on_cancelled)

    def window_rect(self):
        return self.image.window()

    def frame_offset(self) -> float:
        return self.offset_slider.value() / 10.0

    def _on_offset_changed(self, _value: int) -> None:
        offset_mm = self.frame_offset()
        self.offset_label.setText(f"{offset_mm:.1f} mm")
        extent = self._max_area_mm[1] if self._max_area_mm and len(self._max_area_mm) > 1 else 0.0
        # Line shows only the pending cut (slider beyond the previewed offset) → clears after Preview.
        base = self._previewed_offset or 0.0
        delta = offset_mm - base
        self.image.set_offset_indicator(delta / extent if (extent and delta > 0) else None)

    # ── preview flow ──────────────────────────────────────────────────

    def _preview_dpi(self) -> int:
        dpis = self._caps.supported_dpi
        return min(dpis) if dpis else _PREVIEW_FALLBACK_DPI

    def _on_preview(self) -> None:
        req = ScanRequest(
            device_id=self._device.id,
            params=ScanParams(
                dpi=self._preview_dpi(),
                depth=8,
                capture_ir=False,
                autofocus=False,
                auto_exposure=False,
                window=None,
                frame_offset_mm=self.frame_offset(),
                frame=self.frame_spin.value(),
            ),
            output_folder="",
            filename_pattern="",
            output_format="TIFF",
        )
        try:
            self._controller.start_preview(req)
        except Exception as e:
            self.status.setText(f"Cannot preview: {e}")
            return
        self._preview_offset_pending = self.frame_offset()
        self.preview_btn.setEnabled(False)
        self.progress.setRange(0, 0)  # indeterminate busy indicator
        self.progress.setVisible(True)
        self.status.setText(f"Previewing frame {self.frame_spin.value()}…")

    @pyqtSlot(object)
    def _on_preview_ready(self, rgb) -> None:
        self.progress.setVisible(False)
        self.preview_btn.setEnabled(True)
        try:
            pixmap = QPixmap.fromImage(ImageConverter.to_qimage(rgb))
        except Exception as e:
            self.status.setText(f"Could not display preview: {e}")
            return
        self._previewed_offset = self._preview_offset_pending
        self.image.set_frame(pixmap)
        self._on_offset_changed(self.offset_slider.value())  # rebase the line to the applied offset
        self.status.setText("Preview reflects the offset. Nudge the slider (dashed line = extra cut) then Preview again, or drag a window.")

    @pyqtSlot(str)
    def _on_error(self, msg) -> None:
        self.progress.setVisible(False)
        self.preview_btn.setEnabled(True)
        self.status.setText(f"Preview failed: {msg}")

    @pyqtSlot()
    def _on_cancelled(self) -> None:
        self.progress.setVisible(False)
        self.preview_btn.setEnabled(True)
        self.status.setText("Preview cancelled.")

    def _update_readout(self, rect) -> None:
        area = scan_window_to_area(rect, self._max_area_mm)
        if area is None:
            self.readout.setText("Window: full frame (default)")
        else:
            tl_x, tl_y, br_x, br_y = area
            self.readout.setText(f"Window: {br_x - tl_x:.1f} × {br_y - tl_y:.1f} mm  at  ({tl_x:.1f}, {tl_y:.1f}) mm")

    def closeEvent(self, ev) -> None:
        for signal, slot in (
            (self._controller.scan_preview_ready, self._on_preview_ready),
            (self._controller.scan_error, self._on_error),
            (self._controller.scan_cancelled, self._on_cancelled),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(ev)
