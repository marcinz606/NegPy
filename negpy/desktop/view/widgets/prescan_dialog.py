# SPDX-License-Identifier: GPL-3.0-or-later
"""Modal: run a Plustek Prescan, set a crop window, return TA-normalized scan_window."""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from negpy.desktop.converters import ImageConverter
from negpy.desktop.view.widgets.scan_window_label import ScanWindowLabel
from negpy.desktop.workers.scan_worker import PrescanRequest
from negpy.infrastructure.scanners.base import ScannerDevice
from negpy.infrastructure.scanners.params import crop_to_scan_window
from negpy.infrastructure.scanners.result import ScanResult


def _preview_u8(rgb: np.ndarray) -> np.ndarray:
    """Scale scan RGB to uint8 for the crop widget (percentile stretch)."""
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    out = np.empty(a.shape[:2] + (3,), dtype=np.uint8)
    channels = a.shape[2]
    for c in range(3):
        ch = a[..., min(c, channels - 1)]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        if hi <= lo:
            out[..., c] = 0
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


class PrescanCropDialog(QDialog):
    """Acquire a 1200 dpi full-window preview and let the user set the scan crop."""

    def __init__(
        self,
        controller,
        device: ScannerDevice,
        *,
        initial_window: tuple[float, float, float, float] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._device = device
        caps = device.capabilities
        # TA / backend space (what ScanParams.window stores).
        self._scan_window: tuple[float, float, float, float] | None = initial_window
        self._prescan_mirror_x = bool(caps.prescan_mirror_x)
        self._busy = False

        self.setWindowTitle("Prescan — set crop")
        self.setModal(True)
        self.resize(720, 560)

        root = QVBoxLayout(self)
        self._status = QLabel("Starting Prescan…")
        root.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        root.addWidget(self._progress)

        self._label = ScanWindowLabel()
        root.addWidget(self._label, 1)

        row = QHBoxLayout()
        self._retry_btn = QPushButton("Rescan")
        self._retry_btn.setEnabled(False)
        self._clear_btn = QPushButton("Clear crop")
        self._clear_btn.setEnabled(False)
        self._cancel_btn = QPushButton("Cancel")
        self._ok_btn = QPushButton("Use crop")
        self._ok_btn.setEnabled(False)
        self._ok_btn.setDefault(True)
        row.addWidget(self._retry_btn)
        row.addWidget(self._clear_btn)
        row.addStretch()
        row.addWidget(self._cancel_btn)
        row.addWidget(self._ok_btn)
        root.addLayout(row)

        self._retry_btn.clicked.connect(self._start_prescan)
        self._clear_btn.clicked.connect(self._on_clear_crop)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._ok_btn.clicked.connect(self.accept)
        self._label.windowChanged.connect(self._on_window_changed)

        self._controller.scan_prescan_ready.connect(self._on_prescan_ready)
        self._controller.scan_prescan_error.connect(self._on_prescan_error)
        self._controller.scan_progress.connect(self._on_progress)
        self._controller.scan_cancelled.connect(self._on_prescan_cancelled)

        self._start_prescan()

    def scan_window(self) -> tuple[float, float, float, float] | None:
        """TA-normalized window for ScanParams, or None for full frame."""
        return self._scan_window

    def _start_prescan(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._status.setText("Scanning preview at 1200 dpi…")
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._retry_btn.setEnabled(False)
        self._ok_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)
        self._label.set_frame(QPixmap())
        try:
            self._controller.start_prescan(
                PrescanRequest(
                    device_id=self._device.id,
                    prescan_dpi=self._device.capabilities.prescan_dpi,
                )
            )
        except Exception as exc:
            self._busy = False
            self._status.setText(str(exc))
            self._retry_btn.setEnabled(True)

    def _on_progress(self, value: float) -> None:
        if not self._busy:
            return
        self._progress.setValue(int(max(0.0, min(1.0, float(value))) * 100))

    def _on_prescan_ready(self, result: object) -> None:
        if not self._busy:
            return
        self._busy = False
        self._progress.setVisible(False)
        self._retry_btn.setEnabled(True)
        self._clear_btn.setEnabled(True)
        if not isinstance(result, ScanResult):
            self._status.setText("Prescan returned no image")
            return
        u8 = _preview_u8(result.rgb)
        qimg = ImageConverter.to_qimage(u8)
        self._label.set_frame(QPixmap.fromImage(qimg))
        if self._scan_window is None:
            default_crop = self._device.capabilities.prescan_default_crop
            if default_crop is not None:
                self._scan_window = default_crop
        if self._scan_window is not None:
            image_rect = crop_to_scan_window(self._scan_window, mirror_x=self._prescan_mirror_x)
            self._label.set_window(image_rect)
        self._ok_btn.setEnabled(True)
        self._status.setText("Drag the rectangle to set the scan crop")

    def _on_prescan_error(self, message: str) -> None:
        if not self._busy:
            return
        self._busy = False
        self._progress.setVisible(False)
        self._retry_btn.setEnabled(True)
        self._status.setText(message or "Prescan failed")

    def _on_prescan_cancelled(self) -> None:
        if not self._busy:
            return
        self._busy = False
        self._progress.setVisible(False)
        self._retry_btn.setEnabled(True)
        self._status.setText("Prescan cancelled")

    def _on_window_changed(self, rect: object) -> None:
        if rect is None:
            self._scan_window = None
            return
        self._scan_window = crop_to_scan_window(tuple(rect), mirror_x=self._prescan_mirror_x)  # type: ignore[arg-type]

    def _on_clear_crop(self) -> None:
        self._scan_window = None
        self._label.clear_window()
        self._ok_btn.setEnabled(True)

    def _on_cancel(self) -> None:
        if self._busy:
            self._controller.cancel_scan()
        self.reject()

    def reject(self) -> None:
        self._disconnect_controller()
        super().reject()

    def accept(self) -> None:
        # Sync from widget in case the last drag did not emit.
        rect = self._label.window()
        if rect is not None:
            self._scan_window = crop_to_scan_window(rect, mirror_x=self._prescan_mirror_x)
        self._disconnect_controller()
        super().accept()

    def _disconnect_controller(self) -> None:
        for signal, slot in (
            (self._controller.scan_prescan_ready, self._on_prescan_ready),
            (self._controller.scan_prescan_error, self._on_prescan_error),
            (self._controller.scan_progress, self._on_progress),
            (self._controller.scan_cancelled, self._on_prescan_cancelled),
        ):
            try:
                signal.disconnect(slot)
            except TypeError:
                pass

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._busy:
            self._controller.cancel_scan()
        self._disconnect_controller()
        super().closeEvent(event)
