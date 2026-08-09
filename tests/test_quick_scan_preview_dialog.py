"""Offline tests for the single-shot preview dialog (frame-less devices, e.g. Plustek).

Constructs the real QuickScanPreviewDialog against a light fake controller under an
offscreen Qt platform. Proves the preview/result flow and — since both preview
dialogs now share their signal wiring via RollPreviewSignalsMixin — that this
dialog's connect/disconnect actually works, not just StripPreviewDialog's.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.widgets.quick_scan_preview_dialog import QuickScanPreviewDialog
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import ScanMode
from negpy.infrastructure.scanners.roll import RollPreview

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _device() -> ScannerDevice:
    caps = ScannerCapabilities(
        ir_channel=False,
        supported_dpi=(1200, 2400, 7200),
        supported_depths=(8, 16),
        sources=(ScanMode.NEGATIVE,),
        max_area_mm=(36.0, 24.0),
        adapter_frame_capacity=None,  # the whole point: no feeder
    )
    return ScannerDevice(id="plustek:libusb:001:008", vendor="Plustek", model="OpticFilm", capabilities=caps)


class _FakeController(QObject):
    scan_roll_preview_ready = pyqtSignal(object)
    scan_roll_preview_finished = pyqtSignal()
    scan_error = pyqtSignal(str)
    scan_cancelled = pyqtSignal()

    def __init__(self, *, raise_on_preview: bool = False) -> None:
        super().__init__()
        self.preview_reqs: list = []
        self._raise = raise_on_preview

    def start_roll_preview(self, req) -> None:
        if self._raise:
            raise RuntimeError("A scanner request is already active")
        self.preview_reqs.append(req)

    def deliver(self, slot: int = 1, *, rgb=None, error: str | None = None) -> None:
        rgb = np.zeros((8, 8, 3), dtype=np.uint8) if rgb is None and error is None else rgb
        self.scan_roll_preview_ready.emit(RollPreview(slot=slot, rgb=rgb, error=error))
        self.scan_roll_preview_finished.emit()


def test_preview_requests_the_single_implicit_slot():
    controller = _FakeController()
    dialog = QuickScanPreviewDialog(controller, _device())

    dialog._on_preview()

    assert controller.preview_reqs[0].slots == (1,)


def test_preview_uses_lowest_supported_dpi():
    controller = _FakeController()
    dialog = QuickScanPreviewDialog(controller, _device())

    dialog._on_preview()

    assert controller.preview_reqs[0].dpi == 1200


def test_preview_result_shows_the_frame_and_clears_busy_state():
    controller = _FakeController()
    dialog = QuickScanPreviewDialog(controller, _device())

    dialog._on_preview()
    assert dialog.preview_btn.isEnabled() is False
    controller.deliver()

    assert dialog.preview_btn.isEnabled() is True
    assert dialog.label.has_frame()
    assert dialog.status.text() == ""


def test_preview_failure_reports_status_and_clears_busy_state():
    controller = _FakeController()
    dialog = QuickScanPreviewDialog(controller, _device())

    dialog._on_preview()
    controller.deliver(error="carriage jammed")

    assert dialog.preview_btn.isEnabled() is True
    assert "carriage jammed" in dialog.status.text()


def test_busy_scanner_reports_status_without_starting_preview():
    controller = _FakeController(raise_on_preview=True)
    dialog = QuickScanPreviewDialog(controller, _device())

    dialog._on_preview()

    assert "busy" in dialog.status.text().lower()
    assert dialog.preview_btn.isEnabled() is True


def test_initial_window_is_restored_but_no_image_until_previewed():
    rect = (0.1, 0.1, 0.5, 0.5)
    dialog = QuickScanPreviewDialog(_FakeController(), _device(), initial_window=rect)

    assert dialog.window() == rect
    assert dialog.label.has_frame() is False


def test_clear_removes_the_window():
    rect = (0.1, 0.1, 0.5, 0.5)
    dialog = QuickScanPreviewDialog(_FakeController(), _device(), initial_window=rect)

    dialog.clear_btn.click()

    assert dialog.window() is None


def test_scan_button_sets_scan_requested_and_accepts():
    dialog = QuickScanPreviewDialog(_FakeController(), _device())

    dialog._on_scan_clicked()

    assert dialog.scan_requested() is True
    assert dialog.result() == dialog.Accepted


def test_use_button_does_not_set_scan_requested():
    dialog = QuickScanPreviewDialog(_FakeController(), _device())

    dialog.ok_btn.click()

    assert dialog.scan_requested() is False
    assert dialog.result() == dialog.Accepted


def test_close_disconnects_preview_signals_without_error():
    controller = _FakeController()
    dialog = QuickScanPreviewDialog(controller, _device())

    dialog.close()

    # Disconnected: delivering a result now must not raise or touch the dialog.
    controller.deliver()
