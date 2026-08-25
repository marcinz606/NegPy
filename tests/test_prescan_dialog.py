"""Offline tests for the Prescan crop dialog.

Constructs the real PrescanCropDialog against a light fake controller under an
offscreen Qt platform. Proves the footer grammar it shares with the other two
preview dialogs, and the Scan exit it did not use to have.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.widgets.prescan_dialog import PrescanCropDialog
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import ScanMode
from negpy.infrastructure.scanners.result import ScanResult

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _device() -> ScannerDevice:
    caps = ScannerCapabilities(
        ir_channel=True,
        supported_dpi=(1200, 3600),
        supported_depths=(16,),
        sources=(ScanMode.TRANSPARENCY,),
        max_area_mm=(36.33, 25.0),
        prescan=True,
        prescan_dpi=1200,
        prescan_default_crop=(0.0, 0.35, 1.0, 0.65),
    )
    return ScannerDevice(id="plustek:usb:07b3:1825:002:006", vendor="PLUSTEK", model="OpticFilm 8200i SE", capabilities=caps)


class _FakeController(QObject):
    scan_prescan_ready = pyqtSignal(object)
    scan_prescan_error = pyqtSignal(str)
    scan_progress = pyqtSignal(float, str)
    scan_cancelled = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.prescan_reqs: list = []
        self.cancels = 0

    def start_prescan(self, req) -> None:
        self.prescan_reqs.append(req)

    def cancel_scan(self) -> None:
        self.cancels += 1


def _result() -> ScanResult:
    rgb = np.linspace(0, 65535, 32 * 48 * 3, dtype=np.float32).reshape(32, 48, 3).astype(np.uint16)
    return ScanResult(rgb=rgb, ir=None, dpi=1200, device_model="OpticFilm 8200i SE")


def _ready(controller: _FakeController, dialog: PrescanCropDialog) -> None:
    controller.scan_prescan_ready.emit(_result())
    assert dialog._label.pixmap() is not None


def test_the_footer_reads_like_the_other_preview_dialogs() -> None:
    controller = _FakeController()
    dialog = PrescanCropDialog(controller, _device())
    _ready(controller, dialog)

    assert dialog._clear_btn.text() == "Clear crop"
    assert dialog._ok_btn.text() == "Apply crop"
    assert dialog._scan_btn.text() == "Scan frame"


def test_re_acquisition_is_not_one_of_the_exits() -> None:
    """Rescan sat among Cancel and Use; the other two dialogs keep acquisition up top."""
    controller = _FakeController()
    dialog = PrescanCropDialog(controller, _device())

    footer = dialog.layout().itemAt(dialog.layout().count() - 1).layout()
    in_footer = {footer.itemAt(i).widget() for i in range(footer.count())}

    assert dialog._retry_btn not in in_footer
    assert {dialog._clear_btn, dialog._cancel_btn, dialog._ok_btn, dialog._scan_btn} <= in_footer


def test_scan_is_the_exit_the_dialog_never_had() -> None:
    """Framing a crop used to end at the panel, and a second trip to the Scan button."""
    controller = _FakeController()
    dialog = PrescanCropDialog(controller, _device())
    _ready(controller, dialog)

    assert dialog.scan_requested() is False
    assert dialog._scan_btn.isDefault() is True

    dialog._scan_btn.click()

    assert dialog.scan_requested() is True
    assert dialog.result() == PrescanCropDialog.DialogCode.Accepted.value


def test_apply_crop_leaves_without_asking_for_a_scan() -> None:
    controller = _FakeController()
    dialog = PrescanCropDialog(controller, _device())
    _ready(controller, dialog)

    dialog._ok_btn.click()

    assert dialog.scan_requested() is False
    assert dialog.scan_window() == (0.0, 0.35, 1.0, 0.65)


def test_neither_exit_is_offered_before_a_preview_arrives() -> None:
    controller = _FakeController()
    dialog = PrescanCropDialog(controller, _device())

    assert dialog._ok_btn.isEnabled() is False
    assert dialog._scan_btn.isEnabled() is False

    _ready(controller, dialog)

    assert dialog._ok_btn.isEnabled() is True
    assert dialog._scan_btn.isEnabled() is True


def test_the_pass_and_its_message_share_one_reserved_row() -> None:
    """Status and progress were two rows that came and went above the crop."""
    controller = _FakeController()
    dialog = PrescanCropDialog(controller, _device())

    assert dialog._strip.showing() == "progress"

    controller.scan_progress.emit(0.5, "Prescanning")
    assert dialog._strip._bar.value() == 50

    _ready(controller, dialog)
    assert dialog._strip.showing() == "message"
    assert "Drag the rectangle" in dialog._strip.message()


def test_a_failed_prescan_says_so_in_the_same_row() -> None:
    controller = _FakeController()
    dialog = PrescanCropDialog(controller, _device())
    reserved = dialog._strip.height()

    controller.scan_prescan_error.emit("lamp did not warm up")

    assert dialog._strip.showing() == "message"
    assert "lamp did not warm up" in dialog._strip.message()
    assert dialog._strip.height() == reserved
    assert dialog._ok_btn.isEnabled() is False
