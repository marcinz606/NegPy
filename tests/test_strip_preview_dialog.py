"""Offline tests for the whole-strip preview dialog.

Constructs the real StripPreviewDialog against a light fake controller under an
offscreen Qt platform. Proves the per-frame tiles, the Use gating, the getters
the sidebar reads on accept, and the single-flight "Preview all" chain.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.widgets.strip_preview_dialog import (
    StripPreviewDialog,
    _display_to_scan_rect,
    _preview_positive,
    _scan_to_display_rect,
)
from negpy.infrastructure.scanners.base import ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import ScanMode

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _device(capacity: int) -> ScannerDevice:
    caps = ScannerCapabilities(
        ir_channel=False,
        supported_dpi=(1000, 4000),
        supported_depths=(8, 14),
        sources=(ScanMode.NEGATIVE,),
        max_area_mm=(25.0, 38.0),
        adapter_frame_capacity=capacity,
        can_eject=True,
    )
    return ScannerDevice(id="coolscan3:usb:libusb:001:050", vendor="Nikon", model="LS-50", capabilities=caps)


class _FakeController(QObject):
    scan_preview_ready = pyqtSignal(object)
    scan_error = pyqtSignal(str)
    scan_cancelled = pyqtSignal()

    def __init__(self, *, raise_on_preview: bool = False) -> None:
        super().__init__()
        self.preview_reqs: list = []
        self._raise = raise_on_preview

    def start_preview(self, req) -> None:
        if self._raise:
            raise RuntimeError("A scanner request is already active")
        self.preview_reqs.append(req)


def _rgb():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_builds_one_tile_per_frame() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(6))
    assert sorted(dialog._tiles) == [1, 2, 3, 4, 5, 6]


def test_preview_uses_lowest_supported_dpi() -> None:
    caps = ScannerCapabilities(
        ir_channel=False,
        supported_dpi=(90, 400, 4000),
        supported_depths=(8, 14),
        sources=(ScanMode.NEGATIVE,),
        max_area_mm=(25.0, 38.0),
        adapter_frame_capacity=6,
        can_eject=True,
    )
    device = ScannerDevice(id="coolscan3:usb:libusb:001:050", vendor="Nikon", model="LS-50", capabilities=caps)
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, device)

    dialog._on_preview_one(1)

    assert controller.preview_reqs[0].params.dpi == 90


def test_offset_indicator_grows_from_the_left_matching_content_shift() -> None:
    # +offset shifts re-scanned content toward the rotated preview's right, so the
    # cut band must grow from the LEFT (new content) — same direction as the slider.
    dialog = StripPreviewDialog(_FakeController(), _device(3), initial_offset=4.0)
    dialog._refresh_offset_indicators()

    label = dialog._tiles[1].label
    assert label._offset_edge == "left"
    assert label._offset_frac == pytest.approx(4.0 / 38.0, abs=1e-3)  # extent = max_area_mm[1]


def test_preview_positive_inverts_and_levels() -> None:
    neg = np.zeros((4, 4, 3), dtype=np.uint8)
    neg[:, :2, :] = 20  # low negative value = scene shadow → should become bright
    neg[:, 2:, :] = 200  # high negative value = scene highlight → should become dark
    pos = _preview_positive(neg)
    assert pos.dtype == np.uint8
    assert pos.shape == neg.shape
    assert pos[:, :2, :].mean() > pos[:, 2:, :].mean()  # inverted


def test_scan_button_marks_scan_requested_and_use_does_not() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(3))
    assert dialog.scan_requested() is False

    dialog._on_scan_clicked()
    assert dialog.scan_requested() is True

    other = StripPreviewDialog(_FakeController(), _device(3))
    other.accept()
    assert other.scan_requested() is False


def test_tile_dims_scale_to_height_at_landscape_aspect() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(3))  # max_area_mm=(25, 38) → aspect 1.52
    assert dialog._tile_aspect == pytest.approx(38.0 / 25.0, abs=1e-3)

    w, h = dialog._tile_dims(400)
    assert h == 400 - 8  # window height minus padding
    assert w == int(h * dialog._tile_aspect)  # width tracks height, landscape

    _, small_h = dialog._tile_dims(10)
    assert small_h == 140  # floored


def test_scan_button_is_gated_on_a_frame_being_checked() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(3))
    for tile in dialog._tiles.values():
        tile.checkbox.setChecked(False)
    assert dialog.scan_btn.isEnabled() is False
    dialog._tiles[2].checkbox.setChecked(True)
    assert dialog.scan_btn.isEnabled() is True


def test_use_is_disabled_when_no_frame_is_checked() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(3))
    for tile in dialog._tiles.values():
        tile.checkbox.setChecked(False)
    assert dialog.ok_btn.isEnabled() is False
    dialog._tiles[2].checkbox.setChecked(True)
    assert dialog.ok_btn.isEnabled() is True


def test_getters_reflect_selection_windows_and_offset() -> None:
    scan_rect = (0.1, 0.2, 0.6, 0.7)  # a window on an unchecked frame is still kept
    dialog = StripPreviewDialog(
        _FakeController(),
        _device(3),
        initial_windows={2: scan_rect},
        initial_selected=(1, 3),
        initial_offset=0.5,
    )
    assert dialog.selected_frames() == (1, 3)
    windows = dialog.frame_windows()
    assert set(windows) == {2}
    assert windows[2] == pytest.approx(scan_rect)  # round-trips through the display rotation
    assert dialog.frame_offset() == 0.5


def test_portrait_preview_is_shown_landscape() -> None:
    # The scanner raster is portrait; the strip view rotates it so the frame reads
    # landscape, as it sits on the film strip.
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3))
    dialog._on_preview_one(1)
    controller.scan_preview_ready.emit(np.zeros((60, 20, 3), dtype=np.uint8))  # tall (H > W)
    pixmap = dialog._tiles[1].label._pixmap
    assert pixmap is not None
    assert pixmap.width() > pixmap.height()


def test_rect_transform_round_trips() -> None:
    scan = (0.1, 0.2, 0.6, 0.7)
    assert _display_to_scan_rect(_scan_to_display_rect(scan)) == pytest.approx(scan)


def test_scan_top_maps_to_display_right() -> None:
    # The feed-axis offset cuts the scan top (small fy); after rotation that lands on
    # the display's right (short) edge — where the offset indicator draws.
    assert _scan_to_display_rect((0.0, 0.0, 1.0, 0.25)) == pytest.approx((0.75, 0.0, 1.0, 1.0))


def test_preview_all_chains_one_frame_at_a_time() -> None:
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3))

    dialog._on_preview_all()
    assert [r.params.frame for r in controller.preview_reqs] == [1]  # only one in flight

    controller.scan_preview_ready.emit(_rgb())
    assert [r.params.frame for r in controller.preview_reqs] == [1, 2]

    controller.scan_preview_ready.emit(_rgb())
    assert [r.params.frame for r in controller.preview_reqs] == [1, 2, 3]

    controller.scan_preview_ready.emit(_rgb())  # last frame done → chain stops
    assert [r.params.frame for r in controller.preview_reqs] == [1, 2, 3]
    assert dialog.preview_all_btn.isEnabled() is True


def test_preview_ready_routes_to_the_in_flight_tile() -> None:
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3))

    dialog._on_preview_one(2)
    controller.scan_preview_ready.emit(_rgb())

    assert dialog._tiles[2].label.has_frame() is True
    assert dialog._tiles[1].label.has_frame() is False
    assert dialog._tiles[3].label.has_frame() is False


def test_preview_error_continues_the_chain() -> None:
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3))

    dialog._on_preview_all()  # frame 1 in flight
    controller.scan_error.emit("film jam")  # frame 1 glitches → move on, don't abort

    assert [r.params.frame for r in controller.preview_reqs] == [1, 2]

    controller.scan_error.emit("film jam")  # frame 2 → frame 3
    controller.scan_error.emit("film jam")  # frame 3 → chain ends

    assert [r.params.frame for r in controller.preview_reqs] == [1, 2, 3]
    assert dialog.preview_all_btn.isEnabled() is True


def test_start_preview_busy_is_handled_gracefully() -> None:
    controller = _FakeController(raise_on_preview=True)
    dialog = StripPreviewDialog(controller, _device(3))

    dialog._on_preview_one(1)  # must not raise

    assert controller.preview_reqs == []
    assert dialog.preview_all_btn.isEnabled() is True
