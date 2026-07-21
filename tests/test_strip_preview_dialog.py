"""Offline tests for the whole-strip preview dialog.

Constructs the real StripPreviewDialog against a light fake controller under an
offscreen Qt platform. Proves the per-frame tiles, the Use gating, the getters
the sidebar reads on accept, and the roll-preview request/result flow.
"""

from __future__ import annotations

import dataclasses
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
from negpy.infrastructure.scanners.roll import RollPreview

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

    def deliver(self, slot: int, *, rgb=None, offset: float = 0.0, error: str | None = None) -> None:
        """Hand back one slot the way the worker does."""
        self.scan_roll_preview_ready.emit(
            RollPreview(slot=slot, rgb=_rgb() if rgb is None and error is None else rgb, offset=offset, error=error)
        )

    def deliver_all(self, slots, *, offset: float = 0.0) -> None:
        for slot in slots:
            self.deliver(slot, offset=offset)
        self.scan_roll_preview_finished.emit()


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

    assert controller.preview_reqs[0].dpi == 90


def test_offset_indicator_grows_from_the_right_tracking_the_slider() -> None:
    # The tile is anchored to the next scan, so +offset slides content toward the
    # display's LEFT and the band grows from the RIGHT: film past the frame
    # boundary the transport cannot deliver at this offset.
    dialog = StripPreviewDialog(_FakeController(), _device(3), initial_offset=4.0)
    dialog._refresh_offset_indicators()

    ((frac, edge),) = dialog._tiles[1].label._offset_indicators
    assert edge == "right"
    assert frac == pytest.approx(4.0 / 38.0, abs=1e-3)  # extent = max_area_mm[1]


def test_preview_offsets_follow_drift_per_frame_position() -> None:
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3), initial_offset=1.0, initial_offset_modifier=0.2)

    dialog._on_preview_all()

    # Offsets go out as fractions of one pitch; the session converts and clamps.
    offsets = controller.preview_reqs[0].offsets
    assert [offsets[f] * 38.0 for f in (1, 2, 3)] == pytest.approx([1.0, 1.2, 1.4])


def test_negative_drift_floors_the_offset_at_zero() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(3), initial_offset=0.3, initial_offset_modifier=-0.25)

    assert dialog._offset_for_frame(1) == pytest.approx(0.3)
    assert dialog._offset_for_frame(2) == pytest.approx(0.05)
    assert dialog._offset_for_frame(3) == 0.0  # below 0 is physically impossible


def test_drift_slider_spans_plus_minus_two_point_five() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(3), initial_offset_modifier=2.5)
    assert (dialog.drift_slider.minimum(), dialog.drift_slider.maximum()) == (-250, 250)
    assert dialog.frame_offset_modifier() == pytest.approx(2.5)


def test_negative_drift_pins_the_line_to_the_edge_on_floored_frames() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(3))

    dialog.drift_slider.setValue(-50)  # -0.50 mm/frame, base 0

    assert dialog._tiles[1].label._offset_indicators == []  # raw 0: genuinely no offset
    ((f2, e2),) = dialog._tiles[2].label._offset_indicators
    assert (e2, f2) == ("right", 0.0)  # floored → line pinned at the edge
    ((f3, e3),) = dialog._tiles[3].label._offset_indicators
    assert (e3, f3) == ("right", 0.0)


def test_drift_slider_updates_indicators_live_per_frame() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(3))

    dialog.drift_slider.setValue(50)  # +0.50 mm/frame

    assert dialog.drift_label.text() == "+0.50 mm/frame"
    assert dialog._tiles[1].label._offset_indicators == []  # frame 1: no drift yet
    ((f2, e2),) = dialog._tiles[2].label._offset_indicators
    assert (e2, f2) == ("right", pytest.approx(0.5 / 38.0, abs=1e-3))
    ((f3, e3),) = dialog._tiles[3].label._offset_indicators
    assert (e3, f3) == ("right", pytest.approx(1.0 / 38.0, abs=1e-3))


def test_offset_slider_is_non_negative_up_to_ten_mm() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(3), initial_offset=-2.0)

    assert (dialog.offset_slider.minimum(), dialog.offset_slider.maximum()) == (0, 100)
    assert dialog.frame_offset() == 0.0  # a stale negative setting clamps to 0


def test_indicator_is_absolute_per_frame_not_relative_to_the_shown_preview() -> None:
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3), initial_offset=2.0)

    # Preview frame 1 at the current offset — the band must still show the
    # absolute 2.0 mm cut, not disappear because "nothing changed".
    dialog._on_preview_one(1)
    controller.deliver(1, offset=2.0 / 38.0)

    ((frac, edge),) = dialog._tiles[1].label._offset_indicators
    assert edge == "right"
    assert frac == pytest.approx(2.0 / 38.0, abs=1e-3)


def test_offset_and_drift_sliders_reset_to_zero_on_double_click() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(3), initial_offset=2.0, initial_offset_modifier=0.5)

    dialog.offset_slider.mouseDoubleClickEvent(None)
    dialog.drift_slider.mouseDoubleClickEvent(None)

    assert dialog.frame_offset() == 0.0
    assert dialog.frame_offset_modifier() == 0.0
    assert dialog.offset_label.text() == "0.0 mm"  # valueChanged fired → labels refreshed
    assert dialog.drift_label.text() == "+0.00 mm/frame"


def test_preview_dpi_dropdown_defaults_to_lowest_and_flows_into_requests() -> None:
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3))  # supported_dpi=(1000, 4000)
    assert dialog._preview_dpi() == 1000

    dialog.preview_dpi_combo.setCurrentIndex(1)
    dialog._on_preview_one(1)

    assert controller.preview_reqs[0].dpi == 4000


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


def test_tiles_have_constant_size_at_landscape_aspect() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(3))  # max_area_mm=(25, 38) → aspect 1.52
    assert dialog._tile_aspect == pytest.approx(38.0 / 25.0, abs=1e-3)

    w, h = dialog._tile_size()
    assert h == 140
    assert w == int(h * dialog._tile_aspect)
    assert (dialog._tiles[1].label.width(), dialog._tiles[1].label.height()) == (w, h)

    dialog.resize(2000, 1000)  # tiles must not track the window size
    assert (dialog._tiles[1].label.width(), dialog._tiles[1].label.height()) == (w, h)


def test_tiles_wrap_in_rows_of_six() -> None:
    dialog = StripPreviewDialog(_FakeController(), _device(40))  # full-roll adapter
    grid = dialog._scroll.widget().layout()

    def pos(frame: int) -> tuple[int, int]:
        row, col, _rowspan, _colspan = grid.getItemPosition(grid.indexOf(dialog._tiles[frame].widget))
        return row, col

    assert pos(1) == (0, 0)
    assert pos(6) == (0, 5)
    assert pos(7) == (1, 0)
    assert pos(40) == (6, 3)


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
        initial_offset_modifier=0.05,
    )
    assert dialog.selected_frames() == (1, 3)
    windows = dialog.frame_windows()
    assert set(windows) == {2}
    assert windows[2] == pytest.approx(scan_rect)  # round-trips through the display rotation
    assert dialog.frame_offset() == 0.5
    assert dialog.frame_offset_modifier() == 0.05


def test_portrait_preview_is_shown_landscape() -> None:
    # The scanner raster is portrait; the strip view rotates it so the frame reads
    # landscape, as it sits on the film strip.
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3))
    dialog._on_preview_one(1)
    controller.deliver(1, rgb=np.zeros((60, 20, 3), dtype=np.uint8))  # tall (H > W)
    pixmap = dialog._tiles[1].label._pixmap
    assert pixmap is not None
    assert pixmap.width() > pixmap.height()


def test_rect_transform_round_trips() -> None:
    scan = (0.1, 0.2, 0.6, 0.7)
    assert _display_to_scan_rect(_scan_to_display_rect(scan)) == pytest.approx(scan)


def test_scan_top_maps_to_display_left() -> None:
    # The scan top (small fy) is the frame's feed-axis start, adjacent to the
    # PREVIOUS frame; after the -90 rotation it lands on the display's LEFT edge,
    # so tiles laid out 1..N left-to-right read continuously like the strip.
    assert _scan_to_display_rect((0.0, 0.0, 1.0, 0.25)) == pytest.approx((0.0, 0.0, 0.25, 1.0))


def test_preview_all_asks_for_the_whole_strip_in_one_request() -> None:
    # The session owns the traversal: one request for every slot, results streamed
    # back per slot. A transport that reads the strip in one pass can honour this.
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3))

    dialog._on_preview_all()

    assert [r.slots for r in controller.preview_reqs] == [(1, 2, 3)]
    assert dialog.preview_all_btn.isEnabled() is False  # locked until finished

    controller.deliver_all((1, 2, 3))

    assert all(dialog._tiles[f].label.has_frame() for f in (1, 2, 3))
    assert dialog.preview_all_btn.isEnabled() is True


def test_previews_route_by_their_own_slot_number() -> None:
    """Results carry their slot, so out-of-order delivery still lands correctly."""
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3))

    dialog._on_preview_all()
    controller.deliver(3)
    controller.deliver(1)

    assert dialog._tiles[1].label.has_frame() is True
    assert dialog._tiles[2].label.has_frame() is False
    assert dialog._tiles[3].label.has_frame() is True


def test_a_failed_slot_does_not_abort_the_strip() -> None:
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3))

    dialog._on_preview_all()
    controller.deliver(1, error="film jam")
    controller.deliver(2)
    controller.deliver(3)
    controller.scan_roll_preview_finished.emit()

    assert dialog._tiles[1].label.has_frame() is False
    assert all(dialog._tiles[f].label.has_frame() for f in (2, 3))
    assert "1" in dialog.status.text()  # the failed slot is named
    assert dialog.preview_all_btn.isEnabled() is True


def test_a_terminal_error_unlocks_the_dialog() -> None:
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3))

    dialog._on_preview_all()
    controller.scan_error.emit("scanner went away")

    assert dialog.preview_all_btn.isEnabled() is True
    assert "scanner went away" in dialog.status.text()


def test_a_second_preview_is_refused_while_one_is_running() -> None:
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _device(3))

    dialog._on_preview_all()
    dialog._on_preview_one(2)

    assert len(controller.preview_reqs) == 1


def test_start_preview_busy_is_handled_gracefully() -> None:
    controller = _FakeController(raise_on_preview=True)
    dialog = StripPreviewDialog(controller, _device(3))

    dialog._on_preview_one(1)  # must not raise

    assert controller.preview_reqs == []
    assert dialog.preview_all_btn.isEnabled() is True


def _pitch_device(capacity: int, pitch: float = 37.83) -> ScannerDevice:
    device = _device(capacity)
    return ScannerDevice(
        id=device.id,
        vendor=device.vendor,
        model=device.model,
        capabilities=dataclasses.replace(device.capabilities, frame_pitch_mm=pitch),
    )


def test_current_preview_sits_flush_left_and_ends_at_the_boundary() -> None:
    # The tile is anchored to the next scan: a raster previewed at the current
    # offset starts at the tile's left edge and ends where delivery blacks out.
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _pitch_device(3), initial_offset=4.8)

    dialog._on_preview_one(1)
    controller.deliver(1, offset=4.8 / 37.83)

    start, end = dialog._tiles[1].label._coverage
    assert start == pytest.approx(0.0, abs=1e-4)
    assert end == pytest.approx(1.0 - 4.8 / 37.83, abs=1e-4)
    assert dialog._tiles[1].previewed_offset == pytest.approx(4.8 / 37.83)


def test_moving_the_slider_slides_stale_previews_live() -> None:
    # No re-scan needed to see the effect: a raster previewed at x re-places at
    # (x - y, 1 - y) when the slider reads y, so content slides under the cursor.
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _pitch_device(3), initial_offset=2.0)

    dialog._on_preview_one(1)
    controller.deliver(1, offset=2.0 / 37.83)
    dialog.offset_slider.setValue(48)  # 4.8 mm

    start, end = dialog._tiles[1].label._coverage
    assert start == pytest.approx((2.0 - 4.8) / 37.83, abs=1e-4)
    assert end == pytest.approx(1.0 - 4.8 / 37.83, abs=1e-4)

    dialog.offset_slider.setValue(0)  # back down: the un-previewed head is a gap
    start, end = dialog._tiles[1].label._coverage
    assert start == pytest.approx(2.0 / 37.83, abs=1e-4)
    assert end == pytest.approx(1.0, abs=1e-4)


def test_preview_at_zero_offset_covers_the_whole_frame() -> None:
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _pitch_device(3))

    dialog._on_preview_one(1)
    controller.deliver(1, offset=0.0)

    assert dialog._tiles[1].label._coverage == (0.0, 1.0)


def test_offset_indicator_lands_on_the_edge_of_its_own_preview() -> None:
    # The cut line and the previewed content must agree once the preview is current
    # — the whole "did the offset apply?" confusion was these two disagreeing.
    controller = _FakeController()
    dialog = StripPreviewDialog(controller, _pitch_device(3), initial_offset=4.8)
    label = dialog._tiles[1].label

    dialog._on_preview_one(1)
    controller.deliver(1, offset=4.8 / 37.83)

    ((frac, _edge),) = label._offset_indicators
    assert 1.0 - frac == pytest.approx(label._coverage[1], abs=1e-4)  # band starts at the raster end


def test_indicator_uses_the_reported_pitch_over_the_scan_extent() -> None:
    dialog = StripPreviewDialog(_FakeController(), _pitch_device(3), initial_offset=4.0)
    dialog._refresh_offset_indicators()

    ((frac, _edge),) = dialog._tiles[1].label._offset_indicators
    assert frac == pytest.approx(4.0 / 37.83, abs=1e-4)  # not / 38.0 (max_area_mm)


def test_offset_past_the_frame_budget_warns_about_picture_loss() -> None:
    # The 20260719 scan-2 incident: 5.5 mm offset silently cost ~3.7 mm of every
    # frame's tail (delivery ends one pitch past the frame start, frame is 36 mm).
    dialog = StripPreviewDialog(_FakeController(), _pitch_device(3), initial_offset=5.5)
    dialog._refresh_offset_indicators()

    text = dialog.status.text()
    assert "cuts into the frame" in text
    assert "3.7 mm" in text  # 5.5 - (37.83 - 36.0)
    assert "1, 2, 3" in text


def test_offset_within_the_frame_budget_does_not_warn() -> None:
    dialog = StripPreviewDialog(_FakeController(), _pitch_device(3), initial_offset=5.5)
    dialog._refresh_offset_indicators()

    dialog.offset_slider.setValue(14)  # 1.4 mm — inside pitch - 36 mm budget

    assert dialog.status.text() == ""  # warning cleared once the offset is safe


def test_drift_past_the_pitch_is_clamped_and_flagged() -> None:
    dialog = StripPreviewDialog(_FakeController(), _pitch_device(14), initial_offset=8.0, initial_offset_modifier=2.5)

    assert dialog._offset_for_frame(1) == pytest.approx(8.0)
    assert dialog._offset_for_frame(14) == pytest.approx(36.83)  # pitch - 1 mm, not 40.5
    assert "14" in dialog.status.text()
