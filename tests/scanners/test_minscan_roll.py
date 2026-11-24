"""MinscanRollSession: the Minolta index/prescan path behind the Preview button."""

from __future__ import annotations

# The package import is intentionally guarded by importorskip below.
# ruff: noqa: E402

import threading

import numpy as np
import pytest

pytest.importorskip("minscan")

from negpy.infrastructure.scanners.base import ScanMode, ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.minscan_backend import MinscanRollSession, _crop_result, _frame_group_slot
from negpy.infrastructure.scanners.result import ScanResult


def _device(capacity: int | None) -> ScannerDevice:
    caps = ScannerCapabilities(
        ir_channel=True,
        supported_dpi=(4800, 2400) if capacity else (3200, 1600),
        supported_depths=(16,),
        sources=(ScanMode.NEGATIVE,),
        max_area_mm=(24.0, 36.0) if capacity else (56.0, 84.0),
        adapter_frame_capacity=capacity,
        frame_pitch_mm=38.0 if capacity else 0.0,
    )
    return ScannerDevice(id="minscan:/dev/sg1", vendor="Minolta", model="DiMAGE", capabilities=caps)


class _FakeMinscanSession:
    def __init__(self, *, fail_on: set[int] | None = None, cancel_on: int | None = None) -> None:
        self.previewed: list = []
        self.closed = False
        self._fail_on = fail_on or set()
        self._cancel_on = cancel_on

    def preview(self, params, progress, cancel) -> ScanResult:
        self.previewed.append(params)
        if self._cancel_on is not None and self._cancel_on == params.frame:
            cancel.set()
            raise RuntimeError("cancelled mid-preview")
        if params.frame in self._fail_on:
            raise RuntimeError(f"slot {params.frame} jammed")
        return ScanResult(rgb=np.zeros((4, 4, 3), dtype=np.uint16), ir=None, dpi=600, device_model="DiMAGE")

    def scan(self, params, progress, cancel) -> ScanResult:
        raise AssertionError("roll preview must use the hardware prescan, not a full scan")

    def close(self) -> None:
        self.closed = True


def _session(session, *, capacity: int | None = 6, dpi: int = 300, film_format: str | None = "135") -> MinscanRollSession:
    return MinscanRollSession(
        session,
        _device(capacity),
        dpi=dpi,
        film_format=film_format,
        film_type="negative",
    )


def test_35mm_previews_every_slot_through_the_index_scan() -> None:
    session = _FakeMinscanSession()
    roll = _session(session)

    previews = list(roll.preview((1, 2, 3, 4, 5, 6), cancel=threading.Event()))

    assert [p.slot for p in previews] == [1, 2, 3, 4, 5, 6]
    assert all(p.rgb is not None and p.error is None for p in previews)
    assert [p.frame for p in session.previewed] == [1, 2, 3, 4, 5, 6]
    assert all(p.depth == 16 for p in session.previewed)
    assert all(p.autofocus is False for p in session.previewed)
    assert all(p.capture_ir is False for p in session.previewed)
    assert all(p.film_format == "135" for p in session.previewed)


def test_120_has_one_slot() -> None:
    session = _FakeMinscanSession()
    roll = _session(session, capacity=None, film_format="69")

    assert roll.slot_count == 1
    (preview,) = list(roll.preview((1,), cancel=threading.Event()))
    assert preview.rgb is not None
    assert session.previewed[0].frame == 1


def test_a_failed_slot_does_not_stop_the_rest() -> None:
    session = _FakeMinscanSession(fail_on={2})
    previews = list(_session(session).preview((1, 2, 3), cancel=threading.Event()))

    assert [p.slot for p in previews] == [1, 2, 3]
    assert previews[1].rgb is None
    assert "jammed" in previews[1].error
    assert previews[2].rgb is not None


def test_cancel_stops_before_the_next_slot() -> None:
    session = _FakeMinscanSession()
    cancel = threading.Event()
    cancel.set()

    assert list(_session(session).preview((1, 2, 3), cancel=cancel)) == []
    assert session.previewed == []


def test_offsets_are_not_used_and_the_session_closes() -> None:
    session = _FakeMinscanSession()
    roll = _session(session)
    roll.set_offset(2, 0.5)
    roll.approve(2)

    list(roll.preview((2,), cancel=threading.Event()))
    roll.close()

    assert session.previewed[0].frame_offset_mm == 0.0
    assert session.closed is True


def test_35mm_frame_mapping_matches_minscan_parse_frame() -> None:
    """Even frames are A, odd frames are B; frames 1 and 2 share group 1."""
    assert [_frame_group_slot("35mm", n) for n in range(1, 7)] == [
        (1, 1),
        (1, 0),
        (2, 1),
        (2, 0),
        (3, 1),
        (3, 0),
    ]


def test_a_scan_window_crops_the_returned_raster() -> None:
    rgb = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3)
    result = ScanResult(rgb=rgb, ir=None, dpi=3200, device_model="DiMAGE")

    cropped = _crop_result(result, (1 / 6, 0.25, 4 / 6, 0.75))

    assert cropped.rgb.shape == (2, 3, 3)
    assert cropped.rgb[0, 0, 0] == rgb[1, 1, 0]


def test_no_window_leaves_the_raster_untouched() -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint16)
    result = ScanResult(rgb=rgb, ir=None, dpi=3200, device_model="DiMAGE")

    assert _crop_result(result, None) is result
