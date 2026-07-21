"""PerFrameRollSession: a whole strip built from single-frame scans."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from negpy.infrastructure.scanners.base import ScanMode, ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.per_frame_roll import PerFrameRollSession
from negpy.infrastructure.scanners.result import ScanResult

_PITCH = 38.0


def _device(capacity: int = 3, pitch: float = _PITCH) -> ScannerDevice:
    caps = ScannerCapabilities(
        ir_channel=False,
        supported_dpi=(1000,),
        supported_depths=(8, 16),
        sources=(ScanMode.NEGATIVE,),
        max_area_mm=(25.0, 38.0),
        adapter_frame_capacity=capacity,
        frame_pitch_mm=pitch,
    )
    return ScannerDevice(id="coolscan3:test", vendor="Nikon", model="LS-50", capabilities=caps)


class _FakeBackend:
    def __init__(self, *, fail_on: set[int] | None = None, cancel_on: int | None = None) -> None:
        self.params_seen: list = []
        self._fail_on = fail_on or set()
        self._cancel_on = cancel_on

    def scan(self, device_id, params, progress, cancel) -> ScanResult:
        self.params_seen.append(params)
        if self._cancel_on == params.frame:
            cancel.set()
            raise RuntimeError("cancelled mid-scan")
        if params.frame in self._fail_on:
            raise RuntimeError(f"slot {params.frame} jammed")
        return ScanResult(rgb=np.zeros((4, 4, 3), dtype=np.uint16), ir=None, dpi=params.dpi, device_model="LS-50")


def _session(backend, *, device=None, dpi: int = 1000) -> PerFrameRollSession:
    return PerFrameRollSession(backend, device or _device(), dpi=dpi)


def test_yields_one_preview_per_slot_in_order() -> None:
    backend = _FakeBackend()
    previews = list(_session(backend).preview((1, 2, 3), cancel=threading.Event()))

    assert [p.slot for p in previews] == [1, 2, 3]
    assert all(p.rgb is not None and p.error is None for p in previews)
    assert [p.frame for p in backend.params_seen] == [1, 2, 3]


def test_previews_are_cheap_reads() -> None:
    """Previews are for looking at: 8-bit, no IR, no autofocus or metering delay."""
    backend = _FakeBackend()
    list(_session(backend, dpi=400).preview((1,), cancel=threading.Event()))

    (params,) = backend.params_seen
    assert (params.dpi, params.depth) == (400, 8)
    assert params.capture_ir is False
    assert params.autofocus is False
    assert params.auto_exposure is False
    assert params.window is None


def test_a_failed_slot_is_yielded_and_the_strip_continues() -> None:
    backend = _FakeBackend(fail_on={2})
    previews = list(_session(backend).preview((1, 2, 3), cancel=threading.Event()))

    assert [p.slot for p in previews] == [1, 2, 3]
    assert previews[1].rgb is None
    assert "jammed" in previews[1].error
    assert previews[2].rgb is not None  # slot 3 still ran


def test_cancel_before_a_slot_stops_the_strip() -> None:
    backend = _FakeBackend()
    cancel = threading.Event()
    cancel.set()

    assert list(_session(backend).preview((1, 2, 3), cancel=cancel)) == []
    assert backend.params_seen == []


def test_cancel_during_a_slot_is_not_reported_as_a_slot_failure() -> None:
    backend = _FakeBackend(cancel_on=2)
    previews = list(_session(backend).preview((1, 2, 3), cancel=threading.Event()))

    assert [p.slot for p in previews] == [1]  # no error entry for the cancelled slot


def test_offsets_convert_from_pitch_fractions_to_mm() -> None:
    backend = _FakeBackend()
    session = _session(backend)
    session.set_offset(2, 0.25)

    previews = list(session.preview((1, 2), cancel=threading.Event()))

    assert [p.frame_offset_mm for p in backend.params_seen] == pytest.approx([0.0, 0.25 * _PITCH])
    assert previews[1].offset == pytest.approx(0.25)


def test_offsets_are_clamped_short_of_the_frame_boundary() -> None:
    """Past one pitch the window collapses and the scan comes back empty."""
    backend = _FakeBackend()
    session = _session(backend)
    session.set_offset(1, 1.5)  # 1.5 pitches — unreachable

    (preview,) = list(session.preview((1,), cancel=threading.Event()))

    assert backend.params_seen[0].frame_offset_mm == pytest.approx(_PITCH - 1.0)
    assert preview.offset == pytest.approx((_PITCH - 1.0) / _PITCH)


def test_negative_offsets_floor_at_zero() -> None:
    backend = _FakeBackend()
    session = _session(backend)
    session.set_offset(1, -0.5)  # the transport cannot back up

    (preview,) = list(session.preview((1,), cancel=threading.Event()))

    assert backend.params_seen[0].frame_offset_mm == 0.0
    assert preview.offset == 0.0


def test_an_unknown_pitch_disables_offsets_rather_than_guessing() -> None:
    caps = ScannerCapabilities(
        ir_channel=False,
        supported_dpi=(1000,),
        supported_depths=(8,),
        sources=(ScanMode.NEGATIVE,),
        max_area_mm=(0.0, 0.0),
        adapter_frame_capacity=2,
        frame_pitch_mm=0.0,
    )
    device = ScannerDevice(id="x:1", vendor="v", model="m", capabilities=caps)
    backend = _FakeBackend()
    session = PerFrameRollSession(backend, device, dpi=1000)
    session.set_offset(1, 0.5)

    (preview,) = list(session.preview((1,), cancel=threading.Event()))

    assert backend.params_seen[0].frame_offset_mm == 0.0
    assert preview.offset == 0.0


def test_slot_count_follows_the_adapter_capacity() -> None:
    assert _session(_FakeBackend(), device=_device(capacity=40)).slot_count == 40


def test_approve_and_close_are_no_ops() -> None:
    session = _session(_FakeBackend())
    session.approve(1)
    session.close()
    session.close()
