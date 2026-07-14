from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest

from negpy.infrastructure.scanners.params import (
    RegisteredScanGeometry,
    ScannerCaptureState,
    ScanParams,
)
from negpy.infrastructure.scanners.result import ScanResult, SplitIrAlignment, SplitSourceCapture
from negpy.services.scanning.full_negative_capture import FullNegativeCapturer
from negpy.services.scanning.roll_service import FrameRegistration


PITCH_ROWS = 8
PHASE_ROW = 3


def _state() -> ScannerCaptureState:
    return ScannerCaptureState(
        focus_position=216,
        exposure_multiplier=1.0,
        red_exposure_us=1370.0,
        green_exposure_us=1290.0,
        blue_exposure_us=1120.0,
    )


def _source(rows: np.ndarray, *, state: ScannerCaptureState | None = None) -> ScanResult:
    height = len(rows)
    rgb = np.broadcast_to(rows[:, None, None], (height, 4, 3)).astype(np.uint16, copy=True)
    ir = np.broadcast_to((rows + 100)[:, None], (height, 4)).astype(np.uint16, copy=True)
    return ScanResult(
        rgb=rgb,
        ir=ir,
        dpi=4000,
        device_model="Nikon LS-5000",
        ir_alignment=SplitIrAlignment(mode="identity", dx_px=0.0, dy_px=0.0),
        ir_valid_mask=np.ones((height, 4), dtype=np.bool_),
        capture_state=_state() if state is None else state,
    )


def _registration() -> FrameRegistration:
    phase_mm = (PHASE_ROW + 0.25) * 25.4 / 4000
    return FrameRegistration(
        frame=37,
        target_start_row=110,
        usable_tail_row=595.0,
        confidence="high",
        geometry=RegisteredScanGeometry(
            frame=37,
            subframe_mm=phase_mm,
            br_y_device_px=4938,
        ),
    )


def _recipe() -> ScanParams:
    return ScanParams(
        dpi=4000,
        depth=16,
        capture_ir=True,
        autofocus=False,
        samples_per_scan=4,
        auto_exposure=False,
    )


class _Scanner:
    def __init__(self, results: list[ScanResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, ScanParams, object, threading.Event]] = []

    def run_scan(self, device_id, params, progress, cancel):
        self.calls.append((device_id, params, progress, cancel))
        return self.results.pop(0)


def test_frame_37_capture_uses_one_full_window_then_only_the_bounded_frame_38_tail() -> None:
    scanner = _Scanner(
        [
            _source(np.arange(PITCH_ROWS, dtype=np.uint16)),
            _source(np.arange(PITCH_ROWS, PITCH_ROWS + PHASE_ROW, dtype=np.uint16)),
        ]
    )
    stop = threading.Event()

    captured = FullNegativeCapturer(scanner=scanner, pitch_rows=PITCH_ROWS).capture(
        device_id="coolscan3:usb:test",
        registration=_registration(),
        recipe=_recipe(),
        stop=stop,
    )

    assert len(scanner.calls) == 2
    assert [call[0] for call in scanner.calls] == ["coolscan3:usb:test"] * 2
    first_params = scanner.calls[0][1]
    second_params = scanner.calls[1][1]
    assert first_params.registered_geometry == RegisteredScanGeometry(
        frame=37,
        subframe_mm=0.0,
        br_y_device_px=PITCH_ROWS - 1,
    )
    assert second_params.registered_geometry == RegisteredScanGeometry(
        frame=38,
        subframe_mm=0.0,
        br_y_device_px=PHASE_ROW - 1,
    )
    assert first_params.autofocus and first_params.auto_exposure
    assert first_params.capture_state is None
    assert second_params.autofocus is False and second_params.auto_exposure is False
    assert second_params.capture_state == _state()
    assert first_params.preserve_full_canvas and second_params.preserve_full_canvas
    assert first_params.samples_per_scan == second_params.samples_per_scan == 4
    assert scanner.calls[0][3] is scanner.calls[1][3]
    assert scanner.calls[0][3] is not stop
    np.testing.assert_array_equal(
        captured.result.rgb[:, 0, 0],
        np.arange(PHASE_ROW, PHASE_ROW + PITCH_ROWS, dtype=np.uint16),
    )
    assert captured.first_source.rgb.shape[0] == PITCH_ROWS
    assert captured.second_source.rgb.shape[0] == PHASE_ROW


def test_stop_requested_during_first_transfer_is_honored_before_the_successor_transfer() -> None:
    stop = threading.Event()

    class StoppingScanner(_Scanner):
        def run_scan(self, device_id, params, progress, cancel):
            result = super().run_scan(device_id, params, progress, cancel)
            stop.set()
            return result

    scanner = StoppingScanner([_source(np.arange(PITCH_ROWS, dtype=np.uint16))])

    with pytest.raises(RuntimeError, match="cancelled between transfers"):
        FullNegativeCapturer(scanner=scanner, pitch_rows=PITCH_ROWS).capture(
            device_id="coolscan3:usb:test",
            registration=_registration(),
            recipe=_recipe(),
            stop=stop,
        )

    assert len(scanner.calls) == 1
    assert scanner.calls[0][3].is_set() is False


def test_missing_actual_capture_state_fails_before_the_successor_transport_moves() -> None:
    first = replace(
        _source(np.arange(PITCH_ROWS, dtype=np.uint16)),
        capture_state=None,
    )
    scanner = _Scanner([first])

    with pytest.raises(RuntimeError, match="missing the actual focus/exposure state"):
        FullNegativeCapturer(scanner=scanner, pitch_rows=PITCH_ROWS).capture(
            device_id="coolscan3:usb:test",
            registration=_registration(),
            recipe=_recipe(),
            stop=threading.Event(),
        )

    assert len(scanner.calls) == 1


def test_each_validated_source_can_be_committed_before_the_next_step() -> None:
    scanner = _Scanner(
        [
            _source(np.arange(PITCH_ROWS, dtype=np.uint16)),
            _source(np.arange(PITCH_ROWS, PITCH_ROWS + PHASE_ROW, dtype=np.uint16)),
        ]
    )
    committed: list[tuple[str, int, int, int]] = []

    def commit(label: str, physical_frame: int, source: ScanResult) -> None:
        committed.append((label, physical_frame, source.rgb.shape[0], len(scanner.calls)))

    FullNegativeCapturer(scanner=scanner, pitch_rows=PITCH_ROWS).capture(
        device_id="coolscan3:usb:test",
        registration=_registration(),
        recipe=_recipe(),
        stop=threading.Event(),
        on_source=commit,
    )

    assert committed == [
        ("first", 37, PITCH_ROWS, 1),
        ("second", 38, PHASE_ROW, 2),
    ]


def test_persisted_raw_split_planes_are_released_before_capture_returns() -> None:
    def with_split(source: ScanResult) -> ScanResult:
        assert source.ir is not None
        return replace(
            source,
            split_source=SplitSourceCapture(
                rgb4x=source.rgb,
                rgb1x_proxy=source.rgb.copy(),
                ir1x=source.ir.copy(),
            ),
        )

    scanner = _Scanner(
        [
            with_split(_source(np.arange(PITCH_ROWS, dtype=np.uint16))),
            with_split(_source(np.arange(PITCH_ROWS, PITCH_ROWS + PHASE_ROW, dtype=np.uint16))),
        ]
    )
    persisted: list[str] = []

    def persist(label: str, _physical_frame: int, source: ScanResult) -> None:
        assert source.split_source is not None
        persisted.append(label)

    captured = FullNegativeCapturer(scanner=scanner, pitch_rows=PITCH_ROWS).capture(
        device_id="coolscan3:usb:test",
        registration=_registration(),
        recipe=_recipe(),
        stop=threading.Event(),
        on_source=persist,
    )

    assert persisted == ["first", "second"]
    assert captured.first_source.split_source is None
    assert captured.second_source.split_source is None


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"ir_alignment": None}, "alignment failed QC"),
        ({"ir_valid_mask": None}, "missing IR or its validity mask"),
        (
            {
                "capture_state": ScannerCaptureState(
                    focus_position=0,
                    exposure_multiplier=1.0,
                    red_exposure_us=1370.0,
                    green_exposure_us=1290.0,
                    blue_exposure_us=1120.0,
                )
            },
            "uncalibrated focus position",
        ),
    ),
)
def test_invalid_first_source_alignment_or_mask_fails_before_successor_transport(
    changes: dict[str, object],
    message: str,
) -> None:
    first = replace(
        _source(np.arange(PITCH_ROWS, dtype=np.uint16)),
        **changes,
    )
    scanner = _Scanner([first])

    with pytest.raises(RuntimeError, match=message):
        FullNegativeCapturer(scanner=scanner, pitch_rows=PITCH_ROWS).capture(
            device_id="coolscan3:usb:test",
            registration=_registration(),
            recipe=_recipe(),
            stop=threading.Event(),
        )

    assert len(scanner.calls) == 1


def test_clipped_first_source_is_warning_only_and_does_not_block_successor() -> None:
    first = _source(np.arange(PITCH_ROWS, dtype=np.uint16))
    first = replace(first, rgb=np.full_like(first.rgb, np.iinfo(np.uint16).max))
    second = _source(np.arange(PITCH_ROWS, PITCH_ROWS + PHASE_ROW, dtype=np.uint16))
    scanner = _Scanner([first, second])

    captured = FullNegativeCapturer(scanner=scanner, pitch_rows=PITCH_ROWS).capture(
        device_id="coolscan3:usb:test",
        registration=_registration(),
        recipe=_recipe(),
        stop=threading.Event(),
    )

    assert captured.result.rgb.shape[0] == PITCH_ROWS
    assert len(scanner.calls) == 2


def test_repeated_row_smear_in_first_source_fails_before_successor_transport() -> None:
    height, width = 160, 96
    rows = np.arange(height, dtype=np.uint32)[:, None]
    columns = np.arange(width, dtype=np.uint32)[None, :]
    plane = (4000 + rows * 200 + columns * 200).astype(np.uint16)
    plane[80:] = plane[80]
    rgb = np.stack((plane, plane + 250, plane + 500), axis=2).astype(np.uint16)
    ir = (plane + 750).astype(np.uint16)
    first = ScanResult(
        rgb=rgb,
        ir=ir,
        dpi=4000,
        device_model="Nikon LS-5000",
        ir_alignment=SplitIrAlignment(mode="identity", dx_px=0.0, dy_px=0.0),
        ir_valid_mask=np.ones((height, width), dtype=np.bool_),
        capture_state=_state(),
    )
    scanner = _Scanner([first])

    with pytest.raises(RuntimeError, match="first source.*stopped-transport smear"):
        FullNegativeCapturer(scanner=scanner, pitch_rows=height).capture(
            device_id="coolscan3:usb:test",
            registration=_registration(),
            recipe=_recipe(),
            stop=threading.Event(),
        )

    assert len(scanner.calls) == 1


def test_reconstructed_frame_clipping_is_warning_only() -> None:
    pitch, phase = 200, 100
    first = _source(np.arange(pitch, dtype=np.uint16))
    second = _source(np.arange(pitch, pitch + phase, dtype=np.uint16))
    first_rgb = first.rgb.copy()
    second_rgb = second.rgb.copy()
    first_rgb[phase : phase + 2] = np.iinfo(np.uint16).max
    second_rgb[0] = np.iinfo(np.uint16).max
    scanner = _Scanner(
        [
            replace(first, rgb=first_rgb),
            replace(second, rgb=second_rgb),
        ]
    )
    phase_mm = (phase + 0.25) * 25.4 / 4000
    registration = replace(
        _registration(),
        geometry=RegisteredScanGeometry(
            frame=37,
            subframe_mm=phase_mm,
            br_y_device_px=4938,
        ),
    )

    captured = FullNegativeCapturer(scanner=scanner, pitch_rows=pitch).capture(
        device_id="coolscan3:usb:test",
        registration=registration,
        recipe=_recipe(),
        stop=threading.Event(),
    )

    assert captured.result.rgb.shape[0] == pitch
    assert len(scanner.calls) == 2


def test_reconstructed_frame_runs_final_stopped_transport_smear_qc() -> None:
    pitch, phase, width = 160, 40, 96
    rows = np.arange(pitch, dtype=np.uint32)[:, None]
    columns = np.arange(width, dtype=np.uint32)[None, :]
    plane = (4000 + rows * 200 + columns * 200).astype(np.uint16)
    plane[130:] = plane[130]
    first_rgb = np.stack((plane, plane + 250, plane + 500), axis=2).astype(np.uint16)
    second_plane = np.broadcast_to(plane[130], (phase, width)).copy()
    second_rgb = np.stack(
        (second_plane, second_plane + 250, second_plane + 500),
        axis=2,
    ).astype(np.uint16)

    def source(rgb: np.ndarray) -> ScanResult:
        height = rgb.shape[0]
        return ScanResult(
            rgb=rgb,
            ir=rgb[:, :, 0].copy(),
            dpi=4000,
            device_model="Nikon LS-5000",
            ir_alignment=SplitIrAlignment(mode="identity", dx_px=0.0, dy_px=0.0),
            ir_valid_mask=np.ones((height, width), dtype=np.bool_),
            capture_state=_state(),
        )

    scanner = _Scanner([source(first_rgb), source(second_rgb)])
    phase_mm = (phase + 0.25) * 25.4 / 4000
    registration = replace(
        _registration(),
        geometry=RegisteredScanGeometry(
            frame=37,
            subframe_mm=phase_mm,
            br_y_device_px=4938,
        ),
    )

    with pytest.raises(RuntimeError, match="reconstructed frame.*stopped-transport smear"):
        FullNegativeCapturer(scanner=scanner, pitch_rows=pitch).capture(
            device_id="coolscan3:usb:test",
            registration=registration,
            recipe=_recipe(),
            stop=threading.Event(),
        )

    assert len(scanner.calls) == 2
