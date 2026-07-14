"""Fail-closed contracts needed by adjacent-window full-negative capture."""

import cv2
import numpy as np
import pytest
import threading

from negpy.infrastructure.scanners.params import ScannerCaptureState, ScanParams
from negpy.infrastructure.scanners.sane_backend import SaneBackend, _align_split_ir_full_canvas
from tests.scanners.test_coolscan_ir import (
    COOLSCAN3_OPT,
    FakeOption,
    SplitCaptureFakeSaneDev,
    _make_backend,
)


def _state() -> ScannerCaptureState:
    return ScannerCaptureState(
        focus_position=216,
        exposure_multiplier=1.0,
        red_exposure_us=1370.0,
        green_exposure_us=1290.0,
        blue_exposure_us=1120.0,
    )


def test_locked_capture_state_cannot_rerun_focus_or_metering() -> None:
    with pytest.raises(ValueError, match="locked capture state.*autofocus.*auto-exposure"):
        ScanParams(
            dpi=4000,
            depth=16,
            capture_ir=True,
            autofocus=True,
            auto_exposure=True,
            capture_state=_state(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("focus_position", -1),
        ("exposure_multiplier", 0.0),
        ("red_exposure_us", float("nan")),
        ("green_exposure_us", -10.0),
        ("blue_exposure_us", float("inf")),
    ),
)
def test_capture_state_rejects_values_that_cannot_be_replayed(field: str, value: float) -> None:
    values = {
        "focus_position": 216,
        "exposure_multiplier": 1.0,
        "red_exposure_us": 1370.0,
        "green_exposure_us": 1290.0,
        "blue_exposure_us": 1120.0,
    }
    values[field] = value

    with pytest.raises(ValueError):
        ScannerCaptureState(**values)


def _shifted(array: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(array)
    height, width = array.shape[:2]
    out[dy:, dx:] = array[: height - dy, : width - dx]
    return out


def test_split_ir_alignment_preserves_the_full_rgb_canvas_and_marks_invalid_borders() -> None:
    rng = np.random.default_rng(812)
    height, width = 120, 160
    dx, dy = 2, 3
    plane = rng.integers(2000, 62000, size=(height, width), dtype=np.uint16)
    reference = np.repeat(plane[:, :, None], 3, axis=2)
    target_ir = rng.integers(1000, 64000, size=(height, width), dtype=np.uint16)
    proxy = _shifted(reference, dx=dx, dy=dy)
    moving_ir = _shifted(target_ir, dx=dx, dy=dy)

    rgb, ir, valid, alignment = _align_split_ir_full_canvas(reference, proxy, moving_ir)

    assert rgb.shape == reference.shape
    assert ir.shape == target_ir.shape
    assert valid.shape == target_ir.shape and valid.dtype == np.bool_
    assert np.array_equal(rgb, reference)
    assert not valid[-dy:, :].any()
    assert not valid[:, -dx:].any()
    assert valid[: height - dy, : width - dx].all()
    error = np.abs(ir[: height - dy, : width - dx].astype(np.int32) - target_ir[: height - dy, : width - dx])
    assert float(np.percentile(error, 99)) <= 2.0
    assert alignment.mode == "phase-ecc"


def test_identity_alignment_marks_python_sane_missing_tail_row_without_cropping_rgb() -> None:
    rng = np.random.default_rng(813)
    reference = rng.integers(1000, 64000, size=(13, 10, 3), dtype=np.uint16)
    proxy = reference[:-1].copy()
    ir_source = rng.integers(1000, 64000, size=(12, 10), dtype=np.uint16)

    rgb, ir, valid, alignment = _align_split_ir_full_canvas(reference, proxy, ir_source)

    assert np.array_equal(rgb, reference)
    assert np.array_equal(ir[:-1], ir_source)
    assert not valid[-1].any()
    assert valid[:-1].all()
    assert alignment.mode == "identity"


def test_fractional_ir_alignment_marks_pixels_without_full_bilinear_support_invalid() -> None:
    rng = np.random.default_rng(815)
    height, width = 160, 200
    reference = rng.integers(2000, 62000, size=(height, width, 3), dtype=np.uint16)
    source_ir = rng.integers(1000, 64000, size=(height, width), dtype=np.uint16)
    fractional_shift = np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=np.float32)
    proxy = cv2.warpAffine(
        reference,
        fractional_shift,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    moving_ir = cv2.warpAffine(
        source_ir,
        fractional_shift,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    _rgb, _ir, valid, alignment = _align_split_ir_full_canvas(reference, proxy, moving_ir)

    assert alignment.mode == "phase-ecc"
    assert 0.25 < alignment.dx_px < 0.75
    assert 0.25 < alignment.dy_px < 0.75
    assert valid[2:-2, 2:-2].all()
    assert not valid[-1].any()
    assert not valid[:, -1].any()


class _StatefulSplitDevice(SplitCaptureFakeSaneDev):
    _INTERNAL = SplitCaptureFakeSaneDev._INTERNAL + ("option_values",)

    def __init__(self, multi_rgb: np.ndarray, single_rgbi: np.ndarray, *, opt_map=None) -> None:
        options = dict(COOLSCAN3_OPT)
        options.update(
            {
                "exposure": FakeOption(),
                "red_exposure": FakeOption(),
                "green_exposure": FakeOption(),
                "blue_exposure": FakeOption(),
                "focus": FakeOption(),
            }
        )
        if opt_map is not None:
            options = opt_map
        super().__init__(multi_rgb, single_rgbi, opt_map=options)
        object.__setattr__(
            self,
            "option_values",
            {
                "exposure": 1.0,
                "red_exposure": 1200.0,
                "green_exposure": 1200.0,
                "blue_exposure": 1000.0,
                "focus": 0,
                "autofocus": False,
                "ae": False,
            },
        )

    def __getattr__(self, name: str):
        values = object.__getattribute__(self, "option_values")
        if name in values:
            return values[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name not in self._INTERNAL and "option_values" in self.__dict__ and name in self.option_values:
            self.option_values[name] = value
        super().__setattr__(name, value)

    def start(self) -> None:
        if self.option_values["autofocus"]:
            self.option_values["focus"] = 216
        if self.option_values["ae"]:
            self.option_values.update(
                {
                    "exposure": 1.0,
                    "red_exposure": 1370.0,
                    "green_exposure": 1290.0,
                    "blue_exposure": 1120.0,
                }
            )
        super().start()


def _stateful_device(*, opt_map=None) -> _StatefulSplitDevice:
    rng = np.random.default_rng(814)
    multi_rgb = rng.integers(1000, 64000, size=(24, 20, 3), dtype=np.uint16)
    ir = rng.integers(1000, 64000, size=(24, 20), dtype=np.uint16)
    return _StatefulSplitDevice(multi_rgb, np.dstack((multi_rgb, ir)), opt_map=opt_map)


def test_full_canvas_scan_returns_actual_state_and_raw_split_sources() -> None:
    dev = _stateful_device()
    result = _make_backend(dev).scan(
        "coolscan3:usb:test",
        ScanParams(
            dpi=4000,
            depth=16,
            capture_ir=True,
            samples_per_scan=4,
            frame=3,
            autofocus=True,
            auto_exposure=True,
            preserve_full_canvas=True,
        ),
        None,
        threading.Event(),
    )

    assert result.capture_state == _state()
    assert result.split_source is not None
    assert np.array_equal(result.split_source.rgb4x, result.rgb)
    assert result.ir_valid_mask is not None and result.ir_valid_mask.all()


def test_locked_state_is_replayed_with_focus_and_metering_disabled() -> None:
    dev = _stateful_device()
    result = _make_backend(dev).scan(
        "coolscan3:usb:test",
        ScanParams(
            dpi=4000,
            depth=16,
            capture_ir=True,
            samples_per_scan=4,
            frame=4,
            autofocus=False,
            auto_exposure=False,
            capture_state=_state(),
            preserve_full_canvas=True,
        ),
        None,
        threading.Event(),
    )

    for settings in dev.start_settings:
        assert settings["focus"] == 216
        assert settings["exposure"] == 1.0
        assert settings["red_exposure"] == 1370.0
        assert settings["green_exposure"] == 1290.0
        assert settings["blue_exposure"] == 1120.0
        assert settings["autofocus"] is False
        assert settings["ae"] is False
    assert result.capture_state == _state()


def test_locked_state_missing_option_fails_before_transport_moves() -> None:
    probe = _stateful_device()
    options = dict(probe.opt_map)
    del options["green_exposure"]
    dev = _stateful_device(opt_map=options)
    backend: SaneBackend = _make_backend(dev)

    with pytest.raises(RuntimeError, match="green_exposure"):
        backend.scan(
            "coolscan3:usb:test",
            ScanParams(
                dpi=4000,
                depth=16,
                capture_ir=True,
                samples_per_scan=4,
                frame=4,
                autofocus=False,
                auto_exposure=False,
                capture_state=_state(),
                preserve_full_canvas=True,
            ),
            None,
            threading.Event(),
        )

    assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y"} for event in dev.events)


def test_locked_uncalibrated_focus_fails_before_transport_moves() -> None:
    dev = _stateful_device()
    uncalibrated = ScannerCaptureState(
        focus_position=0,
        exposure_multiplier=1.0,
        red_exposure_us=1370.0,
        green_exposure_us=1290.0,
        blue_exposure_us=1120.0,
    )

    with pytest.raises(RuntimeError, match="uncalibrated focus position"):
        _make_backend(dev).scan(
            "coolscan3:usb:test",
            ScanParams(
                dpi=4000,
                depth=16,
                capture_ir=True,
                samples_per_scan=4,
                frame=4,
                autofocus=False,
                auto_exposure=False,
                capture_state=uncalibrated,
                preserve_full_canvas=True,
            ),
            None,
            threading.Event(),
        )

    assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y"} for event in dev.events)


def test_full_canvas_autofocus_missing_fails_before_transport_moves() -> None:
    probe = _stateful_device()
    options = dict(probe.opt_map)
    del options["autofocus"]
    dev = _stateful_device(opt_map=options)

    with pytest.raises(RuntimeError, match="autofocus"):
        _make_backend(dev).scan(
            "coolscan3:usb:test",
            ScanParams(
                dpi=4000,
                depth=16,
                capture_ir=True,
                samples_per_scan=4,
                frame=3,
                autofocus=True,
                auto_exposure=True,
                preserve_full_canvas=True,
            ),
            None,
            threading.Event(),
        )

    assert not any(event[0] == "set" and event[1] in {"frame", "subframe", "br_y"} for event in dev.events)


def test_full_canvas_autofocus_write_failure_fails_loud() -> None:
    class RejectingAutofocusDevice(_StatefulSplitDevice):
        def __setattr__(self, name: str, value) -> None:
            if name == "autofocus" and value is True:
                raise RuntimeError("autofocus command rejected")
            super().__setattr__(name, value)

    probe = _stateful_device()
    dev = RejectingAutofocusDevice(
        probe.multi_rgb,
        probe.single_rgbi,
        opt_map=probe.opt_map,
    )

    with pytest.raises(RuntimeError, match="Could not enable autofocus.*rejected"):
        _make_backend(dev).scan(
            "coolscan3:usb:test",
            ScanParams(
                dpi=4000,
                depth=16,
                capture_ir=True,
                samples_per_scan=4,
                frame=3,
                autofocus=True,
                auto_exposure=True,
                preserve_full_canvas=True,
            ),
            None,
            threading.Event(),
        )

    assert dev.start_settings == []


def test_full_canvas_rejects_an_uncalibrated_focus_after_autofocus() -> None:
    class UncalibratedAutofocusDevice(_StatefulSplitDevice):
        def start(self) -> None:
            if self.option_values["ae"]:
                self.option_values.update(
                    {
                        "exposure": 1.0,
                        "red_exposure": 1370.0,
                        "green_exposure": 1290.0,
                        "blue_exposure": 1120.0,
                    }
                )
            SplitCaptureFakeSaneDev.start(self)

    probe = _stateful_device()
    dev = UncalibratedAutofocusDevice(
        probe.multi_rgb,
        probe.single_rgbi,
        opt_map=probe.opt_map,
    )

    with pytest.raises(RuntimeError, match="uncalibrated focus position"):
        _make_backend(dev).scan(
            "coolscan3:usb:test",
            ScanParams(
                dpi=4000,
                depth=16,
                capture_ir=True,
                samples_per_scan=4,
                frame=3,
                autofocus=True,
                auto_exposure=True,
                preserve_full_canvas=True,
            ),
            None,
            threading.Event(),
        )

    assert len(dev.start_settings) == 1
