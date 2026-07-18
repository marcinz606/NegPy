#!/usr/bin/env python3
"""Capture the exact dual RGBI source pair required by portable Digital ICE.

Nikon's captured caller consumes a distinct 285 dpi RGBI prepass before the
main RGBI image.  This runner reproduces that *acquisition boundary*
without loading Nikon code:

* one libsane initialization and one device handle;
* a dedicated full-aperture 16-bit RGBI prepass at 285 dpi;
* a full-aperture 16-bit RGBI main scan at 4000 dpi;
* one sample per scan in both epochs, never RGBI4x averaging or meter data;
* focus/exposure measured by the prepass and replayed for the main pass;
* raw ``sane_read`` bytes, avoiding python-sane's known final-RGBI-row loss;
* immutable full-aperture arrays;
* a transactional receipt containing every option write, shape, digest, and
  ordering check. The final bundle is not published until it verifies itself.

Default invocation is scanner-free and prints the exact live command:

    uv run python -m negpy.infrastructure.scanners.dice_dual_source_runner

Live acquisition requires an explicit physical confirmation:

    uv run python -m negpy.infrastructure.scanners.dice_dual_source_runner \
      --live --confirm-film-stationary
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import shlex
import sys
import tempfile
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from negpy.infrastructure.scanners.params import ScannerCaptureState


NATIVE_OPTICAL_DPI = 4000
PREPASS_DPI = 285
NATIVE_MAIN_DPI = 4000
ROLL_FULL_APERTURE = (0, 0, 3945, 5958)  # inclusive tl_x, tl_y, br_x, br_y
MOUNTED_FULL_APERTURE = (0, 0, 3945, 5781)
FULL_APERTURE = ROLL_FULL_APERTURE
SUPPORTED_VENDOR = "nikon"
SUPPORTED_MODELS = frozenset(
    {
        "ls5000",
        "ls5000ed",
        "supercoolscan5000ed",
        "nikonsupercoolscan5000ed",
    }
)

SANE_STATUS_GOOD = 0
SANE_STATUS_EOF = 5
SANE_ACTION_GET_VALUE = 0
SANE_ACTION_SET_VALUE = 1
SANE_TYPE_BOOL = 0
SANE_TYPE_INT = 1
SANE_TYPE_FIXED = 2
SANE_TYPE_STRING = 3
SANE_TYPE_BUTTON = 4
SANE_CONSTRAINT_NONE = 0
SANE_CONSTRAINT_RANGE = 1
SANE_CONSTRAINT_WORD_LIST = 2
SANE_CONSTRAINT_STRING_LIST = 3
SANE_CAP_SOFT_SELECT = 1 << 0
SANE_CAP_INACTIVE = 1 << 5
SANE_INFO_RELOAD_OPTIONS = 1 << 1
SANE_FRAME_RGB = 1


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalise_option_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _strip_net_prefix(device_id: str) -> str:
    if not device_id.startswith("net:"):
        return device_id
    rest = device_id[4:]
    if rest.startswith("["):
        close = rest.find("]:")
        return rest[close + 2 :] if close >= 0 else device_id
    _, separator, backend = rest.partition(":")
    return backend if separator else device_id


def _normalise_identity_text(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


@dataclass(frozen=True)
class ScannerIdentity:
    device_id: str
    vendor: str
    model: str
    kind: str

    def __post_init__(self) -> None:
        for name in ("device_id", "vendor", "model", "kind"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"scanner identity {name} must be a non-empty string")

    @property
    def is_supported_ls5000(self) -> bool:
        return (
            _normalise_identity_text(self.vendor) == SUPPORTED_VENDOR
            and _normalise_identity_text(self.model) in SUPPORTED_MODELS
            and _strip_net_prefix(self.device_id).startswith("coolscan3:")
        )

    def require_supported_ls5000(self) -> None:
        if not self.is_supported_ls5000:
            raise RuntimeError(
                "portable Digital ICE requires a Nikon Super Coolscan 5000 ED; "
                f"found vendor={self.vendor!r}, model={self.model!r}, "
                f"device={self.device_id!r}"
            )


@dataclass(frozen=True)
class PixelWindow:
    tl_x: int
    tl_y: int
    br_x: int
    br_y: int

    def __post_init__(self) -> None:
        values = (self.tl_x, self.tl_y, self.br_x, self.br_y)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("pixel-window coordinates must be non-negative integers")
        if self.br_x < self.tl_x or self.br_y < self.tl_y:
            raise ValueError("pixel-window bottom-right must not precede top-left")

    @property
    def native_width(self) -> int:
        return self.br_x - self.tl_x + 1

    @property
    def native_height(self) -> int:
        return self.br_y - self.tl_y + 1

    def output_shape(self, dpi: int) -> tuple[int, int]:
        if type(dpi) is not int or dpi <= 0:
            raise ValueError("dpi must be a positive integer")
        pitch = NATIVE_OPTICAL_DPI // dpi
        if pitch <= 0 or NATIVE_OPTICAL_DPI // pitch != dpi:
            raise ValueError(f"{dpi} dpi is not an exact LS-5000 integer-pitch mode")
        return self.native_height // pitch, self.native_width // pitch


@dataclass(frozen=True)
class DiceDualSourcePlan:
    """One bounded 285 dpi to 4000 dpi acquisition for Digital ICE."""

    window: PixelWindow = PixelWindow(*FULL_APERTURE)
    prepass_dpi: int = PREPASS_DPI
    main_dpi: int = NATIVE_MAIN_DPI
    depth: int = 16
    frame: int | None = None
    subframe_mm: float | None = None
    transport: str = "roll"

    @classmethod
    def for_transport(
        cls,
        transport: str = "roll",
        *,
        frame: int | None = None,
        subframe_mm: float | None = None,
    ) -> DiceDualSourcePlan:
        """Build the exact plan for a roll adapter or mounted holder."""

        if transport not in {"roll", "mounted"}:
            raise ValueError("transport must be 'roll' or 'mounted'")
        if transport == "mounted" and (frame is not None or subframe_mm is not None):
            raise ValueError("mounted transport cannot select a roll frame or subframe")
        window = PixelWindow(
            *(MOUNTED_FULL_APERTURE if transport == "mounted" else ROLL_FULL_APERTURE)
        )

        return cls(
            window=window,
            frame=frame,
            subframe_mm=subframe_mm,
            transport=transport,
        )

    def __post_init__(self) -> None:
        if self.transport not in {"roll", "mounted"}:
            raise ValueError("transport must be 'roll' or 'mounted'")
        if self.prepass_dpi != PREPASS_DPI:
            raise ValueError(f"the DICE source contract requires a {PREPASS_DPI} dpi prepass")
        if self.main_dpi != NATIVE_MAIN_DPI:
            raise ValueError(f"the DICE source contract requires a {NATIVE_MAIN_DPI} dpi main scan")
        if self.depth != 16:
            raise ValueError("the DICE source contract requires 16-bit samples")
        if self.frame is not None and (type(self.frame) is not int or self.frame < 1):
            raise ValueError("frame must be a positive integer or None")
        if self.subframe_mm is not None:
            if self.frame is None:
                raise ValueError("subframe_mm requires a selected roll frame")
            if not math.isfinite(self.subframe_mm) or self.subframe_mm < 0:
                raise ValueError("subframe_mm must be finite and non-negative")

    @property
    def prepass_full_shape(self) -> tuple[int, int]:
        return self.window.output_shape(self.prepass_dpi)

    @property
    def main_full_shape(self) -> tuple[int, int]:
        return self.window.output_shape(self.main_dpi)

    def semantic_dict(self) -> dict[str, object]:
        return {
            "native_optical_dpi": NATIVE_OPTICAL_DPI,
            "window": asdict(self.window),
            "prepass": {
                "dpi": self.prepass_dpi,
                "full_shape_hw": list(self.prepass_full_shape),
                "source": "dedicated_single_sample_rgbi",
            },
            "main": {
                "dpi": self.main_dpi,
                "full_shape_hw": list(self.main_full_shape),
                "source": "dedicated_single_sample_rgbi",
            },
            "depth": self.depth,
            "samples_per_scan": 1,
            "channels": ["red", "green", "blue", "infrared"],
            "orientation": "scanner_native_portrait",
            "transport": self.transport,
            "frame": self.frame,
            "subframe_mm": self.subframe_mm,
            "same_frame_rule": "one handle; fixed window and transport; prepass state replayed for main",
        }


@dataclass(frozen=True)
class OptionInfo:
    name: str
    value_type: int
    active: bool
    settable: bool
    constraint: tuple[float | int | str, ...] | None = None
    range_constraint: tuple[float, float, float] | None = None

    def supports(self, value: bool | int | float | str) -> bool:
        if self.constraint is not None:
            return any(
                candidate == value
                or (
                    isinstance(candidate, (int, float))
                    and isinstance(value, (int, float))
                    and math.isclose(float(candidate), float(value), rel_tol=0.0, abs_tol=1e-9)
                )
                for candidate in self.constraint
            )
        if self.range_constraint is None:
            return True
        lower, upper, quantum = self.range_constraint
        numeric = float(value)
        if not lower <= numeric <= upper:
            return False
        if quantum <= 0:
            return True
        steps = (numeric - lower) / quantum
        return math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9)


@dataclass(frozen=True)
class RawFrame:
    rgbi: np.ndarray
    bytes_per_line: int
    bytes_read: int
    frame_format: int = SANE_FRAME_RGB
    last_frame: bool = True
    depth: int = 16


class RawSaneDevice(Protocol):
    device_id: str
    identity: ScannerIdentity

    def options(self) -> dict[str, OptionInfo]: ...
    def set_option(self, name: str, value: bool | int | float | str) -> None: ...
    def get_option(self, name: str) -> bool | int | float | str: ...
    def read_rgbi(
        self,
        *,
        expected_shape: tuple[int, int],
        label: str,
        progress: Callable[[float], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> RawFrame: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...


class CaptureCancelled(RuntimeError):
    """The caller cancelled before a complete dual-source pair existed."""


class BundleVerificationError(RuntimeError):
    """A capture bundle failed its pre-publication integrity gate."""


@dataclass(frozen=True)
class DualSourceCapture:
    prepass_rgbi: np.ndarray
    main_rgbi: np.ndarray
    capture_state: ScannerCaptureState
    scanner_identity: ScannerIdentity
    same_frame_id: str
    events: tuple[dict[str, object], ...]
    assertions: dict[str, bool]

    def __post_init__(self) -> None:
        if not self.same_frame_id.strip():
            raise ValueError("same_frame_id must be non-empty")
        if not self.assertions.get("all_passed", False):
            raise ValueError("dual-source capture cannot be constructed from failed assertions")
        if not isinstance(self.scanner_identity, ScannerIdentity):
            raise TypeError("scanner_identity must be a ScannerIdentity")
        self.scanner_identity.require_supported_ls5000()
        object.__setattr__(self, "prepass_rgbi", _immutable_rgbi(self.prepass_rgbi, "prepass"))
        object.__setattr__(self, "main_rgbi", _immutable_rgbi(self.main_rgbi, "main"))


def _immutable_rgbi(array: np.ndarray, label: str) -> np.ndarray:
    if array.dtype != np.dtype(np.uint16):
        raise TypeError(f"{label} RGBI array must have dtype uint16")
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"{label} RGBI array must have shape HxWx4")
    owned = np.array(array, dtype=np.uint16, order="C", copy=True)
    owned.flags.writeable = False
    return owned


def _decode_rgbi16(payload: bytes | bytearray, *, width: int, height: int, bytes_per_line: int) -> np.ndarray:
    """Decode every raw SANE row, including rows python-sane discards."""

    expected_bpl = width * 4 * 2
    if bytes_per_line != expected_bpl:
        raise RuntimeError(f"raw RGBI bytes_per_line {bytes_per_line} != expected {expected_bpl}")
    expected_bytes = bytes_per_line * height
    if len(payload) != expected_bytes:
        raise RuntimeError(f"raw RGBI payload has {len(payload)} bytes; expected {expected_bytes}")
    return np.frombuffer(payload, dtype=np.dtype("=u2")).reshape(height, width, 4).copy()


def _validate_raw_frame(frame: RawFrame, *, expected_shape: tuple[int, int], label: str) -> None:
    expected_array_shape = (*expected_shape, 4)
    expected_bpl = expected_shape[1] * 4 * 2
    expected_bytes = expected_bpl * expected_shape[0]
    failures: list[str] = []
    if frame.rgbi.shape != expected_array_shape:
        failures.append(f"shape={frame.rgbi.shape}, expected {expected_array_shape}")
    if frame.rgbi.dtype != np.uint16:
        failures.append(f"dtype={frame.rgbi.dtype}, expected uint16")
    if frame.bytes_per_line != expected_bpl:
        failures.append(f"bytes_per_line={frame.bytes_per_line}, expected {expected_bpl}")
    if frame.bytes_read != expected_bytes:
        failures.append(f"bytes_read={frame.bytes_read}, expected {expected_bytes}")
    if frame.frame_format != SANE_FRAME_RGB:
        failures.append(f"format={frame.frame_format}, expected RGB")
    if not frame.last_frame:
        failures.append("last_frame is false")
    if frame.depth != 16:
        failures.append(f"depth={frame.depth}, expected 16")
    if failures:
        raise RuntimeError(f"{label} raw RGBI frame refused: " + "; ".join(failures))


def _read_state(device: RawSaneDevice) -> ScannerCaptureState:
    try:
        state = ScannerCaptureState(
            focus_position=int(device.get_option("focus")),
            exposure_multiplier=float(device.get_option("exposure")),
            red_exposure_us=float(device.get_option("red_exposure")),
            green_exposure_us=float(device.get_option("green_exposure")),
            blue_exposure_us=float(device.get_option("blue_exposure")),
        )
    except Exception as exc:
        raise RuntimeError(f"could not read locked focus/exposure state: {exc}") from exc
    if state.focus_position <= 0:
        raise RuntimeError(f"scanner returned uncalibrated focus position {state.focus_position}")
    return state


def _preflight(device: RawSaneDevice, plan: DiceDualSourcePlan) -> dict[str, OptionInfo]:
    device.identity.require_supported_ls5000()
    options = device.options()
    required = {
        "depth": plan.depth,
        "resolution": plan.prepass_dpi,
        "preview": False,
        "negative": False,
        "samples_per_scan": 1,
        "infrared": True,
        "autofocus": True,
        "ae": True,
        "focus": 1,
        "exposure": 1.0,
        "red_exposure": 1.0,
        "green_exposure": 1.0,
        "blue_exposure": 1.0,
        "tl_x": plan.window.tl_x,
        "tl_y": plan.window.tl_y,
        "br_x": plan.window.br_x,
        "br_y": plan.window.br_y,
    }
    if plan.frame is not None:
        required["frame"] = plan.frame
        required["frame_count"] = 1
    if plan.subframe_mm is not None:
        required["subframe"] = plan.subframe_mm
    failures: list[str] = []
    for name, value in required.items():
        info = options.get(name)
        if info is None:
            failures.append(f"option {name!r} is missing")
            continue
        if not info.active:
            failures.append(f"option {name!r} is inactive")
        if not info.settable:
            failures.append(f"option {name!r} is not settable")
        # Exposure/focus values above only prove writability; the measured
        # values are not known until after the prepass.
        if name not in {"focus", "exposure", "red_exposure", "green_exposure", "blue_exposure"} and not info.supports(value):
            failures.append(f"option {name!r} cannot accept {value!r}")
    resolution = options.get("resolution")
    if resolution is not None and not resolution.supports(plan.main_dpi):
        failures.append(f"option 'resolution' cannot accept main dpi {plan.main_dpi}")
    if failures:
        raise RuntimeError("dual RGBI preflight failed before scanner mutation: " + "; ".join(failures))
    return options


def _derive_assertions(events: list[dict[str, object]], plan: DiceDualSourcePlan) -> dict[str, bool]:
    starts = [index for index, event in enumerate(events) if event["event"] == "read_begin"]
    ends = [index for index, event in enumerate(events) if event["event"] == "read_end"]
    sets = [(index, event) for index, event in enumerate(events) if event["event"] == "set"]
    values_by_option = {
        name: [event["value"] for _, event in sets if event["option"] == name]
        for name in (
            "resolution",
            "samples_per_scan",
            "infrared",
            "depth",
            "preview",
            "negative",
            "autofocus",
            "ae",
            "frame_count",
        )
    }
    frame_sets = [index for index, event in sets if event["option"] == "frame"]
    transport_after_first = any(
        index > starts[0]
        and event["option"] in {"frame", "subframe", "tl_x", "tl_y", "br_x", "br_y"}
        for index, event in sets
    ) if starts else True
    source_kinds = [
        event.get("source") for event in events if event["event"] == "read_begin"
    ]
    checks = {
        "exactly_two_reads": len(starts) == 2 and len(ends) == 2,
        "prepass_then_main": len(starts) == 2 and len(ends) == 2 and starts[0] < ends[0] < starts[1] < ends[1],
        "resolution_prepass_then_main": values_by_option["resolution"]
        == [plan.prepass_dpi, plan.main_dpi],
        "single_sample_both_epochs": values_by_option["samples_per_scan"] == [1],
        "dedicated_rgbi_sources": source_kinds == [
            "dedicated_single_sample_rgbi",
            "dedicated_single_sample_rgbi",
        ],
        "raw_positive_rgbi_contract": (
            values_by_option["preview"] == [False]
            and values_by_option["negative"] == [False]
            and values_by_option["depth"] == [plan.depth]
            and values_by_option["infrared"] == [False, True]
        ),
        "meter_then_lock_controls": (
            values_by_option["autofocus"] == [True, False]
            and values_by_option["ae"] == [True, False]
        ),
        "every_option_write_verified": all(
            event.get("readback_verified") is True for _, event in sets
        ),
        "one_or_zero_frame_write_before_prepass": (
            (plan.frame is None and not frame_sets) or (plan.frame is not None and len(frame_sets) == 1 and frame_sets[0] < starts[0])
        ) if starts else False,
        "no_transport_write_after_prepass_started": not transport_after_first,
        "frame_counter_only_reset_to_one": all(
            value == 1 for value in values_by_option["frame_count"]
        ),
        "raw_reader_preserved_prepass_rows": any(
            event["event"] == "read_end" and event["epoch"] == "prepass" and event["shape"] == [*plan.prepass_full_shape, 4]
            for event in events
        ),
        "raw_reader_preserved_main_rows": any(
            event["event"] == "read_end" and event["epoch"] == "main" and event["shape"] == [*plan.main_full_shape, 4]
            for event in events
        ),
        "locked_state_verified_after_main": any(
            event["event"] == "capture_state_verified" for event in events
        ),
    }
    checks["all_passed"] = all(checks.values())
    return checks


def validate_dual_source_capture(
    capture: DualSourceCapture,
    plan: DiceDualSourcePlan,
) -> None:
    """Re-derive the acquisition evidence before a downstream exactness claim."""

    if not isinstance(capture, DualSourceCapture):
        raise TypeError("capture must be a DualSourceCapture")
    if not isinstance(plan, DiceDualSourcePlan):
        raise TypeError("plan must be a DiceDualSourcePlan")
    capture.scanner_identity.require_supported_ls5000()
    if capture.prepass_rgbi.shape != (*plan.prepass_full_shape, 4):
        raise BundleVerificationError("prepass RGBI shape does not match the capture plan")
    if capture.main_rgbi.shape != (*plan.main_full_shape, 4):
        raise BundleVerificationError("main RGBI shape does not match the capture plan")
    if capture.prepass_rgbi.dtype != np.uint16 or capture.main_rgbi.dtype != np.uint16:
        raise BundleVerificationError("dual-source RGBI inputs must use uint16 samples")
    if not capture.same_frame_id.strip():
        raise BundleVerificationError("dual-source capture has no same-frame identity")
    derived = _derive_assertions(list(capture.events), plan)
    if not derived["all_passed"] or capture.assertions != derived:
        raise BundleVerificationError(
            "dual-source acquisition evidence does not reproduce its assertions"
        )


def acquire_dual_sources(
    device: RawSaneDevice,
    plan: DiceDualSourcePlan,
    *,
    progress: Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
) -> DualSourceCapture:
    """Acquire the dedicated prepass then locked main on one SANE handle."""

    options = _preflight(device, plan)
    cancel = cancel or threading.Event()
    events: list[dict[str, object]] = []
    same_frame_id = f"ls5000-{uuid.uuid4().hex}"
    prepass_bytes = plan.prepass_full_shape[0] * plan.prepass_full_shape[1] * 4 * 2
    main_bytes = plan.main_full_shape[0] * plan.main_full_shape[1] * 4 * 2
    total_bytes = prepass_bytes + main_bytes
    last_progress = 0.0

    def record(event: str, **fields: object) -> None:
        events.append({"sequence": len(events) + 1, "ts": _now(), "event": event, **fields})

    def check_cancelled() -> None:
        if cancel.is_set():
            device.cancel()
            raise CaptureCancelled("Digital ICE dual-source acquisition was cancelled")

    def report(value: float) -> None:
        nonlocal last_progress
        value = min(1.0, max(last_progress, float(value)))
        last_progress = value
        if progress is not None:
            progress(value)

    def epoch_progress(completed_bytes: int, epoch_bytes: int) -> Callable[[float], None]:
        def update(fraction: float) -> None:
            check_cancelled()
            report((completed_bytes + epoch_bytes * min(1.0, max(0.0, fraction))) / total_bytes)

        return update

    def values_equal(actual: object, requested: object) -> bool:
        if isinstance(actual, bool) or isinstance(requested, bool):
            return actual is requested
        if isinstance(actual, (int, float)) and isinstance(requested, (int, float)):
            return math.isclose(float(actual), float(requested), rel_tol=0.0, abs_tol=1.0 / 65536.0)
        return actual == requested

    def set_value(name: str, value: bool | int | float | str) -> None:
        check_cancelled()
        device.set_option(name, value)
        readback = device.get_option(name)
        verified = values_equal(readback, value)
        record(
            "set",
            option=name,
            value=value,
            readback=readback,
            readback_verified=verified,
        )
        if not verified:
            raise RuntimeError(
                f"SANE option {name!r} read back as {readback!r}; requested {value!r}"
            )

    check_cancelled()
    report(0.0)
    record(
        "acquisition_begin",
        same_frame_id=same_frame_id,
        orientation="scanner_native_portrait",
    )
    try:
        # Disable infrared before lowering samples so a stale RGBI4x state
        # cannot make the first write fail or wedge the scanner.
        set_value("preview", False)
        set_value("infrared", False)
        set_value("samples_per_scan", 1)
        set_value("depth", plan.depth)
        set_value("negative", False)
        if plan.frame is not None:
            set_value("frame", plan.frame)
            set_value("frame_count", 1)
        if plan.subframe_mm is not None:
            set_value("subframe", plan.subframe_mm)
        for name, value in (
            ("tl_x", plan.window.tl_x),
            ("tl_y", plan.window.tl_y),
            ("br_x", plan.window.br_x),
            ("br_y", plan.window.br_y),
        ):
            set_value(name, value)
        set_value("infrared", True)
        set_value("resolution", plan.prepass_dpi)
        set_value("autofocus", True)
        set_value("ae", True)

        record(
            "read_begin",
            epoch="prepass",
            source="dedicated_single_sample_rgbi",
            expected_shape=[*plan.prepass_full_shape, 4],
        )
        prepass = device.read_rgbi(
            expected_shape=plan.prepass_full_shape,
            label="prepass",
            progress=epoch_progress(0, prepass_bytes),
            cancel=cancel,
        )
        _validate_raw_frame(prepass, expected_shape=plan.prepass_full_shape, label="prepass")
        record(
            "read_end",
            epoch="prepass",
            shape=list(prepass.rgbi.shape),
            dtype=np.dtype(prepass.rgbi.dtype).name,
            bytes=prepass.bytes_read,
            sha256=_sha256_bytes(memoryview(np.ascontiguousarray(prepass.rgbi)).cast("B")),
        )
        state = _read_state(device)
        record("capture_state_read", **asdict(state))

        # sane_read consumes the roll exposure counter. A mounted holder has
        # this option inactive; a roll adapter needs one counter reset without
        # changing frame, subframe, window, or physical orientation.
        frame_count = options.get("frame_count")
        if frame_count is not None and frame_count.active and frame_count.settable:
            set_value("frame_count", 1)

        set_value("autofocus", False)
        set_value("ae", False)
        for name, value in (
            ("focus", state.focus_position),
            ("exposure", state.exposure_multiplier),
            ("red_exposure", state.red_exposure_us),
            ("green_exposure", state.green_exposure_us),
            ("blue_exposure", state.blue_exposure_us),
        ):
            set_value(name, value)
        set_value("resolution", plan.main_dpi)

        record(
            "read_begin",
            epoch="main",
            source="dedicated_single_sample_rgbi",
            expected_shape=[*plan.main_full_shape, 4],
        )
        main = device.read_rgbi(
            expected_shape=plan.main_full_shape,
            label="main",
            progress=epoch_progress(prepass_bytes, main_bytes),
            cancel=cancel,
        )
        _validate_raw_frame(main, expected_shape=plan.main_full_shape, label="main")
        record(
            "read_end",
            epoch="main",
            shape=list(main.rgbi.shape),
            dtype=np.dtype(main.rgbi.dtype).name,
            bytes=main.bytes_read,
            sha256=_sha256_bytes(memoryview(np.ascontiguousarray(main.rgbi)).cast("B")),
        )
        replayed_state = _read_state(device)
        if replayed_state != state:
            raise RuntimeError(f"main acquisition did not retain locked capture state: {replayed_state!r} != {state!r}")
        record("capture_state_verified", **asdict(replayed_state))

        assertions = _derive_assertions(events, plan)
        if not assertions["all_passed"]:
            raise RuntimeError(f"dual RGBI ordering assertions failed: {assertions}")
        report(1.0)
        return DualSourceCapture(
            prepass_rgbi=prepass.rgbi,
            main_rgbi=main.rgbi,
            capture_state=state,
            scanner_identity=device.identity,
            same_frame_id=same_frame_id,
            events=tuple(events),
            assertions=assertions,
        )
    except BaseException:
        device.cancel()
        raise


class _SaneRange(ctypes.Structure):
    _fields_ = [("minimum", ctypes.c_int32), ("maximum", ctypes.c_int32), ("quant", ctypes.c_int32)]


class _SaneConstraint(ctypes.Union):
    _fields_ = [
        ("string_list", ctypes.POINTER(ctypes.c_char_p)),
        ("word_list", ctypes.POINTER(ctypes.c_int32)),
        ("range", ctypes.POINTER(_SaneRange)),
    ]


class _SaneOptionDescriptor(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("title", ctypes.c_char_p),
        ("description", ctypes.c_char_p),
        ("value_type", ctypes.c_int),
        ("unit", ctypes.c_int),
        ("size", ctypes.c_int32),
        ("cap", ctypes.c_int32),
        ("constraint_type", ctypes.c_int),
        ("constraint", _SaneConstraint),
    ]


class _SaneParameters(ctypes.Structure):
    _fields_ = [
        ("frame_format", ctypes.c_int),
        ("last_frame", ctypes.c_int32),
        ("bytes_per_line", ctypes.c_int32),
        ("pixels_per_line", ctypes.c_int32),
        ("lines", ctypes.c_int32),
        ("depth", ctypes.c_int32),
    ]


class _SaneDevice(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("vendor", ctypes.c_char_p), ("model", ctypes.c_char_p), ("kind", ctypes.c_char_p)]


def _decode_c_string(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if value is not None else ""


class Libsane:
    """Minimal raw SANE binding used only where python-sane loses RGBI rows."""

    def __init__(self, library_path: str | None = None) -> None:
        candidates = [
            library_path,
            ctypes.util.find_library("sane"),
            "/opt/homebrew/opt/sane-backends/lib/libsane.1.dylib",
            "/usr/local/lib/libsane.so.1",
            "libsane.so.1",
        ]
        last_error: OSError | None = None
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self._lib = ctypes.CDLL(candidate)
                self.library_path = str(candidate)
                break
            except OSError as error:
                last_error = error
        else:
            raise RuntimeError(f"could not load libsane: {last_error}")
        self._configure_signatures()
        version = ctypes.c_int32()
        self._check(self._lib.sane_init(ctypes.byref(version), None), "sane_init")
        self.version_code = int(version.value)
        self._closed = False

    def _configure_signatures(self) -> None:
        lib = self._lib
        lib.sane_init.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_void_p]
        lib.sane_init.restype = ctypes.c_int
        lib.sane_exit.argtypes = []
        lib.sane_exit.restype = None
        device_list_type = ctypes.POINTER(ctypes.POINTER(_SaneDevice))
        lib.sane_get_devices.argtypes = [ctypes.POINTER(device_list_type), ctypes.c_int32]
        lib.sane_get_devices.restype = ctypes.c_int
        lib.sane_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.sane_open.restype = ctypes.c_int
        lib.sane_close.argtypes = [ctypes.c_void_p]
        lib.sane_close.restype = None
        lib.sane_get_option_descriptor.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.sane_get_option_descriptor.restype = ctypes.POINTER(_SaneOptionDescriptor)
        lib.sane_control_option.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
        lib.sane_control_option.restype = ctypes.c_int
        lib.sane_start.argtypes = [ctypes.c_void_p]
        lib.sane_start.restype = ctypes.c_int
        lib.sane_get_parameters.argtypes = [ctypes.c_void_p, ctypes.POINTER(_SaneParameters)]
        lib.sane_get_parameters.restype = ctypes.c_int
        lib.sane_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        lib.sane_read.restype = ctypes.c_int
        lib.sane_cancel.argtypes = [ctypes.c_void_p]
        lib.sane_cancel.restype = None
        lib.sane_strstatus.argtypes = [ctypes.c_int]
        lib.sane_strstatus.restype = ctypes.c_char_p

    def _check(self, status: int, action: str) -> None:
        if status != SANE_STATUS_GOOD:
            message = _decode_c_string(self._lib.sane_strstatus(status))
            raise RuntimeError(f"{action} failed with SANE status {status}: {message}")

    def list_device_identities(self) -> list[ScannerIdentity]:
        device_list = ctypes.POINTER(ctypes.POINTER(_SaneDevice))()
        self._check(self._lib.sane_get_devices(ctypes.byref(device_list), 0), "sane_get_devices")
        result: list[ScannerIdentity] = []
        index = 0
        while device_list[index]:
            descriptor = device_list[index].contents
            result.append(
                ScannerIdentity(
                    device_id=_decode_c_string(descriptor.name),
                    vendor=_decode_c_string(descriptor.vendor),
                    model=_decode_c_string(descriptor.model),
                    kind=_decode_c_string(descriptor.kind),
                )
            )
            index += 1
        return result

    def list_devices(self) -> list[str]:
        return [identity.device_id for identity in self.list_device_identities()]

    def require_ls5000(self, device_id: str) -> ScannerIdentity:
        matches = [
            identity
            for identity in self.list_device_identities()
            if identity.device_id == device_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one SANE descriptor for {device_id!r}, found {len(matches)}"
            )
        identity = matches[0]
        identity.require_supported_ls5000()
        return identity

    def discover_ls5000(self) -> ScannerIdentity:
        devices = self.list_device_identities()
        matches = [identity for identity in devices if identity.is_supported_ls5000]
        if len(matches) != 1:
            descriptions = [
                f"{identity.device_id} ({identity.vendor} {identity.model})"
                for identity in devices
            ]
            raise RuntimeError(
                "expected exactly one Nikon Super Coolscan 5000 ED, found "
                f"{len(matches)} (all devices: {descriptions or 'none'})"
            )
        return matches[0]

    def open(
        self,
        device_id: str,
        *,
        identity: ScannerIdentity | None = None,
    ) -> LibsaneRawDevice:
        identity = self.require_ls5000(device_id) if identity is None else identity
        if identity.device_id != device_id:
            raise ValueError("scanner identity does not match the requested device")
        identity.require_supported_ls5000()
        handle = ctypes.c_void_p()
        self._check(self._lib.sane_open(device_id.encode(), ctypes.byref(handle)), f"sane_open({device_id!r})")
        try:
            return LibsaneRawDevice(self, identity, handle)
        except BaseException:
            self._lib.sane_close(handle)
            raise

    def close(self) -> None:
        if not self._closed:
            self._lib.sane_exit()
            self._closed = True


class LibsaneRawDevice:
    def __init__(
        self,
        owner: Libsane,
        identity: ScannerIdentity,
        handle: ctypes.c_void_p,
    ) -> None:
        self._owner = owner
        self._lib = owner._lib
        self.identity = identity
        self.device_id = identity.device_id
        self._handle = handle
        self._closed = False
        self._options: dict[str, tuple[int, _SaneOptionDescriptor]] = {}
        self._refresh_options()

    def _check(self, status: int, action: str) -> None:
        self._owner._check(status, action)

    def _refresh_options(self) -> None:
        count_value = ctypes.c_int32()
        info = ctypes.c_int32()
        self._check(
            self._lib.sane_control_option(self._handle, 0, SANE_ACTION_GET_VALUE, ctypes.byref(count_value), ctypes.byref(info)),
            "read SANE option count",
        )
        options: dict[str, tuple[int, _SaneOptionDescriptor]] = {}
        for index in range(1, int(count_value.value)):
            pointer = self._lib.sane_get_option_descriptor(self._handle, index)
            if not pointer or not pointer.contents.name:
                continue
            descriptor = pointer.contents
            options[_normalise_option_name(_decode_c_string(descriptor.name))] = (index, descriptor)
        self._options = options

    @staticmethod
    def _fixed(raw: int) -> float:
        return raw / 65536.0

    def _option_info(self, name: str, descriptor: _SaneOptionDescriptor) -> OptionInfo:
        constraint: tuple[float | int | str, ...] | None = None
        range_constraint: tuple[float, float, float] | None = None
        if descriptor.constraint_type == SANE_CONSTRAINT_WORD_LIST and descriptor.constraint.word_list:
            words = descriptor.constraint.word_list
            count = int(words[0])
            values: list[float | int] = []
            for index in range(1, count + 1):
                raw = int(words[index])
                values.append(self._fixed(raw) if descriptor.value_type == SANE_TYPE_FIXED else raw)
            constraint = tuple(values)
        elif descriptor.constraint_type == SANE_CONSTRAINT_RANGE and descriptor.constraint.range:
            raw = descriptor.constraint.range.contents
            if descriptor.value_type == SANE_TYPE_FIXED:
                range_constraint = (self._fixed(raw.minimum), self._fixed(raw.maximum), self._fixed(raw.quant))
            else:
                range_constraint = (float(raw.minimum), float(raw.maximum), float(raw.quant))
        elif descriptor.constraint_type == SANE_CONSTRAINT_STRING_LIST and descriptor.constraint.string_list:
            values_string: list[str] = []
            index = 0
            while descriptor.constraint.string_list[index]:
                values_string.append(_decode_c_string(descriptor.constraint.string_list[index]))
                index += 1
            constraint = tuple(values_string)
        return OptionInfo(
            name=name,
            value_type=descriptor.value_type,
            active=not bool(descriptor.cap & SANE_CAP_INACTIVE),
            settable=bool(descriptor.cap & SANE_CAP_SOFT_SELECT),
            constraint=constraint,
            range_constraint=range_constraint,
        )

    def options(self) -> dict[str, OptionInfo]:
        return {name: self._option_info(name, descriptor) for name, (_, descriptor) in self._options.items()}

    def _descriptor(self, name: str) -> tuple[int, _SaneOptionDescriptor]:
        normalized = _normalise_option_name(name)
        try:
            return self._options[normalized]
        except KeyError as exc:
            raise RuntimeError(f"SANE option {normalized!r} is unavailable") from exc

    def set_option(self, name: str, value: bool | int | float | str) -> None:
        index, descriptor = self._descriptor(name)
        info = ctypes.c_int32()
        if descriptor.value_type in (SANE_TYPE_BOOL, SANE_TYPE_INT, SANE_TYPE_FIXED):
            if descriptor.value_type == SANE_TYPE_FIXED:
                raw_value = round(float(value) * 65536.0)
            else:
                raw_value = int(value)
            payload: object = ctypes.c_int32(raw_value)
        elif descriptor.value_type == SANE_TYPE_STRING:
            encoded = str(value).encode()
            if len(encoded) + 1 > descriptor.size:
                raise RuntimeError(f"value for SANE option {name!r} exceeds its {descriptor.size}-byte buffer")
            payload = ctypes.create_string_buffer(encoded, descriptor.size)
        elif descriptor.value_type == SANE_TYPE_BUTTON:
            payload = ctypes.c_int32()
        else:
            raise RuntimeError(f"unsupported SANE option type {descriptor.value_type} for {name!r}")
        self._check(
            self._lib.sane_control_option(self._handle, index, SANE_ACTION_SET_VALUE, ctypes.byref(payload), ctypes.byref(info)),
            f"set SANE option {name}={value!r}",
        )
        if info.value & SANE_INFO_RELOAD_OPTIONS:
            self._refresh_options()

    def get_option(self, name: str) -> bool | int | float | str:
        index, descriptor = self._descriptor(name)
        info = ctypes.c_int32()
        if descriptor.value_type in (SANE_TYPE_BOOL, SANE_TYPE_INT, SANE_TYPE_FIXED):
            payload: object = ctypes.c_int32()
        elif descriptor.value_type == SANE_TYPE_STRING:
            payload = ctypes.create_string_buffer(descriptor.size)
        else:
            raise RuntimeError(f"SANE option {name!r} cannot be read as a scalar")
        self._check(
            self._lib.sane_control_option(self._handle, index, SANE_ACTION_GET_VALUE, ctypes.byref(payload), ctypes.byref(info)),
            f"read SANE option {name}",
        )
        if descriptor.value_type == SANE_TYPE_BOOL:
            return bool(payload.value)  # type: ignore[attr-defined]
        if descriptor.value_type == SANE_TYPE_INT:
            return int(payload.value)  # type: ignore[attr-defined]
        if descriptor.value_type == SANE_TYPE_FIXED:
            return self._fixed(int(payload.value))  # type: ignore[attr-defined]
        return bytes(payload.value).decode("utf-8", errors="replace")  # type: ignore[attr-defined]

    def read_rgbi(
        self,
        *,
        expected_shape: tuple[int, int],
        label: str,
        progress: Callable[[float], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> RawFrame:
        if cancel is not None and cancel.is_set():
            raise CaptureCancelled(f"{label} scan cancelled before start")
        self._check(self._lib.sane_start(self._handle), f"start {label} scan")
        parameters = _SaneParameters()
        self._check(self._lib.sane_get_parameters(self._handle, ctypes.byref(parameters)), f"read {label} parameters")
        height, width = expected_shape
        expected_bpl = width * 4 * 2
        failures = []
        if parameters.frame_format != SANE_FRAME_RGB:
            failures.append(f"format={parameters.frame_format}, expected RGB")
        if parameters.last_frame != 1:
            failures.append(f"last_frame={parameters.last_frame}, expected 1")
        if parameters.depth != 16:
            failures.append(f"depth={parameters.depth}, expected 16")
        if (parameters.lines, parameters.pixels_per_line) != expected_shape:
            failures.append(f"shape={(parameters.lines, parameters.pixels_per_line)}, expected {expected_shape}")
        if parameters.bytes_per_line != expected_bpl:
            failures.append(f"bytes_per_line={parameters.bytes_per_line}, expected {expected_bpl}")
        if failures:
            raise RuntimeError(f"{label} SANE metadata refused: " + "; ".join(failures))

        expected_bytes = expected_bpl * height
        payload = bytearray(expected_bytes)
        offset = 0
        chunk = (ctypes.c_ubyte * (1024 * 1024))()
        if progress is not None:
            progress(0.0)
        while True:
            if cancel is not None and cancel.is_set():
                self.cancel()
                raise CaptureCancelled(f"{label} scan cancelled")
            delivered = ctypes.c_int32()
            status = self._lib.sane_read(self._handle, chunk, len(chunk), ctypes.byref(delivered))
            count = int(delivered.value)
            if count < 0 or offset + count > expected_bytes:
                raise RuntimeError(f"{label} raw SANE read exceeded declared frame size")
            if count:
                payload[offset : offset + count] = ctypes.string_at(chunk, count)
                offset += count
                if progress is not None:
                    progress(offset / expected_bytes)
            if status == SANE_STATUS_EOF:
                break
            self._check(status, f"read {label} RGBI bytes")
            if count == 0:
                raise RuntimeError(f"{label} raw SANE read returned success with zero bytes")
        if offset != expected_bytes:
            raise RuntimeError(f"{label} raw SANE frame ended at {offset} bytes; expected {expected_bytes}")
        if progress is not None:
            progress(1.0)
        array = _decode_rgbi16(payload, width=width, height=height, bytes_per_line=parameters.bytes_per_line)
        return RawFrame(
            rgbi=array,
            bytes_per_line=parameters.bytes_per_line,
            bytes_read=offset,
            frame_format=parameters.frame_format,
            last_frame=bool(parameters.last_frame),
            depth=parameters.depth,
        )

    def cancel(self) -> None:
        if not self._closed:
            self._lib.sane_cancel(self._handle)

    def close(self) -> None:
        if not self._closed:
            self._lib.sane_close(self._handle)
            self._closed = True


def _write_npy(path: Path, array: np.ndarray) -> dict[str, object]:
    with path.open("wb") as stream:
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": np.dtype(array.dtype).name,
        "array_payload_sha256": _sha256_array_payload(array),
    }


def _sha256_array_payload(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    for start in range(0, array.shape[0], 64):
        rows = np.ascontiguousarray(array[start : start + 64])
        digest.update(memoryview(rows).cast("B"))
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_capture_bundle(bundle: str | Path) -> dict[str, object]:
    """Verify every bound artifact before a bundle is made visible."""

    root = Path(bundle)
    try:
        receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
        manifest_path = root / str(receipt["manifest"])
        if manifest_path.parent != root or manifest_path.name != "manifest.json":
            raise BundleVerificationError("receipt names an unsafe manifest path")
        if receipt.get("manifest_sha256") != _sha256_file(manifest_path):
            raise BundleVerificationError("manifest SHA-256 does not match receipt")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "negpy.dice-dual-rgbi.v1":
            raise BundleVerificationError("capture manifest schema is unsupported")
        if manifest.get("assertions", {}).get("all_passed") is not True:
            raise BundleVerificationError("capture assertions did not all pass")
        plan = manifest.get("plan", {})
        if not isinstance(plan, dict):
            raise BundleVerificationError("capture plan is not an object")
        prepass_plan = plan.get("prepass", {})
        main_plan = plan.get("main", {})
        if (
            not isinstance(prepass_plan, dict)
            or not isinstance(main_plan, dict)
            or prepass_plan.get("dpi") != PREPASS_DPI
            or main_plan.get("dpi") != NATIVE_MAIN_DPI
            or prepass_plan.get("source") != "dedicated_single_sample_rgbi"
            or main_plan.get("source") != "dedicated_single_sample_rgbi"
            or plan.get("samples_per_scan") != 1
            or plan.get("orientation") != "scanner_native_portrait"
            or plan.get("depth") != 16
            or plan.get("channels") != ["red", "green", "blue", "infrared"]
            or plan.get("transport") not in {"roll", "mounted"}
        ):
            raise BundleVerificationError("capture plan is outside the exact DICE input contract")
        window_data = plan.get("window")
        if not isinstance(window_data, dict):
            raise BundleVerificationError("capture plan has no pixel window")
        window = PixelWindow(
            tl_x=window_data.get("tl_x"),
            tl_y=window_data.get("tl_y"),
            br_x=window_data.get("br_x"),
            br_y=window_data.get("br_y"),
        )
        expected_shapes = {
            "prepass_rgbi": [*window.output_shape(PREPASS_DPI), 4],
            "main_rgbi": [*window.output_shape(NATIVE_MAIN_DPI), 4],
        }
        if prepass_plan.get("full_shape_hw") != expected_shapes["prepass_rgbi"][:2]:
            raise BundleVerificationError("prepass shape is inconsistent with the pixel window")
        if main_plan.get("full_shape_hw") != expected_shapes["main_rgbi"][:2]:
            raise BundleVerificationError("main shape is inconsistent with the pixel window")
        identity_data = manifest.get("scanner_identity")
        if not isinstance(identity_data, dict):
            raise BundleVerificationError("capture manifest has no scanner identity")
        identity = ScannerIdentity(
            device_id=identity_data.get("device_id"),
            vendor=identity_data.get("vendor"),
            model=identity_data.get("model"),
            kind=identity_data.get("kind"),
        )
        identity.require_supported_ls5000()
        if identity.device_id != manifest.get("device_id"):
            raise BundleVerificationError("scanner identity does not match the device ID")
        if not isinstance(manifest.get("same_frame_id"), str) or not manifest[
            "same_frame_id"
        ].strip():
            raise BundleVerificationError("capture manifest has no same-frame identity")
        capture_state = manifest.get("capture_state")
        if not isinstance(capture_state, dict):
            raise BundleVerificationError("capture manifest has no locked scanner state")
        ScannerCaptureState(**capture_state)
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {"prepass_rgbi", "main_rgbi"}:
            raise BundleVerificationError("capture bundle has the wrong artifact roles")
        for role, metadata in artifacts.items():
            path = root / str(metadata["path"])
            if path.parent != root or path.name != f"{role}.npy":
                raise BundleVerificationError(f"{role} has an unsafe artifact path")
            if _sha256_file(path) != metadata.get("sha256"):
                raise BundleVerificationError(f"{role} file SHA-256 mismatch")
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            if list(array.shape) != metadata.get("shape") or np.dtype(array.dtype).name != metadata.get("dtype"):
                raise BundleVerificationError(f"{role} metadata mismatch")
            if array.dtype != np.dtype(np.uint16) or array.ndim != 3 or array.shape[2] != 4:
                raise BundleVerificationError(f"{role} is not uint16 HxWx4")
            if list(array.shape) != expected_shapes[role]:
                raise BundleVerificationError(
                    f"{role} shape {list(array.shape)} does not match plan "
                    f"{expected_shapes[role]}"
                )
            if _sha256_array_payload(array) != metadata.get("array_payload_sha256"):
                raise BundleVerificationError(f"{role} payload SHA-256 mismatch")
    except BundleVerificationError:
        raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise BundleVerificationError(f"capture bundle is malformed: {error}") from error
    return manifest


def write_capture_bundle(
    output_dir: str | Path,
    *,
    device_id: str,
    plan: DiceDualSourcePlan,
    capture: DualSourceCapture,
    run_id: str | None = None,
) -> Path:
    validate_dual_source_capture(capture, plan)
    if device_id != capture.scanner_identity.device_id:
        raise BundleVerificationError("device ID does not match the captured scanner identity")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = datetime.now().astimezone().strftime("dice-dual-%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:8]}"
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one safe path component")
    final = root / run_id
    if final.exists():
        raise FileExistsError(f"capture bundle already exists: {final}")
    partial = Path(tempfile.mkdtemp(prefix=f".{run_id}.", suffix=".partial", dir=root))
    try:
        arrays = {
            "prepass_rgbi": capture.prepass_rgbi,
            "main_rgbi": capture.main_rgbi,
        }
        artifacts = {role: _write_npy(partial / f"{role}.npy", array) for role, array in arrays.items()}
        manifest = {
            "schema": "negpy.dice-dual-rgbi.v1",
            "created_at": _now(),
            "device_id": device_id,
            "scanner_identity": asdict(capture.scanner_identity),
            "reader": "direct libsane sane_read; no python-sane arr_snap and no Nikon runtime",
            "native_byteorder": sys.byteorder,
            "same_frame_id": capture.same_frame_id,
            "scanner_handle_lifecycle": "caller must release the scanner before image processing",
            "plan": plan.semantic_dict(),
            "capture_state": asdict(capture.capture_state),
            "events": list(capture.events),
            "assertions": capture.assertions,
            "artifacts": artifacts,
        }
        manifest_path = partial / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {
            "schema": "negpy.dice-dual-rgbi-receipt.v1",
            "manifest": manifest_path.name,
            "manifest_sha256": _sha256_file(manifest_path),
        }
        receipt_path = partial / "receipt.json"
        with receipt_path.open("w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(partial)
        verify_capture_bundle(partial)
        os.replace(partial, final)
        _fsync_directory(root)
    except BaseException:
        for path in partial.iterdir():
            path.unlink(missing_ok=True)
        partial.rmdir()
        raise
    return final


def load_capture_bundle(bundle: str | Path) -> tuple[DualSourceCapture, DiceDualSourcePlan]:
    """Reload a persisted bundle for processing after the scanner is released.

    A roll batch acquires every selected frame first and processes afterwards,
    so the pair crosses a bundle on disk in between.  The reload re-runs the
    full integrity gate: ``verify_capture_bundle`` re-hashes both arrays and the
    manifest, and ``validate_dual_source_capture`` re-derives the acquisition
    assertions from the round-tripped event log.  Anything that cannot prove it
    is the same verified capture fails loud instead of processing.
    """

    root = Path(bundle)
    manifest = verify_capture_bundle(root)
    try:
        plan_data = manifest["plan"]
        if not isinstance(plan_data, dict):
            raise BundleVerificationError("capture bundle plan is not an object")
        window_data = plan_data["window"]
        if not isinstance(window_data, dict):
            raise BundleVerificationError("capture bundle plan has no pixel window")
        # The recorded window is authoritative: verify_capture_bundle has
        # already proven the arrays and stated shapes consistent with it, and
        # DiceDualSourcePlan.__post_init__ re-enforces every other invariant
        # of the DICE source contract on reconstruction.
        plan = DiceDualSourcePlan(
            window=PixelWindow(
                tl_x=window_data["tl_x"],
                tl_y=window_data["tl_y"],
                br_x=window_data["br_x"],
                br_y=window_data["br_y"],
            ),
            frame=plan_data.get("frame"),
            subframe_mm=plan_data.get("subframe_mm"),
            transport=str(plan_data["transport"]),
        )
        prepass = np.load(root / "prepass_rgbi.npy", allow_pickle=False, mmap_mode="r")
        main = np.load(root / "main_rgbi.npy", allow_pickle=False, mmap_mode="r")
        capture = DualSourceCapture(
            prepass_rgbi=prepass,
            main_rgbi=main,
            capture_state=ScannerCaptureState(**manifest["capture_state"]),
            scanner_identity=ScannerIdentity(**manifest["scanner_identity"]),
            same_frame_id=str(manifest["same_frame_id"]),
            events=tuple(manifest["events"]),
            assertions=dict(manifest["assertions"]),
        )
    except BundleVerificationError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise BundleVerificationError(
            f"capture bundle cannot be reloaded: {error}"
        ) from error
    validate_dual_source_capture(capture, plan)
    return capture, plan


def exact_next_command(
    *,
    output_dir: str | Path,
    device_id: str | None = None,
    frame: int | None = None,
    subframe_mm: float | None = None,
    transport: str = "roll",
) -> str:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "negpy.infrastructure.scanners.dice_dual_source_runner",
        "--live",
        "--confirm-film-stationary",
        "--out-dir",
        str(output_dir),
    ]
    if transport != "roll":
        command.extend(("--transport", transport))
    if device_id is not None:
        command.extend(("--device", device_id))
    if frame is not None:
        command.extend(("--frame", str(frame)))
    if subframe_mm is not None:
        command.extend(("--subframe-mm", str(subframe_mm)))
    return shlex.join(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="perform both scans; default only prints the bounded plan")
    parser.add_argument(
        "--confirm-film-stationary",
        action="store_true",
        help="required with --live: confirms film is loaded, aligned, and no other scanner client is running",
    )
    parser.add_argument(
        "--transport",
        choices=("roll", "mounted"),
        default="roll",
        help="loaded holder type; mounted uses the MA-21 physical aperture",
    )
    parser.add_argument("--device", help="explicit SANE device id; default discovers exactly one coolscan3 device")
    parser.add_argument("--frame", type=int, help="optional roll-adapter frame; omit for a mounted holder")
    parser.add_argument("--subframe-mm", type=float, help="optional registered roll subframe; requires --frame")
    parser.add_argument("--out-dir", default="dice-dual-rgbi-results")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = DiceDualSourcePlan.for_transport(
            args.transport,
            frame=args.frame,
            subframe_mm=args.subframe_mm,
        )
    except ValueError as error:
        parser.error(str(error))
    next_command = exact_next_command(
        output_dir=args.out_dir,
        device_id=args.device,
        frame=args.frame,
        subframe_mm=args.subframe_mm,
        transport=args.transport,
    )
    if not args.live:
        print(json.dumps({"plan": plan.semantic_dict(), "next_command": next_command}, indent=2, sort_keys=True))
        return 0
    if not args.confirm_film_stationary:
        parser.error("--live requires --confirm-film-stationary")

    sane = Libsane()
    device: LibsaneRawDevice | None = None
    try:
        identity = (
            sane.require_ls5000(args.device)
            if args.device is not None
            else sane.discover_ls5000()
        )
        device_id = identity.device_id
        device = sane.open(device_id, identity=identity)
        capture = acquire_dual_sources(device, plan)
        bundle = write_capture_bundle(args.out_dir, device_id=device_id, plan=plan, capture=capture)
    except KeyboardInterrupt:
        if device is not None:
            device.cancel()
        raise
    finally:
        if device is not None:
            device.close()
        sane.close()
    print(f"capture: {bundle}")
    print(f"manifest: {bundle / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
