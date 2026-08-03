"""How PieusbBackend reports a scan's outcome.

The split that matters is ScannerService's: `TransientScanError` earns three
attempts, anything else fails fast, and a cancelled scan is neither. pieusb
reports by exception type, so these assert on type rather than on message text.
"""

import threading

import numpy as np
import pytest

from negpy.infrastructure.scanners.base import TransientScanError
from negpy.infrastructure.scanners.params import ScanParams

pytest.importorskip("pieusb")

from negpy.infrastructure.scanners import pieusb_backend  # noqa: E402
from negpy.infrastructure.scanners.pieusb_backend import PieusbBackend, _as_scan_error  # noqa: E402

_PARAMS = ScanParams(dpi=5000, depth=16, capture_ir=True)


def _pieusb_exc(name: str, *args):
    import pieusb.exceptions as exceptions

    return getattr(exceptions, name)(*args)


# ── classification ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(_pieusb_exc("TransportError", "USB status 0x08"), id="transport"),
        pytest.param(_pieusb_exc("Timeout", "command 0x08 timed out"), id="timeout"),
        pytest.param(_pieusb_exc("WarmingUp", "still warming up"), id="warming-up"),
    ],
)
def test_transport_trouble_is_typed_transient(exc: Exception) -> None:
    assert isinstance(_as_scan_error(exc, _PARAMS), TransientScanError)


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(_pieusb_exc("DeviceNotReady", "not ready"), id="not-ready"),
        pytest.param(_pieusb_exc("CheckCondition", 0x05, 0x20, 0x00), id="check-condition"),
        pytest.param(_pieusb_exc("PieusbError", "GET SHADING PARMS returned no entries"), id="protocol"),
        pytest.param(_pieusb_exc("ParamError", "sharpen and fast_infrared are exclusive"), id="bad-options"),
        pytest.param(MemoryError(), id="out-of-memory"),
        pytest.param(ValueError("Invalid value provided to option 'resolution'"), id="unexpected"),
    ],
)
def test_real_errors_fail_fast(exc: BaseException) -> None:
    """A retry would cost another full scan and end the same way."""
    reported = _as_scan_error(exc, _PARAMS)
    assert isinstance(reported, Exception)
    assert not isinstance(reported, TransientScanError)


def test_a_stalled_interface_is_transient_but_a_vanished_device_is_not() -> None:
    """Both are USBError, which pieusb's transport does not wrap — errno decides.

    A retry re-opens the device, which clears a stall; it cannot help once the
    device is gone, because _devices_map still holds the dead handle.
    """
    import errno

    import usb.core

    # pyusb's third argument is the errno; the second is libusb's own code.
    stalled = usb.core.USBError("Resource busy", -6, errno.EBUSY)
    vanished = usb.core.USBError("No such device", -4, errno.ENODEV)

    assert isinstance(_as_scan_error(stalled, _PARAMS), TransientScanError)
    vanished_report = _as_scan_error(vanished, _PARAMS)
    assert not isinstance(vanished_report, TransientScanError)
    assert "refresh" in str(vanished_report)


def test_a_usb_error_without_an_errno_makes_no_claim_about_the_cause() -> None:
    """pieusb raises some USBErrors with no errno; the message must not overreach."""
    import usb.core

    reported = _as_scan_error(usb.core.USBError("something went wrong"), _PARAMS)

    assert not isinstance(reported, TransientScanError)
    assert "refresh" not in str(reported)
    assert "something went wrong" in str(reported)


def test_the_sense_data_survives_into_the_message() -> None:
    """An unrecognised refusal is only diagnosable from its sense triple."""
    reported = _as_scan_error(_pieusb_exc("CheckCondition", 0x05, 0x20, 0x00), _PARAMS)
    assert "0x05" in str(reported) and "0x20" in str(reported)


def test_out_of_memory_names_what_to_change() -> None:
    """MemoryError alone tells the user nothing they can act on."""
    reported = _as_scan_error(MemoryError(), _PARAMS)
    assert "5000" in str(reported) and "16" in str(reported)


def test_an_empty_image_is_reported_as_missing_film() -> None:
    exc = _pieusb_exc("PieusbError", "GET PARAMETERS reported an empty image (0x0)")
    assert "film" in str(_as_scan_error(exc, _PARAMS)).lower()


# ── outcome triage ────────────────────────────────────────────────────────


class _FakeScanner:
    """Enough Scanner for PieusbBackend.scan: options are plain attributes."""

    def __init__(self, outcome) -> None:
        self._outcome = outcome

    def __enter__(self) -> "_FakeScanner":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def scan(self, on_update, on_complete) -> None:
        from pieusb.types import ScanPhase, UpdateData

        on_update(UpdateData(phase=ScanPhase.SCANNING, progress=0.5))
        if self._outcome is not None:
            on_complete(self._outcome)

    def wait(self, timeout=None) -> bool:
        return True

    def cancel(self) -> None:
        return None


class _FakeInquiry:
    max_scan_w = 10000
    max_scan_h = 10000
    model_str = "ProScan 10T"


class _FakeInfo:
    inquiry = _FakeInquiry()


def _backend_with(monkeypatch: pytest.MonkeyPatch, outcome) -> PieusbBackend:
    monkeypatch.setattr("pieusb.scanner.Scanner", lambda info: _FakeScanner(outcome))
    monkeypatch.setattr(pieusb_backend, "_require_pieusb", lambda: None)
    backend = PieusbBackend()
    backend._devices_map = {"pieusb:1:2": _FakeInfo()}  # type: ignore[dict-item]
    return backend


def _scan(backend: PieusbBackend, progress=None):
    return backend.scan("pieusb:1:2", _PARAMS, progress or (lambda *_: None), threading.Event())


def _pieusb_result(**kwargs):
    from pieusb.types import ScanResult

    return ScanResult(**kwargs)


def test_a_clean_scan_returns_the_image(monkeypatch: pytest.MonkeyPatch) -> None:
    rgb = np.ones((4, 3, 3), dtype=np.uint16)
    backend = _backend_with(monkeypatch, _pieusb_result(rgb=rgb, width=3, height=4))
    seen: list[tuple[float, str]] = []

    result = _scan(backend, progress=lambda fraction, phase: seen.append((fraction, phase)))

    assert result.rgb.shape == (4, 3, 3)
    assert result.dpi == _PARAMS.dpi
    assert result.device_model == "ProScan 10T"
    assert seen == [(0.5, "Scanning")]


def test_a_transient_failure_reaches_the_caller_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without this the service cannot retry: the type is its only signal."""
    error = _pieusb_exc("TransportError", "USB status 0x08")
    backend = _backend_with(monkeypatch, _pieusb_result(rgb=None, error=error))

    with pytest.raises(TransientScanError) as excinfo:
        _scan(backend)
    assert excinfo.value.__cause__ is error  # the traceback survives for the log


def test_a_real_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    error = _pieusb_exc("CheckCondition", 0x05, 0x20, 0x00)
    backend = _backend_with(monkeypatch, _pieusb_result(rgb=None, error=error))

    with pytest.raises(Exception) as excinfo:
        _scan(backend)
    assert not isinstance(excinfo.value, TransientScanError)
    assert excinfo.value.__cause__ is error


def test_a_scan_cancelled_by_the_worker_reports_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    """pieusb stops at a chunk boundary, so the poll sees it finish, not cancel."""
    backend = _backend_with(monkeypatch, _pieusb_result(rgb=None, cancelled=True))

    with pytest.raises(Exception, match="[Cc]ancel") as excinfo:
        _scan(backend)
    assert not isinstance(excinfo.value, TransientScanError)


def test_a_missing_outcome_is_not_disguised_as_a_failed_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """on_complete never ran, so the device state is unknown rather than failed."""
    backend = _backend_with(monkeypatch, None)

    with pytest.raises(RuntimeError, match="did not report an outcome"):
        _scan(backend)


def test_an_empty_result_without_an_error_is_still_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend_with(monkeypatch, _pieusb_result(rgb=None))

    with pytest.raises(RuntimeError, match="no image data"):
        _scan(backend)
