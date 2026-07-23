"""Optional coolscanpy-backed roll-scanning adapter for the Nikon Coolscan LS-5000.

coolscanpy (https://github.com/rohanpandula/coolscanpy) is a standalone,
SANE-free library that talks to an LS-5000 plus SA-21/SA-30 roll feeder
directly over USB: whole-roll preview, per-slot spacing correction, and
batch fine-scanning with receipts. NegPy consumes it exactly like
python-sane or gphoto2 elsewhere in this package -- an optional dependency,
imported lazily, entirely absent by default. See `available()`.

This module owns every place coolscanpy itself gets imported or driven for
the roll workflow: opening a device, opening its roll extension, and
translating coolscanpy's exceptions. `negpy.services.roll.service` builds
the file-writing workflow on top of it but never imports coolscanpy
directly, matching how `negpy.services.capture.service` never imports
`gphoto2` directly either.

--------------------------------------------------------------------------
INTEGRATION POINT for a future re-point
--------------------------------------------------------------------------
`open_roll()` below is the one place a device handle gets resolved for the
roll workflow. It currently goes straight to `coolscanpy.open()`. NegPy's
maintainer has a generic SANE-based coolscan route planned upstream, which
is expected to add a real backend-selection seam to
`negpy.services.scanning.service.ScannerService` (today `_get_backend()`
just hardcodes `SaneBackend()`). When that seam lands, re-pointing this
adapter at it should only mean changing how `open_roll()` resolves a
device -- everything built on top of the returned `RollHandle` (this
module's exception translation, and all of
`negpy/services/roll/service.py`) is unaffected, since it never talks to
coolscanpy except through this one function and the `RollHandle` it
returns.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:
    import coolscanpy
    from coolscanpy.protocol.ls5000_single_pass.capture_process import (
        ManualFrameApproval as CoolscanManualFrameApproval,
    )
    from coolscanpy.types import ProgressCallback as CoolscanProgressCallback


def available() -> bool:
    """True if the optional `coolscanpy` dependency is importable.

    A cheap, side-effect-free presence check -- mirrors the
    `ScanSidebar._sane_available()` check the plain Scan panel already uses
    for python-sane, so callers (a future roll panel, a service, a test) can
    degrade gracefully without ever importing coolscanpy just to ask whether
    it exists.
    """
    return importlib.util.find_spec("coolscanpy") is not None


class RollHandle:
    """One open coolscanpy `Device` + `Roll`, released together on `close()`.

    Construct via `open_roll()`, not directly. Every method here is a thin
    forward to the underlying `coolscanpy.Roll`, translating coolscanpy's
    typed exceptions (rooted at `PyCoolscanError`) to plain `RuntimeError`s
    -- matching how `SaneBackend.scan()` already reports failures elsewhere
    in this package (a message string the scan worker forwards verbatim to
    the GUI's status label). NegPy's scanner layer has no typed exception
    vocabulary of its own for a caller to catch selectively; a caller that
    wants coolscanpy's own richer hierarchy can still recover it from
    `__cause__`, since every translation chains `from error`.
    """

    def __init__(self, device: "coolscanpy.Device", roll: "coolscanpy.Roll") -> None:
        self._device = device
        self._roll = roll

    def preview(
        self, slots: Iterable[int] | None = None, *, on_progress: "CoolscanProgressCallback | None" = None
    ) -> "list[coolscanpy.Thumbnail]":
        """One whole-roll transport read. See `coolscanpy.Roll.preview`."""
        try:
            return self._roll.preview(slots, on_progress=on_progress)
        except Exception as error:
            raise _translate(error) from error

    def restore_preview_session(
        self,
        payload: str,
        slots: Iterable[int] | None = None,
    ) -> "list[coolscanpy.Thumbnail]":
        """Restore one content-verified preview without scanner I/O."""

        try:
            return self._roll.restore_preview_session(payload, slots)
        except Exception as error:
            raise _translate(error) from error

    def set_spacing_offset(self, slot: int, offset_rows: int) -> None:
        try:
            self._roll.set_spacing_offset(slot, offset_rows)
        except Exception as error:
            raise _translate(error) from error

    def approve(self, slot: int) -> "CoolscanManualFrameApproval":
        """Approve one reviewed slot and return Coolscanpy's bound receipt."""

        try:
            return self._roll.approve(slot)
        except Exception as error:
            raise _translate(error) from error

    def needs_approval(self, slot: int) -> bool:
        try:
            return self._roll.needs_approval(slot)
        except Exception as error:
            raise _translate(error) from error

    def scan_many(self, slots: Iterable[int], *, on_progress: "CoolscanProgressCallback | None" = None) -> Iterator["coolscanpy.Frame"]:
        """Batch fine-scan `slots` in one transport reservation.

        A `RuntimeError` raised here whose `__cause__` is a
        `coolscanpy.SafeStopRequested` means `safe_stop()` was already
        called deliberately and should be treated as a clean stop, not
        surfaced as a failure -- see `is_safe_stop()`.
        """
        try:
            yield from self._roll.scan_many(slots, on_progress=on_progress)
        except Exception as error:
            raise _translate(error) from error

    def safe_stop(self) -> None:
        """Request a graceful stop; the frame already in flight still finishes."""
        self._roll.safe_stop()

    def close(self) -> None:
        """Release the roll, then its device after ownership is confirmed."""

        self._roll.close()
        self._device.close()

    def __enter__(self) -> "RollHandle":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def list_devices() -> "list[coolscanpy.DeviceInfo]":
    """Enumerate attached LS-5000 units. Empty list, never raises, if none found."""
    import coolscanpy

    return coolscanpy.get_devices()


def eject(device_id: str) -> bool:
    """Eject one direct-USB roll after its coolscanpy reservation is closed."""

    from negpy.infrastructure.scanners.sane_backend import scanimage_eject_direct_usb

    return scanimage_eject_direct_usb(device_id)


def open_roll(
    device_id: str | None = None,
    *,
    material: "coolscanpy.Material | None" = None,
    attempts_root: str | Path | None = None,
) -> RollHandle:
    """Open one LS-5000 and its roll-feeder extension.

    `device_id` follows `coolscanpy.open()`: omitted (`None`) picks "the one
    attached unit", an explicit id (from `list_devices()`) disambiguates
    when more than one is attached. `material` defaults to
    `coolscanpy.Material.COLOR_NEGATIVE`, NegPy's live-accepted workflow.
    Coolscanpy's host-native B&W fine-scan route is implemented and covered
    without hardware, but still awaits its first live macOS acceptance with
    conventional silver B&W film.
    """
    import coolscanpy

    resolved_material = material if material is not None else coolscanpy.Material.COLOR_NEGATIVE
    try:
        device = coolscanpy.open(device_id or "ls5000")
    except Exception as error:
        raise _translate(error) from error

    try:
        roll_kwargs = {"material": resolved_material}
        if attempts_root is not None:
            roll_kwargs["attempts_root"] = Path(attempts_root)
        roll = device.roll(**roll_kwargs)
    except Exception as error:
        device.close()
        raise _translate(error) from error

    return RollHandle(device, roll)


def _translate(error: BaseException) -> RuntimeError:
    """Flatten any coolscanpy failure to a plain `RuntimeError`.

    Covers the typed `PyCoolscanError` hierarchy (`RollMismatch`,
    `FingerprintRefused`, `ManualReviewRequired`, `SafeStopRequested`, ...)
    and the handful of plain `ValueError`/`RuntimeError` coolscanpy itself
    raises (e.g. "no roll adapter is attached"), so nothing coolscanpy-shaped
    ever crosses out of this module. `raise _translate(error) from error`
    always preserves the original as `__cause__` -- see `is_safe_stop()`.
    """
    return RuntimeError(str(error))


def is_safe_stop(error: BaseException) -> bool:
    """True if `error` is this module's translation of a deliberate
    `RollHandle.safe_stop()` outcome, not a genuine failure.

    `error` is normally the `RuntimeError` a caller of `scan_many()` just
    caught; every translation in this module chains `from error`, so the
    original `coolscanpy.SafeStopRequested` (when that is what happened)
    is always sitting in `__cause__`. Imports coolscanpy lazily like the
    rest of this module -- safe here because an error this module raised
    already means coolscanpy was reachable.
    """
    import coolscanpy

    return isinstance(getattr(error, "__cause__", None), coolscanpy.SafeStopRequested)
