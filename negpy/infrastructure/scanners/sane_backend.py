import math
import subprocess
import sys
import threading
from typing import Callable

import numpy as np

from negpy.infrastructure.scanners.base import (
    ScanMode,
    ScannerCapabilities,
    ScannerDevice,
    ScannerUnavailable,
    TransientScanError,
)
from negpy.infrastructure.scanners.params import ScanParams, clamp_frame_offset_mm
from negpy.infrastructure.scanners.result import ScanResult
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_SOURCE_MAP: dict[str, ScanMode] = {
    "negative": ScanMode.NEGATIVE,
    "negative film": ScanMode.NEGATIVE,
    "color negative": ScanMode.NEGATIVE,
    "positive": ScanMode.POSITIVE,
    "positive film": ScanMode.POSITIVE,
    "slide": ScanMode.POSITIVE,
    "transparency": ScanMode.TRANSPARENCY,
    "transparency adapter": ScanMode.TRANSPARENCY,
    "transparency unit": ScanMode.TRANSPARENCY,
    "tpu": ScanMode.TRANSPARENCY,
    "film": ScanMode.TRANSPARENCY,
}

CANONICAL_DPI_STOPS = (75, 150, 300, 600, 1200, 2400, 3600, 4800, 6400, 7200, 9600)

# Legacy SANE option py_names that expose a dedicated infrared channel/scan.
# Their presence-only capability behavior predates Coolscan support.
_IR_OPTION_NAMES = ("ir", "preview_ir")

# coolscan3's exact boolean option carries IR inline as a fourth sample while
# still reporting SANE_FRAME_RGB. Other Nikon backends use the same spelling
# with different semantics, so it must stay backend-scoped.
_COOLSCAN3_IR_OPTION_NAME = "infrared"

# Vendor eject/unload action option (SANE_TYPE_BUTTON on the coolscan3 backend).
# Presence-only, like _find_ir_option — a device without it has no eject.
_EJECT_OPTION_NAMES = ("eject",)

# python-sane 2.9.2 cannot activate a SANE_TYPE_BUTTON — setattr raises "Buttons
# don't have values", set_option raises "...can't be set", set_auto_option raises
# "Invalid argument" (all verified on an LS-50). So eject is pressed by shelling
# out to `scanimage --eject`, which performs the C-level sane_control_option
# SET_VALUE. scanimage then runs a spurious sane_start that exits non-zero with
# "out of documents"; the eject has already fired, so that exit is expected.
_EJECT_TIMEOUT_S = 30.0
_EJECT_BENIGN_STDERR_MARKERS = ("out of documents", "no documents", "no more documents")

# Standard SANE option-unit enum values. Kept local so capability detection
# does not require importing the optional python-sane extension.
_SANE_UNIT_PIXEL = 1
_SANE_UNIT_MM = 3

_PIEUSB_PREFIX = "pieusb:"
_COOLSCAN3_PREFIX = "coolscan3:"

# Stable SANE status strings for transport glitches worth one retry (a Coolscan's
# USB link occasionally hiccups mid-strip). A real error — bad option, missing
# frame — carries a different message and must fail fast.
_TRANSIENT_IO_MARKERS = ("error during device i/o", "device busy")


def _as_scan_error(exc: Exception, message: str) -> Exception:
    """Re-type a SANE failure so the service can retry without reading messages."""
    msg = str(exc).lower()
    cls = TransientScanError if any(marker in msg for marker in _TRANSIENT_IO_MARKERS) else RuntimeError
    return cls(message)


def _strip_net_prefix(device_id: str) -> str:
    """Drop a leading `net:<host>:` so backend-prefix checks work over saned.

    Handles both `net:10.0.0.100:coolscan3:...` and bracketed IPv6 hosts
    (`net:[2001:db8::1]:coolscan3:...`).
    """
    if not device_id.startswith("net:"):
        return device_id
    rest = device_id[len("net:") :]
    if rest.startswith("["):  # bracketed IPv6 host
        close = rest.find("]:")
        return rest[close + 2 :] if close > 0 else device_id
    host, sep, backend_part = rest.partition(":")
    return backend_part if sep and host else device_id


def _mode_has_rgbi(opt) -> bool:
    """True if the device offers an RGBI scan mode (RGB + infrared, e.g. pieusb)."""
    if "mode" not in opt:
        return False
    constraint = opt["mode"].constraint
    if not isinstance(constraint, (list, tuple)):
        return False
    return any(str(v).strip().lower() == "rgbi" for v in constraint)


def _infer_film_scanner(opt, device_id: str) -> bool:
    """Heuristically decide a device is a dedicated film scanner lacking a `source` option.

    Signals: an RGBI mode, an `invert` option described as negative-film correction, or a
    known film-scanner backend prefix. Kept narrow to avoid matching plain flatbeds.
    """
    if _mode_has_rgbi(opt):
        return True
    invert = opt["invert"] if "invert" in opt else None
    if invert is not None:
        desc = str(getattr(invert, "desc", "") or "").lower()
        if "negative" in desc and "film" in desc:
            return True
    # Preserve pieusb's historical direct-ID inference. Coolscan support is
    # saned-aware because the tested scanner is remote.
    return device_id.startswith(_PIEUSB_PREFIX) or _strip_net_prefix(device_id).startswith(_COOLSCAN3_PREFIX)


def _resolve_install_hint() -> str:
    if sys.platform == "darwin":
        return "Install: brew install sane-backends && uv sync"
    if sys.platform.startswith("linux"):
        return "Install: sudo apt install libsane-dev && uv sync"
    return "Scanner support is not available on this platform."


def _preload_libsane() -> None:
    """Load libsane.so.1 globally before the _sane C extension is dlopened.

    AppImages set LD_LIBRARY_PATH to their own _internal/ dir. Without this,
    the dynamic linker may fail to find the host's libsane.so.1 when resolving
    _sane.so's DT_NEEDED entries, even though ldconfig knows where it is.
    Loading it explicitly with RTLD_GLOBAL puts it in the process symbol table
    first so _sane.so can bind to it correctly.
    """
    import ctypes
    import ctypes.util

    name = ctypes.util.find_library("sane") or "libsane.so.1"
    try:
        ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        logger.debug(f"preloaded {name}")
    except OSError as e:
        logger.warning(f"could not preload libsane ({name}): {e}")


def _detect_dpi(opt) -> tuple[int, ...]:
    if "resolution" not in opt:
        return ()
    constraint = opt["resolution"].constraint
    # python-sane: list == enumerated values, tuple == (min, max, step) range.
    if isinstance(constraint, list):
        return tuple(sorted(int(c) for c in constraint))
    if isinstance(constraint, tuple) and len(constraint) >= 2:
        lo, hi = constraint[0], constraint[1]
        dpi = tuple(s for s in CANONICAL_DPI_STOPS if lo <= s <= hi)
        return dpi or tuple(CANONICAL_DPI_STOPS)
    return CANONICAL_DPI_STOPS


def _detect_depths(opt) -> tuple[int, ...]:
    if "depth" not in opt:
        return (8, 16)
    constraint = opt["depth"].constraint
    if not isinstance(constraint, list):
        return (8, 16)
    # Drop lineart (1-bit) — useless for film and clutters the UI.
    depths = tuple(sorted(int(d) for d in constraint if int(d) >= 8))
    return depths or (8, 16)


def _detect_explicit_sources(opt) -> tuple[ScanMode, ...]:
    if "source" not in opt:
        return ()
    constraint = opt["source"].constraint
    if not isinstance(constraint, list):
        return ()
    modes: set[ScanMode] = set()
    for s in constraint:
        s_stripped = str(s).strip().lower()
        s_base = s_stripped.split("(")[0].strip() if "(" in s_stripped else s_stripped
        mode = _SOURCE_MAP.get(s_base)
        if mode is not None:
            modes.add(mode)
    return tuple(sorted(modes, key=lambda m: list(ScanMode).index(m)))


def _detect_max_area(opt) -> tuple[float, float]:
    # opt keys are py_names (hyphens → underscores). constraint is a
    # (min, max, step) range. Pixel maxima are inclusive coordinates.
    def _upper_and_unit(name: str) -> tuple[float, int | None]:
        if name not in opt:
            return (-1.0, None)
        option = opt[name]
        constraint = option.constraint
        if isinstance(constraint, (list, tuple)) and len(constraint) >= 2:
            return (float(constraint[1]), getattr(option, "unit", None))
        return (-1.0, None)

    (br_x, unit_x), (br_y, unit_y) = _upper_and_unit("br_x"), _upper_and_unit("br_y")
    if br_x <= 0 or br_y <= 0 or unit_x != unit_y:
        return (36.0, 25.0)
    if unit_x == _SANE_UNIT_MM:
        return (br_x, br_y)
    if unit_x == _SANE_UNIT_PIXEL:
        supported_dpi = _detect_dpi(opt)
        if supported_dpi:
            native_dpi = max(supported_dpi)
            return ((br_x + 1.0) * 25.4 / native_dpi, (br_y + 1.0) * 25.4 / native_dpi)
    return (36.0, 25.0)  # default 35mm frame


def _axis_extent(opt, names: tuple[str, ...]) -> tuple[float | None, bool]:
    """Max value and int-ness of the first present option (coolscan3 px vs SANE_FIXED mm)."""
    for name in names:
        if name not in opt:
            continue
        constraint = opt[name].constraint
        hi = None
        if isinstance(constraint, tuple) and len(constraint) >= 2:
            hi = constraint[1]
        elif isinstance(constraint, list) and constraint:
            hi = max(constraint)
        if hi is not None:
            return float(hi), isinstance(hi, int)
    return None, False


def _window_to_option_values(opt, window: tuple[float, float, float, float]) -> dict[str, int | float]:
    """Normalized window (0..1) → geometry option values in each option's native type
    (coolscan3 int px, SANE_FIXED mm). tl emitted before br so a full default narrows, never inverts."""
    x1, y1, x2, y2 = window
    x_hi, x_int = _axis_extent(opt, ("br_x", "tl_x"))
    y_hi, y_int = _axis_extent(opt, ("br_y", "tl_y"))
    values: dict[str, int | float] = {}
    for name, frac, hi, is_int in (
        ("tl_x", x1, x_hi, x_int),
        ("br_x", x2, x_hi, x_int),
        ("tl_y", y1, y_hi, y_int),
        ("br_y", y2, y_hi, y_int),
    ):
        if hi is None or name not in opt:
            continue
        value = frac * hi
        values[name] = int(round(value)) if is_int else float(value)
    return values


def _feed_pitch_mm(opt) -> float:
    """One frame pitch along the feed axis: the `subframe` option's range max.

    coolscan3 positions the scan at frame_pitch x (N-1) + subframe, and caps
    subframe at exactly one pitch (37.83 mm on an LS-50) — past that you would
    simply index the next frame. 0.0 when the device has no usable range.
    """
    if "subframe" not in opt:
        return 0.0
    constraint = opt["subframe"].constraint
    if isinstance(constraint, tuple) and len(constraint) >= 2 and constraint[1]:
        return float(constraint[1])
    return 0.0


def _frame_extent_cap(opt, offset_mm: float) -> float | None:
    """Feed-axis fraction scannable under a positive offset (None = no cap).

    The scanner delivers film only up to one pitch past the frame start — any
    frame, mid-strip included (measured offset + delivered ≈ 38.0 mm on an
    LS-50) — and pads the overrun with black, so the window must shrink by the
    offset. The subframe range max is the pitch, just under the measured limit.
    """
    if offset_mm <= 0:
        return None
    pitch = _feed_pitch_mm(opt)
    if pitch <= 0:
        return None
    return max(0.0, 1.0 - offset_mm / pitch)


def _apply_frame_offset(dev, offset_mm: float) -> None:
    """Set coolscan3 `subframe` (feed-axis offset, mm) if present. Absent → skip; set fails → raise.

    Always written, including 0.0 — options latch on an open handle, so a scan
    on a held session must reset a previous frame's offset, not skip it.
    """
    opt = dev.opt if hasattr(dev, "opt") else {}
    if "subframe" not in opt:
        return
    try:
        dev.subframe = float(offset_mm)
    except Exception as e:
        raise RuntimeError(f"Could not set frame offset (subframe)={offset_mm}: {e}") from e


def _find_ir_option(opt) -> str | None:
    """Return a legacy dedicated-IR option, preserving presence-only behavior."""
    for key in opt:
        if str(key).lower().replace("-", "_").strip("_") in _IR_OPTION_NAMES:
            return str(key)
    return None


def _find_eject_option(opt) -> str | None:
    """Return the device's vendor eject/unload action option, if any."""
    for key in opt:
        if str(key).lower().replace("-", "_").strip("_") in _EJECT_OPTION_NAMES:
            return str(key)
    return None


def _scanimage_eject(device_id: str) -> None:
    """Press the vendor eject button via `scanimage --eject`.

    The only working path: python-sane cannot activate a SANE_TYPE_BUTTON (see
    _EJECT_* notes above). The device must already be closed — SANE allows a
    single open handle and scanimage opens its own. scanimage runs a spurious
    scan after the press and exits non-zero with "out of documents"; the eject
    has fired by then, so that specific failure is treated as success.
    """
    try:
        proc = subprocess.run(
            ["scanimage", "-d", device_id, "--eject"],
            capture_output=True,
            text=True,
            timeout=_EJECT_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Cannot eject: `scanimage` (sane-utils) is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Eject timed out after {_EJECT_TIMEOUT_S:g}s for {device_id!r}") from exc

    if proc.returncode == 0:
        return
    stderr = (proc.stderr or "").lower()
    if any(marker in stderr for marker in _EJECT_BENIGN_STDERR_MARKERS):
        return
    detail = (proc.stderr or "").strip() or f"exit code {proc.returncode}"
    raise RuntimeError(f"scanimage --eject failed for {device_id!r}: {detail}")


def _sane_container_depth(requested_depth: int) -> int:
    """The container SANE ships `requested_depth` samples in — 8 or 16 bits.

    coolscan3 takes the scanner's native depth on the `depth` option (10, 12 or
    14 on some Coolscans; an LS-50 offers 8 and 14), then reports 16 back from
    get_parameters() and rescales the samples to fill the wider container. So a
    14-bit request yields full-range uint16, and the container — never the
    requested value — decides the dtype.
    """
    return 8 if requested_depth <= 8 else 16


def _option_is_usable(option) -> bool:
    """Fail-closed capability probe: is the option active and settable?"""
    for method_name in ("is_active", "is_settable"):
        method = getattr(option, method_name, None)
        if not callable(method):
            continue
        try:
            if not bool(method()):
                return False
        except Exception:
            return False
    return True


def _find_coolscan3_ir_option(opt, device_id: str) -> str | None:
    """Return coolscan3's exact inline-IR option for direct or saned IDs."""
    if not _strip_net_prefix(device_id).startswith(_COOLSCAN3_PREFIX):
        return None
    for key in opt:
        if str(key).lower().replace("-", "_").strip("_") == _COOLSCAN3_IR_OPTION_NAME:
            return str(key)
    return None


def _detect_ir(opt, device_id: str = "") -> bool:
    if _mode_has_rgbi(opt):
        return True
    if _find_ir_option(opt) is not None:
        return True
    coolscan_ir = _find_coolscan3_ir_option(opt, device_id)
    return coolscan_ir is not None and _option_is_usable(opt[coolscan_ir])


def _has_usable_option(opt, name: str) -> bool:
    """True if `name` is present AND the device will actually accept it.

    Presence alone is not capability: a backend advertises an option it has
    compiled in, then marks it SANE_CAP_INACTIVE for devices that lack the
    feature. coolscan3 does exactly this — an LS-50 carries `ae`/`samples_per_scan`
    but reports them inactive. Gating on presence would offer a control whose
    every non-default value SaneBackend.scan() then refuses.
    """
    return name in opt and _option_is_usable(opt[name])


def _detect_auto_exposure(opt) -> bool:
    """True if the device exposes usable hardware auto-exposure (SANE `ae`).
    UI-gating only — scan() fails loud on its own if an unavailable option is requested."""
    return _has_usable_option(opt, "ae")


def _detect_eject(opt) -> bool:
    """True when the device exposes a usable eject/unload action."""
    option_name = _find_eject_option(opt)
    return option_name is not None and _option_is_usable(opt[option_name])


def _detect_adapter_frame_capacity(opt) -> int | None:
    """Return the adapter's advertised transport bound, not an exposure count."""
    if "frame" not in opt:
        return None
    constraint = opt["frame"].constraint
    if isinstance(constraint, tuple) and len(constraint) >= 2:
        capacity = int(constraint[1])
        return capacity if capacity > 0 else None
    if isinstance(constraint, list) and constraint:
        capacity = max(int(value) for value in constraint)
        return capacity if capacity > 0 else None
    return None


def _detect_adapter_frame_control(opt) -> bool:
    """True when SANE exposes a frame-position control.

    Presence is intentional: Coolscan marks this option inactive and reports a
    1..0 constraint while a feeder is parked. Capacity detection stays
    conservative, but callers still need to distinguish that state from a device
    with no frame transport control at all.
    """
    return "frame" in opt


def _require_writable_option(opt, option_name: str, absent_message: str) -> None:
    """Fail before scanner mutation when a requested SANE option cannot be written."""
    if option_name not in opt:
        raise RuntimeError(absent_message)
    is_active = getattr(opt[option_name], "is_active", None)
    if callable(is_active):
        try:
            active = bool(is_active())
        except Exception as e:
            raise RuntimeError(f"Could not determine whether requested SANE option {option_name!r} is active: {e}") from e
        if not active:
            raise RuntimeError(f"Requested SANE option {option_name!r} is inactive")
    is_settable = getattr(opt[option_name], "is_settable", None)
    if callable(is_settable):
        try:
            settable = bool(is_settable())
        except Exception as e:
            raise RuntimeError(f"Could not determine whether requested SANE option {option_name!r} is settable: {e}") from e
        if not settable:
            raise RuntimeError(f"Requested SANE option {option_name!r} is not settable")


def _caps_from_options(opt, device_id: str = "") -> ScannerCapabilities:
    """Build ScannerCapabilities from a SANE option map. Pure — no `sane` import."""
    sources = _detect_explicit_sources(opt)
    if not sources and _infer_film_scanner(opt, device_id):
        # Dedicated film scanner with no `source` option (e.g. pieusb). `sources` is only a
        # detection/UI gate — never applied to the device — so populate it to unblock scanning.
        sources = (ScanMode.NEGATIVE, ScanMode.POSITIVE, ScanMode.TRANSPARENCY)
    return ScannerCapabilities(
        ir_channel=_detect_ir(opt, device_id),
        supported_dpi=_detect_dpi(opt),
        supported_depths=_detect_depths(opt),
        sources=sources,
        max_area_mm=_detect_max_area(opt),
        auto_exposure=_detect_auto_exposure(opt),
        adapter_frame_capacity=_detect_adapter_frame_capacity(opt),
        adapter_frame_control=_detect_adapter_frame_control(opt),
        can_eject=_detect_eject(opt),
        frame_pitch_mm=_feed_pitch_mm(opt),
    )


def _split_rgbi(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an RGBI scan `(H, W, 4)` into RGB `(H, W, 3)` and IR `(H, W)`."""
    return arr[:, :, :3], arr[:, :, 3]


def _validate_inline_rgbi_parameters(
    *,
    frame_format,
    last_frame,
    pixels_per_line: int,
    lines: int,
    returned_depth: int,
    bytes_per_line: int,
    requested_depth: int,
    context: str,
) -> None:
    """Reject SANE metadata that cannot describe an inline RGBI frame."""
    if frame_format != "color":
        raise RuntimeError(f"{context} reported SANE frame format {frame_format!r}; expected 'color'")
    if type(last_frame) not in (bool, int) or last_frame != 1:
        raise RuntimeError(f"{context} reported last_frame={last_frame!r}; expected true for one inline RGBI frame")
    if type(returned_depth) is not int or returned_depth not in (8, 16):
        raise RuntimeError(f"{context} reported unusable sample depth {returned_depth!r}; expected 8 or 16 bits")
    expected_container = _sane_container_depth(requested_depth)
    if returned_depth != expected_container:
        raise RuntimeError(
            f"{context} reported sample depth {returned_depth}, but the scan requested {requested_depth} "
            f"(expected a {expected_container}-bit container)"
        )
    if type(pixels_per_line) is not int or pixels_per_line <= 0 or type(lines) is not int or lines <= 0:
        raise RuntimeError(f"{context} reported invalid frame dimensions {pixels_per_line!r}x{lines!r}")
    expected_bytes_per_line = pixels_per_line * 4 * math.ceil(returned_depth / 8)
    if type(bytes_per_line) is not int or bytes_per_line != expected_bytes_per_line:
        raise RuntimeError(
            f"{context} reported bytes_per_line {bytes_per_line!r}; expected {expected_bytes_per_line} "
            f"for {pixels_per_line} pixels, four channels, and {returned_depth}-bit samples"
        )


def _reinterpret_channels(arr: np.ndarray, width: int, lines: int) -> np.ndarray:
    """Recover the true channel count of a frame python-sane misread.

    python-sane's C reader hardcodes 3 samples/pixel for every non-gray frame
    and reads the stream in `3 * width`-sample chunks, so a 4-sample
    inline-RGBI stream (pieusb/coolscan3 convention: SANE_FRAME_RGB with
    bytes_per_line = 4 x pixels_per_line x sample size) arrives misshaped.
    Worse, a partial final chunk is *discarded* at EOF: when `4 * lines` is
    not a multiple of 3, the stream's trailing `(4 * lines mod 3) * width`
    samples are lost. The loss is confined to the tail of the last row, so we
    drop that one edge row rather than lose the IR plane.
    """
    if width <= 0 or lines <= 0:
        return arr
    total = int(arr.size)
    expected = 4 * width * lines
    missing = expected - total
    if missing in (width, 2 * width):
        flat = arr.reshape(-1)
        # Every sample through the penultimate row is present. Reshape only that
        # complete prefix as a zero-copy view; padding a full 188 MB RGBI frame
        # merely to discard its incomplete last row can double peak RAM.
        complete = (lines - 1) * width * 4
        return flat[:complete].reshape(lines - 1, width, 4)
    if total % (width * lines):
        return arr
    nch = total // (width * lines)
    if nch not in (1, 3, 4):
        return arr
    if arr.ndim == 3 and arr.shape == (lines, width, nch):
        return arr
    return arr.reshape(lines, width, nch)


def _snap_progress_callback(progress: Callable[[float], None] | None) -> Callable[[int, int], None]:
    """Adapt python-sane's `arr_snap(progress=(current, total))` per-line callback
    to NegPy's fractional `progress(float)` callable, so the UI progress bar moves
    during the blocking read instead of only jumping 0% -> 100%.

    Forwarding only — does NOT raise to abort the read early. python-sane 2.9.2's
    C reader `Py_DECREF`s the callback's return value before checking
    `PyErr_Occurred()`; `Py_DECREF(NULL)` after a raised exception is undefined
    behaviour and segfaults against the real compiled `_sane` extension.
    Cancellation stays on the pre-start/post-read `threading.Event` checks in
    SaneBackend.scan() — a mid-read cancel is honored once arr_snap() returns.
    """

    def _cb(current: int, total: int) -> None:
        if progress is None or not total or total <= 0:
            return
        try:
            progress(min(1.0, max(0.0, current / total)))
        except Exception:
            pass

    return _cb


class SaneSession:
    """Exclusive hold on one scanner: opened once, N scans, released once.

    The handover seam for batch/roll workflows that must own the device for a
    whole strip — SANE hardware is single-open, and the Coolscan feeder
    auto-parks after any session closes mid-roll. While a session is open the
    backend refuses scan()/eject() on the device and list_devices() reuses the
    cached entry instead of probing (a probe would open the held device).
    eject() ends the session: the handle must close before scanimage's own open.
    """

    def __init__(self, backend: "SaneBackend", device_id: str, dev, opened_id: str, device: ScannerDevice | None) -> None:
        self._backend = backend
        self._dev = dev
        self.device_id = device_id
        self.opened_id = opened_id
        self.device = device
        self.closed = False

    def __enter__(self) -> "SaneSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def scan(
        self,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult:
        """Scan one frame on the held handle. Blocks until complete or cancelled."""
        if self.closed:
            raise RuntimeError(f"Scanner session for {self.device_id} is closed")
        return self._backend._scan_on_device(self._dev, self.device_id, params, progress, cancel)

    def eject(self) -> bool:
        """Press the vendor eject action, if any. Always ends the session.

        Capability-gated like SaneBackend.eject(): returns False as a clean
        no-op when the device has no usable 'eject' option. The handle is
        closed either way — scanimage needs the single open slot.
        """
        if self.closed:
            raise RuntimeError(f"Scanner session for {self.device_id} is closed")
        option_map = self._dev.opt if hasattr(self._dev, "opt") else {}
        eject_option = _find_eject_option(option_map)
        has_eject = eject_option is not None and _option_is_usable(option_map[eject_option])
        self.close()
        if not has_eject:
            return False
        _scanimage_eject(self.opened_id)
        return True

    def close(self) -> None:
        """Release the device. Idempotent."""
        if self.closed:
            return
        self.closed = True
        try:
            self._dev.close()
        finally:
            self._backend._release_session(self)


class SaneBackend:
    """python-sane implementation of ScannerBackend. Only module that imports `sane`."""

    def __init__(self) -> None:
        if sys.platform.startswith("linux"):
            _preload_libsane()
        try:
            import sane  # noqa: F811
        except ImportError:
            raise ScannerUnavailable(f"python-sane not importable. {_resolve_install_hint()}") from None
        self._sane = sane
        self._sane_initialized = False
        self._devices_cache: list[ScannerDevice] | None = None
        # Stale device id -> its post-re-enumeration id (see _open_device).
        self._id_remap: dict[str, str] = {}
        # device_id -> exclusive session holding the device (see open_session).
        self._active_sessions: dict[str, SaneSession] = {}
        self._session_lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        if self._sane_initialized:
            return
        self._sane.init()
        self._sane_initialized = True

    def list_devices(self) -> list[ScannerDevice]:
        if self._devices_cache is not None:
            return self._devices_cache

        try:
            self._ensure_initialized()
        except Exception as e:
            logger.error(f"SANE init failed: {e}")
            return []

        raw_devices = self._sane.get_devices()
        logger.info(f"SANE found {len(raw_devices)} raw device(s): {[r[0] for r in raw_devices]}")
        devices: list[ScannerDevice] = []
        for raw in raw_devices:
            held = self._session_holding(raw[0])
            if held is not None:
                # Single-open hardware — probing would open the held device.
                if held.device is not None:
                    devices.append(held.device)
                else:
                    logger.warning(f"Device {raw[0]} is held by an active session — skipping probe")
                continue
            try:
                dev = self._sane.open(raw[0])
                caps = self._detect_caps(dev, raw[0])
                dev.close()
                if caps.sources:
                    devices.append(
                        ScannerDevice(
                            id=raw[0],
                            vendor=raw[1] if len(raw) > 1 else "Unknown",
                            model=raw[2] if len(raw) > 2 else raw[0],
                            capabilities=caps,
                        )
                    )
                else:
                    logger.warning(f"Device {raw[0]} has no recognised film sources — skipping")
            except Exception as e:
                logger.warning(f"Could not probe device {raw[0]}: {e}")

        # Sort so film-capable devices come first
        devices.sort(key=lambda d: (len(d.capabilities.sources) == 0, d.model))
        self._devices_cache = devices
        return devices

    def refresh_devices(self) -> list[ScannerDevice]:
        """Clear cache and rescan.

        Enough to pick up film loaded after NegPy started, despite coolscan3(5)
        BUGS claiming the --frame option is fixed at backend init: coolscan3
        senses the adapter in cs3_full_inquiry(), which sane_open() calls, so
        the re-open below rebuilds the frame option from the live strip. Do not
        add a sane.exit()/init() cycle here — sane_exit() frees the device list
        that open handles still point into.
        """
        self._devices_cache = None
        return self.list_devices()

    def _open_device(self, device_id: str):
        """Open a device, self-healing across USB re-enumeration.

        A mid-session USB re-enumeration changes the libusb address embedded in
        the SANE id (observed on the LS-50: ...:003:006 → ...:003:007), so the
        cached id goes stale and sane.open() raises "Invalid argument". On
        failure, re-list, remap to the same physical scanner, retry once, and
        remember the remap so later opens skip straight to the fresh id. Returns
        (dev, opened_id) — callers that keep addressing the device (eject) must
        use opened_id, not the stale one they passed in.
        """
        target = self._id_remap.get(device_id, device_id)
        try:
            return self._sane.open(target), target
        except Exception:
            fresh_id = self._find_reenumerated_id(device_id)
            if fresh_id is None or fresh_id == target:
                raise
            dev = self._sane.open(fresh_id)
            self._id_remap[device_id] = fresh_id
            logger.info(f"Scanner {device_id} re-enumerated; remapped to {fresh_id}")
            return dev, fresh_id

    def open_session(self, device_id: str) -> SaneSession:
        """Open an exclusive scanning session — the batch/roll handover seam.

        The device is opened once (self-healing via _open_device) and stays
        open until SaneSession.close()/eject(). One session per device.
        """
        try:
            self._ensure_initialized()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize SANE before opening a session: {exc}") from exc

        with self._session_lock:
            if self._session_holding(device_id) is not None:
                raise RuntimeError(f"Scanner {device_id} is already held by an active session")
            dev, opened_id = self._open_device(device_id)
            device = next((d for d in (self._devices_cache or []) if d.id == device_id), None)
            session = SaneSession(self, device_id, dev, opened_id, device)
            self._active_sessions[device_id] = session
        return session

    def _session_holding(self, device_id: str) -> SaneSession | None:
        """The active session holding `device_id`, matching stale or remapped ids.

        Lock-free read: the only check-then-act race worth serializing is
        open_session's duplicate check, which holds _session_lock itself.
        """
        for session in self._active_sessions.values():
            if device_id in (session.device_id, session.opened_id):
                return session
        return None

    def _release_session(self, session: SaneSession) -> None:
        with self._session_lock:
            if self._active_sessions.get(session.device_id) is session:
                del self._active_sessions[session.device_id]

    def _find_reenumerated_id(self, device_id: str) -> str | None:
        """After an open failure, re-enumerate and return the scanner's new id.

        Matches by vendor+model when the stale device is still cached, else by
        the sole device sharing the backend/transport prefix (the single-scanner
        case). Returns None when no unambiguous match exists.
        """
        stale = {d.id: d for d in (self._devices_cache or [])}.get(device_id)
        try:
            fresh = self.refresh_devices()
        except Exception:
            return None
        if stale is not None:
            same = [d for d in fresh if d.vendor == stale.vendor and d.model == stale.model]
            if len(same) == 1:
                return same[0].id
        prefix = device_id.rsplit(":", 2)[0]
        same_prefix = [d for d in fresh if d.id.rsplit(":", 2)[0] == prefix]
        if len(same_prefix) == 1:
            return same_prefix[0].id
        return None

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult:
        """Execute a one-shot scan via SANE (open, scan, close). Blocks until complete or cancelled."""

        try:
            self._ensure_initialized()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize SANE before scanning: {exc}") from exc

        if self._session_holding(device_id) is not None:
            raise RuntimeError(f"Scanner {device_id} is held by an active session — scan through that session")

        try:
            dev, _ = self._open_device(device_id)
        except Exception as e:
            raise _as_scan_error(e, f"Failed to open scanner {device_id}: {e}") from e

        try:
            return self._scan_on_device(dev, device_id, params, progress, cancel)
        finally:
            try:
                dev.close()
            except Exception:
                pass

    def _scan_on_device(
        self,
        dev,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult:
        """Scan one frame on an already-open handle. sane_cancel()s the frame when
        done (required between frames on a held session) but never closes."""
        try:
            # IR capture strategy decides the scan mode (RGBI yields a 4th channel inline).
            ir_strategy = self._ir_strategy(dev, device_id) if params.capture_ir else None
            if params.capture_ir and ir_strategy is None:
                raise RuntimeError("IR capture requested but the device exposes no usable infrared mechanism")

            # Not every backend has a `mode` option (coolscan3 exposes none) —
            # only touch options the device has.
            if hasattr(dev, "opt") and "mode" in dev.opt:
                dev.mode = "RGBI" if (ir_strategy == "rgbi" or ir_strategy == "rgbi-hw3") else "Color"
            dev.depth = params.depth
            dev.resolution = params.dpi

            # Validate the complete requested option set before applying any of
            # it, so a missing option never leaves the feeder half-positioned.
            option_map = dev.opt if hasattr(dev, "opt") else {}
            ir_opt = _find_coolscan3_ir_option(option_map, device_id) if ir_strategy == "option" else None
            required_options: list[tuple[str, str]] = []
            if params.frame is not None:
                required_options.append(("frame", f"Frame {params.frame} requested but the device has no frame-selection option"))
            if params.auto_exposure:
                required_options.append(("ae", "Auto-exposure requested but the device has no 'ae' option"))
            if ir_strategy == "option":
                if ir_opt is None:
                    raise RuntimeError("IR option strategy selected but the device's IR option is unavailable")
                required_options.append((ir_opt, "IR option strategy selected but the device's IR option is unavailable"))
            for option_name, absent_message in required_options:
                _require_writable_option(option_map, option_name, absent_message)

            # Position the film before autofocus, auto-exposure, or scan start.
            # The scan blacks out at the frame boundary — below 0 is unreachable.
            offset_mm = clamp_frame_offset_mm(params.frame_offset_mm, _feed_pitch_mm(option_map))
            if params.frame is not None:
                try:
                    dev.frame = params.frame
                except Exception as e:
                    raise RuntimeError(f"Could not set frame={params.frame}: {e}") from e

            _apply_frame_offset(dev, offset_mm)

            # Autofocus where the device supports it (the LS-5000 powers up at
            # an uncalibrated focus position — unfocused otherwise).
            if params.autofocus and hasattr(dev, "opt") and "autofocus" in dev.opt:
                try:
                    dev.autofocus = True
                except Exception as e:
                    raise RuntimeError(f"Could not enable autofocus: {e}") from e

            # Hardware auto-exposure must meter the already-positioned frame.
            if params.auto_exposure:
                try:
                    dev.ae = True
                except Exception as e:
                    raise RuntimeError(f"Could not enable auto-exposure: {e}") from e

            # Inline IR via a boolean option (coolscan3 `infrared`): the 4th
            # sample rides in the same frame, no mode/source change. IR was
            # explicitly requested — fail loud rather than silently drop it.
            if ir_strategy == "option":
                if ir_opt is None:
                    raise RuntimeError("IR option strategy selected but the device's IR option is unavailable")
                try:
                    setattr(dev, ir_opt, True)
                except Exception as e:
                    raise RuntimeError(f"IR capture requested but enabling option {ir_opt!r} failed: {e}") from e

            if device_id.startswith(_PIEUSB_PREFIX):
                self._set_pieusb_flags(dev, params.capture_ir)

            # offset + extent must stay within one pitch — the overrun comes back black.
            window = params.window
            extent_cap = _frame_extent_cap(option_map, offset_mm)
            if extent_cap is not None:
                x1, y1, x2, y2 = window if window is not None else (0.0, 0.0, 1.0, 1.0)
                y2 = min(y2, extent_cap)
                window = (x1, min(y1, y2), x2, y2)
            if window is not None:
                for name, value in _window_to_option_values(option_map, window).items():
                    if hasattr(dev, name):
                        setattr(dev, name, value)

            if progress:
                try:
                    progress(0.0)
                except Exception:
                    pass

            if cancel.is_set():
                dev.cancel()
                raise RuntimeError("Scan cancelled before start")

            dev.start()
            rgb_array = None
            ir_array = None
            ir_valid_mask: np.ndarray | None = None

            # Frame geometry truth, for channel reinterpretation below
            # (python-sane assumes 3 samples/pixel; see _reinterpret_channels).
            try:
                frame_format, last_frame, (px_per_line, n_lines), returned_depth, bytes_per_line = dev.get_parameters()
            except Exception:
                frame_format = None
                last_frame = None
                px_per_line = n_lines = -1
                returned_depth = bytes_per_line = -1

            if ir_strategy == "option":
                _validate_inline_rgbi_parameters(
                    frame_format=frame_format,
                    last_frame=last_frame,
                    pixels_per_line=px_per_line,
                    lines=n_lines,
                    returned_depth=returned_depth,
                    bytes_per_line=bytes_per_line,
                    requested_depth=params.depth,
                    context=f"Inline infrared frame for strategy {ir_strategy!r}",
                )

            # Read RGB frame. Use arr_snap() (numpy path) — snap() goes via PIL
            # which is 8-bit only and silently truncates 16-bit RGB buffers.
            try:
                rgb_array = dev.arr_snap(progress=_snap_progress_callback(progress))
            except Exception as e:
                dev.cancel()
                raise _as_scan_error(e, f"RGB scan failed: {e}") from e

            # Inline-IR frames carry infrared as the 4th channel — recover the
            # true shape (python-sane misreads 4-sample frames) and split it off.
            if ir_strategy == "option" or ir_strategy == "rgbi-hw3":
                rgb_array = _reinterpret_channels(rgb_array, px_per_line, n_lines)
                if rgb_array.ndim == 3 and rgb_array.shape[2] == 4:
                    rgb_array, ir_array = _split_rgbi(rgb_array)
                else:
                    dev.cancel()
                    raise RuntimeError(f"IR strategy '{ir_strategy}' yielded no 4th channel (shape={rgb_array.shape})")
            elif ir_strategy == "rgbi" and rgb_array.ndim == 3 and rgb_array.shape[2] == 4:
                # Generic RGBI devices: native four-channel ndarrays split directly.
                rgb_array, ir_array = _split_rgbi(rgb_array)

            expected_dtype = np.uint16 if _sane_container_depth(params.depth) == 16 else np.uint8
            if rgb_array.dtype != expected_dtype:
                logger.warning(
                    f"Scanner returned {rgb_array.dtype} for depth={params.depth}; "
                    f"shape={rgb_array.shape}, min={rgb_array.min()}, max={rgb_array.max()}"
                )

            if cancel.is_set():
                dev.cancel()
                raise RuntimeError("Scan cancelled")

            # Legacy IR: separate scan via an IR source string (Plustek).
            if ir_strategy == "source":
                try:
                    old_source = dev.source
                    ir_source = self._get_ir_source(dev)
                    if ir_source:
                        dev.source = ir_source
                    dev.start()
                    ir_array = dev.arr_snap()
                    dev.source = old_source
                except Exception as e:
                    logger.warning(f"IR scan failed, continuing without IR: {e}")
                    ir_array = None

            if progress:
                try:
                    progress(1.0)
                except Exception:
                    pass

            if ir_array is not None and ir_valid_mask is None:
                ir_valid_mask = np.ones(ir_array.shape[:2], dtype=np.bool_)

            # Look up real vendor/model from cached device list (dev itself has no such attrs).
            sd = next((d for d in (self._devices_cache or []) if d.id == device_id), None)
            model = f"{sd.vendor} {sd.model}" if sd else device_id

            return ScanResult(
                rgb=rgb_array,
                ir=ir_array[:, :, 0] if ir_array is not None and ir_array.ndim == 3 else ir_array,
                dpi=params.dpi,
                device_model=model,
                ir_valid_mask=ir_valid_mask,
            )

        finally:
            try:
                dev.cancel()
            except Exception:
                pass

    def eject(self, device_id: str) -> bool:
        """Trigger the device's vendor eject action, if it exposes one.

        Mirrors Nikon Scan's behaviour of ejecting film at completion instead of
        leaving it parked (the LS-5000 feeder auto-parks a few minutes after any
        session closes; a parked feeder needs a power-cycle to recover mid-roll).
        Capability-gated: returns False as a clean no-op when the device has no
        active, settable 'eject' option.

        python-sane cannot press a SANE_TYPE_BUTTON, so once capability is
        confirmed the handle is closed and the button is pressed via `scanimage
        --eject` (see _scanimage_eject). Raises when the open/close or the eject fail.
        """

        try:
            self._ensure_initialized()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize SANE before eject: {exc}") from exc

        if self._session_holding(device_id) is not None:
            raise RuntimeError(f"Scanner {device_id} is held by an active session — eject through that session")

        try:
            dev, opened_id = self._open_device(device_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to open scanner {device_id} to eject: {exc}") from exc

        try:
            option_map = dev.opt if hasattr(dev, "opt") else {}
            eject_option = _find_eject_option(option_map)
            has_eject = eject_option is not None and _option_is_usable(option_map[eject_option])
        except Exception as exc:
            try:
                dev.close()
            except Exception:
                pass
            raise RuntimeError(f"Could not inspect eject capability on {device_id!r}: {exc}") from exc

        # Close before pressing: SANE allows a single open handle and scanimage opens its own.
        try:
            dev.close()
        except Exception as exc:
            raise RuntimeError(f"Could not close scanner device {device_id!r} before eject: {exc}") from exc

        if not has_eject:
            return False

        # opened_id, not device_id: a re-enumeration may have remapped the address.
        _scanimage_eject(opened_id)
        return True

    def _set_pieusb_flags(self, dev, capture_ir) -> None:
        """Apply hardware-specific optimizations for pieusb scanners."""
        opts = {
            "sharpen": True,
            "shading_analysis": True,
            "advance": True,
            "calibration": "from internal test",
            "correct_shading": True,
        }
        if capture_ir:
            opts["clean_image"] = False
            opts["correct_infrared"] = True

        for name, val in opts.items():
            try:
                setattr(dev, name, val)
            except Exception as e:
                logger.warning(f"Could not set SANE pieusb option {name}={val}: {e}")

    def _detect_caps(self, dev, device_id: str = "") -> ScannerCapabilities:
        """Read dev.opt to build ScannerCapabilities."""
        opt = dev.opt if hasattr(dev, "opt") else {}
        return _caps_from_options(opt, device_id)

    @staticmethod
    def _ir_strategy(dev, device_id) -> str | None:
        """How to capture IR for this device: 'rgbi' (RGBI scan mode), 'option'
        (boolean IR option, 4th channel inline — coolscan3), 'source' (Plustek
        second scan), 'rgbi-hw3' (just 4th channel inline) or None."""
        opt = dev.opt if hasattr(dev, "opt") else {}
        backend_id = _strip_net_prefix(device_id)
        if device_id.startswith(_PIEUSB_PREFIX):
            return "rgbi-hw3"
        if _mode_has_rgbi(opt):
            return "rgbi"
        # Inline-boolean IR is a coolscan3 contract (4th sample in the same
        # frame). coolscan2 exposes the same option name but delivers IR as a
        # separate later frame — do not claim it here.
        if backend_id.startswith(_COOLSCAN3_PREFIX) and _find_coolscan3_ir_option(opt, device_id) is not None:
            return "option"
        if SaneBackend._get_ir_source(dev):
            return "source"
        return None

    @staticmethod
    def _get_ir_source(dev) -> str | None:
        """Find an IR-specific source string if available."""
        if not hasattr(dev, "opt") or "source" not in dev.opt:
            return None
        constraint = dev.opt["source"].constraint
        if not isinstance(constraint, (list, tuple)):
            return None
        for s in constraint:
            s_lower = str(s).strip().lower()
            if "ir" in s_lower:
                return str(s)
        return None
