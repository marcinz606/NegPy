import json
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from negpy.infrastructure.scanners.base import ScanMode, ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import ScannerCaptureState, ScanParams
from negpy.infrastructure.scanners.result import ScanResult, SplitIrAlignment, SplitSourceCapture
from negpy.infrastructure.scanners.split_alignment_policy import (
    SPLIT_MAX_CHANNEL_SPREAD_PX as _SPLIT_MAX_CHANNEL_SPREAD_PX,
    SPLIT_MAX_ECC_DIVERGENCE_PX as _SPLIT_MAX_ECC_DIVERGENCE_PX,
    SPLIT_MAX_ECC_REFINEMENT_DELTA_PX as _SPLIT_MAX_ECC_REFINEMENT_DELTA_PX,
    SPLIT_MAX_HORIZONTAL_SHIFT_PX as _SPLIT_MAX_HORIZONTAL_SHIFT_PX,
    SPLIT_MAX_VERTICAL_SHIFT_PX as _SPLIT_MAX_VERTICAL_SHIFT_PX,
    SPLIT_MIN_ECC as _SPLIT_MIN_ECC,
    SPLIT_MIN_OVERLAP_FRACTION as _SPLIT_MIN_OVERLAP_FRACTION,
    SPLIT_MIN_PHASE_RESPONSE as _SPLIT_MIN_PHASE_RESPONSE,
    SPLIT_MIN_TEXTURE_STD as _SPLIT_MIN_TEXTURE_STD,
    SPLIT_MULTISCALE_HIGH_MAX_DIMENSION as _SPLIT_MULTISCALE_HIGH_MAX_DIMENSION,
    SPLIT_MULTISCALE_HIGH_MIN_RESPONSE as _SPLIT_MULTISCALE_HIGH_MIN_RESPONSE,
    SPLIT_MULTISCALE_LOW_MAX_DIMENSION as _SPLIT_MULTISCALE_LOW_MAX_DIMENSION,
    SPLIT_MULTISCALE_LOW_MIN_RESPONSE as _SPLIT_MULTISCALE_LOW_MIN_RESPONSE,
    SPLIT_MULTISCALE_MAX_CHANNEL_SHIFT_PX as _SPLIT_MULTISCALE_MAX_CHANNEL_SHIFT_PX,
    SPLIT_MULTISCALE_MAX_CHANNEL_SPREAD_PX as _SPLIT_MULTISCALE_MAX_CHANNEL_SPREAD_PX,
    SPLIT_MULTISCALE_MAX_CROSS_SCALE_DELTA_PX as _SPLIT_MULTISCALE_MAX_CROSS_SCALE_DELTA_PX,
    SPLIT_MULTISCALE_MIN_ASPECT_RATIO as _SPLIT_MULTISCALE_MIN_ASPECT_RATIO,
    SPLIT_MULTISCALE_PERIODIC_ALIAS_DOMINANCE as _SPLIT_MULTISCALE_PERIODIC_ALIAS_DOMINANCE,
    SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_CHANNELS as _SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_CHANNELS,
    SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_HORIZONTAL_PX as _SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_HORIZONTAL_PX,
    SPLIT_PHASE_ONLY_MAX_CHANNEL_SPREAD_PX as _SPLIT_PHASE_ONLY_MAX_CHANNEL_SPREAD_PX,
    SPLIT_PHASE_ONLY_MAX_SHIFT_PX as _SPLIT_PHASE_ONLY_MAX_SHIFT_PX,
    SPLIT_PHASE_ONLY_MIN_RESPONSE as _SPLIT_PHASE_ONLY_MIN_RESPONSE,
    SPLIT_TILED_PHASE_MAX_CHANNEL_SPREAD_PX as _SPLIT_TILED_PHASE_MAX_CHANNEL_SPREAD_PX,
    SPLIT_TILED_PHASE_MAX_SHIFT_PX as _SPLIT_TILED_PHASE_MAX_SHIFT_PX,
    SPLIT_TILED_PHASE_MAX_TILE_SPREAD_PX as _SPLIT_TILED_PHASE_MAX_TILE_SPREAD_PX,
    SPLIT_TILED_PHASE_MIN_MEDIAN_RESPONSE as _SPLIT_TILED_PHASE_MIN_MEDIAN_RESPONSE,
    SPLIT_TILED_PHASE_MIN_SUPPORT as _SPLIT_TILED_PHASE_MIN_SUPPORT,
    SPLIT_TILED_PHASE_MIN_TILE_RESPONSE as _SPLIT_TILED_PHASE_MIN_TILE_RESPONSE,
)
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

# Vendor eject/unload action option (SANE_TYPE_BUTTON on the coolscan3
# backend; confirmed reachable via `scanimage --eject` on the LS-5000).
# Presence-only, like _find_ir_option — a device without it simply has no
# eject capability to gate on.
_EJECT_OPTION_NAMES = ("eject",)

# Standard SANE option-unit enum values. Keep these local so capability
# detection does not require importing the optional python-sane extension.
_SANE_UNIT_PIXEL = 1
_SANE_UNIT_MM = 3

_PIEUSB_PREFIX = "pieusb:"
_COOLSCAN3_PREFIX = "coolscan3:"
_SPLIT_DIAGNOSTICS_ENV = "NEGPY_SPLIT_DIAGNOSTICS_DIR"
_CAPTURE_STATE_OPTIONS = (
    "focus",
    "exposure",
    "red_exposure",
    "green_exposure",
    "blue_exposure",
)


def _persist_split_diagnostics(
    directory: str,
    *,
    reference_rgb: np.ndarray,
    proxy_rgb: np.ndarray,
    ir: np.ndarray,
    error: RuntimeError,
) -> None:
    """Persist opt-in raw split captures after an alignment refusal."""

    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "multisample-rgb.npy": reference_rgb,
        "single-pass-rgb-proxy.npy": proxy_rgb,
        "single-pass-ir.npy": ir,
    }
    for filename, array in arrays.items():
        np.save(root / filename, array, allow_pickle=False)
    manifest = {
        "error": f"{type(error).__name__}: {error}",
        "arrays": {filename: {"shape": list(array.shape), "dtype": np.dtype(array.dtype).name} for filename, array in arrays.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


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
    # intentionally saned-aware because the tested scanner is remote.
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


def _option_is_usable(option) -> bool:
    """Fail-closed capability probe for the new coolscan3 inline option."""
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


def _detect_multi_sample(opt) -> bool:
    """True if the device exposes a samples-per-scan option (hardware
    multi-sampling, e.g. Nikon Coolscan). UI-gating only — the actual
    fail-loud enforcement lives in SaneBackend.scan()."""
    return "samples_per_scan" in opt


def _detect_auto_exposure(opt) -> bool:
    """True if the device exposes hardware auto-exposure metering (SANE `ae`).

    Presence-only, like `_detect_multi_sample` — the fail-loud enforcement of
    an unavailable option lives in SaneBackend.scan()."""
    return "ae" in opt


def _detect_registered_geometry(opt) -> bool:
    """True if the device exposes both registration-window options
    (`subframe` + `br_y`) needed to honor ScanParams.registered_geometry."""
    return "subframe" in opt and "br_y" in opt


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


def _option_is_active(opt, option_name: str) -> bool:
    """True when a SANE option exists and is currently active.

    Used for mid-scan decisions (e.g. whether coolscan3's `frame_count` needs
    a reset between split captures). Unknown/error states count as inactive:
    skipping the write is safe — a later start on a truly-consumed counter
    fails loud without moving film — while writing an inactive option raises.
    """
    if option_name not in opt:
        return False
    is_active = getattr(opt[option_name], "is_active", None)
    if not callable(is_active):
        return True
    try:
        return bool(is_active())
    except Exception:
        return False


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


def _require_supported_option_value(opt, option_name: str, value: int) -> None:
    """Reject a numeric option value that violates its SANE constraint."""
    constraint = opt[option_name].constraint
    if isinstance(constraint, list):
        supported = value in constraint
    elif isinstance(constraint, tuple) and len(constraint) >= 2:
        minimum, maximum = float(constraint[0]), float(constraint[1])
        supported = minimum <= value <= maximum
        if supported and len(constraint) >= 3:
            quantum = float(constraint[2])
            if quantum > 0:
                steps = (value - minimum) / quantum
                supported = math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9)
    else:
        return
    if not supported:
        raise RuntimeError(f"Requested {option_name}={value} is not supported by SANE constraint {constraint!r}")


def _apply_capture_state(dev, state: ScannerCaptureState) -> None:
    if state.focus_position <= 0:
        raise RuntimeError(f"Cannot replay uncalibrated focus position {state.focus_position}")
    values = {
        "focus": state.focus_position,
        "exposure": state.exposure_multiplier,
        "red_exposure": state.red_exposure_us,
        "green_exposure": state.green_exposure_us,
        "blue_exposure": state.blue_exposure_us,
        "autofocus": False,
        "ae": False,
    }
    for name, value in values.items():
        try:
            setattr(dev, name, value)
        except Exception as exc:
            raise RuntimeError(f"Could not replay locked scanner capture state {name}={value}: {exc}") from exc


def _read_capture_state(dev) -> ScannerCaptureState:
    try:
        state = ScannerCaptureState(
            focus_position=int(dev.focus),
            exposure_multiplier=float(dev.exposure),
            red_exposure_us=float(dev.red_exposure),
            green_exposure_us=float(dev.green_exposure),
            blue_exposure_us=float(dev.blue_exposure),
        )
        if state.focus_position <= 0:
            raise ValueError(f"scanner reported uncalibrated focus position {state.focus_position}")
        return state
    except Exception as exc:
        raise RuntimeError(f"Could not read the scanner focus/exposure state after capture: {exc}") from exc


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
        multi_sample=_detect_multi_sample(opt),
        adapter_frame_capacity=_detect_adapter_frame_capacity(opt),
        auto_exposure=_detect_auto_exposure(opt),
        registered_geometry=_detect_registered_geometry(opt),
        can_eject=_detect_eject(opt),
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
    if returned_depth != requested_depth:
        raise RuntimeError(f"{context} reported sample depth {returned_depth}, but the scan requested {requested_depth}")
    if type(pixels_per_line) is not int or pixels_per_line <= 0 or type(lines) is not int or lines <= 0:
        raise RuntimeError(f"{context} reported invalid frame dimensions {pixels_per_line!r}x{lines!r}")
    expected_bytes_per_line = pixels_per_line * 4 * math.ceil(returned_depth / 8)
    if type(bytes_per_line) is not int or bytes_per_line != expected_bytes_per_line:
        raise RuntimeError(
            f"{context} reported bytes_per_line {bytes_per_line!r}; expected {expected_bytes_per_line} "
            f"for {pixels_per_line} pixels, four channels, and {returned_depth}-bit samples"
        )


def _validate_registered_geometry_readback(
    dev,
    *,
    frame: int | None,
    subframe_mm: float,
    br_y_device_px: int,
) -> None:
    """Prove that SANE retained the coupled transport settings we wrote.

    A successful option assignment is not enough for archival positioning:
    backends may quantize a fixed-point value or accept a write while keeping
    an older value.  Read all three coordinates back before autofocus,
    metering, or scan start and refuse any disagreement.
    """

    expected: list[tuple[str, int | float]] = [
        ("subframe", subframe_mm),
        ("br_y", br_y_device_px),
    ]
    if frame is not None:
        expected.insert(0, ("frame", frame))

    for option_name, requested in expected:
        try:
            actual = getattr(dev, option_name)
        except Exception as error:
            raise RuntimeError(f"Could not read back registered geometry {option_name}") from error
        if option_name == "subframe":
            if (
                type(actual) not in (int, float)
                or not math.isfinite(float(actual))
                or abs(float(actual) - float(requested)) > (2.0 / 65_536.0)
            ):
                raise RuntimeError(f"Registered geometry readback disagrees for subframe: requested {requested}, got {actual!r}")
        elif type(actual) is not int or actual != requested:
            raise RuntimeError(f"Registered geometry readback disagrees for {option_name}: requested {requested}, got {actual!r}")


def _is_ls5000_silver_bw_recipe(
    *,
    is_coolscan3: bool,
    params: ScanParams,
) -> bool:
    geometry = params.registered_geometry
    return bool(
        is_coolscan3
        and geometry is not None
        and geometry.br_y_device_px == 5_958
        and params.dpi == 4_000
        and params.depth == 16
        and params.samples_per_scan == 4
        and not params.capture_ir
        and params.area is None
    )


def _validate_ls5000_silver_bw_parameters(
    *,
    frame_format,
    last_frame,
    pixels_per_line: int,
    lines: int,
    returned_depth: int,
    bytes_per_line: int,
) -> None:
    """Reject a B&W archival transfer before reading a wrong-sized stream."""

    context = "LS-5000 B&W RGB4x frame"
    if frame_format != "color":
        raise RuntimeError(f"{context} reported SANE frame format {frame_format!r}; expected 'color'")
    if type(last_frame) not in (bool, int) or last_frame != 1:
        raise RuntimeError(f"{context} reported last_frame={last_frame!r}; expected true")
    if (pixels_per_line, lines) != (3_946, 5_959):
        raise RuntimeError(f"{context} reported {pixels_per_line!r}x{lines!r}; expected 3946x5959")
    if returned_depth != 16:
        raise RuntimeError(f"{context} reported sample depth {returned_depth!r}; expected 16")
    expected_bytes_per_line = 3_946 * 3 * 2
    if bytes_per_line != expected_bytes_per_line:
        raise RuntimeError(f"{context} reported bytes_per_line {bytes_per_line!r}; expected {expected_bytes_per_line}")


def _validate_ls5000_silver_bw_option_readback(
    dev,
    *,
    frame: int | None,
    infrared_option_name: str,
    subframe_mm: float,
    br_y_device_px: int,
) -> None:
    """Prove the complete RGB4x transfer recipe immediately before start."""

    if type(frame) is not int:
        raise RuntimeError("LS-5000 B&W RGB4x scanning requires an exact frame position")

    expected_integer_options = (
        ("resolution", 4_000),
        ("depth", 16),
        ("samples_per_scan", 4),
    )
    for option_name, requested in expected_integer_options:
        try:
            actual = getattr(dev, option_name)
        except Exception as error:
            raise RuntimeError(f"Could not read back LS-5000 B&W option {option_name}") from error
        if type(actual) is not int or actual != requested:
            raise RuntimeError(
                f"LS-5000 B&W option readback disagrees for {option_name}: requested {requested}, got {actual!r}"
            )

    for option_name in ("autofocus", "ae"):
        try:
            actual = getattr(dev, option_name)
        except Exception as error:
            raise RuntimeError(f"Could not read back LS-5000 B&W option {option_name}") from error
        if type(actual) not in (bool, int) or not bool(actual):
            raise RuntimeError(
                f"LS-5000 B&W option readback disagrees for {option_name}: requested True, got {actual!r}"
            )

    try:
        infrared = getattr(dev, infrared_option_name)
    except Exception as error:
        raise RuntimeError("Could not read back LS-5000 B&W option infrared") from error
    if type(infrared) not in (bool, int) or bool(infrared):
        raise RuntimeError(
            f"LS-5000 B&W option readback disagrees for infrared: requested False, got {infrared!r}"
        )

    _validate_registered_geometry_readback(
        dev,
        frame=frame,
        subframe_mm=subframe_mm,
        br_y_device_px=br_y_device_px,
    )


# Split-capture registration acceptance gates come from the dependency-neutral
# policy module. Private aliases remain here for compatibility with existing
# backend-focused tests and downstream diagnostics.


@dataclass(frozen=True)
class _PhaseScaleEvidence:
    max_dimension: int
    width: int
    height: int
    normalized_ref: tuple[np.ndarray, ...]
    normalized_mov: tuple[np.ndarray, ...]
    channel_shifts_px: tuple[tuple[float, float], ...]
    responses: tuple[float, ...]

    @property
    def aggregate_shift_px(self) -> tuple[float, float]:
        return (
            sum(shift[0] for shift in self.channel_shifts_px) / len(self.channel_shifts_px),
            sum(shift[1] for shift in self.channel_shifts_px) / len(self.channel_shifts_px),
        )

    @property
    def channel_spread_px(self) -> float:
        return max(
            max(shift[0] for shift in self.channel_shifts_px) - min(shift[0] for shift in self.channel_shifts_px),
            max(shift[1] for shift in self.channel_shifts_px) - min(shift[1] for shift in self.channel_shifts_px),
        )


@dataclass(frozen=True)
class _TiledScaleEvidence:
    max_dimension: int
    channel_shifts_px: tuple[tuple[float, float], ...]
    responses: tuple[float, ...]
    support_counts: tuple[int, ...]
    tile_shift_spreads_px: tuple[float, ...]

    @property
    def aggregate_shift_px(self) -> tuple[float, float]:
        return (
            sum(shift[0] for shift in self.channel_shifts_px) / len(self.channel_shifts_px),
            sum(shift[1] for shift in self.channel_shifts_px) / len(self.channel_shifts_px),
        )

    @property
    def channel_spread_px(self) -> float:
        return max(
            max(shift[0] for shift in self.channel_shifts_px) - min(shift[0] for shift in self.channel_shifts_px),
            max(shift[1] for shift in self.channel_shifts_px) - min(shift[1] for shift in self.channel_shifts_px),
        )


def _short_wide_scale_dimensions(height: int, width: int) -> tuple[int, int]:
    native_max = max(height, width)
    high = min(_SPLIT_MULTISCALE_HIGH_MAX_DIMENSION, native_max)
    low = min(_SPLIT_MULTISCALE_LOW_MAX_DIMENSION, max(16, high // 2))
    return low, high


def _measure_phase_scale(
    reference_rgb: np.ndarray,
    proxy_rgb: np.ndarray,
    *,
    target_max_dimension: int,
    full_scale: float,
) -> _PhaseScaleEvidence:
    native_height, native_width = reference_rgb.shape[:2]
    resize_scale = max(native_height, native_width) / target_max_dimension
    if resize_scale > 1.0:
        estimate_width = max(8, round(native_width / resize_scale))
        estimate_height = max(8, round(native_height / resize_scale))
        estimate_size = (estimate_width, estimate_height)
        ref_small = cv2.resize(reference_rgb, estimate_size, interpolation=cv2.INTER_AREA)
        mov_small = cv2.resize(proxy_rgb, estimate_size, interpolation=cv2.INTER_AREA)
    else:
        estimate_width, estimate_height = native_width, native_height
        ref_small, mov_small = reference_rgb, proxy_rgb

    window = cv2.createHanningWindow((estimate_width, estimate_height), cv2.CV_32F)
    normalized_ref: list[np.ndarray] = []
    normalized_mov: list[np.ndarray] = []
    channel_shifts: list[tuple[float, float]] = []
    responses: list[float] = []
    for channel in range(3):
        pair: list[np.ndarray] = []
        for image in (ref_small, mov_small):
            plane = np.ascontiguousarray(image[:, :, channel], dtype=np.float32) / full_scale
            plane = cv2.GaussianBlur(plane, (0, 0), 1.0)
            deviation = float(plane.std())
            if not math.isfinite(deviation) or deviation < _SPLIT_MIN_TEXTURE_STD:
                raise RuntimeError(
                    f"Coolscan split capture has too little texture to register (channel {channel} std={deviation:.2e} of full scale)"
                )
            pair.append((plane - float(plane.mean())) / deviation)
        normalized_ref.append(pair[0])
        normalized_mov.append(pair[1])
        # phaseCorrelate applies an optimal-size Hanning window in place.
        # Every scale is reused by the cross-scale and tiled decisions.
        (channel_dx, channel_dy), response = cv2.phaseCorrelate(pair[0].copy(), pair[1].copy(), window)
        channel_shifts.append(
            (
                float(channel_dx) * native_width / estimate_width,
                float(channel_dy) * native_height / estimate_height,
            )
        )
        responses.append(float(response))

    return _PhaseScaleEvidence(
        max_dimension=max(estimate_width, estimate_height),
        width=estimate_width,
        height=estimate_height,
        normalized_ref=tuple(normalized_ref),
        normalized_mov=tuple(normalized_mov),
        channel_shifts_px=tuple(channel_shifts),
        responses=tuple(responses),
    )


def _multiscale_global_confident(evidence: tuple[_PhaseScaleEvidence, ...]) -> bool:
    if len(evidence) < 2:
        return False
    aggregates: list[tuple[float, float]] = []
    for index, scale in enumerate(evidence):
        minimum_response = _SPLIT_MULTISCALE_LOW_MIN_RESPONSE if index == 0 else _SPLIT_MULTISCALE_HIGH_MIN_RESPONSE
        if (
            len(scale.channel_shifts_px) != 3
            or len(scale.responses) != 3
            or not all(math.isfinite(value) for shift in scale.channel_shifts_px for value in shift)
            or not all(math.isfinite(response) and response >= minimum_response for response in scale.responses)
            or any(max(abs(shift[0]), abs(shift[1])) > _SPLIT_MULTISCALE_MAX_CHANNEL_SHIFT_PX for shift in scale.channel_shifts_px)
            or scale.channel_spread_px > _SPLIT_MULTISCALE_MAX_CHANNEL_SPREAD_PX
        ):
            return False
        aggregates.append(scale.aggregate_shift_px)
    return (
        max(shift[0] for shift in aggregates) - min(shift[0] for shift in aggregates) <= _SPLIT_MULTISCALE_MAX_CROSS_SCALE_DELTA_PX
        and max(shift[1] for shift in aggregates) - min(shift[1] for shift in aggregates) <= _SPLIT_MULTISCALE_MAX_CROSS_SCALE_DELTA_PX
    )


def _has_clear_periodic_global_alias(evidence: tuple[_PhaseScaleEvidence, ...]) -> bool:
    if len(evidence) < 2:
        return False
    scale_signs: list[int] = []
    for scale in evidence:
        qualifying: list[float] = []
        for (dx, dy), response in zip(scale.channel_shifts_px, scale.responses, strict=True):
            if (
                math.isfinite(dx)
                and math.isfinite(dy)
                and math.isfinite(response)
                and response >= _SPLIT_MIN_PHASE_RESPONSE
                and abs(dx) >= _SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_HORIZONTAL_PX
                and abs(dx) >= _SPLIT_MULTISCALE_PERIODIC_ALIAS_DOMINANCE * max(abs(dy), 1.0)
            ):
                qualifying.append(dx)
        if len(qualifying) < _SPLIT_MULTISCALE_PERIODIC_ALIAS_MIN_CHANNELS:
            return False
        signs = {1 if dx > 0.0 else -1 for dx in qualifying}
        if len(signs) != 1:
            return False
        scale_signs.append(signs.pop())
    return len(set(scale_signs)) == 1


def _measure_tiled_scale(
    scale: _PhaseScaleEvidence,
    *,
    native_width: int,
    native_height: int,
) -> _TiledScaleEvidence:
    channel_shifts: list[tuple[float, float]] = []
    responses: list[float] = []
    support_counts: list[int] = []
    tile_spreads: list[float] = []
    for reference, moving in zip(scale.normalized_ref, scale.normalized_mov, strict=True):
        candidates: list[tuple[float, float, float]] = []
        for tile_row in range(2):
            top = tile_row * scale.height // 2
            bottom = (tile_row + 1) * scale.height // 2
            for tile_column in range(4):
                left = tile_column * scale.width // 4
                right = (tile_column + 1) * scale.width // 4
                ref_tile = np.ascontiguousarray(reference[top:bottom, left:right]).copy()
                mov_tile = np.ascontiguousarray(moving[top:bottom, left:right]).copy()
                tile_window = cv2.createHanningWindow((ref_tile.shape[1], ref_tile.shape[0]), cv2.CV_32F)
                (tile_dx, tile_dy), tile_response = cv2.phaseCorrelate(ref_tile, mov_tile, tile_window)
                if (
                    math.isfinite(tile_dx)
                    and math.isfinite(tile_dy)
                    and math.isfinite(tile_response)
                    and tile_response >= _SPLIT_TILED_PHASE_MIN_TILE_RESPONSE
                ):
                    candidates.append((float(tile_dx), float(tile_dy), float(tile_response)))
        support_counts.append(len(candidates))
        if not candidates:
            channel_shifts.append((float("nan"), float("nan")))
            responses.append(float("nan"))
            tile_spreads.append(float("inf"))
            continue
        median_dx = float(np.median([candidate[0] for candidate in candidates]))
        median_dy = float(np.median([candidate[1] for candidate in candidates]))
        responses.append(float(np.median([candidate[2] for candidate in candidates])))
        tile_spreads.append(
            max(
                max(abs(candidate[0] - median_dx) for candidate in candidates),
                max(abs(candidate[1] - median_dy) for candidate in candidates),
            )
        )
        channel_shifts.append(
            (
                median_dx * native_width / scale.width,
                median_dy * native_height / scale.height,
            )
        )
    return _TiledScaleEvidence(
        max_dimension=scale.max_dimension,
        channel_shifts_px=tuple(channel_shifts),
        responses=tuple(responses),
        support_counts=tuple(support_counts),
        tile_shift_spreads_px=tuple(tile_spreads),
    )


def _multiscale_tiled_confident(evidence: tuple[_TiledScaleEvidence, ...]) -> bool:
    if len(evidence) < 2:
        return False
    aggregates: list[tuple[float, float]] = []
    for scale in evidence:
        if (
            len(scale.channel_shifts_px) != 3
            or len(scale.responses) != 3
            or len(scale.support_counts) != 3
            or len(scale.tile_shift_spreads_px) != 3
            or not all(math.isfinite(value) for shift in scale.channel_shifts_px for value in shift)
            or not all(math.isfinite(response) and response >= _SPLIT_TILED_PHASE_MIN_MEDIAN_RESPONSE for response in scale.responses)
            or not all(count >= _SPLIT_TILED_PHASE_MIN_SUPPORT for count in scale.support_counts)
            or not all(math.isfinite(spread) and spread <= _SPLIT_TILED_PHASE_MAX_TILE_SPREAD_PX for spread in scale.tile_shift_spreads_px)
            or any(max(abs(shift[0]), abs(shift[1])) > _SPLIT_MULTISCALE_MAX_CHANNEL_SHIFT_PX for shift in scale.channel_shifts_px)
            or scale.channel_spread_px > _SPLIT_MULTISCALE_MAX_CHANNEL_SPREAD_PX
        ):
            return False
        aggregates.append(scale.aggregate_shift_px)
    return (
        max(shift[0] for shift in aggregates) - min(shift[0] for shift in aggregates) <= _SPLIT_MULTISCALE_MAX_CROSS_SCALE_DELTA_PX
        and max(shift[1] for shift in aggregates) - min(shift[1] for shift in aggregates) <= _SPLIT_MULTISCALE_MAX_CROSS_SCALE_DELTA_PX
    )


def _estimate_split_shift(reference_rgb: np.ndarray, proxy_rgb: np.ndarray) -> tuple[float, float, dict]:
    """Estimate the (dx, dy) translation of `proxy_rgb` relative to
    `reference_rgb` at full resolution, fail-closed.

    Short-wide film tails use two-scale, per-channel phase evidence so a
    near-zero global lock wins before periodic tiles are considered; a clear
    sprocket alias may use tiles only when both resolutions agree. Other
    geometries retain phase plus translation-only ECC corroboration. Raises
    RuntimeError whenever the registration cannot be trusted and returns all
    persisted metrics needed for independent revalidation.
    """
    height, width = reference_rgb.shape[:2]
    full_scale = float(np.iinfo(reference_rgb.dtype).max) if np.issubdtype(reference_rgb.dtype, np.integer) else 1.0
    if width / height >= _SPLIT_MULTISCALE_MIN_ASPECT_RATIO:
        scale_dimensions = _short_wide_scale_dimensions(height, width)
        multiscale = tuple(
            _measure_phase_scale(
                reference_rgb,
                proxy_rgb,
                target_max_dimension=target,
                full_scale=full_scale,
            )
            for target in scale_dimensions
        )
        if _multiscale_global_confident(multiscale):
            selected_dx, selected_dy = multiscale[-1].aggregate_shift_px
            return (
                selected_dx,
                selected_dy,
                {
                    "mode": "multiscale-global",
                    "phase_responses": multiscale[-1].responses,
                    "channel_spread_px": multiscale[-1].channel_spread_px,
                    "ecc_coefficient": None,
                    "tile_support_counts": (),
                    "tile_shift_spread_px": None,
                    "estimator_version": 2,
                    "multiscale_max_dimensions": tuple(scale.max_dimension for scale in multiscale),
                    "multiscale_channel_shifts_px": tuple(scale.channel_shifts_px for scale in multiscale),
                    "multiscale_responses": tuple(scale.responses for scale in multiscale),
                    "multiscale_tile_support_counts": (),
                    "multiscale_tile_shift_spreads_px": (),
                    "multiscale_global_alias_shifts_px": (),
                },
            )
        if _has_clear_periodic_global_alias(multiscale):
            tiled_multiscale = tuple(_measure_tiled_scale(scale, native_width=width, native_height=height) for scale in multiscale)
            if _multiscale_tiled_confident(tiled_multiscale):
                selected_dx, selected_dy = tiled_multiscale[-1].aggregate_shift_px
                return (
                    selected_dx,
                    selected_dy,
                    {
                        "mode": "multiscale-tiled",
                        "phase_responses": tiled_multiscale[-1].responses,
                        "channel_spread_px": tiled_multiscale[-1].channel_spread_px,
                        "ecc_coefficient": None,
                        "tile_support_counts": tiled_multiscale[-1].support_counts,
                        "tile_shift_spread_px": max(tiled_multiscale[-1].tile_shift_spreads_px),
                        "estimator_version": 2,
                        "multiscale_max_dimensions": tuple(scale.max_dimension for scale in tiled_multiscale),
                        "multiscale_channel_shifts_px": tuple(scale.channel_shifts_px for scale in tiled_multiscale),
                        "multiscale_responses": tuple(scale.responses for scale in tiled_multiscale),
                        "multiscale_tile_support_counts": tuple(scale.support_counts for scale in tiled_multiscale),
                        "multiscale_tile_shift_spreads_px": tuple(scale.tile_shift_spreads_px for scale in tiled_multiscale),
                        "multiscale_global_alias_shifts_px": tuple(scale.channel_shifts_px for scale in multiscale),
                    },
                )
        raise RuntimeError(
            "Could not safely align the short-wide Coolscan infrared capture: "
            "cross-scale global phase was not a tight near-zero consensus and no stable periodic-alias tile consensus was established"
        )
    scale = max(1.0, max(height, width) / 1024.0)
    if scale > 1.0:
        estimate_width = max(8, round(width / scale))
        estimate_height = max(8, round(height / scale))
        estimate_size = (estimate_width, estimate_height)
        ref_small = cv2.resize(reference_rgb, estimate_size, interpolation=cv2.INTER_AREA)
        mov_small = cv2.resize(proxy_rgb, estimate_size, interpolation=cv2.INTER_AREA)
    else:
        estimate_width, estimate_height = width, height
        ref_small, mov_small = reference_rgb, proxy_rgb

    window = cv2.createHanningWindow((estimate_width, estimate_height), cv2.CV_32F)

    normalized_ref: list[np.ndarray] = []
    normalized_mov: list[np.ndarray] = []
    shifts: list[tuple[float, float]] = []
    responses: list[float] = []
    for channel in range(3):
        pair: list[np.ndarray] = []
        for image in (ref_small, mov_small):
            plane = np.ascontiguousarray(image[:, :, channel], dtype=np.float32) / full_scale
            plane = cv2.GaussianBlur(plane, (0, 0), 1.0)
            deviation = float(plane.std())
            if not math.isfinite(deviation) or deviation < _SPLIT_MIN_TEXTURE_STD:
                raise RuntimeError(
                    f"Coolscan split capture has too little texture to register (channel {channel} std={deviation:.2e} of full scale)"
                )
            pair.append((plane - float(plane.mean())) / deviation)
        normalized_ref.append(pair[0])
        normalized_mov.append(pair[1])
        # OpenCV may apply the Hanning window to both source arrays in place
        # when their dimensions are already optimal for the DFT.  These
        # normalized planes are reused by tiled correlation and ECC below, so
        # phase correlation must receive disposable copies.
        (channel_dx, channel_dy), response = cv2.phaseCorrelate(pair[0].copy(), pair[1].copy(), window)
        shifts.append((float(channel_dx), float(channel_dy)))
        responses.append(float(response))

    channel_spread = max(
        max(s[0] for s in shifts) - min(s[0] for s in shifts),
        max(s[1] for s in shifts) - min(s[1] for s in shifts),
    )
    finite = all(math.isfinite(v) for s in shifts for v in s) and all(math.isfinite(r) for r in responses)
    if not finite or min(responses) < _SPLIT_MIN_PHASE_RESPONSE:
        raise RuntimeError(
            "Could not safely align the Coolscan infrared capture: phase correlation "
            f"did not lock (responses={[f'{r:.3f}' for r in responses]}, "
            f"minimum required {_SPLIT_MIN_PHASE_RESPONSE})"
        )
    if channel_spread > _SPLIT_MAX_CHANNEL_SPREAD_PX:
        raise RuntimeError(
            "Could not safely align the Coolscan infrared capture: R/G/B channels "
            f"disagree on the shift (spread={channel_spread:.2f} px at estimation scale, "
            f"shifts={[(f'{s[0]:.2f}', f'{s[1]:.2f}') for s in shifts]})"
        )

    phase_dx = sum(s[0] for s in shifts) / 3.0
    phase_dy = sum(s[1] for s in shifts) / 3.0

    phase_full_dx = phase_dx * width / estimate_width
    phase_full_dy = phase_dy * height / estimate_height
    if max(abs(phase_full_dx), abs(phase_full_dy)) > _SPLIT_TILED_PHASE_MAX_SHIFT_PX:
        # A short, wide film tail can be dominated by periodic sprocket holes;
        # full-width phase correlation then locks one perforation period away.
        # Local 2x4 tiles break that periodicity.  Use their median only when
        # every channel has broad, tight support for a near-zero displacement.
        tile_shifts: list[tuple[float, float]] = []
        tile_responses: list[float] = []
        tile_support: list[int] = []
        maximum_tile_spread = 0.0
        tiled_confident = estimate_height >= 16 and estimate_width >= 32
        for reference, moving in zip(normalized_ref, normalized_mov, strict=True):
            candidates: list[tuple[float, float, float]] = []
            for tile_row in range(2):
                top = tile_row * estimate_height // 2
                bottom = (tile_row + 1) * estimate_height // 2
                for tile_column in range(4):
                    left = tile_column * estimate_width // 4
                    right = (tile_column + 1) * estimate_width // 4
                    # phaseCorrelate applies its Hanning window in place.  A
                    # tile view would therefore alter the normalized arrays
                    # later consumed by ECC and bias a genuine translation.
                    ref_tile = np.ascontiguousarray(reference[top:bottom, left:right]).copy()
                    mov_tile = np.ascontiguousarray(moving[top:bottom, left:right]).copy()
                    tile_window = cv2.createHanningWindow((ref_tile.shape[1], ref_tile.shape[0]), cv2.CV_32F)
                    (tile_dx, tile_dy), tile_response = cv2.phaseCorrelate(ref_tile, mov_tile, tile_window)
                    if (
                        math.isfinite(tile_dx)
                        and math.isfinite(tile_dy)
                        and math.isfinite(tile_response)
                        and tile_response >= _SPLIT_TILED_PHASE_MIN_TILE_RESPONSE
                    ):
                        candidates.append((float(tile_dx), float(tile_dy), float(tile_response)))
            tile_support.append(len(candidates))
            if len(candidates) < _SPLIT_TILED_PHASE_MIN_SUPPORT:
                tiled_confident = False
                continue
            median_dx = float(np.median([item[0] for item in candidates]))
            median_dy = float(np.median([item[1] for item in candidates]))
            median_response = float(np.median([item[2] for item in candidates]))
            spread = max(
                max(abs(item[0] - median_dx) for item in candidates),
                max(abs(item[1] - median_dy) for item in candidates),
            )
            maximum_tile_spread = max(maximum_tile_spread, spread)
            if median_response < _SPLIT_TILED_PHASE_MIN_MEDIAN_RESPONSE or spread > _SPLIT_TILED_PHASE_MAX_TILE_SPREAD_PX:
                tiled_confident = False
            tile_shifts.append((median_dx, median_dy))
            tile_responses.append(median_response)
        if len(tile_shifts) == 3:
            tiled_channel_spread = max(
                max(item[0] for item in tile_shifts) - min(item[0] for item in tile_shifts),
                max(item[1] for item in tile_shifts) - min(item[1] for item in tile_shifts),
            )
            tiled_dx = sum(item[0] for item in tile_shifts) / 3.0
            tiled_dy = sum(item[1] for item in tile_shifts) / 3.0
            tiled_full_dx = tiled_dx * width / estimate_width
            tiled_full_dy = tiled_dy * height / estimate_height
            tiled_confident = (
                tiled_confident
                and tiled_channel_spread <= _SPLIT_TILED_PHASE_MAX_CHANNEL_SPREAD_PX
                and max(abs(tiled_full_dx), abs(tiled_full_dy)) <= _SPLIT_TILED_PHASE_MAX_SHIFT_PX
            )
            if tiled_confident:
                return (
                    tiled_full_dx,
                    tiled_full_dy,
                    {
                        "mode": "tiled-phase",
                        "phase_responses": tuple(tile_responses),
                        "channel_spread_px": float(tiled_channel_spread),
                        "ecc_coefficient": None,
                        "tile_support_counts": tuple(tile_support),
                        "tile_shift_spread_px": float(maximum_tile_spread),
                    },
                )

    # Translation-only ECC on the mean-RGB proxy corroborates (and sub-pixel
    # refines) the Fourier-domain estimate from actual pixel overlap. ECC's
    # gradient descent is only trustworthy near zero displacement — seeded
    # with a large shift it can settle in a local optimum and the zero-fill
    # bands bias its correlation (observed on the corpus cross-session pair).
    # So crop both images to the phase-predicted common overlap first and let
    # ECC measure only the sub-pixel residual.
    seed_dx, seed_dy = int(round(phase_dx)), int(round(phase_dy))
    ref_top, ref_bottom = max(0, -seed_dy), estimate_height + min(0, -seed_dy)
    ref_left, ref_right = max(0, -seed_dx), estimate_width + min(0, -seed_dx)
    overlap_height, overlap_width = ref_bottom - ref_top, ref_right - ref_left
    if overlap_height < max(8, _SPLIT_MIN_OVERLAP_FRACTION * estimate_height) or overlap_width < max(
        8, _SPLIT_MIN_OVERLAP_FRACTION * estimate_width
    ):
        raise RuntimeError(
            "Could not safely align the Coolscan infrared capture: the phase shift "
            f"({phase_dx:.2f}, {phase_dy:.2f}) leaves insufficient common overlap "
            f"({overlap_width}x{overlap_height} of {estimate_width}x{estimate_height})"
        )
    ref_mean = (normalized_ref[0] + normalized_ref[1] + normalized_ref[2]) / 3.0
    mov_mean = (normalized_mov[0] + normalized_mov[1] + normalized_mov[2]) / 3.0
    ref_overlap = np.ascontiguousarray(ref_mean[ref_top:ref_bottom, ref_left:ref_right])
    mov_overlap = np.ascontiguousarray(mov_mean[ref_top + seed_dy : ref_bottom + seed_dy, ref_left + seed_dx : ref_right + seed_dx])
    warp = np.asarray([[1.0, 0.0, phase_dx - seed_dx], [0.0, 1.0, phase_dy - seed_dy]], dtype=np.float32)
    ecc_mask = np.full(ref_overlap.shape, 255, dtype=np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        ecc, warp = cv2.findTransformECC(ref_overlap, mov_overlap, warp, cv2.MOTION_TRANSLATION, criteria, ecc_mask, 5)
    except cv2.error as e:
        raise RuntimeError(
            "Could not safely align the Coolscan infrared capture: ECC refinement "
            f"did not converge from phase seed ({phase_dx:.2f}, {phase_dy:.2f}): {e}"
        ) from e
    ecc = float(ecc)
    ecc_dx, ecc_dy = seed_dx + float(warp[0, 2]), seed_dy + float(warp[1, 2])
    if not all(math.isfinite(v) for v in (ecc, ecc_dx, ecc_dy)) or ecc < _SPLIT_MIN_ECC:
        raise RuntimeError(
            f"Could not safely align the Coolscan infrared capture: ECC coefficient {ecc:.3f} is below the {_SPLIT_MIN_ECC} acceptance gate"
        )
    mode = "phase-ecc"
    ecc_phase_divergence_dx = (ecc_dx - phase_dx) * width / estimate_width
    ecc_phase_divergence_dy = (ecc_dy - phase_dy) * height / estimate_height
    ecc_phase_divergence = max(abs(ecc_phase_divergence_dx), abs(ecc_phase_divergence_dy))
    # ECC may refine the phase estimate, never relocate it.  Its gradient
    # descent measures photometric agreement, and the multisampled and
    # single-pass captures differ photometrically by design (sample
    # averaging and per-pass exposure state), so on a weakly textured axis
    # its optimum can sit pixels away from the true translation while still
    # scoring a high coefficient (observed live on 2026-07-12: phase said
    # ~0 on all three channels while ECC slid to -11 native px).  Phase
    # locks independently on all three RGB channels; accept ECC's value
    # only while it stays sub-pixel-close to that consensus.
    if ecc_phase_divergence <= _SPLIT_MAX_ECC_REFINEMENT_DELTA_PX:
        selected_dx, selected_dy = ecc_dx, ecc_dy
    else:
        selected_dx, selected_dy = phase_dx, phase_dy
    if ecc_phase_divergence > _SPLIT_MAX_ECC_DIVERGENCE_PX:
        phase_full_dx = phase_dx * width / estimate_width
        phase_full_dy = phase_dy * height / estimate_height
        phase_only_confident = (
            min(responses) >= _SPLIT_PHASE_ONLY_MIN_RESPONSE
            and channel_spread <= _SPLIT_PHASE_ONLY_MAX_CHANNEL_SPREAD_PX
            and max(abs(phase_full_dx), abs(phase_full_dy)) <= _SPLIT_PHASE_ONLY_MAX_SHIFT_PX
        )
        if not phase_only_confident:
            raise RuntimeError(
                "Could not safely align the Coolscan infrared capture: ECC refinement "
                f"({ecc_dx:.2f}, {ecc_dy:.2f}) diverges from the phase estimate "
                f"({phase_dx:.2f}, {phase_dy:.2f})"
            )
        # A damaged/clear frame can give ECC a high-scoring local optimum even
        # when Fourier phase locks independently on all three RGB channels.
        # Accept that evidence only under the much tighter near-zero gates
        # above; weak, spread, or materially displaced phase estimates still
        # fail closed.
        mode = "phase-only"
        selected_dx, selected_dy = phase_dx, phase_dy

    dx = selected_dx * width / estimate_width
    dy = selected_dy * height / estimate_height
    metrics = {
        "mode": mode,
        "phase_responses": tuple(responses),
        "channel_spread_px": float(channel_spread),
        "ecc_coefficient": ecc,
        "tile_support_counts": (),
        "tile_shift_spread_px": None,
    }
    return dx, dy, metrics


def _align_split_ir_full_canvas(
    reference_rgb: np.ndarray,
    proxy_rgb: np.ndarray,
    ir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SplitIrAlignment]:
    """Register a single-pass Coolscan IR plane to a multisampled RGB frame.

    The RGB channels captured alongside IR are a registration proxy.  Estimate
    their translation relative to the multisampled RGB and apply that exact
    transform to IR.  Keep the complete multisampled RGB canvas and return a
    boolean validity mask for IR; downstream repair must never interpret a
    synthetic border or python-sane's missing RGBI tail row as a defect.
    """
    if reference_rgb.ndim != 3 or reference_rgb.shape[2] != 3:
        raise RuntimeError(f"Multisampled RGB has an unexpected shape {reference_rgb.shape}")
    if proxy_rgb.ndim != 3 or proxy_rgb.shape[2] != 3 or ir.ndim != 2:
        raise RuntimeError(f"Single-pass RGB/IR shapes are invalid: rgb={proxy_rgb.shape}, ir={ir.shape}")
    if proxy_rgb.shape[:2] != ir.shape:
        raise RuntimeError(f"Single-pass RGB and IR dimensions differ: {proxy_rgb.shape[:2]} vs {ir.shape}")

    # Both captures use one unchanged scan window, so the only legitimate
    # geometry difference is python-sane's known dropped RGBI tail row.
    # Anything else means the two captures did not see the same window —
    # cropping that away would silently hide a positioning fault.
    if proxy_rgb.shape[1] != reference_rgb.shape[1]:
        raise RuntimeError(
            f"Coolscan split captures have different widths ({reference_rgb.shape[1]} vs {proxy_rgb.shape[1]}) — refusing to align"
        )
    height_loss = reference_rgb.shape[0] - proxy_rgb.shape[0]
    if height_loss not in (0, 1):
        raise RuntimeError(
            "Coolscan split captures have irreconcilable heights "
            f"({reference_rgb.shape[0]} vs {proxy_rgb.shape[0]}); only the known "
            "one-row RGBI tail loss is permitted"
        )

    height, width = proxy_rgb.shape[:2]
    reference_height = reference_rgb.shape[0]
    registration_reference = reference_rgb[:height]

    # The common case is byte-identical geometry; avoid introducing a
    # needless resample when the two RGB captures already coincide.
    if np.array_equal(registration_reference, proxy_rgb):
        full_scale = float(np.iinfo(registration_reference.dtype).max) if np.issubdtype(registration_reference.dtype, np.integer) else 1.0
        for channel in range(3):
            deviation = float(np.asarray(registration_reference[:, :, channel], dtype=np.float32).std()) / full_scale
            if not math.isfinite(deviation) or deviation < _SPLIT_MIN_TEXTURE_STD:
                raise RuntimeError(
                    f"Coolscan split capture has too little texture to register (channel {channel} std={deviation:.2e} of full scale)"
                )
        aligned_ir = np.zeros((reference_height, width), dtype=ir.dtype)
        valid_ir = np.zeros((reference_height, width), dtype=np.bool_)
        aligned_ir[:height] = ir
        valid_ir[:height] = True
        return (
            reference_rgb,
            aligned_ir,
            valid_ir,
            SplitIrAlignment(mode="identity", dx_px=0.0, dy_px=0.0),
        )
    if height < 8 or width < 8:
        raise RuntimeError(f"Coolscan split capture is too small to register ({width}x{height})")

    dx, dy, metrics = _estimate_split_shift(registration_reference, proxy_rgb)
    # Phase correlation on a truly integer translation can land a few
    # hundredths away from the exact pixel because of edge/window effects.
    # Snapping only that tiny numerical residue avoids needlessly blurring IR.
    if abs(dx - round(dx)) < 0.05:
        dx = float(round(dx))
    if abs(dy - round(dy)) < 0.05:
        dy = float(round(dy))
    if not all(math.isfinite(value) for value in (dx, dy)):
        raise RuntimeError(f"Could not safely align the Coolscan infrared capture (non-finite shift=({dx}, {dy}))")
    if abs(dx) > _SPLIT_MAX_HORIZONTAL_SHIFT_PX:
        raise RuntimeError(
            "Could not safely align the Coolscan infrared capture "
            f"(horizontal shift {dx:.2f} px exceeds {_SPLIT_MAX_HORIZONTAL_SHIFT_PX:.1f} px)"
        )
    if abs(dy) > _SPLIT_MAX_VERTICAL_SHIFT_PX:
        raise RuntimeError(
            "Could not safely align the Coolscan infrared capture "
            f"(vertical shift {dy:.2f} px exceeds {_SPLIT_MAX_VERTICAL_SHIFT_PX:.1f} px)"
        )

    transform = np.asarray([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    aligned_ir = cv2.warpAffine(
        ir,
        transform,
        (width, reference_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Warp a continuous coverage plane with the same bilinear kernel as IR.
    # A nearest-neighbour mask can label a border pixel valid even when IR
    # blended one real scanner sample with constant padding.  Only a coverage
    # value of one proves every non-zero interpolation weight came from the
    # scanner canvas.
    ir_coverage = cv2.warpAffine(
        np.ones((height, width), dtype=np.float32),
        transform,
        (width, reference_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_ir = ir_coverage >= np.float32(1.0 - 1e-6)
    if not valid_ir.any():
        raise RuntimeError(f"Coolscan infrared alignment has no common overlap (shift=({dx:.2f}, {dy:.2f}))")

    alignment = SplitIrAlignment(
        mode=metrics["mode"],
        dx_px=dx,
        dy_px=dy,
        phase_responses=metrics["phase_responses"],
        channel_spread_px=metrics["channel_spread_px"],
        ecc_coefficient=metrics["ecc_coefficient"],
        tile_support_counts=metrics["tile_support_counts"],
        tile_shift_spread_px=metrics["tile_shift_spread_px"],
        estimator_version=metrics.get("estimator_version"),
        multiscale_max_dimensions=metrics.get("multiscale_max_dimensions", ()),
        multiscale_channel_shifts_px=metrics.get("multiscale_channel_shifts_px", ()),
        multiscale_responses=metrics.get("multiscale_responses", ()),
        multiscale_tile_support_counts=metrics.get("multiscale_tile_support_counts", ()),
        multiscale_tile_shift_spreads_px=metrics.get("multiscale_tile_shift_spreads_px", ()),
        multiscale_global_alias_shifts_px=metrics.get("multiscale_global_alias_shifts_px", ()),
    )
    logger.info(
        "aligned single-pass Coolscan IR to multisampled RGB: "
        f"dx={dx:.2f}px dy={dy:.2f}px phase_responses="
        f"{[f'{r:.3f}' for r in alignment.phase_responses]} "
        f"channel_spread={alignment.channel_spread_px:.2f}px "
        f"ecc={alignment.ecc_coefficient if alignment.ecc_coefficient is not None else 'n/a'}"
    )
    return reference_rgb, aligned_ir, valid_ir, alignment


def _align_split_ir(
    reference_rgb: np.ndarray,
    proxy_rgb: np.ndarray,
    ir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, SplitIrAlignment]:
    """Legacy common-overlap split alignment.

    Ordinary scans retain their established cropped contract.  Full-negative
    capture explicitly opts into `_align_split_ir_full_canvas` so acquisition
    never discards photograph and can persist the validity mask.
    """

    full_rgb, full_ir, valid, alignment = _align_split_ir_full_canvas(reference_rgb, proxy_rgb, ir)
    rows, columns = np.nonzero(valid)
    if rows.size == 0 or columns.size == 0:
        raise RuntimeError("Coolscan infrared alignment has no common overlap")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    return full_rgb[top:bottom, left:right], full_ir[top:bottom, left:right], alignment


def _reinterpret_channels(arr: np.ndarray, width: int, lines: int) -> np.ndarray:
    """Recover the true channel count of a frame python-sane misread.

    python-sane's C reader hardcodes 3 samples/pixel for every non-gray frame
    and reads the stream in `3 * width`-sample chunks, so a 4-sample
    inline-RGBI stream (pieusb/coolscan3 convention: SANE_FRAME_RGB with
    bytes_per_line = 4 x pixels_per_line x sample size) arrives misshaped.
    Worse, a partial final chunk is *discarded* at EOF: when `4 * lines` is
    not a multiple of 3, the stream's trailing `(4 * lines mod 3) * width`
    samples are lost (both 1489- and 5959-line LS-5000 frames hit this).
    The loss is confined to the tail of the last row, so we pad the missing
    samples and drop that one edge row rather than lose the IR plane.
    """
    if width <= 0 or lines <= 0:
        return arr
    total = int(arr.size)
    expected = 4 * width * lines
    missing = expected - total
    if missing in (width, 2 * width):
        flat = arr.reshape(-1)
        # Every sample through the penultimate row is present.  Reshape only
        # that complete prefix as a zero-copy view; padding a full 188 MB RGBI
        # frame merely to discard its incomplete last row can double peak RAM.
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

    Forwarding only — does NOT raise to abort the read early, even though
    NegPy's cancel Event is available at the call site. python-sane 2.9.2's C
    reader (`_sane.c` `SaneDev_snap`) invokes this callback once per output
    line and checks `PyErr_Occurred()` afterward to support early abort, but
    it unconditionally `Py_DECREF`s the callback's return value *before* that
    check:

        PyObject *result = PyObject_Call(progress, progArgs, NULL);
        Py_DECREF(result);              // result is NULL if the call raised
        Py_DECREF(progArgs);
        if (PyErr_Occurred()) { ... }

    `Py_DECREF` requires a non-NULL argument (`Py_XDECREF` is the NULL-safe
    variant), so `Py_DECREF(NULL)` here is undefined behaviour and segfaults
    in practice once this callback is driven by the real compiled `_sane`
    extension rather than a pure-Python fake. Raising from this callback
    against real hardware would therefore very likely crash the desktop app
    on cancel instead of cleanly aborting the scan, so we deliberately don't.
    Cancellation stays on the pre-start/post-read `threading.Event` checks
    already in `SaneBackend.scan()` — a mid-read cancel click is still
    honored, just only once the current blocking `arr_snap()` call returns.
    """

    def _cb(current: int, total: int) -> None:
        if progress is None or not total or total <= 0:
            return
        try:
            progress(min(1.0, max(0.0, current / total)))
        except Exception:
            pass

    return _cb


def _progress_segment(
    progress: Callable[[float], None] | None,
    start: float,
    end: float,
) -> Callable[[float], None] | None:
    """Map one capture's 0..1 progress into a larger multi-capture span."""
    if progress is None:
        return None

    def _mapped(fraction: float) -> None:
        progress(start + (end - start) * min(1.0, max(0.0, fraction)))

    return _mapped


class SaneBackend:
    """python-sane implementation of ScannerBackend. Only module that imports `sane`."""

    def __init__(self) -> None:
        if sys.platform.startswith("linux"):
            _preload_libsane()
        try:
            import sane  # noqa: F811
        except ImportError:
            raise ImportError(f"python-sane not importable. {_resolve_install_hint()}") from None
        self._sane = sane
        self._sane_initialized = False
        self._devices_cache: list[ScannerDevice] | None = None

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
        """Clear cache and rescan."""
        self._devices_cache = None
        return self.list_devices()

    def probe_device(self, device_id: str) -> ScannerDevice | None:
        """Strictly probe one freshly enumerated device.

        Unlike best-effort device listing, transport and option-inspection
        failures are propagated so callers can distinguish an unavailable
        scanner stack from a device ID that is genuinely absent.
        """

        try:
            self._ensure_initialized()
        except Exception as exc:
            raise RuntimeError(f"SANE initialization failed: {exc}") from exc

        try:
            raw_devices = self._sane.get_devices()
        except Exception as exc:
            raise RuntimeError(f"SANE device enumeration failed: {exc}") from exc

        raw = next((candidate for candidate in raw_devices if candidate[0] == device_id), None)
        if raw is None:
            return None

        try:
            dev = self._sane.open(device_id)
        except Exception as exc:
            raise RuntimeError(f"Could not open scanner device {device_id!r} during fresh probe: {exc}") from exc

        try:
            caps = self._detect_caps(dev, device_id)
        except Exception as exc:
            try:
                dev.close()
            except Exception:
                pass
            raise RuntimeError(f"Could not inspect scanner device {device_id!r} during fresh probe: {exc}") from exc
        try:
            dev.close()
        except Exception as exc:
            raise RuntimeError(f"Could not close scanner device {device_id!r} after fresh probe: {exc}") from exc

        if not caps.sources:
            return None
        return ScannerDevice(
            id=device_id,
            vendor=raw[1] if len(raw) > 1 else "Unknown",
            model=raw[2] if len(raw) > 2 else device_id,
            capabilities=caps,
        )

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult:
        """Execute a scan via SANE. Blocks until complete or cancelled."""

        if params.registered_geometry is not None and params.area is not None:
            raise RuntimeError("A generic scan area cannot be combined with registered geometry")

        try:
            self._ensure_initialized()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize SANE before scanning: {exc}") from exc

        try:
            dev = self._sane.open(device_id)
        except Exception as e:
            raise RuntimeError(f"Failed to open scanner {device_id}: {e}") from e

        try:
            # IR capture strategy decides the scan mode (RGBI yields a 4th channel inline).
            ir_strategy = self._ir_strategy(dev, device_id) if params.capture_ir else None
            if params.capture_ir and ir_strategy is None:
                raise RuntimeError("IR capture requested but the device exposes no usable infrared mechanism")
            split_coolscan_ir = (
                ir_strategy == "option" and params.samples_per_scan > 1 and _strip_net_prefix(device_id).startswith(_COOLSCAN3_PREFIX)
            )
            is_coolscan3 = _strip_net_prefix(device_id).startswith(_COOLSCAN3_PREFIX)
            strict_ls5000_bw = _is_ls5000_silver_bw_recipe(
                is_coolscan3=is_coolscan3,
                params=params,
            )
            if strict_ls5000_bw and (params.autofocus is not True or params.auto_exposure is not True):
                raise RuntimeError(
                    "LS-5000 B&W RGB4x scanning requires autofocus=True and auto_exposure=True"
                )
            if params.preserve_full_canvas and not split_coolscan_ir:
                raise RuntimeError("Full-canvas IR preservation requires a split coolscan3 RGB+IR capture")
            if params.capture_state is not None and not is_coolscan3:
                raise RuntimeError("Locked scanner capture state is supported only by coolscan3")
            if params.capture_state is not None and params.capture_state.focus_position <= 0:
                raise RuntimeError(f"Cannot replay uncalibrated focus position {params.capture_state.focus_position}")

            # Configure SANE parameters. Not every backend has a `mode` option
            # (coolscan3 exposes none) — only touch options the device has.
            if hasattr(dev, "opt") and "mode" in dev.opt:
                dev.mode = "RGBI" if ir_strategy == "rgbi" else "Color"
            dev.depth = params.depth
            dev.resolution = params.dpi

            # Registered geometry is an inseparable transport position/window
            # pair. Keep legacy params.frame support for callers that only need
            # frame selection, and reject contradictory duplicate positions.
            geometry = params.registered_geometry
            requested_frame = params.frame
            if geometry is not None and geometry.frame is not None:
                if requested_frame is not None and requested_frame != geometry.frame:
                    raise RuntimeError(
                        f"Conflicting frame positions requested: frame={requested_frame}, registered geometry frame={geometry.frame}"
                    )
                requested_frame = geometry.frame

            # Validate the complete requested option set before applying any
            # of it. A missing window option must not leave the feeder moved
            # to a frame with only half of its registered geometry applied.
            option_map = dev.opt if hasattr(dev, "opt") else {}
            ir_opt = _find_coolscan3_ir_option(option_map, device_id) if ir_strategy == "option" or strict_ls5000_bw else None
            required_options: list[tuple[str, str]] = []
            if requested_frame is not None:
                required_options.append(
                    (
                        "frame",
                        f"Frame {requested_frame} requested but the device has no frame-selection option",
                    )
                )
            if geometry is not None:
                required_options.extend(
                    (
                        ("subframe", "Registered geometry requested but the device has no 'subframe' option"),
                        ("br_y", "Registered geometry requested but the device has no 'br_y' option"),
                    )
                )
            if params.auto_exposure:
                required_options.append(("ae", "Auto-exposure requested but the device has no 'ae' option"))
            if params.samples_per_scan > 1:
                # An unsupported archival multisample request must fail here,
                # before any frame/geometry write repositions the film.
                required_options.append(
                    (
                        "samples_per_scan",
                        f"{params.samples_per_scan}x multi-sampling requested but the device has no samples-per-scan option",
                    )
                )
            if ir_strategy == "option":
                if ir_opt is None:
                    raise RuntimeError("IR option strategy selected but the device's IR option is unavailable")
                required_options.append(
                    (
                        ir_opt,
                        "IR option strategy selected but the device's IR option is unavailable",
                    )
                )
            elif strict_ls5000_bw:
                if ir_opt is None:
                    raise RuntimeError("LS-5000 B&W RGB4x scanning requires the scanner's 'infrared' option")
                required_options.append(
                    (
                        ir_opt,
                        "LS-5000 B&W RGB4x scanning requires the scanner's 'infrared' option",
                    )
                )
                required_options.append(
                    (
                        "autofocus",
                        "LS-5000 B&W RGB4x scanning requires the scanner's 'autofocus' option",
                    )
                )
            if split_coolscan_ir and requested_frame is not None:
                # Only a roll-frame workflow decrements/checks the exposure
                # counter, so only it needs `frame_count` active and reset
                # between the two captures. A true single-frame holder marks
                # the option inactive and must not be rejected for that.
                required_options.append(
                    (
                        "frame_count",
                        "Coolscan roll multisample+IR capture requires the 'frame-count' option",
                    )
                )
            if params.preserve_full_canvas or params.capture_state is not None:
                required_options.extend((name, f"Locked Coolscan capture requires the {name!r} option") for name in _CAPTURE_STATE_OPTIONS)
            if params.preserve_full_canvas:
                required_options.extend(
                    (
                        ("autofocus", "Full-negative capture requires the 'autofocus' option"),
                        ("ae", "Full-negative capture requires the 'ae' option"),
                    )
                )
            if params.capture_state is not None:
                required_options.extend(
                    (
                        ("autofocus", "Locked Coolscan capture requires the 'autofocus' option"),
                        ("ae", "Locked Coolscan capture requires the 'ae' option"),
                    )
                )
            for option_name, absent_message in required_options:
                _require_writable_option(option_map, option_name, absent_message)
            if params.samples_per_scan > 1:
                _require_supported_option_value(option_map, "samples_per_scan", params.samples_per_scan)

            # Position the film and shorten its inclusive device-pixel window
            # before autofocus, auto-exposure, or scan start.
            if requested_frame is not None:
                try:
                    dev.frame = requested_frame
                except Exception as e:
                    raise RuntimeError(f"Could not set frame={requested_frame}: {e}") from e

            if geometry is not None:
                for option_name, value in (
                    ("subframe", geometry.subframe_mm),
                    ("br_y", geometry.br_y_device_px),
                ):
                    try:
                        setattr(dev, option_name, value)
                    except Exception as e:
                        raise RuntimeError(f"Could not set registered geometry {option_name}={value}: {e}") from e
                _validate_registered_geometry_readback(
                    dev,
                    frame=requested_frame,
                    subframe_mm=geometry.subframe_mm,
                    br_y_device_px=geometry.br_y_device_px,
                )

            # A previous scan may have left coolscan3's inline-IR flag on.
            # Disable it before autofocus or auto-exposure can observe the
            # wrong channel layout, then prove it stayed off before start.
            if strict_ls5000_bw:
                if ir_opt is None:
                    raise RuntimeError("LS-5000 B&W RGB4x scanning lost its preflighted infrared option")
                try:
                    setattr(dev, ir_opt, False)
                except Exception as e:
                    raise RuntimeError(f"Could not explicitly disable infrared for LS-5000 B&W scanning: {e}") from e

            if params.capture_state is not None:
                _apply_capture_state(dev, params.capture_state)

            # Autofocus where the device supports it (the LS-5000 powers up
            # at an uncalibrated focus position — unfocused otherwise).
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

            # Hardware multi-sampling: explicitly requested for archival
            # quality — refuse to silently degrade to a single pass. The
            # option's presence/activity was already validated pre-positioning.
            if params.samples_per_scan > 1:
                try:
                    dev.samples_per_scan = params.samples_per_scan
                except Exception as e:
                    raise RuntimeError(f"Could not set samples-per-scan={params.samples_per_scan}: {e}") from e

            # Inline IR via a boolean option (coolscan3 `infrared`): the 4th
            # sample rides in the same frame, no mode/source change involved.
            # IR was explicitly requested — fail loud rather than silently
            # returning a scan without the channel.
            if ir_strategy == "option":
                if ir_opt is None:
                    raise RuntimeError("IR option strategy selected but the device's IR option is unavailable")
                try:
                    # The LS-5000 wedges when coolscan3 requests multisampling
                    # on the IR window.  Capture the multisampled RGB first;
                    # the 1x RGBI capture follows on this same open reservation.
                    setattr(dev, ir_opt, not split_coolscan_ir)
                except Exception as e:
                    raise RuntimeError(f"IR capture requested but enabling option {ir_opt!r} failed: {e}") from e
            # Apply hardware-specific optimizations
            if device_id.startswith(_PIEUSB_PREFIX):
                self._set_pieusb_flags(dev, params.capture_ir)

            # Set scan area if specified
            if params.area is not None:
                tl_x, tl_y, br_x, br_y = params.area
                if hasattr(dev, "tl_x"):
                    dev.tl_x = tl_x
                if hasattr(dev, "tl_y"):
                    dev.tl_y = tl_y
                if hasattr(dev, "br_x"):
                    dev.br_x = br_x
                if hasattr(dev, "br_y"):
                    dev.br_y = br_y

            # Emit start progress
            if progress:
                try:
                    progress(0.0)
                except Exception:
                    pass

            if cancel.is_set():
                dev.cancel()
                raise RuntimeError("Scan cancelled before start")

            if strict_ls5000_bw:
                if geometry is None:
                    raise RuntimeError("LS-5000 B&W RGB4x scanning lost its registered geometry")
                _validate_ls5000_silver_bw_option_readback(
                    dev,
                    frame=requested_frame,
                    infrared_option_name=ir_opt,
                    subframe_mm=geometry.subframe_mm,
                    br_y_device_px=geometry.br_y_device_px,
                )

            # Start scan
            dev.start()
            rgb_array = None
            ir_array = None
            ir_alignment: SplitIrAlignment | None = None
            ir_valid_mask: np.ndarray | None = None
            capture_state: ScannerCaptureState | None = None
            split_source: SplitSourceCapture | None = None

            # Frame geometry truth, for channel reinterpretation below
            # (python-sane assumes 3 samples/pixel; see _reinterpret_channels).
            try:
                frame_format, last_frame, (px_per_line, n_lines), returned_depth, bytes_per_line = dev.get_parameters()
            except Exception:
                frame_format = None
                last_frame = None
                px_per_line = n_lines = -1
                returned_depth = bytes_per_line = -1

            if strict_ls5000_bw:
                _validate_ls5000_silver_bw_parameters(
                    frame_format=frame_format,
                    last_frame=last_frame,
                    pixels_per_line=px_per_line,
                    lines=n_lines,
                    returned_depth=returned_depth,
                    bytes_per_line=bytes_per_line,
                )

            if not split_coolscan_ir and ir_strategy == "option":
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
            # progress is forwarded per-line so the UI bar moves during the
            # read; see _snap_progress_callback for why it can't also cancel
            # the read early.
            try:
                first_progress = _progress_segment(progress, 0.0, 0.75) if split_coolscan_ir else progress
                rgb_array = dev.arr_snap(progress=_snap_progress_callback(first_progress))
            except Exception as e:
                dev.cancel()
                raise RuntimeError(f"RGB scan failed: {e}") from e

            if strict_ls5000_bw and (
                not isinstance(rgb_array, np.ndarray) or rgb_array.dtype != np.uint16 or rgb_array.shape != (5_959, 3_946, 3)
            ):
                dev.cancel()
                shape = getattr(rgb_array, "shape", None)
                dtype = getattr(rgb_array, "dtype", None)
                raise RuntimeError(f"LS-5000 B&W RGB4x payload disagrees with its frame metadata: shape={shape}, dtype={dtype}")

            if params.preserve_full_canvas:
                capture_state = _read_capture_state(dev)

            if split_coolscan_ir:
                if ir_opt is None:
                    dev.cancel()
                    raise RuntimeError("Coolscan split capture lost its preflighted infrared option")
                if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
                    dev.cancel()
                    raise RuntimeError(f"Coolscan multisample RGB capture yielded an unexpected shape {rgb_array.shape}")
                if cancel.is_set():
                    dev.cancel()
                    raise RuntimeError("Scan cancelled")

                # sane_read() consumes coolscan3's one-frame counter at EOF —
                # but only when the exposure counter is in play (roll adapter).
                # A single-frame holder marks `frame_count` inactive and never
                # checks it; writing an inactive option would itself error.
                # Reset only that counter; do not touch `frame`, `subframe`, or
                # the scan window, so the feeder never repositions between the
                # RGB and RGBI captures.
                try:
                    fresh_options = dev.opt if hasattr(dev, "opt") else {}
                    if _option_is_active(fresh_options, "frame_count"):
                        dev.frame_count = 1
                    dev.samples_per_scan = 1
                    setattr(dev, ir_opt, True)
                    if params.autofocus and "autofocus" in option_map:
                        dev.autofocus = False
                    if params.auto_exposure and "ae" in option_map:
                        dev.ae = False
                except Exception as e:
                    dev.cancel()
                    raise RuntimeError(f"Could not prepare the single-pass Coolscan IR capture: {e}") from e

                # Last chance to stop without wasting the IR traversal — after
                # this start, cancellation waits for the read to finish (a
                # transfer must never be interrupted mid-flight).
                if cancel.is_set():
                    dev.cancel()
                    raise RuntimeError("Scan cancelled")
                try:
                    dev.start()
                except Exception as e:
                    dev.cancel()
                    raise RuntimeError(f"Infrared scan failed: {e}") from e
                try:
                    ir_format, ir_last_frame, (ir_px_per_line, ir_n_lines), ir_depth, ir_bytes_per_line = dev.get_parameters()
                except Exception:
                    ir_format = None
                    ir_last_frame = None
                    ir_px_per_line = ir_n_lines = -1
                    ir_depth = ir_bytes_per_line = -1
                _validate_inline_rgbi_parameters(
                    frame_format=ir_format,
                    last_frame=ir_last_frame,
                    pixels_per_line=ir_px_per_line,
                    lines=ir_n_lines,
                    returned_depth=ir_depth,
                    bytes_per_line=ir_bytes_per_line,
                    requested_depth=params.depth,
                    context="Single-pass Coolscan infrared frame",
                )
                try:
                    rgbi_array = dev.arr_snap(progress=_snap_progress_callback(_progress_segment(progress, 0.75, 1.0)))
                except Exception as e:
                    dev.cancel()
                    raise RuntimeError(f"Infrared scan failed: {e}") from e

                # Honor a mid-IR-read cancel before spending CV time on a
                # result nobody wants; never return a partial (RGB-only) scan.
                if cancel.is_set():
                    dev.cancel()
                    raise RuntimeError("Scan cancelled")

                rgbi_array = _reinterpret_channels(rgbi_array, ir_px_per_line, ir_n_lines)
                if rgbi_array.ndim != 3 or rgbi_array.shape[2] != 4:
                    dev.cancel()
                    raise RuntimeError(f"Single-pass Coolscan IR capture yielded no 4th channel (shape={rgbi_array.shape})")
                rgb_proxy, ir_array = _split_rgbi(rgbi_array)
                if params.preserve_full_canvas:
                    split_source = SplitSourceCapture(
                        rgb4x=rgb_array,
                        rgb1x_proxy=rgb_proxy,
                        ir1x=ir_array,
                    )
                try:
                    if params.preserve_full_canvas:
                        rgb_array, ir_array, ir_valid_mask, ir_alignment = _align_split_ir_full_canvas(
                            rgb_array,
                            rgb_proxy,
                            ir_array,
                        )
                    else:
                        rgb_array, ir_array, ir_alignment = _align_split_ir(rgb_array, rgb_proxy, ir_array)
                except RuntimeError as exc:
                    diagnostics_dir = os.environ.get(_SPLIT_DIAGNOSTICS_ENV)
                    if diagnostics_dir:
                        try:
                            _persist_split_diagnostics(
                                diagnostics_dir,
                                reference_rgb=rgb_array,
                                proxy_rgb=rgb_proxy,
                                ir=ir_array,
                                error=exc,
                            )
                        except (OSError, ValueError) as diagnostics_exc:
                            logger.warning(f"Could not persist split-capture diagnostics: {diagnostics_exc}")
                    raise

            # Inline-IR frames carry infrared as the 4th channel — recover the
            # true shape (python-sane misreads 4-sample frames) and split it off.
            if not split_coolscan_ir and ir_strategy == "option":
                rgb_array = _reinterpret_channels(rgb_array, px_per_line, n_lines)
                if rgb_array.ndim == 3 and rgb_array.shape[2] == 4:
                    rgb_array, ir_array = _split_rgbi(rgb_array)
                else:
                    # IR was explicitly requested; a long archival scan must
                    # not quietly complete without its defect channel.
                    dev.cancel()
                    raise RuntimeError(f"IR strategy '{ir_strategy}' yielded no 4th channel (shape={rgb_array.shape})")
            elif ir_strategy == "rgbi" and rgb_array.ndim == 3 and rgb_array.shape[2] == 4:
                # Preserve the pre-Coolscan contract for generic RGBI devices:
                # native four-channel ndarrays are split directly, without
                # coolscan3 raw-stream reinterpretation or metadata gates.
                rgb_array, ir_array = _split_rgbi(rgb_array)

            expected_dtype = np.uint16 if params.depth == 16 else np.uint8
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
                ir_alignment=ir_alignment,
                ir_valid_mask=ir_valid_mask,
                capture_state=capture_state,
                split_source=split_source,
            )

        finally:
            try:
                dev.cancel()
            except Exception:
                pass
            try:
                dev.close()
            except Exception:
                pass

    def eject(self, device_id: str) -> bool:
        """Trigger the device's vendor eject action, if it exposes one.

        Mirrors Nikon Scan's own behavior of ejecting film at batch
        completion instead of leaving it parked (the LS-5000 feeder
        auto-parks a few minutes after any session closes; a parked feeder
        needs a power-cycle to recover mid-roll). Capability-gated: returns
        False as a clean no-op when the device has no active, settable
        'eject' option — e.g. a backend other than coolscan3. Raises when
        the option is present but triggering it, or the surrounding
        open/close, fails; callers that must not fail an otherwise-complete
        run over a transport hiccup are responsible for catching and
        logging.
        """

        try:
            self._ensure_initialized()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize SANE before eject: {exc}") from exc

        try:
            dev = self._sane.open(device_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to open scanner {device_id} to eject: {exc}") from exc

        try:
            option_map = dev.opt if hasattr(dev, "opt") else {}
            eject_option = _find_eject_option(option_map)
            if eject_option is None or not _option_is_usable(option_map[eject_option]):
                triggered = False
            else:
                setattr(dev, eject_option, True)
                triggered = True
        except Exception as exc:
            try:
                dev.close()
            except Exception:
                pass
            raise RuntimeError(f"Could not trigger eject on {device_id!r}: {exc}") from exc

        try:
            dev.close()
        except Exception as exc:
            raise RuntimeError(f"Could not close scanner device {device_id!r} after eject: {exc}") from exc
        return triggered

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
            opts["clean_image"] = True
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
        second scan), 'internal' (applied by the Backend/Scanner itself) or None."""
        opt = dev.opt if hasattr(dev, "opt") else {}
        backend_id = _strip_net_prefix(device_id)
        if device_id.startswith(_PIEUSB_PREFIX):
            return "internal"
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
