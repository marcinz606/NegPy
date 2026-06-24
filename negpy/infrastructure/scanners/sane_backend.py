import sys
import threading
from typing import Callable

import numpy as np

from negpy.infrastructure.scanners.base import ScanMode, ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import ScanParams
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

# SANE option py_names (underscored) that expose a dedicated infrared channel/scan.
_IR_OPTION_NAMES = ("ir", "preview_ir")

# Backends for dedicated film scanners that expose no `source` option.
_FILM_BACKEND_PREFIXES = ("pieusb:",)


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
    return device_id.startswith(_FILM_BACKEND_PREFIXES)


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
    # opt keys are py_names (hyphens → underscores). constraint is a (min, max, step) range.
    def _upper(name: str) -> float:
        if name not in opt:
            return -1.0
        constraint = opt[name].constraint
        if isinstance(constraint, (list, tuple)) and len(constraint) >= 2:
            return float(constraint[1])
        return -1.0

    br_x, br_y = _upper("br_x"), _upper("br_y")
    if br_x > 0 and br_y > 0:
        return (br_x, br_y)
    return (36.0, 25.0)  # default 35mm frame


def _detect_ir(opt) -> bool:
    if _mode_has_rgbi(opt):
        return True
    return any(str(key).lower().replace("-", "_").strip("_") in _IR_OPTION_NAMES for key in opt)


def _caps_from_options(opt, device_id: str = "") -> ScannerCapabilities:
    """Build ScannerCapabilities from a SANE option map. Pure — no `sane` import."""
    sources = _detect_explicit_sources(opt)
    if not sources and _infer_film_scanner(opt, device_id):
        # Dedicated film scanner with no `source` option (e.g. pieusb). `sources` is only a
        # detection/UI gate — never applied to the device — so populate it to unblock scanning.
        sources = (ScanMode.NEGATIVE, ScanMode.POSITIVE, ScanMode.TRANSPARENCY)
    return ScannerCapabilities(
        ir_channel=_detect_ir(opt),
        supported_dpi=_detect_dpi(opt),
        supported_depths=_detect_depths(opt),
        sources=sources,
        max_area_mm=_detect_max_area(opt),
    )


def _split_rgbi(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an RGBI scan `(H, W, 4)` into RGB `(H, W, 3)` and IR `(H, W)`."""
    return arr[:, :, :3], arr[:, :, 3]


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

    def list_devices(self) -> list[ScannerDevice]:
        if self._devices_cache is not None:
            return self._devices_cache

        if not self._sane_initialized:
            try:
                self._sane.init()
                self._sane_initialized = True
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

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult:
        """Execute a scan via SANE. Blocks until complete or cancelled."""

        try:
            dev = self._sane.open(device_id)
        except Exception as e:
            raise RuntimeError(f"Failed to open scanner {device_id}: {e}") from e

        try:
            # IR capture strategy decides the scan mode (RGBI yields a 4th channel inline).
            ir_strategy = self._ir_strategy(dev) if params.capture_ir else None

            # Configure SANE parameters
            dev.mode = "RGBI" if ir_strategy == "rgbi" else "Color"
            dev.depth = params.depth
            dev.resolution = params.dpi

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

            # Start scan
            dev.start()
            rgb_array = None
            ir_array = None

            # Read RGB frame. Use arr_snap() (numpy path) — snap() goes via PIL
            # which is 8-bit only and silently truncates 16-bit RGB buffers.
            try:
                rgb_array = dev.arr_snap()
            except Exception as e:
                dev.cancel()
                raise RuntimeError(f"RGB scan failed: {e}") from e

            # RGBI mode returns infrared as the 4th channel inline — split it off.
            if ir_strategy == "rgbi" and rgb_array.ndim == 3 and rgb_array.shape[2] == 4:
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

            # Look up real vendor/model from cached device list (dev itself has no such attrs).
            sd = next((d for d in (self._devices_cache or []) if d.id == device_id), None)
            model = f"{sd.vendor} {sd.model}" if sd else device_id

            return ScanResult(
                rgb=rgb_array,
                ir=ir_array[:, :, 0] if ir_array is not None and ir_array.ndim == 3 else ir_array,
                dpi=params.dpi,
                device_model=model,
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

    def _detect_caps(self, dev, device_id: str = "") -> ScannerCapabilities:
        """Read dev.opt to build ScannerCapabilities."""
        opt = dev.opt if hasattr(dev, "opt") else {}
        return _caps_from_options(opt, device_id)

    @staticmethod
    def _ir_strategy(dev) -> str | None:
        """How to capture IR for this device: 'rgbi' (4th channel), 'source' (Plustek
        second scan), or None."""
        opt = dev.opt if hasattr(dev, "opt") else {}
        if _mode_has_rgbi(opt):
            return "rgbi"
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
