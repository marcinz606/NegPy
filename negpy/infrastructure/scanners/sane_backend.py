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
                caps = self._detect_caps(dev)
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
            # Configure SANE parameters
            dev.mode = "Color"
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

            # IR scan (if requested and supported)
            ir_channel = params.capture_ir and self._has_ir(dev)

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

            expected_dtype = np.uint16 if params.depth == 16 else np.uint8
            if rgb_array.dtype != expected_dtype:
                logger.warning(
                    f"Scanner returned {rgb_array.dtype} for depth={params.depth}; "
                    f"shape={rgb_array.shape}, min={rgb_array.min()}, max={rgb_array.max()}"
                )

            if cancel.is_set():
                dev.cancel()
                raise RuntimeError("Scan cancelled")

            # Read IR frame if applicable
            if ir_channel:
                try:
                    # Switch to IR source
                    old_source = dev.source
                    # For Plustek: set --ir parameter if available
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

    def _detect_caps(self, dev) -> ScannerCapabilities:
        """Read dev.opt to build ScannerCapabilities."""
        opt = dev.opt if hasattr(dev, "opt") else {}

        dpi = ()
        if "resolution" in opt:
            constraint = opt["resolution"].constraint
            if constraint is None:
                dpi = CANONICAL_DPI_STOPS
            elif isinstance(constraint, (list, tuple)):
                dpi = tuple(sorted(int(c) for c in constraint))
            else:
                # Range constraint: intersect with canonical stops
                lo, hi = constraint
                dpi = tuple(s for s in CANONICAL_DPI_STOPS if lo <= s <= hi)
                if not dpi:
                    dpi = tuple(CANONICAL_DPI_STOPS)

        depth = (8, 16)
        if "depth" in opt:
            constraint = opt["depth"].constraint
            if isinstance(constraint, (list, tuple)):
                depth = tuple(sorted(int(d) for d in constraint))

        sources: tuple[ScanMode, ...] = ()
        if "source" in opt:
            constraint = opt["source"].constraint
            if isinstance(constraint, (list, tuple)):
                modes: set[ScanMode] = set()
                for s in constraint:
                    s_stripped = str(s).strip().lower()
                    if "(" in s_stripped:
                        s_base = s_stripped.split("(")[0].strip()
                    else:
                        s_base = s_stripped
                    mode = _SOURCE_MAP.get(s_base)
                    if mode is not None:
                        modes.add(mode)
                sources = tuple(sorted(modes, key=lambda m: list(ScanMode).index(m)))

        ir_channel = False
        for key in opt:
            key_lower = str(key).lower()
            if key_lower in ("ir", "--ir", "preview-ir", "--preview-ir"):
                ir_channel = True
                break

        max_area = (36.0, 25.0)  # default 35mm frame
        br_x = -1.0
        br_y = -1.0
        if "br-x" in opt:
            constraint = opt["br-x"].constraint
            if isinstance(constraint, (list, tuple)) and len(constraint) >= 2:
                br_x = float(constraint[1])
        if "br-y" in opt:
            constraint = opt["br-y"].constraint
            if isinstance(constraint, (list, tuple)) and len(constraint) >= 2:
                br_y = float(constraint[1])
        if br_x > 0 and br_y > 0:
            max_area = (br_x, br_y)

        return ScannerCapabilities(
            ir_channel=ir_channel,
            supported_dpi=dpi,
            supported_depths=depth,
            sources=sources,
            max_area_mm=max_area,
        )

    @staticmethod
    def _has_ir(dev) -> bool:
        if not hasattr(dev, "opt"):
            return False
        for key in dev.opt:
            if str(key).lower().strip("-") in ("ir", "preview-ir"):
                return True
        return False

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
