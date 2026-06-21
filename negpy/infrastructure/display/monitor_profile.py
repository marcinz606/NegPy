"""Cross-platform monitor ICC profile detection.

`QScreen` exposes no color space in PyQt6, and PIL's `get_display_profile` only
works on Windows, so the display profile must be read per-OS:

- Linux: colord over D-Bus (works under Wayland and X11)
- macOS: ColorSync via CoreGraphics (ctypes)
- Windows: PIL's display profile

Every backend returns raw ICC bytes or None (treat the display as sRGB); none
ever raises.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from negpy.kernel.system.logging import get_logger

if TYPE_CHECKING:
    from PyQt6.QtDBus import QDBusConnection

logger = get_logger(__name__)


def detect_monitor_icc(screen: object) -> Optional[bytes]:
    """ICC profile bytes for the screen the window is on, or None for sRGB."""
    try:
        if sys.platform.startswith("linux"):
            return _detect_colord(screen)
        if sys.platform == "darwin":
            return _detect_macos()
        if sys.platform == "win32":
            return _detect_windows()
    except Exception as e:
        logger.debug("Monitor profile detection failed: %s", e)
    return None


_CM = "org.freedesktop.ColorManager"


def _get_property(bus: "QDBusConnection", path: str, iface: str, name: str) -> object:
    """Read a D-Bus property via org.freedesktop.DBus.Properties.Get.

    colord's interfaces aren't introspected by QtDBus, so ``QDBusInterface.property``
    returns None — the explicit Get call is required.
    """
    from PyQt6.QtDBus import QDBusInterface

    props = QDBusInterface(_CM, path, "org.freedesktop.DBus.Properties", bus)
    args = props.call("Get", iface, name).arguments()
    return args[0] if args else None


def _detect_colord(screen: object) -> Optional[bytes]:
    """Read the active display's assigned profile from colord via D-Bus."""
    from PyQt6.QtDBus import QDBusConnection, QDBusInterface, QDBusObjectPath

    def as_path(v: object) -> Optional[str]:
        if isinstance(v, QDBusObjectPath):
            return v.path()
        return v if isinstance(v, str) else None

    bus = QDBusConnection.systemBus()
    if not bus.isConnected():
        return None
    manager = QDBusInterface(_CM, "/org/freedesktop/ColorManager", _CM, bus)
    if not manager.isValid():
        return None

    args = manager.call("GetDevicesByKind", "display").arguments()
    device_paths = [p for p in (as_path(x) for x in args[0])] if args else []
    device_paths = [p for p in device_paths if p]
    if not device_paths:
        return None

    device_path = _match_device(bus, device_paths, screen) or device_paths[0]

    profiles = _get_property(bus, device_path, f"{_CM}.Device", "Profiles")
    profile_path = as_path(profiles[0]) if isinstance(profiles, (list, tuple)) and profiles else None
    if not profile_path:
        return None

    filename = _get_property(bus, profile_path, f"{_CM}.Profile", "Filename")
    if not filename or not isinstance(filename, str):
        return None
    data = Path(filename).read_bytes()
    return data or None


def _match_device(bus: "QDBusConnection", device_paths: list, screen: object) -> Optional[str]:
    """Pick the colord device matching the QScreen by model/vendor; None if no match."""
    model = (getattr(screen, "model", lambda: "")() or "").strip().lower()
    vendor = (getattr(screen, "manufacturer", lambda: "")() or "").strip().lower()
    if not model and not vendor:
        return None
    for path in device_paths:
        dev_model = str(_get_property(bus, path, f"{_CM}.Device", "Model") or "").strip().lower()
        dev_vendor = str(_get_property(bus, path, f"{_CM}.Device", "Vendor") or "").strip().lower()
        if model and dev_model and (model in dev_model or dev_model in model):
            return path
        if vendor and dev_vendor and vendor == dev_vendor and not model:
            return path
    return None


def _detect_macos() -> Optional[bytes]:
    """ColorSync profile for the main display, else assume Display P3.

    Modern Macs ship P3 panels, so P3 is a closer fallback than sRGB when the
    OS read fails.
    """
    try:
        data = _colorsync_icc()
    except Exception as e:
        logger.debug("ColorSync read failed: %s", e)
        data = None
    if data:
        return data

    from negpy.domain.models import ColorSpace
    from negpy.infrastructure.display.color_mgmt import icc_bytes_for_space

    return icc_bytes_for_space(ColorSpace.P3_D65.value)


def _colorsync_icc() -> Optional[bytes]:
    """Read the main display's ColorSync ICC data via CoreGraphics (ctypes)."""
    import ctypes
    import ctypes.util

    cg_path = ctypes.util.find_library("CoreGraphics") or (
        "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
    )
    cf_path = ctypes.util.find_library("CoreFoundation") or (
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    cg = ctypes.cdll.LoadLibrary(cg_path)
    cf = ctypes.cdll.LoadLibrary(cf_path)

    cg.CGMainDisplayID.restype = ctypes.c_uint32
    cg.CGDisplayCopyColorSpace.restype = ctypes.c_void_p
    cg.CGDisplayCopyColorSpace.argtypes = [ctypes.c_uint32]
    cg.CGColorSpaceCopyICCData.restype = ctypes.c_void_p
    cg.CGColorSpaceCopyICCData.argtypes = [ctypes.c_void_p]
    cf.CFDataGetLength.restype = ctypes.c_long
    cf.CFDataGetLength.argtypes = [ctypes.c_void_p]
    cf.CFDataGetBytePtr.restype = ctypes.c_void_p
    cf.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
    cf.CFRelease.argtypes = [ctypes.c_void_p]

    colorspace = cg.CGDisplayCopyColorSpace(cg.CGMainDisplayID())
    if not colorspace:
        return None
    try:
        data_ref = cg.CGColorSpaceCopyICCData(colorspace)
        if not data_ref:
            return None
        try:
            length = cf.CFDataGetLength(data_ref)
            ptr = cf.CFDataGetBytePtr(data_ref)
            if not ptr or length <= 0:
                return None
            return ctypes.string_at(ptr, length)
        finally:
            cf.CFRelease(data_ref)
    finally:
        cf.CFRelease(colorspace)


def _detect_windows() -> Optional[bytes]:
    """Read the OS display profile via PIL (Windows-only API)."""
    from PIL import ImageCms

    prof = ImageCms.get_display_profile()
    if prof is None:
        return None
    return prof.tobytes() or None
