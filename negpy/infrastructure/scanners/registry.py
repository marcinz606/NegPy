import sys
from pathlib import Path
from typing import Callable

from negpy.infrastructure.scanners.base import ScannerBackend, ScannerUnavailable


def _make_sane() -> ScannerBackend:
    from negpy.infrastructure.scanners.sane_backend import SaneBackend

    return SaneBackend()


def _make_plustek() -> ScannerBackend:
    try:
        from negpy.infrastructure.scanners.plustek_backend import PlustekBackend
    except ImportError as exc:
        raise ScannerUnavailable(
            "Plustek USB backend failed to import. Install with: uv sync --group plustek. "
            "On Windows, bind WinUSB with Zadig for OpticFilm 8200i SE (07b3:1825) — "
            "see docs/PLUSTEK_WINDOWS.md."
        ) from exc

    from negpy.kernel.system.paths import get_default_user_dir

    calib_cache = Path(get_default_user_dir()) / "plustek_calib"
    return PlustekBackend(calib_cache=calib_cache)


def _make_pieusb() -> ScannerBackend:
    from negpy.infrastructure.scanners.pieusb_backend import PieusbBackend

    return PieusbBackend()


DEFAULT_BACKEND_ID = "plustek" if sys.platform == "win32" else "sane"

# id -> (display label, factory). Insertion order drives the sidebar dropdown. SANE is
# Unix-only (python-sane); Windows ships Plustek USB and PIEUSB.
BACKENDS: dict[str, tuple[str, Callable[[], ScannerBackend]]] = {
    "plustek": ("pyOpticfilm (Plustek)", _make_plustek),
    "pieusb": ("PIEUSB", _make_pieusb),
}
if sys.platform != "win32":
    BACKENDS = {
        "sane": ("SANE", _make_sane),
        **BACKENDS,
    }


def backend_choices() -> list[tuple[str, str]]:
    """(id, label) pairs for the dropdown, in registration order."""
    return [(bid, label) for bid, (label, _factory) in BACKENDS.items()]


def create_backend(backend_id: str) -> ScannerBackend:
    """Instantiate the backend for `backend_id`, falling back to the default for
    an unknown/removed id — persisted settings may name a backend that no longer
    ships, and that must not crash a scan."""
    _label, factory = BACKENDS.get(backend_id) or BACKENDS[DEFAULT_BACKEND_ID]
    return factory()
