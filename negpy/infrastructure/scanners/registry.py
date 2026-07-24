from typing import Callable

from negpy.infrastructure.scanners.base import ScannerBackend


def _make_sane() -> ScannerBackend:
    from negpy.infrastructure.scanners.sane_backend import SaneBackend

    return SaneBackend()


DEFAULT_BACKEND_ID = "sane"

# id -> (display label, factory). Insertion order drives the sidebar dropdown.
# Adding a backend is one entry here plus its implementation module.
BACKENDS: dict[str, tuple[str, Callable[[], ScannerBackend]]] = {
    "sane": ("SANE", _make_sane),
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
