"""Thin connector from the standalone ``fuji-ble-negpy-trigger`` package into NegPy.

This is the ONLY NegPy module that imports ``fujitrigger``. It mirrors how NegPy already
consumes ``python-sane`` and ``gphoto2``: an optional dependency, imported lazily, entirely
absent by default. Everything above it — the capture worker, the sidebar — talks only
through the functions here, which accept and return plain NegPy-native dicts; no
``fujitrigger`` type escapes this module. Operational errors are translated to plain
``RuntimeError`` with the original preserved as ``__cause__`` (the same contract
``SaneBackend`` and ``GphotoCamera`` already honour).

The device-specific BLE protocol lives in the standalone package; NegPy maintains nothing
here but the glue.
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Optional

from negpy.infrastructure.capture.base import Camera
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def available() -> bool:
    """True when the optional ``fuji-ble-negpy-trigger`` package (import ``fujitrigger``) is present."""
    return importlib.util.find_spec("fujitrigger") is not None


def _ft():
    """Import ``fujitrigger`` lazily, so NegPy runs fine without it."""
    try:
        import fujitrigger  # noqa: PLC0415 — optional dependency, imported on demand
    except ImportError as exc:  # pragma: no cover — depends on the install
        raise RuntimeError(
            "Bluetooth triggering needs the optional fuji-ble-negpy-trigger package. "
            "Install it with `uv sync --group fuji-ble` (macOS and Linux)."
        ) from exc
    return fujitrigger


def _pairing(ft, cfg: dict):
    return ft.Pairing(
        address=cfg.get("address", ""),
        name=cfg.get("name", ""),
        flavor=ft.Flavor(cfg.get("flavor") or "secure"),
        token=cfg.get("token", ""),
        serial=cfg.get("serial", ""),
    )


def make_camera(cfg: dict, cancel: Optional[threading.Event] = None) -> Camera:
    """Build the BLE capture connector from a stored config (pairing fields + ``drop_folder``).

    Returns an object satisfying NegPy's ``Camera`` protocol. The caller owns ``cancel``
    (the worker's stop event) so a cancelled scan aborts the wait for the dropped RAW.
    """
    ft = _ft()
    drop_folder = cfg.get("drop_folder", "")
    if not drop_folder:
        raise RuntimeError("Bluetooth trigger needs a drop folder — the camera's WiFi auto-save folder")
    trigger = ft.FujiBleTrigger(_pairing(ft, cfg))
    return ft.FujiCamera(trigger, ft.FolderWatchAcquirer(drop_folder), cancel=cancel)


def discover(timeout: float = 8.0) -> list[dict]:
    """Advertising Fujifilm bodies as plain dicts (name/address/flavor/serial/token/rssi)."""
    ft = _ft()
    try:
        found = ft.discover(timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — translate to a plain RuntimeError
        raise RuntimeError(f"Bluetooth discovery failed: {exc}") from exc
    return [
        {
            "name": a.name,
            "address": a.address,
            "flavor": a.flavor.value,
            "serial": a.serial.hex(),
            "token": a.token.hex(),
            "rssi": a.rssi,
        }
        for a in found
    ]


def pair(timeout: float = 12.0) -> tuple[bool, str, dict]:
    """Discover + bond a body. Returns ``(ok, message, pairing_dict)``; the dict is empty on failure.

    The body rotates its BLE address and its pairing window is brief, so each candidate is
    tried in turn (strongest signal first) until one bonds. ``pairing_dict`` carries the
    pairing fields the sidebar persists and later hands back to ``make_camera``.
    """
    ft = _ft()
    try:
        found = ft.discover(timeout=timeout)
        if not found:
            return False, "No Fujifilm camera found — Bluetooth on and the camera in pairing mode?", {}
        by_id: dict[str, object] = {}
        for adv in sorted(found, key=lambda a: a.rssi, reverse=True):
            by_id.setdefault(adv.serial.hex() or adv.token.hex() or adv.address, adv)
        last = ""
        for adv in by_id.values():
            pairing = ft.Pairing(address=adv.address, name=adv.name, flavor=adv.flavor, token=adv.token.hex(), serial=adv.serial.hex())
            trigger = ft.FujiBleTrigger(pairing)
            try:
                trigger.connect(timeout=40.0)
            except Exception as exc:  # noqa: BLE001 — try the next candidate address
                last = str(exc)
                trigger.close()
                continue
            trigger.close()
            return (
                True,
                pairing.name or pairing.address,
                {
                    "address": pairing.address,
                    "name": pairing.name,
                    "flavor": pairing.flavor.value,
                    "token": pairing.token,
                    "serial": pairing.serial,
                },
            )
        return False, f"Could not pair: {last}", {}
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 — translate vendor errors to a plain RuntimeError
        raise RuntimeError(f"Bluetooth pairing failed: {exc}") from exc
