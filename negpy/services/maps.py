"""OpenStreetMap tile and Nominatim place lookups for the capture-location picker."""

from __future__ import annotations

import json
import locale
import os
import urllib.parse
import urllib.request
from typing import Any, Optional

from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.version import get_app_version

logger = get_logger("services.maps")

TILE_SIZE = 256
MIN_ZOOM = 2
MAX_ZOOM = 18

_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
_NOMINATIM = "https://nominatim.openstreetmap.org"

# Closing the picker joins whatever request is in flight, so this is also the longest stall a
# user can feel on OK or Cancel.
_TIMEOUT = 3.0

# The OSM tile policy requires an identifying User-Agent and local caching.
_USER_AGENT = f"NegPy/{get_app_version()} (+https://github.com/marcinz606/NegPy)"


def accept_language() -> str:
    """Ask for place names in the user's language: an unreadable script would be written to XMP."""
    tag = (locale.getlocale()[0] or os.environ.get("LANG", "")).split(".")[0].replace("_", "-")
    return f"{tag},en" if tag else "en"


_CITY_KEYS = ("city", "town", "village", "hamlet", "municipality", "suburb")
_STATE_KEYS = ("state", "region", "province", "county")


def tile_cache_path(z: int, x: int, y: int) -> str:
    return os.path.join(APP_CONFIG.cache_dir, "map_tiles", str(z), str(x), f"{y}.png")


def _get(url: str, timeout: float) -> Optional[bytes]:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return None
            return response.read()
    except Exception as exc:
        logger.debug("map request failed (%s): %s", url, exc)
        return None


def fetch_tile(z: int, x: int, y: int, timeout: float = _TIMEOUT) -> Optional[bytes]:
    """One map tile, from the disk cache when it is there, otherwise from OSM."""
    path = tile_cache_path(z, x, y)
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        pass

    data = _get(_TILE_URL.format(z=z, x=x, y=y), timeout)
    if data is None:
        return None

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("map tile cache write failed: %s", exc)

    return data


def _json(url: str, timeout: float) -> Any:
    data = _get(url, timeout)
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def search_places(query: str, limit: int = 8, timeout: float = _TIMEOUT) -> list[dict]:
    """Nominatim hits for a place name. Empty when offline or nothing matches."""
    if not query.strip():
        return []
    params = urllib.parse.urlencode(
        {"q": query.strip(), "format": "json", "addressdetails": 1, "limit": limit, "accept-language": accept_language()}
    )
    payload = _json(f"{_NOMINATIM}/search?{params}", timeout)
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def reverse_place(lat: float, lon: float, timeout: float = _TIMEOUT) -> Optional[dict]:
    """The Nominatim result for a position, or None when it cannot be reached."""
    params = urllib.parse.urlencode(
        {
            "lat": f"{lat:.6f}",
            "lon": f"{lon:.6f}",
            "format": "json",
            "addressdetails": 1,
            "zoom": 10,
            "accept-language": accept_language(),
        }
    )
    payload = _json(f"{_NOMINATIM}/reverse?{params}", timeout)
    return payload if isinstance(payload, dict) and "error" not in payload else None


def place_fields(result: Optional[dict]) -> tuple[str, str, str]:
    """City, state and country from a Nominatim result."""
    if not isinstance(result, dict):
        return "", "", ""
    address = result.get("address")
    if not isinstance(address, dict):
        return "", "", ""

    def first(keys: tuple[str, ...]) -> str:
        for key in keys:
            value = address.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    return first(_CITY_KEYS), first(_STATE_KEYS), first(("country",))


def result_coords(result: Optional[dict]) -> Optional[tuple[float, float]]:
    if not isinstance(result, dict):
        return None
    try:
        return float(result["lat"]), float(result["lon"])
    except (KeyError, TypeError, ValueError):
        return None
