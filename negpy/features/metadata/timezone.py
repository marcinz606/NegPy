"""Capture time zones: the zone at a place, and the UTC offset of a local time in that zone."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones


def zone_at(lat: float, lon: float) -> str:
    """The IANA zone at a place, or "" when there is none. Open sea gives an Etc/GMT zone."""
    import tzfpy  # imported on first use: its boundary index stays in memory for the process

    return tzfpy.get_tz(lon, lat) or ""  # tzfpy takes the longitude first


def zones_at(lat: float, lon: float) -> list[str]:
    """Every zone at a place. A few regions have more than one."""
    import tzfpy

    return list(tzfpy.get_tzs(lon, lat))


def is_zone(name: str) -> bool:
    if not name:
        return False
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


@lru_cache(maxsize=1)
def zone_names() -> tuple[str, ...]:
    """All IANA zone names, sorted, for a picker."""
    return tuple(sorted(available_timezones()))


def capture_zone(override: str, lat: Optional[float], lon: Optional[float]) -> str:
    """The zone of a capture: the chosen zone, else the zone at the capture place, else ""."""
    if is_zone(override):
        return override
    if lat is None or lon is None:
        return ""
    return zone_at(lat, lon)


def resolve_local(naive: datetime, zone: str) -> datetime:
    """A local time in the zone. A repeated hour takes its first occurrence. A skipped hour moves
    forward by the length of the gap."""
    tz = ZoneInfo(zone)
    first = naive.replace(tzinfo=tz, fold=0)
    if first.utcoffset() == naive.replace(tzinfo=tz, fold=1).utcoffset():
        return first
    moved = first.astimezone(timezone.utc).astimezone(tz)
    return first if moved.replace(tzinfo=None) == naive else moved


def offset_text(delta: timedelta) -> str:
    """'+03:00' style text for EXIF OffsetTime tags."""
    minutes = round(delta.total_seconds() / 60)
    sign = "+" if minutes >= 0 else "-"
    return f"{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"
