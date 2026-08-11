"""Persisted Roll Scanning panel settings (stored as a global setting dict)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RollScanSettings:
    """Sticky settings for the Roll Scanning sidebar.

    Persisted via the session repo under the `roll_scan_settings` key,
    mirroring `ScannerSettings` and `ScanlightSettings`. `last_device_id`
    empty means no device remembered yet; per-slot spacing offsets and
    approvals are not persisted here at all -- they live on the open
    coolscanpy `Roll` for as long as it stays open, and `preview()` resets
    them on its own the moment it re-reads the transport.
    """

    last_device_id: str = ""
    output_folder: str = ""
    filename_pattern: str = '{{ date }}_{{ "%03d" % seq }}'

    @classmethod
    def defaults(cls) -> "RollScanSettings":
        return cls()
