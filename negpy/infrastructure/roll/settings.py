"""Persisted Roll Scanning panel settings (stored as a global setting dict)."""

from __future__ import annotations

from dataclasses import dataclass

from negpy.infrastructure.roll.repair import RepairMode


@dataclass(frozen=True)
class RollScanSettings:
    """Sticky settings for the Roll Scanning sidebar.

    Persisted via the session repo under the `roll_scan_settings` key,
    mirroring `ScannerSettings` and `ScanlightSettings`. `last_device_id`
    empty means no device remembered yet; per-slot spacing offsets and
    approvals are not persisted here at all -- they live on the open
    coolscanpy `Roll` for as long as it stays open, and `preview()` resets
    them on its own the moment it re-reads the transport.

    `write_unrepaired`, `write_repaired` and `write_positive` select which of
    the three output tiers a batch scan writes (see
    `negpy.services.roll.service.write_frame`); any combination is valid, and
    they are independent, not a single three-way choice. `write_unrepaired`
    defaults on because it is the archival master and the only tier the
    scanner itself can reproduce -- the other two default off since they are
    derived, regenerable, and (for `write_positive`) subject to a rendering
    pipeline that is still being tuned. `repair_mode` governs Tier 2 (and,
    through it, Tier 3) whenever a repair engine is registered; see
    `negpy.infrastructure.roll.repair`.
    """

    last_device_id: str = ""
    output_folder: str = ""
    filename_pattern: str = '{{ date }}_{{ "%03d" % seq }}'
    write_unrepaired: bool = True
    write_repaired: bool = False
    write_positive: bool = False
    repair_mode: str = RepairMode.EXACT.value

    @classmethod
    def defaults(cls) -> "RollScanSettings":
        return cls()
