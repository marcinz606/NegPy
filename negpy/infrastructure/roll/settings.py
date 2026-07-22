"""Persisted Roll Scanning panel settings (stored as a global setting dict)."""

from __future__ import annotations

from dataclasses import dataclass

from negpy.infrastructure.roll.repair import RepairMode
from negpy.services.roll.exact_color import PositiveColorMode


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
    scanner itself can reproduce. For a new user, repaired and positive also
    default on in Hybrid + Nikon-exact mode so the complete parity workflow
    runs while its frame-bound prepass, infrared-validity data, acquisition
    provenance, and native color-builder evidence are still available. Saved
    user choices override these first-run defaults. `repair_mode` governs Tier
    2 (and, through it, Tier 3) whenever a repair engine is registered; see
    `negpy.infrastructure.roll.repair`. `positive_mode` chooses the Tier-3
    color path. The roll-scanning parity workflow defaults to fail-closed Nikon
    exact color; NegPy's approximate renderer remains an explicit choice.
    """

    last_device_id: str = ""
    output_folder: str = ""
    filename_pattern: str = '{{ date }}_{{ "%03d" % seq }}'
    write_unrepaired: bool = True
    write_repaired: bool = True
    write_positive: bool = True
    repair_mode: str = RepairMode.HYBRID.value
    positive_mode: str = PositiveColorMode.NIKON_EXACT.value

    @classmethod
    def defaults(cls) -> "RollScanSettings":
        return cls()
