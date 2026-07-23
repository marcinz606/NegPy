"""Persisted Roll Scanning panel settings (stored as a global setting dict)."""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    # This is a user-selected ceiling, not a target percentage. The pinned
    # runtime remains the final hard limit; the worker takes the lower value.
    hybrid_synthesis_limit_percent: float = 10.0

    def __post_init__(self) -> None:
        value = self.hybrid_synthesis_limit_percent
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 100.0
        ):
            raise ValueError("hybrid_synthesis_limit_percent must be finite and in [0, 100]")
        object.__setattr__(self, "hybrid_synthesis_limit_percent", float(value))

    @classmethod
    def defaults(cls) -> "RollScanSettings":
        return cls()
