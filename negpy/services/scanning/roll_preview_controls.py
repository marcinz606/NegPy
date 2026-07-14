"""Pure policy shared by roll-preview UI and scanner orchestration.

This module intentionally has no Qt, USB, or SANE imports.  A controller can
therefore validate persisted preview choices without importing the desktop UI.
"""

from enum import StrEnum


class ScanMaterial(StrEnum):
    """Film material choices whose capture recipes differ at the scanner."""

    COLOR_NEGATIVE = "Color negative (C-41)"
    BLACK_AND_WHITE_NEGATIVE = "Black-and-white negative"

    @property
    def captures_infrared(self) -> bool:
        """Whether the capture includes IR for dust-repair input."""

        return self is ScanMaterial.COLOR_NEGATIVE


def boundary_offset_range(slot_id: int) -> tuple[int, int]:
    """Return the inclusive transport-boundary range for a physical roll slot.

    The value addresses a different origin in the scanner's traversal table;
    it is not a pixel crop applied to an existing thumbnail.
    """

    if type(slot_id) is not int or slot_id < 1:
        raise ValueError("slot_id must be a positive 1-based integer")
    return (0, 144) if slot_id == 1 else (-144, 144)


def validate_boundary_offset(slot_id: int, boundary_offset: int) -> int:
    """Validate and return one slot's integer transport-boundary offset."""

    minimum, maximum = boundary_offset_range(slot_id)
    if type(boundary_offset) is not int:
        raise ValueError("boundary_offset must be an integer")
    if not minimum <= boundary_offset <= maximum:
        raise ValueError(f"boundary_offset for slot {slot_id} must be between {minimum} and {maximum}")
    return boundary_offset
