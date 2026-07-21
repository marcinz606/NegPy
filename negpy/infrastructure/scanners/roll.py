"""Whole-strip preview, independent of how a transport delivers it.

Two shapes exist in the wild and this seam covers both: a scanner that scans one
frame at a time (every SANE film scanner — see PerFrameRollSession) and one that
reads the entire strip in a single traversal and slices it afterwards (a Nikon
LS-5000 over raw USB). Preview yields per slot either way; only latency differs.

Fine scanning deliberately stays on ScanWorker.run_batch for now: per-frame crop
windows have no counterpart on a whole-roll transport, so the scan half of this
seam can't be settled until a native implementation exists to settle it against.
"""

import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from negpy.infrastructure.scanners.base import ScannerCapabilities


def effective_pitch_mm(caps: ScannerCapabilities) -> float:
    """Feed-axis distance between frame positions, falling back to the frame's long
    side when the transport does not report one. 0.0 when neither is known."""
    if caps.frame_pitch_mm:
        return caps.frame_pitch_mm
    area = caps.max_area_mm
    return area[1] if area and len(area) > 1 else 0.0


@dataclass(frozen=True)
class RollPreview:
    """One slot's preview. Carries its own slot number so results may arrive in any order."""

    slot: int
    rgb: np.ndarray | None = None
    # Set instead of rgb when this slot alone failed. The strip continues — one bad
    # frame mid-roll must not cost the user the other 39.
    error: str | None = None
    # Effective feed-axis offset the preview was taken at, as a fraction of one frame
    # pitch, after the transport's own clamping. The raster covers [offset, 1.0] of the
    # frame, so the viewer places it there rather than stretching it over the whole tile.
    offset: float = 0.0
    # The transport inferred this slot's boundary rather than measuring it; the user
    # must confirm before it can be scanned. Always False on a per-frame transport.
    needs_approval: bool = False
    warnings: tuple[str, ...] = ()


class RollSession(Protocol):
    """An open strip: preview its slots, nudge their boundaries, release it.

    `close()` is terminal and idempotent. A session does not own file writing.
    """

    slot_count: int
    # Legal `set_offset` bounds as a fraction of one frame pitch. A transport that
    # re-addresses its traversal table can go negative; one that offsets within a
    # frame cannot (the scan blacks out at the boundary — verified on an LS-50).
    offset_range: tuple[float, float]
    # False when previewing costs a whole-strip traversal, so per-slot preview
    # buttons are pointless and the UI should offer Preview all only.
    supports_single_slot_preview: bool

    def preview(self, slots: Iterable[int], *, cancel: threading.Event) -> Iterator[RollPreview]:
        """Yield one RollPreview per slot, as each becomes available."""
        ...

    def set_offset(self, slot: int, offset: float) -> None:
        """Move a slot's frame boundary, as a fraction of one pitch.

        Values the transport cannot reach are clamped; what it actually used comes
        back on RollPreview.offset.
        """
        ...

    def approve(self, slot: int) -> None:
        """Confirm an inferred boundary so the slot may be scanned. No-op where unneeded."""
        ...

    def close(self) -> None: ...
