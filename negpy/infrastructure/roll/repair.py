"""Tier-2 dust/scratch repair engine seam for roll-scanned RGBI frames.

The roll adapter writes a captured frame in up to three tiers (see
`negpy.services.roll.service.write_frame`): the unrepaired RGBI capture, the
same frame with infrared-guided dust/scratch repair applied, and a positive
rendered from the repaired frame. This module is the plug point for the
middle tier -- it does not repair anything itself.

No repair engine ships in this package. `available()` returns False and
`repair()` raises until something calls `register_engine()` with a real
implementation. That keeps Tier 2 honest: a frame this module cannot
actually repair is reported as unavailable rather than written to disk as
"repaired" when it is really just a copy of Tier 1. See
`docs/COOLSCANPY_ROLL_SCANNING.md` for the intended capture -> repair ->
invert ordering and which project is expected to fill this slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np


class RepairMode(StrEnum):
    """A Tier-2 repair variant.

    EXACT heals only the pixels the infrared channel confidently flags as a
    defect. HYBRID additionally routes severe zero-signal regions (where the
    infrared channel carries no usable signal at all, e.g. a dense scratch)
    to an inpainting model. Both are deterministic -- the same frame and mode
    repair to the same result every time -- which is what makes Tier 2 worth
    caching: unlike Tier 3, it never needs to be redone once it has run.
    """

    EXACT = "exact"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class RepairResult:
    """One frame's Tier-2 RGB, plus what the receipt needs to audit or
    reproduce it. The infrared plane is not part of this result -- Tier 2
    retains Tier 1's own infrared plane unchanged (see `service.write_frame`),
    since repair consumes infrared to find defects rather than producing a
    repaired version of it."""

    rgb: np.ndarray
    engine: str
    engine_version: str
    mode: RepairMode


class RepairEngine(Protocol):
    """What a repair implementation must provide to `register_engine()`."""

    def repair(self, rgb: np.ndarray, ir: np.ndarray, mode: RepairMode) -> RepairResult: ...


_engine: RepairEngine | None = None


def available() -> bool:
    """True once a repair engine has been registered.

    Cheap and side-effect-free, mirroring `coolscanpy_roll.available()`, so
    `RollScanningService.write_frame` can decide whether Tier 2 (and, since
    Tier 3 is defined as Tier 2 inverted, Tier 3 too) can be produced before
    doing any work.
    """
    return _engine is not None


def register_engine(engine: RepairEngine) -> None:
    """The integration point a future repair implementation is expected to
    call -- e.g. at import time, guarded the same way `coolscanpy_roll`
    guards its own optional dependency -- to make Tier 2 and Tier 3
    available. Nothing in this codebase calls this today; see the module
    docstring."""
    global _engine
    _engine = engine


def unregister_engine() -> None:
    """Reverts to the unavailable state. Mainly a test teardown hook."""
    global _engine
    _engine = None


def repair(rgb: np.ndarray, ir: np.ndarray, mode: RepairMode) -> RepairResult:
    """Repair one frame's RGB using its infrared plane.

    Raises `RuntimeError` if no engine is registered -- callers on the
    roll-scanning write path check `available()` first so a missing engine
    degrades Tier 2/3 with a clear status instead of raising mid-batch.
    """
    if _engine is None:
        raise RuntimeError("no dust-repair engine is registered; Tier 2 and Tier 3 are unavailable")
    return _engine.repair(rgb, ir, mode)
