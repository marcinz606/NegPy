"""Tier-2 dust/scratch repair engine seam for roll-scanned RGBI frames.

The roll adapter writes a captured frame in up to three tiers (see
`negpy.services.roll.service.write_frame`): the unrepaired RGBI capture, the
same frame with infrared-guided dust/scratch repair applied, and a positive
rendered from the repaired frame. This module is the plug point for the
middle tier -- it does not repair anything itself.

This seam has no intrinsic implementation: `available()` stays false and
`repair()` raises until a bridge calls `register_engine()`. NegPy's
`fauxice_bridge` registers the installed portable Digital ICE runtime during
roll-service import. The explicit registration boundary keeps Tier 2 honest:
a build that cannot load the engine reports repair as unavailable rather than
writing a copy of Tier 1 under a repaired label. See
`docs/COOLSCANPY_ROLL_SCANNING.md` for the intended capture -> repair ->
invert ordering and which project is expected to fill this slot.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Callable, Protocol

import numpy as np

if TYPE_CHECKING:
    from negpy.services.repair.fauxice_hybrid_runner import HybridRuntimeConfig


DIGITAL_ICE_STORAGE_TRANSFORM = "rot90-k1-scanner-native-to-upright-v1"


class RepairCancelled(RuntimeError):
    """A caller-requested repair stop; never a degradable repair failure."""


def _raw_sha256(array: np.ndarray, *, dtype: np.dtype) -> str:
    canonical = np.array(array, dtype=dtype, order="C", copy=True)
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


class RepairMode(StrEnum):
    """A Tier-2 repair variant.

    EXACT heals only the pixels the infrared channel confidently flags as a
    defect. HYBRID additionally routes severe zero-signal regions (where the
    infrared channel carries no usable signal at all, e.g. a dense scratch)
    to a separately pinned inpainting runtime. Receipts bind the backend,
    resolved mode, output hashes, and hybrid disclosure evidence rather than
    making a blanket reproducibility claim across different runtimes.
    """

    EXACT = "exact"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class RepairAcquisition:
    """One hash-bound scanner-native prepass/main pair ready for repair."""

    acquisition_id: str
    slot: int
    reservation_id: str
    capture_attempt_id: str
    storage_transform: str
    evidence_sha256: str
    main_rgbi_sha256: str
    prepass_rgbi_sha256: str
    ir_validity_sha256: str
    main_rgbi: np.ndarray
    prepass_rgbi: np.ndarray
    ir_validity: np.ndarray

    def __post_init__(self) -> None:
        if re.fullmatch(r"dice-[0-9a-f]{64}", self.acquisition_id) is None:
            raise ValueError("repair acquisition identity is malformed")
        if type(self.slot) is not int or not 1 <= self.slot <= 40:
            raise ValueError("repair acquisition slot must be in 1..40")
        for label, value in (
            ("reservation identity", self.reservation_id),
            ("capture-attempt identity", self.capture_attempt_id),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"repair {label} must be non-empty")
        if self.storage_transform != DIGITAL_ICE_STORAGE_TRANSFORM:
            raise ValueError("repair acquisition storage transform is unsupported")
        for label, digest in (
            ("evidence", self.evidence_sha256),
            ("main RGBI", self.main_rgbi_sha256),
            ("prepass RGBI", self.prepass_rgbi_sha256),
            ("IR validity", self.ir_validity_sha256),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"repair acquisition {label} SHA-256 is malformed")

        main = np.array(self.main_rgbi, dtype="<u2", order="C", copy=True)
        prepass = np.array(self.prepass_rgbi, dtype="<u2", order="C", copy=True)
        validity = np.array(
            self.ir_validity,
            dtype=np.bool_,
            order="C",
            copy=True,
        )
        if main.ndim != 3 or main.shape[2] != 4:
            raise ValueError("repair main RGBI must be HxWx4 uint16")
        if prepass.ndim != 3 or prepass.shape[2] != 4:
            raise ValueError("repair prepass RGBI must be HxWx4 uint16")
        if validity.shape != main.shape[:2]:
            raise ValueError("repair IR validity must match the main geometry")
        if _raw_sha256(main, dtype=np.dtype("<u2")) != self.main_rgbi_sha256:
            raise ValueError("repair main RGBI changed after capture")
        if _raw_sha256(prepass, dtype=np.dtype("<u2")) != self.prepass_rgbi_sha256:
            raise ValueError("repair prepass RGBI changed after capture")
        if _raw_sha256(validity, dtype=np.dtype(np.bool_)) != self.ir_validity_sha256:
            raise ValueError("repair IR validity changed after capture")
        for array in (main, prepass, validity):
            array.setflags(write=False)
        object.__setattr__(self, "main_rgbi", main)
        object.__setattr__(self, "prepass_rgbi", prepass)
        object.__setattr__(self, "ir_validity", validity)

    @classmethod
    def from_arrays(
        cls,
        *,
        acquisition_id: str,
        slot: int,
        reservation_id: str,
        capture_attempt_id: str,
        storage_transform: str,
        evidence_sha256: str,
        main_rgbi: np.ndarray,
        prepass_rgbi: np.ndarray,
        ir_validity: np.ndarray,
    ) -> "RepairAcquisition":
        """Test/helper constructor; production passes producer-bound hashes."""

        return cls(
            acquisition_id=acquisition_id,
            slot=slot,
            reservation_id=reservation_id,
            capture_attempt_id=capture_attempt_id,
            storage_transform=storage_transform,
            evidence_sha256=evidence_sha256,
            main_rgbi_sha256=_raw_sha256(main_rgbi, dtype=np.dtype("<u2")),
            prepass_rgbi_sha256=_raw_sha256(prepass_rgbi, dtype=np.dtype("<u2")),
            ir_validity_sha256=_raw_sha256(ir_validity, dtype=np.dtype(np.bool_)),
            main_rgbi=main_rgbi,
            prepass_rgbi=prepass_rgbi,
            ir_validity=ir_validity,
        )

    def storage_rgb(self, scanner_native_rgb: np.ndarray) -> np.ndarray:
        native = np.asarray(scanner_native_rgb)
        if native.dtype != np.uint16 or native.shape != (*self.main_rgbi.shape[:2], 3):
            raise ValueError("repair output must match scanner-native main RGB geometry")
        return np.ascontiguousarray(np.rot90(native, k=1, axes=(0, 1)))

    def storage_mask(self, scanner_native_mask: np.ndarray) -> np.ndarray:
        native = np.asarray(scanner_native_mask)
        if native.dtype != np.bool_ or native.shape != self.main_rgbi.shape[:2]:
            raise ValueError("repair mask must match scanner-native main geometry")
        return np.ascontiguousarray(np.rot90(native, k=1, axes=(0, 1)))


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
    mode_requested: RepairMode
    mode_resolved: RepairMode
    reason: str
    acquisition_id: str | None = None
    slot: int | None = None
    reservation_id: str | None = None
    evidence_sha256: str | None = None
    backend_requested: str | None = None
    backend_used: str | None = None
    backend_selection_reason: str | None = None
    native_output_rgb_sha256: str | None = None
    storage_output_rgb_sha256: str | None = None
    native_synthesis_mask_png: bytes | None = None
    native_synthesis_mask_sha256: str | None = None
    native_synthesis_mask_shape: tuple[int, int] | None = None
    routed_native_synthesis_mask_png: bytes | None = None
    routed_native_synthesis_mask_sha256: str | None = None
    routed_native_synthesis_mask_shape: tuple[int, int] | None = None
    storage_synthesis_mask_png: bytes | None = None
    storage_synthesis_mask_sha256: str | None = None
    storage_synthesis_mask_shape: tuple[int, int] | None = None
    synthesis_mask_transform: str | None = None
    synthesis_fraction: float | None = None
    routing_counts: dict[str, int] | None = None
    hybrid_receipt: bytes | None = None
    hybrid_receipt_sha256: str | None = None
    hybrid_provenance_class: str | None = None
    hybrid_receipt_output_rgb_sha256: str | None = None

    @property
    def degraded(self) -> bool:
        return self.mode_requested is not self.mode_resolved


class RepairEngine(Protocol):
    """What a repair implementation must provide to `register_engine()`."""

    def repair(
        self,
        acquisition: RepairAcquisition,
        mode: RepairMode,
        *,
        hybrid_runtime: "HybridRuntimeConfig | None" = None,
        progress: Callable[[float], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> RepairResult: ...


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


def repair(
    acquisition: RepairAcquisition,
    mode: RepairMode,
    *,
    hybrid_runtime: "HybridRuntimeConfig | None" = None,
    progress: Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
) -> RepairResult:
    """Repair one frame's RGB using its infrared plane.

    Raises `RuntimeError` if no engine is registered -- callers on the
    roll-scanning write path check `available()` first so a missing engine
    degrades Tier 2/3 with a clear status instead of raising mid-batch.
    """
    if _engine is None:
        raise RuntimeError("no dust-repair engine is registered; Tier 2 and Tier 3 are unavailable")
    return _engine.repair(
        acquisition,
        mode,
        hybrid_runtime=hybrid_runtime,
        progress=progress,
        cancel=cancel,
    )
