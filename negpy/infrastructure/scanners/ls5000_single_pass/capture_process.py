"""Process-isolated bridge to NegPy's proven LS-5000 RGBI4x capture worker.

The package owns and integrity-pins the USB worker. This adapter gives the
application a narrow, fail-closed process boundary around it:

* every launch uses an argv list (never a shell command),
* both the worker and bundled replay plan are hash-pinned before launch,
* preview attempts expose the scanner's addressable candidates without a
  count hint, while meter attempts capture one explicit scanner slot,
* selected full frames share one child/reservation and cross a durable
  frame-ready/parent-ACK boundary before the next frame starts, and
* a stop request is observed only at a safe frame boundary.  An active child
  is never signalled or killed by this adapter.

The worker's ``--preview-only`` operation persists the roll preview and 0x8e
table before any frame binding.  Exposure-count labels are not part of this
interface: the adapter never emits ``--expected-frame-count``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec
from pathlib import Path
from statistics import median
from typing import Any, Protocol, Sequence

import numpy as np

from .bundle import (
    CANONICAL_MANIFEST_FILENAME,
    CAPTURE_BUNDLE_SHA256,
    CAPTURE_WORKER_SHA256,
    CaptureBundleIntegrityError,
    canonical_manifest_bytes,
    verify_capture_bundle,
)
from .continuation_plan import (
    CANONICAL_CONTINUATION_PLAN_FILENAME,
    CANONICAL_CONTINUATION_PLAN_SHA256,
    canonical_continuation_plan_bytes,
)
from .plan import (
    CANONICAL_FINE_READ_BYTES,
    CANONICAL_FINE_READ_COUNT,
    CANONICAL_PLAN_FILENAME,
    CANONICAL_PLAN_SHA256,
    canonical_plan_bytes,
)


METER_READ_COUNT = 15
METER_CAPTURE_BYTES = 3_264_000
POWER_CYCLE_RECOVERY = "power-cycle scanner before another attempt"
CAPTURE_HELPER_FLAG = "--ls5000-capture-helper"
PACKAGED_WORKER_MODULE = (
    "negpy.infrastructure.scanners.ls5000_single_pass.worker"
)
REVIEWED_ROLL_FINGERPRINT_VERSION = 2
MANUAL_FRAME_APPROVAL_VERSION = 1
MAX_VISUAL_MEDIAN_HAMMING = 24
MAX_VISUAL_P90_HAMMING = 48
MAX_SELECTED_VISUAL_HAMMING = 48
MIN_VISUAL_LOG_SPAN = 0.5
MIN_DISCRIMINATIVE_FRAME_COUNT = 3
MAX_FRAME_START_MEDIAN_DELTA_ROWS = 16
MAX_FRAME_START_DELTA_ROWS = 32
MAX_NATIVE_ORIGIN_MEDIAN_DELTA = 128
MAX_NATIVE_ORIGIN_DELTA = 256
MAX_PREVIEW_HEIGHT_DELTA_ROWS = 32


def _lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CaptureMode(StrEnum):
    """The safe operations currently exposed by the capture worker."""

    PREVIEW = "preview"
    METER_ONLY = "meter-only"
    FULL = "full"


class CaptureOutcome(StrEnum):
    """Application-level interpretation of a completed child process."""

    COMPLETE = "complete"
    SYNCHRONIZED_REFUSAL = "synchronized-refusal"
    RECOVERY_REQUIRED = "recovery-required"


class BatchAckAction(StrEnum):
    """Parent decision written only after one frame is durably finalized."""

    CONTINUE = "continue"
    STOP = "stop"


class CaptureProcessError(RuntimeError):
    """The adapter refused before a trustworthy worker result was available."""


class CaptureBatchProcessError(CaptureProcessError):
    """A spawned batch failed after preserving its partial frame/release state."""

    def __init__(
        self,
        message: str,
        *,
        outcome: CaptureOutcome,
        paths: BatchSessionPaths,
        frames: Sequence[CaptureAttemptResult],
        returncode: int,
        session_journal: dict[str, Any] | None,
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.paths = paths
        self.frames = tuple(frames)
        self.returncode = returncode
        self.session_journal = session_journal

    @property
    def recovery_required(self) -> bool:
        return self.outcome is CaptureOutcome.RECOVERY_REQUIRED


class CaptureIntegrityError(CaptureProcessError):
    """A pinned executable, plan, or manifest failed verification."""


class CaptureStopped(CaptureProcessError):
    """A stop request prevented the next attempt from launching."""


class _BatchFrameRefused(CaptureProcessError):
    """A bad frame handoff was safely answered with a bound STOP ACK."""

    def __init__(self, message: str, *, slot: int) -> None:
        super().__init__(message)
        self.slot = slot


class _BatchTerminalReceiptObserved(Exception):
    """The child published a terminal session receipt before ``poll()`` noticed."""


@dataclass(frozen=True)
class ReviewedRollFingerprint:
    """Exact reviewed artifacts plus a reread-tolerant physical-roll signature."""

    source_preview_sha256: str
    source_table_sha256: str
    preview_shape: tuple[int, int, int]
    frame_start_rows: tuple[int, ...]
    frame_native_origins: tuple[int, ...]
    frame_visual_hashes: tuple[str, ...]
    frame_visual_log_spans: tuple[float, ...]
    schema_version: int = REVIEWED_ROLL_FINGERPRINT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REVIEWED_ROLL_FINGERPRINT_VERSION:
            raise ValueError("reviewed roll fingerprint schema is unsupported")
        if not _lower_sha256(self.source_preview_sha256) or not _lower_sha256(
            self.source_table_sha256
        ):
            raise ValueError("reviewed roll source identities must be SHA-256 digests")
        if (
            not isinstance(self.preview_shape, tuple)
            or len(self.preview_shape) != 3
            or any(type(value) is not int or value < 1 for value in self.preview_shape)
            or self.preview_shape[2] != 3
        ):
            raise ValueError("reviewed roll preview shape must be a positive HxWx3 tuple")
        count = len(self.frame_start_rows)
        if not 1 <= count <= 40:
            raise ValueError("reviewed roll fingerprint must describe 1..40 slots")
        if (
            len(self.frame_native_origins) != count
            or len(self.frame_visual_hashes) != count
            or len(self.frame_visual_log_spans) != count
        ):
            raise ValueError("reviewed roll fingerprint frame fields must have equal length")
        if any(type(value) is not int or value < 0 for value in self.frame_start_rows):
            raise ValueError("reviewed roll frame starts must be nonnegative integers")
        if any(
            first >= second
            for first, second in zip(self.frame_start_rows, self.frame_start_rows[1:])
        ):
            raise ValueError("reviewed roll frame starts must be strictly increasing")
        if any(type(value) is not int or value < 0 for value in self.frame_native_origins):
            raise ValueError("reviewed roll native origins must be nonnegative integers")
        if any(
            first >= second
            for first, second in zip(
                self.frame_native_origins,
                self.frame_native_origins[1:],
            )
        ):
            raise ValueError("reviewed roll native origins must be strictly increasing")
        if any(not _lower_sha256(value) for value in self.frame_visual_hashes):
            raise ValueError("reviewed roll visual hashes must be SHA-256-sized hex values")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in self.frame_visual_log_spans
        ):
            raise ValueError(
                "reviewed roll visual log spans must be finite nonnegative numbers"
            )

    def binding_payload(self) -> dict[str, Any]:
        return {
            "frame_native_origins": list(self.frame_native_origins),
            "frame_start_rows": list(self.frame_start_rows),
            "frame_visual_hashes": list(self.frame_visual_hashes),
            "frame_visual_log_spans": list(self.frame_visual_log_spans),
            "preview_shape": list(self.preview_shape),
            "schema_version": self.schema_version,
            "source_preview_sha256": self.source_preview_sha256,
            "source_table_sha256": self.source_table_sha256,
        }

    @property
    def binding_sha256(self) -> str:
        return _canonical_json_sha256(self.binding_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.binding_payload(),
            "binding_sha256": self.binding_sha256,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ReviewedRollFingerprint:
        if not isinstance(payload, dict) or set(payload) != {
            "binding_sha256",
            "frame_native_origins",
            "frame_start_rows",
            "frame_visual_hashes",
            "frame_visual_log_spans",
            "preview_shape",
            "schema_version",
            "source_preview_sha256",
            "source_table_sha256",
        }:
            raise ValueError("reviewed roll fingerprint payload is malformed")
        shape = payload.get("preview_shape")
        starts = payload.get("frame_start_rows")
        origins = payload.get("frame_native_origins")
        visual = payload.get("frame_visual_hashes")
        spans = payload.get("frame_visual_log_spans")
        if not all(
            isinstance(value, list)
            for value in (shape, starts, origins, visual, spans)
        ):
            raise ValueError("reviewed roll fingerprint arrays are malformed")
        value = cls(
            source_preview_sha256=payload.get("source_preview_sha256"),  # type: ignore[arg-type]
            source_table_sha256=payload.get("source_table_sha256"),  # type: ignore[arg-type]
            preview_shape=tuple(shape),  # type: ignore[arg-type]
            frame_start_rows=tuple(starts),  # type: ignore[arg-type]
            frame_native_origins=tuple(origins),  # type: ignore[arg-type]
            frame_visual_hashes=tuple(visual),  # type: ignore[arg-type]
            frame_visual_log_spans=tuple(spans),  # type: ignore[arg-type]
            schema_version=payload.get("schema_version"),  # type: ignore[arg-type]
        )
        if payload.get("binding_sha256") != value.binding_sha256:
            raise ValueError("reviewed roll fingerprint binding digest changed")
        return value


@dataclass(frozen=True)
class ManualFrameApproval:
    """Operator approval bound to one reviewed thumbnail and transport choice."""

    reviewed_fingerprint_sha256: str
    slot: int
    boundary_offset_rows: int
    thumbnail_sha256: str
    reviewed_lookup_row: int
    reviewed_native_origin: int
    review_reasons: tuple[str, ...]
    schema_version: int = MANUAL_FRAME_APPROVAL_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANUAL_FRAME_APPROVAL_VERSION:
            raise ValueError("manual frame approval schema is unsupported")
        if not _lower_sha256(self.reviewed_fingerprint_sha256):
            raise ValueError("manual frame approval fingerprint identity is invalid")
        if type(self.slot) is not int or not 1 <= self.slot <= 40:
            raise ValueError("manual frame approval slot must be in 1..40")
        if type(self.boundary_offset_rows) is not int:
            raise TypeError("manual frame approval boundary offset must be an integer")
        minimum_offset = 0 if self.slot == 1 else -144
        if not minimum_offset <= self.boundary_offset_rows <= 144:
            raise ValueError("manual frame approval boundary offset is outside 97-dpi limits")
        if not _lower_sha256(self.thumbnail_sha256):
            raise ValueError("manual frame approval thumbnail identity is invalid")
        if type(self.reviewed_lookup_row) is not int or self.reviewed_lookup_row < 0:
            raise ValueError("manual frame approval lookup row is invalid")
        if type(self.reviewed_native_origin) is not int or self.reviewed_native_origin < 0:
            raise ValueError("manual frame approval native origin is invalid")
        if (
            not isinstance(self.review_reasons, tuple)
            or not self.review_reasons
            or any(type(value) is not str or not value for value in self.review_reasons)
        ):
            raise ValueError("manual frame approval requires explicit review reasons")

    def binding_payload(self) -> dict[str, Any]:
        return {
            "boundary_offset_rows": self.boundary_offset_rows,
            "review_reasons": list(self.review_reasons),
            "reviewed_fingerprint_sha256": self.reviewed_fingerprint_sha256,
            "reviewed_lookup_row": self.reviewed_lookup_row,
            "reviewed_native_origin": self.reviewed_native_origin,
            "schema_version": self.schema_version,
            "slot": self.slot,
            "thumbnail_sha256": self.thumbnail_sha256,
        }

    @property
    def binding_sha256(self) -> str:
        return _canonical_json_sha256(self.binding_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.binding_payload(),
            "binding_sha256": self.binding_sha256,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ManualFrameApproval:
        if not isinstance(payload, dict) or set(payload) != {
            "binding_sha256",
            "boundary_offset_rows",
            "review_reasons",
            "reviewed_fingerprint_sha256",
            "reviewed_lookup_row",
            "reviewed_native_origin",
            "schema_version",
            "slot",
            "thumbnail_sha256",
        }:
            raise ValueError("manual frame approval payload is malformed")
        reasons = payload.get("review_reasons")
        if not isinstance(reasons, list):
            raise ValueError("manual frame approval reasons are malformed")
        value = cls(
            reviewed_fingerprint_sha256=payload.get("reviewed_fingerprint_sha256"),  # type: ignore[arg-type]
            slot=payload.get("slot"),  # type: ignore[arg-type]
            boundary_offset_rows=payload.get("boundary_offset_rows"),  # type: ignore[arg-type]
            thumbnail_sha256=payload.get("thumbnail_sha256"),  # type: ignore[arg-type]
            reviewed_lookup_row=payload.get("reviewed_lookup_row"),  # type: ignore[arg-type]
            reviewed_native_origin=payload.get("reviewed_native_origin"),  # type: ignore[arg-type]
            review_reasons=tuple(reasons),
            schema_version=payload.get("schema_version"),  # type: ignore[arg-type]
        )
        if payload.get("binding_sha256") != value.binding_sha256:
            raise ValueError("manual frame approval binding digest changed")
        return value


@dataclass(frozen=True)
class RollFingerprintComparison:
    """Auditable result of comparing a fresh traversal to reviewed evidence."""

    matches: bool
    reason: str
    compared_frames: int
    preview_height_delta_rows: int | None
    visual_median_hamming: float | None
    visual_p90_hamming: int | None
    frame_start_median_delta_rows: float | None
    frame_start_max_delta_rows: int | None
    native_origin_median_delta: float | None
    native_origin_max_delta: int | None
    discriminative_frames: int
    minimum_discriminative_frames: int
    minimum_visual_log_span: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "compared_frames": self.compared_frames,
            "discriminative_frames": self.discriminative_frames,
            "frame_start_max_delta_rows": self.frame_start_max_delta_rows,
            "frame_start_median_delta_rows": self.frame_start_median_delta_rows,
            "matches": self.matches,
            "minimum_discriminative_frames": self.minimum_discriminative_frames,
            "minimum_visual_log_span": self.minimum_visual_log_span,
            "native_origin_max_delta": self.native_origin_max_delta,
            "native_origin_median_delta": self.native_origin_median_delta,
            "preview_height_delta_rows": self.preview_height_delta_rows,
            "reason": self.reason,
            "visual_median_hamming": self.visual_median_hamming,
            "visual_p90_hamming": self.visual_p90_hamming,
        }


@dataclass(frozen=True)
class SelectedRollFingerprintComparison:
    """A selected slot's own visual proof, independent of roll aggregates."""

    matches: bool
    reason: str
    slot: int
    visual_hamming: int
    maximum_visual_hamming: int
    reviewed_visual_log_span: float
    fresh_visual_log_span: float
    minimum_visual_log_span: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "fresh_visual_log_span": self.fresh_visual_log_span,
            "matches": self.matches,
            "maximum_visual_hamming": self.maximum_visual_hamming,
            "minimum_visual_log_span": self.minimum_visual_log_span,
            "reason": self.reason,
            "reviewed_visual_log_span": self.reviewed_visual_log_span,
            "slot": self.slot,
            "visual_hamming": self.visual_hamming,
        }


def _roll_frame_visual_signature(frame: np.ndarray) -> tuple[str, float]:
    """Return a robust hash and calibrated information span for one slot."""

    array = np.asarray(frame)
    if (
        array.dtype != np.uint16
        or array.ndim != 3
        or array.shape[2] != 3
        or array.shape[0] < 16
        or array.shape[1] < 16
    ):
        raise ValueError("roll fingerprint frames must be uint16 HxWx3 images")
    luminance = np.log1p(array.astype(np.float64)).mean(axis=2)
    y_edges = np.linspace(0, len(luminance), 17, dtype=np.int64)
    x_edges = np.linspace(0, luminance.shape[1], 17, dtype=np.int64)
    grid = np.empty((16, 16), dtype=np.float64)
    for row in range(16):
        for column in range(16):
            block = luminance[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ]
            if block.size == 0:
                raise ValueError("roll fingerprint frame is too small for its grid")
            grid[row, column] = float(block.mean())
    bits = np.packbits(
        (grid > float(np.median(grid))).reshape(-1),
        bitorder="big",
    )
    low, high = np.percentile(grid, (5.0, 95.0))
    return bits.tobytes().hex(), float(high - low)


def _roll_frame_visual_hash(frame: np.ndarray) -> str:
    """Return the gain/noise-tolerant 256-bit part of a slot signature."""

    return _roll_frame_visual_signature(frame)[0]


def build_reviewed_roll_fingerprint(
    rgb: np.ndarray,
    *,
    frame_intervals: Sequence[tuple[int, int]],
    frame_native_origins: Sequence[int],
    source_preview_sha256: str,
    source_table_sha256: str,
) -> ReviewedRollFingerprint:
    """Bind exact preview artifacts to a robust per-slot physical signature."""

    array = np.asarray(rgb)
    intervals = tuple(frame_intervals)
    origins = tuple(frame_native_origins)
    if len(intervals) != len(origins):
        raise ValueError("roll fingerprint intervals and native origins differ")
    starts: list[int] = []
    visual_hashes: list[str] = []
    visual_log_spans: list[float] = []
    for slot, interval in enumerate(intervals, start=1):
        if (
            not isinstance(interval, tuple)
            or len(interval) != 2
            or any(type(value) is not int for value in interval)
        ):
            raise TypeError(f"roll fingerprint interval {slot} is malformed")
        start, end = interval
        if not 0 <= start < end <= len(array):
            raise ValueError(f"roll fingerprint interval {slot} is outside the preview")
        starts.append(start)
        visual_hash, visual_log_span = _roll_frame_visual_signature(array[start:end])
        visual_hashes.append(visual_hash)
        visual_log_spans.append(visual_log_span)
    return ReviewedRollFingerprint(
        source_preview_sha256=source_preview_sha256,
        source_table_sha256=source_table_sha256,
        preview_shape=tuple(int(value) for value in array.shape),
        frame_start_rows=tuple(starts),
        frame_native_origins=origins,
        frame_visual_hashes=tuple(visual_hashes),
        frame_visual_log_spans=tuple(visual_log_spans),
    )


def compare_reviewed_roll_fingerprints(
    reviewed: ReviewedRollFingerprint,
    fresh: ReviewedRollFingerprint,
) -> RollFingerprintComparison:
    """Compare two traversals without requiring analog bytes to be identical."""

    if not isinstance(reviewed, ReviewedRollFingerprint) or not isinstance(
        fresh, ReviewedRollFingerprint
    ):
        raise TypeError("roll fingerprint comparison requires reviewed fingerprints")

    def incompatible(reason: str) -> RollFingerprintComparison:
        return RollFingerprintComparison(
            matches=False,
            reason=reason,
            compared_frames=0,
            preview_height_delta_rows=None,
            visual_median_hamming=None,
            visual_p90_hamming=None,
            frame_start_median_delta_rows=None,
            frame_start_max_delta_rows=None,
            native_origin_median_delta=None,
            native_origin_max_delta=None,
            discriminative_frames=0,
            minimum_discriminative_frames=min(
                MIN_DISCRIMINATIVE_FRAME_COUNT,
                len(reviewed.frame_visual_hashes),
            ),
            minimum_visual_log_span=MIN_VISUAL_LOG_SPAN,
        )

    if reviewed.preview_shape[1:] != fresh.preview_shape[1:]:
        return incompatible("preview-geometry-mismatch")
    preview_height_delta = abs(reviewed.preview_shape[0] - fresh.preview_shape[0])
    if preview_height_delta > MAX_PREVIEW_HEIGHT_DELTA_ROWS:
        return incompatible("preview-height-mismatch")
    if len(reviewed.frame_visual_hashes) != len(fresh.frame_visual_hashes):
        return incompatible("slot-count-mismatch")

    visual = sorted(
        (int(expected, 16) ^ int(observed, 16)).bit_count()
        for expected, observed, expected_span, observed_span in zip(
            reviewed.frame_visual_hashes,
            fresh.frame_visual_hashes,
            reviewed.frame_visual_log_spans,
            fresh.frame_visual_log_spans,
            strict=True,
        )
        if expected_span >= MIN_VISUAL_LOG_SPAN
        and observed_span >= MIN_VISUAL_LOG_SPAN
    )
    discriminative_frames = len(visual)
    minimum_discriminative_frames = min(
        MIN_DISCRIMINATIVE_FRAME_COUNT,
        len(reviewed.frame_visual_hashes),
    )

    starts = sorted(
        abs(expected - observed)
        for expected, observed in zip(
            reviewed.frame_start_rows,
            fresh.frame_start_rows,
            strict=True,
        )
    )
    origins = sorted(
        abs(expected - observed)
        for expected, observed in zip(
            reviewed.frame_native_origins,
            fresh.frame_native_origins,
            strict=True,
        )
    )
    visual_median = float(median(visual)) if visual else None
    visual_p90 = (
        visual[max(0, (9 * len(visual) + 9) // 10 - 1)]
        if visual
        else None
    )
    start_median = float(median(starts))
    start_max = starts[-1]
    origin_median = float(median(origins))
    origin_max = origins[-1]
    if discriminative_frames < minimum_discriminative_frames:
        reason = "visual-signature-indeterminate"
    elif visual_median is None or visual_p90 is None:
        reason = "visual-signature-indeterminate"
    elif (
        visual_median > MAX_VISUAL_MEDIAN_HAMMING
        or visual_p90 > MAX_VISUAL_P90_HAMMING
    ):
        reason = "visual-content-mismatch"
    elif (
        start_median > MAX_FRAME_START_MEDIAN_DELTA_ROWS
        or start_max > MAX_FRAME_START_DELTA_ROWS
    ):
        reason = "frame-lattice-mismatch"
    elif (
        origin_median > MAX_NATIVE_ORIGIN_MEDIAN_DELTA
        or origin_max > MAX_NATIVE_ORIGIN_DELTA
    ):
        reason = "transport-origin-mismatch"
    else:
        reason = "matched"
    return RollFingerprintComparison(
        matches=reason == "matched",
        reason=reason,
        compared_frames=len(visual),
        preview_height_delta_rows=preview_height_delta,
        visual_median_hamming=visual_median,
        visual_p90_hamming=visual_p90,
        frame_start_median_delta_rows=start_median,
        frame_start_max_delta_rows=start_max,
        native_origin_median_delta=origin_median,
        native_origin_max_delta=origin_max,
        discriminative_frames=discriminative_frames,
        minimum_discriminative_frames=minimum_discriminative_frames,
        minimum_visual_log_span=MIN_VISUAL_LOG_SPAN,
    )


def compare_selected_roll_fingerprint(
    reviewed: ReviewedRollFingerprint,
    fresh: ReviewedRollFingerprint,
    *,
    slot: int,
) -> SelectedRollFingerprintComparison:
    """Require a selected slot to match even when roll aggregates tolerate it."""

    if not isinstance(reviewed, ReviewedRollFingerprint) or not isinstance(
        fresh,
        ReviewedRollFingerprint,
    ):
        raise TypeError("selected fingerprint comparison requires reviewed fingerprints")
    if type(slot) is not int or not 1 <= slot <= 40:
        raise ValueError("selected fingerprint slot must be in 1..40")
    if len(reviewed.frame_visual_hashes) != len(fresh.frame_visual_hashes):
        raise ValueError("selected fingerprint slot counts differ")
    if slot > len(reviewed.frame_visual_hashes):
        raise ValueError("selected fingerprint slot is outside the reviewed roll")
    index = slot - 1
    visual_hamming = (
        int(reviewed.frame_visual_hashes[index], 16)
        ^ int(fresh.frame_visual_hashes[index], 16)
    ).bit_count()
    reviewed_span = float(reviewed.frame_visual_log_spans[index])
    fresh_span = float(fresh.frame_visual_log_spans[index])
    if reviewed_span < MIN_VISUAL_LOG_SPAN or fresh_span < MIN_VISUAL_LOG_SPAN:
        reason = "selected-visual-signature-indeterminate"
    elif visual_hamming > MAX_SELECTED_VISUAL_HAMMING:
        reason = "selected-visual-content-mismatch"
    else:
        reason = "matched"
    return SelectedRollFingerprintComparison(
        matches=reason == "matched",
        reason=reason,
        slot=slot,
        visual_hamming=visual_hamming,
        maximum_visual_hamming=MAX_SELECTED_VISUAL_HAMMING,
        reviewed_visual_log_span=reviewed_span,
        fresh_visual_log_span=fresh_span,
        minimum_visual_log_span=MIN_VISUAL_LOG_SPAN,
    )


@dataclass(frozen=True)
class CaptureRequest:
    """One explicit scanner-addressable slot and capture mode.

    Roll exposure counts intentionally do not appear here.  A 36-exposure roll
    can contain a 37th image, and the preview UI may show blank or unusable tail
    slots up to the scanner's capacity.  Selection policy belongs above this
    process boundary.
    """

    mode: CaptureMode
    selected_slot: int | None = None
    boundary_offset_rows: int = 0
    manual_review_approval: ManualFrameApproval | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CaptureMode):
            raise TypeError("mode must be a CaptureMode")
        if isinstance(self.boundary_offset_rows, bool) or not isinstance(
            self.boundary_offset_rows, int
        ):
            raise TypeError("boundary offset must be an integer row count")
        if self.mode is CaptureMode.PREVIEW:
            if self.selected_slot is not None:
                raise ValueError("preview-only requests do not select a scanner slot")
            if self.boundary_offset_rows != 0:
                raise ValueError("preview boundary offset must be zero")
            if self.manual_review_approval is not None:
                raise ValueError("preview requests cannot carry a manual review approval")
            return
        if isinstance(self.selected_slot, bool) or not isinstance(self.selected_slot, int) or not 1 <= self.selected_slot <= 40:
            raise ValueError("selected scanner slot must be an integer in 1..40")
        minimum_offset = 0 if self.selected_slot == 1 else -144
        if not minimum_offset <= self.boundary_offset_rows <= 144:
            raise ValueError(
                f"slot {self.selected_slot} boundary offset must be in "
                f"{minimum_offset}..144 rows"
            )
        approval = self.manual_review_approval
        if approval is not None:
            if not isinstance(approval, ManualFrameApproval):
                raise TypeError("manual review approval has the wrong type")
            if (
                approval.slot != self.selected_slot
                or approval.boundary_offset_rows != self.boundary_offset_rows
            ):
                raise ValueError("manual review approval does not match the requested frame")


@dataclass(frozen=True)
class CaptureBatchRequest:
    """One ordered set of full frames for a future shared USB reservation."""

    frames: tuple[CaptureRequest, ...]
    reviewed_fingerprint: ReviewedRollFingerprint

    def __post_init__(self) -> None:
        if not isinstance(self.frames, tuple):
            raise TypeError("batch frames must be an immutable tuple")
        if not self.frames:
            raise ValueError("batch must contain at least one frame")
        if not all(isinstance(frame, CaptureRequest) for frame in self.frames):
            raise TypeError("every batch frame must be a CaptureRequest")
        if any(frame.mode is not CaptureMode.FULL for frame in self.frames):
            raise ValueError("batch sessions currently support full captures only")
        if not isinstance(self.reviewed_fingerprint, ReviewedRollFingerprint):
            raise TypeError("batch requires one reviewed roll fingerprint")
        for frame in self.frames:
            approval = frame.manual_review_approval
            if (
                approval is not None
                and approval.reviewed_fingerprint_sha256
                != self.reviewed_fingerprint.binding_sha256
            ):
                raise ValueError("manual review approval belongs to another roll preview")
        slots = self.selected_slots
        if tuple(sorted(set(slots))) != slots:
            raise ValueError(
                "batch scanner slots must be unique and strictly increasing"
            )

    @property
    def selected_slots(self) -> tuple[int, ...]:
        """Return statically validated scanner slots without optional typing."""

        return tuple(
            frame.selected_slot
            for frame in self.frames
            if frame.selected_slot is not None
        )


@dataclass(frozen=True)
class AttemptPaths:
    """All durable paths owned by one never-overwritten worker attempt."""

    directory: Path
    output: Path
    journal: Path
    plan: Path
    manifest: Path
    stdout: Path
    stderr: Path


@dataclass(frozen=True)
class BatchSessionPaths:
    """Durable, never-overwritten inputs for one batch child."""

    directory: Path
    job: Path
    first_plan: Path
    continuation_plan: Path
    manifest: Path
    session_journal: Path
    stdout: Path
    stderr: Path


@dataclass(frozen=True)
class PreparedCaptureBatch:
    """Hardware-free description of exactly one worker process."""

    request: CaptureBatchRequest
    paths: BatchSessionPaths
    argv: tuple[str, ...]
    job_sha256: str
    session_id: str


@dataclass(frozen=True)
class CaptureAttemptResult:
    """Validated child result, including a conservative recovery decision."""

    outcome: CaptureOutcome
    request: CaptureRequest
    paths: AttemptPaths
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    journal: dict[str, Any] | None
    journal_error: str | None = None
    batch_session_id: str | None = None
    batch_frame_index: int | None = None
    batch_frame_total: int | None = None
    batch_selected_slots: tuple[int, ...] = ()

    @property
    def recovery_required(self) -> bool:
        return self.outcome is CaptureOutcome.RECOVERY_REQUIRED


@dataclass(frozen=True)
class CaptureBatchResult:
    """One child process, its finalized frames, and final release receipt."""

    outcome: CaptureOutcome
    request: CaptureBatchRequest
    paths: BatchSessionPaths
    frames: tuple[CaptureAttemptResult, ...]
    returncode: int
    stopped: bool
    session_journal: dict[str, Any]
    stdout: str
    stderr: str


class ProcessRunner(Protocol):
    """Injectable child runner; tests can implement this without hardware."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]: ...


class RunningBatchProcess(Protocol):
    """The non-signalled subset of ``subprocess.Popen`` used by the adapter."""

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class BatchProcessSpawner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout: Any,
        stderr: Any,
    ) -> RunningBatchProcess: ...


class BatchFrameHandler(Protocol):
    def __call__(self, result: CaptureAttemptResult) -> BatchAckAction: ...


def _run_subprocess(
    argv: Sequence[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    """Run the worker in its own process group and collect its text output.

    ``start_new_session`` keeps an application-level SIGINT/SIGTERM from being
    forwarded to the scanner child.  The parent may record a stop request, but
    this function waits for the current worker attempt to finish naturally.
    """

    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        start_new_session=True,
    )


def _spawn_batch_subprocess(
    argv: Sequence[str],
    *,
    cwd: Path,
    stdout: Any,
    stderr: Any,
) -> RunningBatchProcess:
    """Start one isolated child without exposing any signal/kill seam."""

    return subprocess.Popen(
        list(argv),
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        shell=False,
        start_new_session=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_exclusive(path: Path, payload: bytes) -> None:
    """Atomically publish a complete never-overwritten handshake file."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_exclusive(temporary, payload)
    try:
        # Linking a fully flushed inode makes the final name visible in one
        # step and, unlike replace(), refuses a stale/replayed destination.
        os.link(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            # The ACK is already atomically visible and the live child may
            # consume it immediately.  Reporting failure after publication
            # could allow an unobserved next frame, so durability sync is
            # best-effort at this point.
            pass
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not path.exists():
                raise


def _is_sha256(value: object) -> bool:
    return _lower_sha256(value)


class CaptureProcessAdapter:
    """Launch one hash-pinned worker process per scanner attempt.

    Calls are serialized because there is one physical transport.  Calling
    :meth:`request_stop` while a child is active only marks the adapter; the
    active attempt finishes, and the next attempt raises :class:`CaptureStopped`.
    """

    def __init__(
        self,
        *,
        worker_path: Path,
        attempts_root: Path,
        expected_worker_sha256: str,
        manifest_path: Path | None = None,
        python_executable: str = sys.executable,
        runner: ProcessRunner = _run_subprocess,
        launcher: Sequence[str] | None = None,
        verify_worker_source: bool = True,
        expected_bundle_sha256: str | None = None,
        manifest_payload: bytes | None = None,
        batch_spawner: BatchProcessSpawner = _spawn_batch_subprocess,
        batch_poll_seconds: float = 0.1,
    ) -> None:
        if not _is_sha256(expected_worker_sha256):
            raise ValueError("expected worker SHA-256 is not a lowercase digest")
        self._worker_path = Path(worker_path).expanduser().resolve()
        self._attempts_root = Path(attempts_root).expanduser().resolve()
        self._manifest_path = (
            self._worker_path.with_name("replay-first-rgbi4-manifest.json")
            if manifest_path is None
            else Path(manifest_path).expanduser().resolve()
        )
        self._python_executable = str(python_executable)
        self._launcher = (
            (self._python_executable, str(self._worker_path))
            if launcher is None
            else tuple(str(item) for item in launcher)
        )
        if not self._launcher:
            raise ValueError("capture worker launcher cannot be empty")
        self._verify_worker_source = bool(verify_worker_source)
        self._expected_worker_sha256 = expected_worker_sha256
        if expected_bundle_sha256 is not None and not _is_sha256(
            expected_bundle_sha256
        ):
            raise ValueError("expected capture bundle SHA-256 is not a lowercase digest")
        self._expected_bundle_sha256 = expected_bundle_sha256
        self._manifest_payload = (
            None if manifest_payload is None else bytes(manifest_payload)
        )
        self._runner = runner
        if batch_poll_seconds < 0:
            raise ValueError("batch_poll_seconds cannot be negative")
        self._batch_spawner = batch_spawner
        self._batch_poll_seconds = float(batch_poll_seconds)
        self._stop_requested = threading.Event()
        self._stop_gate = threading.Lock()
        self._attempt_lock = threading.Lock()

    @classmethod
    def packaged(
        cls,
        attempts_root: Path,
        *,
        runner: ProcessRunner = _run_subprocess,
    ) -> CaptureProcessAdapter:
        """Bind the adapter to NegPy's verified package-owned worker bundle.

        Source checkouts and installed wheels launch an isolated ``python -m``
        child and hash every scanner-facing source first.  Frozen builds
        relaunch the signed app executable with an internal helper flag; the
        helper dispatch occurs before desktop initialization.
        """

        frozen = bool(getattr(sys, "frozen", False))
        try:
            verify_capture_bundle(require_python_sources=not frozen)
            manifest_payload = canonical_manifest_bytes()
        except (CaptureBundleIntegrityError, OSError, ValueError) as error:
            raise CaptureIntegrityError(
                f"packaged capture bundle failed validation: {error}"
            ) from error
        spec = find_spec(PACKAGED_WORKER_MODULE)
        if spec is None or spec.origin is None:
            raise CaptureIntegrityError("packaged capture worker module is missing")
        worker_path = Path(spec.origin).resolve()
        launcher = (
            (sys.executable, CAPTURE_HELPER_FLAG)
            if frozen
            else (sys.executable, "-I", "-m", PACKAGED_WORKER_MODULE)
        )
        return cls(
            worker_path=worker_path,
            attempts_root=attempts_root,
            expected_worker_sha256=CAPTURE_WORKER_SHA256,
            manifest_path=worker_path.with_name(CANONICAL_MANIFEST_FILENAME),
            runner=runner,
            launcher=launcher,
            verify_worker_source=not frozen,
            expected_bundle_sha256=CAPTURE_BUNDLE_SHA256,
            manifest_payload=manifest_payload,
        )

    def request_stop(self) -> None:
        """Stop before the next attempt without touching an active child."""

        with self._stop_gate:
            self._stop_requested.set()

    def clear_stop(self) -> None:
        """Allow launches again after the caller has acknowledged a stop."""

        with self._stop_gate:
            self._stop_requested.clear()

    def prepare_batch_session(
        self,
        request: CaptureBatchRequest,
    ) -> PreparedCaptureBatch:
        """Materialize one child/session contract without launching hardware."""

        if not isinstance(request, CaptureBatchRequest):
            raise TypeError("request must be a CaptureBatchRequest")
        with self._attempt_lock:
            if self._stop_requested.is_set():
                raise CaptureStopped(
                    "capture stopped between sessions; no batch was prepared"
                )
            return self._prepare_batch_session_locked(request)

    def _prepare_batch_session_locked(
        self,
        request: CaptureBatchRequest,
    ) -> PreparedCaptureBatch:
        paths = self._prepare_batch_session_paths(request)
        self._verify_worker()
        self._materialize_pinned_plan(paths.first_plan)
        self._materialize_pinned_continuation_plan(paths.continuation_plan)
        self._materialize_pinned_manifest(paths.manifest)
        session_id = paths.directory.name
        payload = self._batch_job_bytes(request, session_id=session_id)
        _write_exclusive(paths.job, payload)
        argv = self._build_batch_argv(paths)
        return PreparedCaptureBatch(
            request=request,
            paths=paths,
            argv=argv,
            job_sha256=hashlib.sha256(payload).hexdigest(),
            session_id=session_id,
        )

    def run_batch_session(
        self,
        request: CaptureBatchRequest,
        *,
        frame_handler: BatchFrameHandler,
    ) -> CaptureBatchResult:
        """Run one child and ACK only frames the parent finished consuming.

        ``frame_handler`` is the parent-owned finalization boundary.  It must
        validate, decode, promote, and optionally delete the scratch stream
        before returning.  Only then is an ACK written, which is the child's
        permission to begin the next frame.  A stop request changes that ACK
        to ``stop``; it never signals the active scanner child.
        """

        if not isinstance(request, CaptureBatchRequest):
            raise TypeError("request must be a CaptureBatchRequest")
        if not callable(frame_handler):
            raise TypeError("frame_handler must be callable")
        with self._attempt_lock:
            if self._stop_requested.is_set():
                raise CaptureStopped(
                    "capture stopped between sessions; no batch was launched"
                )
            prepared = self._prepare_batch_session_locked(request)
            if self._stop_requested.is_set():
                raise CaptureStopped("capture stopped before batch worker launch")
            return self._run_prepared_batch(prepared, frame_handler)

    def _run_prepared_batch(
        self,
        prepared: PreparedCaptureBatch,
        frame_handler: BatchFrameHandler,
    ) -> CaptureBatchResult:
        paths = prepared.paths
        handled: list[CaptureAttemptResult] = []
        stopped = False
        stopped_unhandled_slot: int | None = None
        handler_error: BaseException | None = None
        monitor_error: BaseException | None = None
        process: RunningBatchProcess | None = None
        returncode: int | None = None
        try:
            with paths.stdout.open("xb") as stdout_handle, paths.stderr.open(
                "xb"
            ) as stderr_handle:
                # Linearize the final stop check with process creation.  If a
                # stop wins this lock no child launches; if Popen wins, that
                # stop applies at the first durable frame boundary.
                with self._stop_gate:
                    if self._stop_requested.is_set():
                        raise CaptureStopped(
                            "capture stopped before batch worker launch"
                        )
                    process = self._batch_spawner(
                        prepared.argv,
                        cwd=paths.directory,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                    )
                try:
                    for frame_index, frame_request in enumerate(
                        prepared.request.frames,
                        start=1,
                    ):
                        frame_result = self._wait_for_batch_frame(
                            process,
                            prepared,
                            frame_request,
                            frame_index=frame_index,
                        )
                        try:
                            action = frame_handler(frame_result)
                            if not isinstance(action, BatchAckAction):
                                raise TypeError(
                                    "frame_handler must return BatchAckAction"
                                )
                        except BaseException as error:
                            handler_error = error
                            action = BatchAckAction.STOP
                        handled.append(frame_result)
                        # Linearize Stop against publishing CONTINUE.  A stop
                        # that wins this lock changes the current boundary to
                        # STOP; one arriving after publication applies to the
                        # newly active frame instead.
                        with self._stop_gate:
                            if self._stop_requested.is_set():
                                action = BatchAckAction.STOP
                            self._write_batch_ack(
                                frame_result,
                                prepared,
                                action=action,
                            )
                        if action is BatchAckAction.STOP:
                            stopped = True
                            break
                except _BatchTerminalReceiptObserved:
                    # A terminal journal is only a wake-up here.  The parent
                    # still joins the child below, then validates the complete
                    # cleanup/release receipt before returning anything to the
                    # UI.  This avoids depending exclusively on Popen.poll(),
                    # which can lag behind an already-exited frozen helper.
                    pass
                except BaseException as error:
                    monitor_error = error
                    if isinstance(error, _BatchFrameRefused):
                        stopped = True
                        stopped_unhandled_slot = error.slot
                finally:
                    # Never signal or abandon a child that may own the USB
                    # reservation.  With no valid ACK it will time out, clean
                    # up, and release; the parent must remain here to observe
                    # that receipt.
                    returncode, wait_error = self._wait_for_batch_exit(process)
                    if monitor_error is None:
                        monitor_error = wait_error
        except OSError as error:
            if process is None:
                raise CaptureProcessError(
                    f"could not launch batch capture worker: {error}"
                ) from error
            monitor_error = monitor_error or error

        if process is None or returncode is None:
            raise CaptureProcessError(
                "batch capture worker did not leave a completed child process"
            ) from monitor_error

        try:
            session_journal = self._load_and_validate_batch_session_journal(
                prepared,
                returncode=returncode,
                handled=handled,
                stopped=stopped,
                stopped_unhandled_slot=stopped_unhandled_slot,
            )
        except BaseException as error:
            cause = error if monitor_error is None else monitor_error
            raise CaptureBatchProcessError(
                "batch child finished without a trustworthy release receipt: "
                f"{error}",
                outcome=CaptureOutcome.RECOVERY_REQUIRED,
                paths=paths,
                frames=handled,
                returncode=returncode,
                session_journal=None,
            ) from cause
        receipt_outcome = (
            CaptureOutcome.SYNCHRONIZED_REFUSAL
            if session_journal.get("recovery_required") == "none"
            else CaptureOutcome.RECOVERY_REQUIRED
        )
        outcome = (
            CaptureOutcome.COMPLETE if returncode == 0 else receipt_outcome
        )
        try:
            stdout = paths.stdout.read_text(encoding="utf-8", errors="replace")
            stderr = paths.stderr.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            stdout = ""
            stderr = ""
            if monitor_error is None:
                monitor_error = error
        if monitor_error is not None:
            if isinstance(monitor_error, (KeyboardInterrupt, SystemExit)):
                raise monitor_error
            raise CaptureBatchProcessError(
                "batch adapter failed after launch; child cleanup and release "
                f"completed before returning: {monitor_error}",
                outcome=receipt_outcome,
                paths=paths,
                frames=handled,
                returncode=returncode,
                session_journal=session_journal,
            ) from monitor_error
        if handler_error is not None:
            raise CaptureBatchProcessError(
                f"batch frame finalization failed after safe stop: {handler_error}",
                outcome=receipt_outcome,
                paths=paths,
                frames=handled,
                returncode=returncode,
                session_journal=session_journal,
            ) from handler_error
        return CaptureBatchResult(
            outcome=outcome,
            request=prepared.request,
            paths=paths,
            frames=tuple(handled),
            returncode=returncode,
            stopped=stopped,
            session_journal=session_journal,
            stdout=stdout,
            stderr=stderr,
        )

    def _wait_for_batch_exit(
        self,
        process: RunningBatchProcess,
    ) -> tuple[int, BaseException | None]:
        """Defer interruption until the scanner child has cleaned up."""

        deferred: BaseException | None = None
        while True:
            try:
                return process.wait(), deferred
            except BaseException as error:
                if deferred is None:
                    deferred = error
                try:
                    returncode = process.poll()
                except BaseException:
                    returncode = None
                if returncode is not None:
                    return returncode, deferred
                if self._batch_poll_seconds:
                    time.sleep(self._batch_poll_seconds)

    def run_attempt(self, request: CaptureRequest) -> CaptureAttemptResult:
        """Run and validate one worker attempt synchronously."""

        if not isinstance(request, CaptureRequest):
            raise TypeError("request must be a CaptureRequest")
        with self._attempt_lock:
            if self._stop_requested.is_set():
                raise CaptureStopped("capture stopped between attempts; no worker was launched")
            paths = self._prepare_attempt_paths(request)
            self._verify_worker()
            self._materialize_pinned_plan(paths.plan)
            self._materialize_pinned_manifest(paths.manifest)
            argv = self._build_argv(request, paths)
            # This is deliberately the final stop check.  Once runner() starts,
            # request_stop() cannot interrupt or signal the active worker.
            if self._stop_requested.is_set():
                raise CaptureStopped("capture stopped before worker launch")
            try:
                completed = self._runner(argv, cwd=paths.directory)
            except OSError as error:
                raise CaptureProcessError(f"could not launch capture worker: {error}") from error

            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            _write_exclusive(paths.stdout, stdout.encode("utf-8", errors="replace"))
            _write_exclusive(paths.stderr, stderr.encode("utf-8", errors="replace"))
            return self._interpret_result(
                request=request,
                paths=paths,
                argv=argv,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
            )

    def _prepare_attempt_paths(self, request: CaptureRequest) -> AttemptPaths:
        self._attempts_root.mkdir(parents=True, exist_ok=True)
        slot_suffix = "" if request.selected_slot is None else f"-slot{request.selected_slot:02d}"
        prefix = f"{request.mode.value}{slot_suffix}-"
        directory = Path(tempfile.mkdtemp(prefix=prefix, dir=self._attempts_root)).resolve()
        return AttemptPaths(
            directory=directory,
            output=directory / "capture.bin",
            journal=directory / "journal.json",
            plan=directory / CANONICAL_PLAN_FILENAME,
            manifest=directory / CANONICAL_MANIFEST_FILENAME,
            stdout=directory / "stdout.txt",
            stderr=directory / "stderr.txt",
        )

    def _prepare_batch_session_paths(
        self,
        request: CaptureBatchRequest,
    ) -> BatchSessionPaths:
        self._attempts_root.mkdir(parents=True, exist_ok=True)
        first, last = request.selected_slots[0], request.selected_slots[-1]
        prefix = f"batch-slot{first:02d}-slot{last:02d}-"
        directory = Path(
            tempfile.mkdtemp(prefix=prefix, dir=self._attempts_root)
        ).resolve()
        return BatchSessionPaths(
            directory=directory,
            job=directory / "batch-job.json",
            first_plan=directory / CANONICAL_PLAN_FILENAME,
            continuation_plan=(
                directory / CANONICAL_CONTINUATION_PLAN_FILENAME
            ),
            manifest=directory / CANONICAL_MANIFEST_FILENAME,
            session_journal=directory / "session-journal.json",
            stdout=directory / "stdout.txt",
            stderr=directory / "stderr.txt",
        )

    def _verify_worker(self) -> None:
        if self._expected_bundle_sha256 is not None:
            try:
                actual_bundle = verify_capture_bundle(
                    require_python_sources=self._verify_worker_source
                )
            except (CaptureBundleIntegrityError, OSError, ValueError) as error:
                raise CaptureIntegrityError(
                    f"capture bundle failed validation before launch: {error}"
                ) from error
            if actual_bundle != self._expected_bundle_sha256:
                raise CaptureIntegrityError(
                    "capture bundle identity changed before launch"
                )
        if not self._verify_worker_source:
            return
        if not self._worker_path.is_file():
            raise CaptureIntegrityError(f"capture worker is not a regular file: {self._worker_path}")
        actual = _sha256_file(self._worker_path)
        if actual != self._expected_worker_sha256:
            raise CaptureIntegrityError(f"capture worker SHA-256 mismatch: expected {self._expected_worker_sha256}, got {actual}")

    def _materialize_pinned_plan(self, destination: Path) -> None:
        try:
            payload = canonical_plan_bytes()
        except (OSError, ValueError) as error:
            raise CaptureIntegrityError(f"bundled capture plan failed validation: {error}") from error
        if hashlib.sha256(payload).hexdigest() != CANONICAL_PLAN_SHA256:
            raise CaptureIntegrityError("bundled capture plan SHA-256 changed after validation")
        _write_exclusive(destination, payload)
        actual = _sha256_file(destination)
        if actual != CANONICAL_PLAN_SHA256:
            raise CaptureIntegrityError(f"materialized plan SHA-256 mismatch: {actual}")

    def _materialize_pinned_continuation_plan(self, destination: Path) -> None:
        try:
            payload = canonical_continuation_plan_bytes()
        except (OSError, ValueError) as error:
            raise CaptureIntegrityError(
                f"bundled continuation plan failed validation: {error}"
            ) from error
        if hashlib.sha256(payload).hexdigest() != CANONICAL_CONTINUATION_PLAN_SHA256:
            raise CaptureIntegrityError(
                "bundled continuation plan SHA-256 changed after validation"
            )
        _write_exclusive(destination, payload)
        actual = _sha256_file(destination)
        if actual != CANONICAL_CONTINUATION_PLAN_SHA256:
            raise CaptureIntegrityError(
                f"materialized continuation plan SHA-256 mismatch: {actual}"
            )

    def _materialize_pinned_manifest(self, destination: Path) -> None:
        try:
            payload = (
                self._manifest_path.read_bytes()
                if self._manifest_payload is None
                else self._manifest_payload
            )
            manifest = json.loads(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CaptureIntegrityError(f"capture manifest could not be read: {error}") from error
        if not isinstance(manifest, dict):
            raise CaptureIntegrityError("capture manifest must be a JSON object")
        if manifest.get("plan_sha256") != CANONICAL_PLAN_SHA256:
            raise CaptureIntegrityError("capture manifest is not bound to the packaged canonical plan")
        _write_exclusive(destination, payload)

    def _build_argv(self, request: CaptureRequest, paths: AttemptPaths) -> tuple[str, ...]:
        argv = [
            *self._launcher,
            "--plan",
            str(paths.plan),
            "--manifest",
            str(paths.manifest),
            "--output",
            str(paths.output),
            "--journal",
            str(paths.journal),
            "--boundary-offset-rows",
            str(request.boundary_offset_rows),
            "--live",
        ]
        if request.mode is CaptureMode.PREVIEW:
            argv.append("--preview-only")
        elif request.mode is CaptureMode.METER_ONLY:
            argv.extend(("--frame", str(request.selected_slot), "--meter-only"))
        else:
            argv.extend(
                (
                    "--frame",
                    str(request.selected_slot),
                    "--reads",
                    str(CANONICAL_FINE_READ_COUNT),
                    "--confirm-full-capture",
                )
            )
        if "--expected-frame-count" in argv:
            raise AssertionError("exposure-count hints must never cross the capture boundary")
        return tuple(argv)

    def _batch_job_bytes(
        self,
        request: CaptureBatchRequest,
        *,
        session_id: str,
    ) -> bytes:
        frames = [
            {
                "ack": f"frame-{frame.selected_slot:03d}/parent-ack.json",
                "boundary_offset_rows": frame.boundary_offset_rows,
                "journal": f"frame-{frame.selected_slot:03d}/journal.json",
                "manual_review_approval": (
                    None
                    if frame.manual_review_approval is None
                    else frame.manual_review_approval.to_payload()
                ),
                "output": f"frame-{frame.selected_slot:03d}/capture.bin",
                "slot": frame.selected_slot,
            }
            for frame in request.frames
        ]
        job = {
            "apply_all_boundary_offsets_before_first_frame": True,
            "capture_plan_sha256": CANONICAL_PLAN_SHA256,
            "continuation_plan_sha256": CANONICAL_CONTINUATION_PLAN_SHA256,
            "frames": frames,
            "parent_ack_required_after_every_frame": True,
            "release_once_after_last_frame": True,
            "reviewed_roll_fingerprint": request.reviewed_fingerprint.to_payload(),
            "schema_version": 2,
            "session_id": session_id,
            "session_contract": "one-process-one-reservation",
        }
        return (
            json.dumps(job, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")

    def _build_batch_argv(self, paths: BatchSessionPaths) -> tuple[str, ...]:
        return (
            *self._launcher,
            "--batch-job",
            str(paths.job),
            "--plan",
            str(paths.first_plan),
            "--continuation-plan",
            str(paths.continuation_plan),
            "--manifest",
            str(paths.manifest),
            "--session-journal",
            str(paths.session_journal),
            "--live",
        )

    def _batch_frame_paths(
        self,
        prepared: PreparedCaptureBatch,
        request: CaptureRequest,
    ) -> AttemptPaths:
        slot = request.selected_slot
        if slot is None:
            raise AssertionError("validated batch frame has no selected slot")
        directory = prepared.paths.directory / f"frame-{slot:03d}"
        return AttemptPaths(
            directory=directory,
            output=directory / "capture.bin",
            journal=directory / "journal.json",
            plan=prepared.paths.first_plan,
            manifest=prepared.paths.manifest,
            stdout=prepared.paths.stdout,
            stderr=prepared.paths.stderr,
        )

    def _wait_for_batch_frame(
        self,
        process: RunningBatchProcess,
        prepared: PreparedCaptureBatch,
        request: CaptureRequest,
        *,
        frame_index: int,
    ) -> CaptureAttemptResult:
        paths = self._batch_frame_paths(prepared, request)
        while True:
            if paths.journal.is_file():
                try:
                    payload = json.loads(paths.journal.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    payload = None
                if isinstance(payload, dict) and payload.get("status") == "frame-complete":
                    try:
                        return self._validate_batch_frame_result(
                            prepared,
                            request,
                            paths,
                            payload,
                            frame_index=frame_index,
                        )
                    except Exception as error:
                        self._write_identifiable_batch_stop(
                            prepared,
                            request,
                            paths,
                            payload,
                            frame_index=frame_index,
                        )
                        slot = request.selected_slot
                        if slot is None:
                            raise AssertionError(
                                "validated batch request lost its slot"
                            )
                        raise _BatchFrameRefused(
                            f"batch frame {frame_index} was refused and safely "
                            f"stopped: {error}",
                            slot=slot,
                        ) from error
            if self._terminal_batch_receipt_is_published(prepared):
                raise _BatchTerminalReceiptObserved
            returncode = process.poll()
            if returncode is not None:
                raise CaptureProcessError(
                    f"batch worker exited {returncode} before frame "
                    f"{frame_index} became ready for parent finalization"
                )
            if self._batch_poll_seconds:
                time.sleep(self._batch_poll_seconds)

    @staticmethod
    def _terminal_batch_receipt_is_published(
        prepared: PreparedCaptureBatch,
    ) -> bool:
        """Return whether this exact child published a terminal failure notice.

        This deliberately does not accept the receipt as valid or release the
        caller.  It only moves the monitor to its mandatory child-join path;
        ``_load_and_validate_batch_session_journal`` remains the authority for
        cleanup, release, recovery, and provenance after the child is reaped.
        """

        try:
            payload = json.loads(
                prepared.paths.session_journal.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return bool(
            isinstance(payload, dict)
            and payload.get("status") in ("failed", "interrupted")
            and payload.get("session_id") == prepared.session_id
            and payload.get("batch_job_sha256") == prepared.job_sha256
        )

    def _validate_batch_frame_result(
        self,
        prepared: PreparedCaptureBatch,
        request: CaptureRequest,
        paths: AttemptPaths,
        payload: dict[str, Any],
        *,
        frame_index: int,
    ) -> CaptureAttemptResult:
        expected_bytes = CANONICAL_FINE_READ_COUNT * CANONICAL_FINE_READ_BYTES
        selected_slots = prepared.request.selected_slots
        expected_batch = {
            "frame_index": frame_index,
            "frame_total": len(selected_slots),
            "selected_slots": list(selected_slots),
            "session_id": prepared.session_id,
        }
        invariants: dict[str, object] = {
            "status": "frame-complete",
            "frame_complete": True,
            "session_reservation_retained": True,
            "unit_released": False,
            "batch_session": expected_batch,
            "capture_mode": "full",
            "requested_frame": request.selected_slot,
            "requested_boundary_offset_rows": request.boundary_offset_rows,
            "expected_reads": CANONICAL_FINE_READ_COUNT,
            "completed_reads": CANONICAL_FINE_READ_COUNT,
            "expected_bytes": expected_bytes,
            "completed_bytes": expected_bytes,
            "disk_bytes": expected_bytes,
            "output": str(paths.output.resolve()),
            "plan_sha256": CANONICAL_PLAN_SHA256,
            "continuation_plan_sha256": CANONICAL_CONTINUATION_PLAN_SHA256,
            "capture_engine_sha256": self._expected_worker_sha256,
            "manual_review_approval": (
                None
                if request.manual_review_approval is None
                else request.manual_review_approval.to_payload()
            ),
            "reviewed_roll_fingerprint_sha256": (
                prepared.request.reviewed_fingerprint.binding_sha256
            ),
        }
        if self._expected_bundle_sha256 is not None:
            invariants["capture_bundle_sha256"] = self._expected_bundle_sha256
        for key, expected in invariants.items():
            if payload.get(key) != expected:
                raise CaptureProcessError(
                    f"batch frame {frame_index} journal {key}="
                    f"{payload.get(key)!r}, expected {expected!r}"
                )
        selection = payload.get("live_frame_selection")
        if not isinstance(selection, dict) or selection.get("frame") != request.selected_slot:
            raise CaptureProcessError(
                f"batch frame {frame_index} live frame selection is malformed"
            )
        roll_identity = selection.get("roll_identity")
        if not isinstance(roll_identity, dict) or set(roll_identity) != {
            "comparison",
            "fresh_fingerprint_sha256",
            "reviewed_fingerprint_sha256",
            "selected_slot_comparison",
        }:
            raise CaptureProcessError(
                f"batch frame {frame_index} roll identity evidence is malformed"
            )
        reviewed_sha = prepared.request.reviewed_fingerprint.binding_sha256
        if roll_identity.get("reviewed_fingerprint_sha256") != reviewed_sha:
            raise CaptureProcessError(
                f"batch frame {frame_index} reviewed roll identity changed"
            )
        if not _lower_sha256(roll_identity.get("fresh_fingerprint_sha256")):
            raise CaptureProcessError(
                f"batch frame {frame_index} fresh roll identity is malformed"
            )
        comparison = roll_identity.get("comparison")
        if (
            not isinstance(comparison, dict)
            or comparison.get("matches") is not True
            or comparison.get("reason") != "matched"
            or type(comparison.get("compared_frames")) is not int
            or comparison["compared_frames"] < 1
            or type(comparison.get("discriminative_frames")) is not int
            or type(comparison.get("minimum_discriminative_frames")) is not int
            or not 1 <= comparison["compared_frames"] <= 40
            or comparison["minimum_discriminative_frames"]
            != min(
                MIN_DISCRIMINATIVE_FRAME_COUNT,
                comparison["compared_frames"],
            )
            or not 1
            <= comparison["minimum_discriminative_frames"]
            <= comparison["discriminative_frames"]
            <= comparison["compared_frames"]
            or comparison.get("minimum_visual_log_span") != MIN_VISUAL_LOG_SPAN
            or not isinstance(comparison.get("visual_median_hamming"), (int, float))
            or comparison["visual_median_hamming"] > MAX_VISUAL_MEDIAN_HAMMING
            or type(comparison.get("visual_p90_hamming")) is not int
            or comparison["visual_p90_hamming"] > MAX_VISUAL_P90_HAMMING
            or not isinstance(
                comparison.get("frame_start_median_delta_rows"),
                (int, float),
            )
            or comparison["frame_start_median_delta_rows"]
            > MAX_FRAME_START_MEDIAN_DELTA_ROWS
            or type(comparison.get("frame_start_max_delta_rows")) is not int
            or comparison["frame_start_max_delta_rows"]
            > MAX_FRAME_START_DELTA_ROWS
            or not isinstance(
                comparison.get("native_origin_median_delta"),
                (int, float),
            )
            or comparison["native_origin_median_delta"]
            > MAX_NATIVE_ORIGIN_MEDIAN_DELTA
            or type(comparison.get("native_origin_max_delta")) is not int
            or comparison["native_origin_max_delta"] > MAX_NATIVE_ORIGIN_DELTA
            or type(comparison.get("preview_height_delta_rows")) is not int
            or comparison["preview_height_delta_rows"]
            > MAX_PREVIEW_HEIGHT_DELTA_ROWS
        ):
            raise CaptureProcessError(
                f"batch frame {frame_index} roll fingerprint comparison did not match"
            )
        selected_comparison = roll_identity.get("selected_slot_comparison")
        if (
            not isinstance(selected_comparison, dict)
            or selected_comparison.get("matches") is not True
            or selected_comparison.get("reason") != "matched"
            or selected_comparison.get("slot") != request.selected_slot
            or type(selected_comparison.get("visual_hamming")) is not int
            or type(selected_comparison.get("maximum_visual_hamming")) is not int
            or selected_comparison["maximum_visual_hamming"]
            != MAX_SELECTED_VISUAL_HAMMING
            or not 0
            <= selected_comparison["visual_hamming"]
            <= selected_comparison["maximum_visual_hamming"]
            or not all(
                isinstance(selected_comparison.get(key), (int, float))
                and not isinstance(selected_comparison.get(key), bool)
                and math.isfinite(float(selected_comparison[key]))
                for key in (
                    "fresh_visual_log_span",
                    "minimum_visual_log_span",
                    "reviewed_visual_log_span",
                )
            )
            or selected_comparison["fresh_visual_log_span"]
            < selected_comparison["minimum_visual_log_span"]
            or selected_comparison["minimum_visual_log_span"]
            != MIN_VISUAL_LOG_SPAN
            or selected_comparison["reviewed_visual_log_span"]
            < selected_comparison["minimum_visual_log_span"]
        ):
            raise CaptureProcessError(
                f"batch frame {frame_index} selected-slot roll fingerprint did not match"
            )
        if payload.get("recovery_required") not in (None, "none"):
            raise CaptureProcessError(
                f"batch frame {frame_index} requested scanner recovery"
            )
        nonce = payload.get("ack_nonce")
        if (
            not isinstance(nonce, str)
            or not 1 <= len(nonce) <= 128
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in nonce)
        ):
            raise CaptureProcessError(
                f"batch frame {frame_index} has no valid parent ACK nonce"
            )
        if not _is_sha256(payload.get("output_sha256")):
            raise CaptureProcessError(
                f"batch frame {frame_index} output SHA-256 is malformed"
            )
        if not paths.output.is_file() or paths.output.stat().st_size != expected_bytes:
            raise CaptureProcessError(
                f"batch frame {frame_index} output is missing or incomplete"
            )
        return CaptureAttemptResult(
            outcome=CaptureOutcome.COMPLETE,
            request=request,
            paths=paths,
            argv=prepared.argv,
            returncode=0,
            stdout="",
            stderr="",
            journal=payload,
            batch_session_id=prepared.session_id,
            batch_frame_index=frame_index,
            batch_frame_total=len(selected_slots),
            batch_selected_slots=selected_slots,
        )

    def _write_batch_ack(
        self,
        result: CaptureAttemptResult,
        prepared: PreparedCaptureBatch,
        *,
        action: BatchAckAction,
    ) -> None:
        if result.journal is None:
            raise CaptureProcessError("cannot ACK a batch frame without a journal")
        slot = result.request.selected_slot
        if slot is None or result.batch_frame_index is None:
            raise CaptureProcessError("cannot ACK a batch frame without identity")
        ack_path = result.paths.directory / "parent-ack.json"
        payload = {
            "ack_nonce": result.journal["ack_nonce"],
            "action": action.value,
            "frame_index": result.batch_frame_index,
            "schema_version": 1,
            "session_id": prepared.session_id,
            "slot": slot,
        }
        _publish_exclusive(
            ack_path,
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )

    def _write_identifiable_batch_stop(
        self,
        prepared: PreparedCaptureBatch,
        request: CaptureRequest,
        paths: AttemptPaths,
        payload: dict[str, Any],
        *,
        frame_index: int,
    ) -> None:
        """Stop a bad frame promptly when its handshake identity is exact."""

        slot = request.selected_slot
        if slot is None:
            raise CaptureProcessError("batch STOP has no selected slot")
        expected_batch = {
            "frame_index": frame_index,
            "frame_total": len(prepared.request.frames),
            "selected_slots": list(prepared.request.selected_slots),
            "session_id": prepared.session_id,
        }
        if payload.get("status") != "frame-complete":
            raise CaptureProcessError("batch STOP source is not frame-complete")
        if payload.get("batch_session") != expected_batch:
            raise CaptureProcessError(
                "bad batch frame has no trustworthy session identity for STOP"
            )
        if payload.get("requested_frame") != slot:
            raise CaptureProcessError(
                "bad batch frame has no trustworthy slot identity for STOP"
            )
        nonce = payload.get("ack_nonce")
        if (
            not isinstance(nonce, str)
            or not 1 <= len(nonce) <= 128
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in nonce
            )
        ):
            raise CaptureProcessError(
                "bad batch frame has no trustworthy nonce for STOP"
            )
        ack = {
            "ack_nonce": nonce,
            "action": BatchAckAction.STOP.value,
            "frame_index": frame_index,
            "schema_version": 1,
            "session_id": prepared.session_id,
            "slot": slot,
        }
        _publish_exclusive(
            paths.directory / "parent-ack.json",
            (json.dumps(ack, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )

    def _load_and_validate_batch_session_journal(
        self,
        prepared: PreparedCaptureBatch,
        *,
        returncode: int,
        handled: Sequence[CaptureAttemptResult],
        stopped: bool,
        stopped_unhandled_slot: int | None = None,
    ) -> dict[str, Any]:
        try:
            payload = json.loads(
                prepared.paths.session_journal.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CaptureProcessError(
                f"batch worker left no trustworthy session journal: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise CaptureProcessError("batch session journal must be an object")
        expected_status = "stopped" if stopped else "complete"
        expected_completed = [result.request.selected_slot for result in handled]
        if stopped_unhandled_slot is not None:
            expected_completed.append(stopped_unhandled_slot)
        invariants: dict[str, object] = {
            "session_id": prepared.session_id,
            "selected_slots": list(prepared.request.selected_slots),
            "batch_job_sha256": prepared.job_sha256,
            "capture_engine_sha256": self._expected_worker_sha256,
            "capture_bundle_sha256": (
                self._expected_bundle_sha256 or CAPTURE_BUNDLE_SHA256
            ),
            "plan_sha256": CANONICAL_PLAN_SHA256,
            "continuation_plan_sha256": CANONICAL_CONTINUATION_PLAN_SHA256,
            "manual_review_approval_sha256_by_slot": {
                str(frame.selected_slot): (
                    None
                    if frame.manual_review_approval is None
                    else frame.manual_review_approval.binding_sha256
                )
                for frame in prepared.request.frames
            },
            "reviewed_roll_fingerprint_sha256": (
                prepared.request.reviewed_fingerprint.binding_sha256
            ),
        }
        for key, expected in invariants.items():
            if payload.get(key) != expected:
                raise CaptureProcessError(
                    f"batch session journal {key}={payload.get(key)!r}, "
                    f"expected {expected!r}"
                )
        completed = payload.get("completed_slots")
        selected = list(prepared.request.selected_slots)
        if (
            not isinstance(completed, list)
            or any(
                isinstance(slot, bool) or not isinstance(slot, int)
                for slot in completed
            )
            or completed != selected[: len(completed)]
            or completed != expected_completed
        ):
            raise CaptureProcessError(
                f"batch session journal completed_slots={completed!r} does not "
                f"match the observed frame prefix {expected_completed!r}"
            )
        status = payload.get("status")
        release_attempts = payload.get("unit_release_attempts")
        unit_released = payload.get("unit_released")
        reservation_acquired = payload.get("reservation_acquired")
        recovery = payload.get("recovery_required")
        if recovery not in ("none", POWER_CYCLE_RECOVERY):
            raise CaptureProcessError(
                f"batch session journal has unknown recovery state {recovery!r}"
            )
        if not isinstance(reservation_acquired, bool):
            raise CaptureProcessError(
                "batch session journal has no reservation-acquired receipt"
            )
        if returncode == 0:
            if status != expected_status or completed != expected_completed:
                raise CaptureProcessError(
                    "completed batch session has the wrong status or frame prefix"
            )
            if (
                reservation_acquired is not True
                or release_attempts != 1
                or unit_released is not True
                or recovery != "none"
            ):
                raise CaptureProcessError(
                    "completed batch session has no successful single-release receipt"
                )
        else:
            if status not in ("failed", "interrupted"):
                raise CaptureProcessError(
                    f"failed batch session has unexpected status {status!r}"
                )
            if (
                isinstance(release_attempts, bool)
                or release_attempts not in (0, 1)
            ):
                raise CaptureProcessError(
                    "failed batch session has an invalid release-attempt count"
                )
            if reservation_acquired is False:
                if (
                    completed
                    or release_attempts != 0
                    or unit_released is not False
                ):
                    raise CaptureProcessError(
                        "failed pre-reservation batch receipt is inconsistent"
                    )
                return payload
            if unit_released is True and (
                release_attempts != 1 or recovery != "none"
            ):
                raise CaptureProcessError(
                    "failed batch release receipt is internally inconsistent"
                )
            if unit_released is not True and recovery != POWER_CYCLE_RECOVERY:
                raise CaptureProcessError(
                    "failed batch without confirmed release must require recovery"
                )
        return payload

    def _interpret_result(
        self,
        *,
        request: CaptureRequest,
        paths: AttemptPaths,
        argv: tuple[str, ...],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> CaptureAttemptResult:
        try:
            journal = self._load_and_validate_journal(paths, request, returncode)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            # A child was launched but did not leave trustworthy synchronization
            # evidence.  Conservatively require recovery instead of guessing.
            return CaptureAttemptResult(
                outcome=CaptureOutcome.RECOVERY_REQUIRED,
                request=request,
                paths=paths,
                argv=argv,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                journal=None,
                journal_error=str(error),
            )

        if returncode == 0:
            outcome = CaptureOutcome.COMPLETE
        elif journal["recovery_required"] == "none":
            outcome = CaptureOutcome.SYNCHRONIZED_REFUSAL
        else:
            outcome = CaptureOutcome.RECOVERY_REQUIRED
        return CaptureAttemptResult(
            outcome=outcome,
            request=request,
            paths=paths,
            argv=argv,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            journal=journal,
        )

    def _load_and_validate_journal(
        self,
        paths: AttemptPaths,
        request: CaptureRequest,
        returncode: int,
    ) -> dict[str, Any]:
        payload = json.loads(paths.journal.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("worker journal must be a JSON object")

        mode = {
            CaptureMode.PREVIEW: "preview-only",
            CaptureMode.METER_ONLY: "meter-only",
            CaptureMode.FULL: "full",
        }[request.mode]
        expected_reads = {
            CaptureMode.PREVIEW: 0,
            CaptureMode.METER_ONLY: METER_READ_COUNT,
            CaptureMode.FULL: CANONICAL_FINE_READ_COUNT,
        }[request.mode]
        expected_bytes = {
            CaptureMode.PREVIEW: 0,
            CaptureMode.METER_ONLY: METER_CAPTURE_BYTES,
            CaptureMode.FULL: CANONICAL_FINE_READ_COUNT * CANONICAL_FINE_READ_BYTES,
        }[request.mode]
        invariants: dict[str, object] = {
            "plan_sha256": CANONICAL_PLAN_SHA256,
            "capture_engine_sha256": self._expected_worker_sha256,
            "output": str(paths.output.resolve()),
            "capture_mode": mode,
            "requested_frame": request.selected_slot,
            "expected_frame_count": None,
            "expected_reads": expected_reads,
            "expected_bytes": expected_bytes,
            "requested_boundary_offset_rows": request.boundary_offset_rows,
        }
        if self._expected_bundle_sha256 is not None:
            invariants["capture_bundle_sha256"] = self._expected_bundle_sha256
        for key, expected in invariants.items():
            if payload.get(key) != expected:
                raise ValueError(f"worker journal {key}={payload.get(key)!r}, expected {expected!r}")

        status = payload.get("status")
        recovery = payload.get("recovery_required")
        if returncode == 0:
            if status != "complete":
                raise ValueError(f"worker exited zero with journal status {status!r}")
            if recovery not in (None, "none"):
                raise ValueError(f"completed worker requested recovery: {recovery!r}")
            if payload.get("completed_reads") != expected_reads or payload.get("completed_bytes") != expected_bytes:
                raise ValueError("completed worker journal has incomplete read or byte counts")
            if payload.get("disk_bytes") != expected_bytes:
                raise ValueError("completed worker journal has the wrong on-disk byte count")
            if payload.get("unit_released") is not True:
                raise ValueError("completed worker did not record unit release")
            if not _is_sha256(payload.get("output_sha256")):
                raise ValueError("completed worker output SHA-256 is missing or malformed")
            if not paths.output.is_file() or paths.output.stat().st_size != expected_bytes:
                raise ValueError("completed worker output file is missing or has the wrong size")
            if request.mode is not CaptureMode.PREVIEW:
                if payload.get("applied_boundary_offset_rows") != request.boundary_offset_rows:
                    raise ValueError(
                        "worker journal applied_boundary_offset_rows does not "
                        "match the requested boundary offset"
                    )
                resolved_row = payload.get("resolved_lookup_row")
                if isinstance(resolved_row, bool) or not isinstance(resolved_row, int) or resolved_row < 0:
                    raise ValueError("worker journal has no valid resolved_lookup_row")
                resolved_origin = payload.get("resolved_native_origin")
                if (
                    isinstance(resolved_origin, bool)
                    or not isinstance(resolved_origin, int)
                    or resolved_origin < 0
                ):
                    raise ValueError("worker journal has no valid resolved_native_origin")
        else:
            if status not in ("failed", "interrupted"):
                raise ValueError(f"failed worker has unexpected journal status {status!r}")
            if recovery not in ("none", POWER_CYCLE_RECOVERY):
                raise ValueError(f"failed worker has unknown recovery state {recovery!r}")
        return payload


__all__ = [
    "BatchAckAction",
    "BatchFrameHandler",
    "BatchProcessSpawner",
    "BatchSessionPaths",
    "CaptureAttemptResult",
    "CaptureBatchProcessError",
    "CaptureBatchResult",
    "CaptureBatchRequest",
    "CAPTURE_HELPER_FLAG",
    "CaptureIntegrityError",
    "CaptureMode",
    "CaptureOutcome",
    "CaptureProcessAdapter",
    "CaptureProcessError",
    "CaptureRequest",
    "CaptureStopped",
    "PreparedCaptureBatch",
]
