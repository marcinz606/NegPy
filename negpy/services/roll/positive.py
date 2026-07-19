"""Tier-3 positive rendering: NegPy's own negative-to-positive pipeline,
called on an in-memory Tier-2 buffer instead of introducing a second one.

`DarkroomEngine.process()` (`negpy.services.rendering.engine`) is how any
imported negative reaches a rendered positive in NegPy today -- it is what
runs when a scan is first opened, what the live canvas re-runs on every
edit, and what the export pipeline (`ImageProcessor.process_export`) calls
after decoding a source file. `ImageProcessor.run_pipeline` is that same
entry point one layer up, taking an in-memory buffer directly rather than a
file path -- it is what `process_export` itself calls after its own decode
step. A roll-scanned frame is already an in-memory buffer, so this module
calls `run_pipeline` straight, skipping the file-decode and the ICC export
encoding `process_export` also does, neither of which apply to a Tier 2
buffer that never touches disk to get here (see `service.write_frame`'s
"never derive an upper tier from a lower one" -- the positive comes from the
same in-memory array Tier 2 optionally writes, not by reading a Tier 2 file
back).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

import negpy
from negpy.domain.models import WorkspaceConfig
from negpy.kernel.image.logic import float_to_uint16, uint16_to_float32
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.rendering.image_processor import ImageProcessor


@dataclass(frozen=True)
class PositiveResult:
    """One frame's Tier-3 RGB, plus what the receipt needs to audit or
    reproduce it."""

    rgb: np.ndarray
    render_intent: str
    process_mode: str
    auto_exposure: bool
    negpy_version: str


def available() -> bool:
    """True if the inversion path can be used.

    Unlike coolscanpy or a Tier-2 repair engine, NegPy's rendering pipeline
    is not an optional external dependency -- it ships with this
    application, so this always returns True today. It exists as a seam
    with the same shape as `coolscanpy_roll.available()` and
    `repair.available()`, so `service.write_frame` has one place to check
    before rendering (and a test simulating "inversion unavailable" has one
    place to patch) instead of a bare `try/except` being the only signal.
    """
    return True


def render_positive(rgb_u16: np.ndarray, *, processor: ImageProcessor) -> PositiveResult:
    """Invert a scanner-linear RGB buffer (Tier 2, or Tier 1 if Tier 2 could
    not be produced -- callers decide what to pass) to a positive.

    Uses a stock, unedited `WorkspaceConfig`: process mode C41 (this
    adapter only ever opens a roll as color negative -- see
    `coolscanpy_roll.open_roll`), render intent PRINT (NegPy's default
    photographic-paper conversion, as opposed to the flat digital-
    intermediate master), and auto exposure on. That is exactly what a user
    would see on first opening this frame's negative in NegPy without
    touching a single slider -- the ordinary default conversion, not a
    custom or maximal-latitude one, since this tier is a regenerable view
    rather than an edit a person has invested time in.

    `processor` is injected rather than constructed here so a caller can
    hold one `ImageProcessor` across a whole batch scan (its constructor
    probes for GPU acceleration, which is worth doing once, not per frame)
    even though rendering itself always runs on the CPU engine below.
    """
    settings = WorkspaceConfig()
    img = uint16_to_float32(np.ascontiguousarray(rgb_u16, dtype=np.uint16))

    rendered, _metrics = processor.run_pipeline(
        img,
        settings,
        # A fresh id per call, deliberately not a stable hash of the frame: the
        # engine's stage cache reuses a prior render whenever both source_hash
        # and the settings hash match, and every Tier-3 render here shares the
        # same stock settings -- a stable/reused hash across two different
        # frames (or two different slots re-scanned in one session) would
        # silently hand back a *previous frame's* rendered pixels instead of
        # this one's. A fresh id guarantees a real render every time; it has
        # no bearing on reproducibility, since the pipeline math itself is a
        # pure function of (image, settings) and this cache is a same-process
        # performance optimization, not a determinism mechanism.
        source_hash=uuid.uuid4().hex,
        render_size_ref=float(APP_CONFIG.preview_render_size),
        prefer_gpu=False,  # CPU engine: deterministic, and this is a batch worker, not the live canvas
        wants_uv_grid=False,
        skip_flatfield=True,  # no flat-field calibration exists for a roll-scanner buffer
    )

    return PositiveResult(
        rgb=float_to_uint16(rendered),
        render_intent=str(settings.exposure.render_intent),
        process_mode=str(settings.process.process_mode),
        auto_exposure=settings.exposure.auto_exposure,
        negpy_version=negpy.__version__,
    )
