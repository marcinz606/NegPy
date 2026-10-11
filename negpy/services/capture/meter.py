"""Exposure meter for a white-light or camera-only scan: one RAW probe sets the shutter.

The medium decides what must not clip. On a negative the clear base becomes the print's black
point, so the base, read in the rebate around the picture, is placed at the target. On a
positive the picture's brightest highlights are. A capture with no rebate meters the picture's
highlights either way: nothing brighter is in the frame. The medium is the preset's, or the
import's own classifier on the probe. Hardware-free: the caller shoots and decodes the probe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from negpy.features.geometry.logic import (
    AUTOCROP_DETECT_RES,
    _normalize_detection_input,
    _refine_roi_to_image,
    _scale_roi,
    detect_film_bounds_with_confidence,
)
from negpy.features.process.logic import detect_process_mode
from negpy.features.process.models import ProcessMode
from negpy.services.capture.calibration import (
    CLIP_CEILING,
    MAX_LINEARITY_FRACTION,
    SATURATION_VALUE,
    SHUTTER_CANDIDATES,
    TARGET_FRACTION,
    true_seconds,
)

#: A clipped region reads as a floor, so the probe steps down this far and meters again.
CLIPPED_STEP_STOPS = -2.0
#: Long edge of the probe's peak map, the RAW zebras' resolution.
PEAK_MAP_EDGE = 960
_SATURATION = SATURATION_VALUE / CLIP_CEILING


@dataclass(frozen=True)
class MeterReading:
    """One probe's verdict: what was metered, where it sits, and the shutter that places it."""

    medium: str  # "negative" or "positive"
    region: str  # "base" or "highlights"
    measured: float  # counts in the metered region, brightest channel; a floor when `clipped`
    clipped: bool
    clipped_fraction: float  # share of the picture at saturation in its brightest channel
    shutter: str  # the probe's
    recommended: str | None  # the ladder rung for the target, never past it; None when no rung is fast enough
    stops: float  # the move `recommended` makes, or the move needed when there is no rung
    # The whole probe's brightest channel, max-pooled to PEAK_MAP_EDGE: a block holds its
    # brightest pixel, so a clip narrower than a block still shows.
    peaks: np.ndarray | None = field(default=None, compare=False, repr=False)


def _peak_map(brightest: np.ndarray) -> np.ndarray:
    """Max-pool to a long edge of PEAK_MAP_EDGE. The frame is zero-padded to whole blocks, so
    every pixel lands in a block and the edge blocks keep their own clips."""
    h, w = brightest.shape
    f = max(1, math.ceil(max(h, w) / PEAK_MAP_EDGE))
    padded = np.pad(brightest, ((0, -h % f), (0, -w % f)))
    hh, ww = padded.shape
    return np.ascontiguousarray(padded.reshape(hh // f, f, ww // f, f).max(axis=(1, 3)))


def raw_clip_mask(reading: MeterReading) -> np.ndarray | None:
    """Where the RAW will clip at `reading.recommended`: the RAW is linear, so a pixel the probe
    read unclipped scales by 2^stops exactly. A pixel the probe clipped has no known level, so it
    stays marked until a probe reads it unclipped. None without a peak map or a recommendation."""
    if reading.peaks is None or reading.recommended is None:
        return None
    return (reading.peaks * np.float32(2.0**reading.stops) >= _SATURATION) | (reading.peaks >= _SATURATION)


def shutter_at_most(seconds: float, candidates: tuple[str, ...]) -> str | None:
    """The slowest candidate whose true exposure is ≤ `seconds` (candidates are fastest-first),
    or None when even the fastest is too slow: a target is a ceiling, never overshot."""
    for c in reversed(candidates):
        if true_seconds(c, candidates) <= seconds:
            return c
    return None


def meter_frame(img: np.ndarray, shutter: str, candidates: tuple[str, ...] = (), medium: str | None = None) -> MeterReading:
    """Meter one decoded linear probe (`HxWx3`, counts to `CLIP_CEILING`) shot at `shutter`.
    `medium` is "negative" or "positive" from the preset, or None for the classifier's call."""
    candidates = tuple(candidates) or SHUTTER_CANDIDATES
    linear = np.clip(img.astype(np.float32) / CLIP_CEILING, 0.0, 1.0)
    if medium is None:
        medium = "positive" if detect_process_mode(linear) == ProcessMode.E6 else "negative"
    det, scale = _normalize_detection_input(linear, AUTOCROP_DETECT_RES)
    det = np.ascontiguousarray(det, dtype=np.float32)
    h, w = det.shape[:2]
    film = detect_film_bounds_with_confidence(det)
    film_roi = film.roi if film.roi is not None else (0, h, 0, w)
    gate, _, _ = _refine_roi_to_image(det, film_roi)
    if gate[1] <= gate[0] or gate[3] <= gate[2]:  # a degenerate picture box on a blank or dark probe
        gate = (0, h, 0, w)
    gy1, gy2, gx1, gx2 = gate
    picture_det = det[gy1:gy2, gx1:gx2].reshape(-1, 3)
    # The picture at the probe's own resolution: a highlight narrower than a detection pixel
    # is averaged away at detection size, and it still clips.
    ly1, ly2, lx1, lx2 = _scale_roi(gate, scale, linear.shape[0], linear.shape[1])
    picture = linear[ly1:ly2, lx1:lx2].reshape(-1, 3)
    frame_brightest = np.max(linear, axis=2)
    brightest = frame_brightest[ly1:ly2, lx1:lx2]
    clipped_fraction = float(np.mean(brightest >= _SATURATION))

    rebate = np.zeros((h, w), bool)
    fy1, fy2, fx1, fx2 = film_roi
    rebate[fy1:fy2, fx1:fx2] = True
    rebate[gy1:gy2, gx1:gx2] = False
    ring = det[rebate]
    # A negative's base is its thinnest film, so a ring darker than the picture is picture.
    if medium == "negative" and ring.size and np.median(ring.mean(axis=1)) > np.median(picture_det.mean(axis=1)):
        region = "base"
        levels = np.median(ring, axis=0)
        clipped = float(levels.max()) >= _SATURATION
    else:
        region = "highlights"
        levels = np.percentile(picture, 99.9, axis=0)
        clipped = clipped_fraction > MAX_LINEARITY_FRACTION  # Calibrate's demosaiced budget
    measured = float(levels.max()) * CLIP_CEILING

    probe_seconds = true_seconds(shutter, candidates)
    needed = CLIPPED_STEP_STOPS if clipped else float(np.log2(TARGET_FRACTION * CLIP_CEILING / max(measured, 1.0)))
    recommended = shutter_at_most(probe_seconds * 2.0**needed, candidates)
    stops = float(np.log2(true_seconds(recommended, candidates) / probe_seconds)) if recommended else needed
    peaks = _peak_map(frame_brightest)
    return MeterReading(medium, region, measured, clipped, clipped_fraction, shutter, recommended, stops, peaks)
