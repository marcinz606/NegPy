"""Exposure meter for a white-light or camera-only scan: one RAW probe sets the shutter.
A negative places its clear base (read in the rebate) at the target, since it becomes the print's black point. A positive,
or a negative with no rebate the probe reads unclipped, places its brightest highlights: a negative's thin end. The medium is the preset's, else the import classifier's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import negpy_core
import numpy as np

from negpy.features.geometry.logic import (
    AUTOCROP_DETECT_RES,
    _normalize_detection_input,
    _scale_roi,
    film_and_picture,
    rebate_mask,
)
from negpy.features.process.logic import detect_process_mode
from negpy.features.process.models import ProcessMode
from negpy.services.capture.calibration import (
    CLIP_CEILING,
    SATURATION_VALUE,
    SHUTTER_CANDIDATES,
    TARGET_FRACTION,
    true_seconds,
)

#: Saturated pixels at this share of the picture are film, not light through a gap.
_BLOWN_SHARE = 0.02
#: A clipped region reads as a floor, so the probe steps down this far and meters again.
CLIPPED_STEP_STOPS = -2.0
#: Long edge of the probe's peak map, the RAW zebras' resolution.
PEAK_MAP_EDGE = 960
#: The probe and the preview frame the same picture when their aspects agree this closely...
ASPECT_TOLERANCE = 0.02
#: ...and the rank correlation of their thumbnails reaches this. Ranks, because the preview
#: is the body's tone-curved JPEG and the probe is linear: any monotonic curve keeps the order.
SAME_FRAMING = 0.8
_THUMB = (64, 43)
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
    # The probe's picture at thumbnail size and its full shape, to check the preview frames it.
    thumb: np.ndarray | None = field(default=None, compare=False, repr=False)
    shape: tuple[int, int] | None = None
    # The rebate ring read as light or as a clipped base, so the picture alone set the exposure.
    ring_saturated: bool = False


def gray_of(rgb: np.ndarray) -> np.ndarray:
    """`rgb.mean(axis=2)` as float32 for an integer or float32 frame."""
    return negpy_core.mean_channels(np.ascontiguousarray(rgb[..., :3], dtype=np.float32))


def ranked_thumb(gray: np.ndarray) -> np.ndarray:
    """A picture at thumbnail size as zero-mean, unit-variance ranks: two thumbnails' mean product
    is their rank correlation, which no monotonic tone curve and no exposure change can move."""
    small = negpy_core.resize_area(np.ascontiguousarray(gray, dtype=np.float32), _THUMB).ravel()
    ranks = np.argsort(np.argsort(small)).astype(np.float32)
    ranks -= ranks.mean()
    return ranks / max(float(ranks.std()), 1e-6)


def framing_mismatch(reading: MeterReading, preview: np.ndarray) -> str | None:
    """Why the probe's RAW map cannot be laid over `preview` (HxWx3), or None when it can: the
    two must have the same aspect and show the same picture. An aspect setting the RAW ignores,
    a crop, a zoomed view or a body that streams another framing all fail one or the other."""
    if reading.thumb is None or reading.shape is None:
        return "the probe kept no picture"
    ph, pw = reading.shape
    vh, vw = preview.shape[:2]
    if abs((pw / ph) / (vw / vh) - 1.0) > ASPECT_TOLERANCE:
        return f"the preview is {vw}:{vh} and the RAW {pw}:{ph}"
    if float(np.mean(ranked_thumb(gray_of(preview)) * reading.thumb)) < SAME_FRAMING:
        return "the preview does not frame the RAW the same way"
    return None


def _peak_map(linear: np.ndarray) -> np.ndarray:
    """The brightest channel max-pooled to a long edge of PEAK_MAP_EDGE. The frame is zero-padded
    to whole blocks, so every pixel lands in a block and the edge blocks keep their own clips."""
    h, w = linear.shape[:2]
    return negpy_core.peak_map(np.ascontiguousarray(linear, dtype=np.float32), max(1, math.ceil(max(h, w) / PEAK_MAP_EDGE)))


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


def _measure_picture(linear: np.ndarray):
    """(detection image, film ROI, picture gate, meter_picture's (clipped fraction, blown, thin end))
    of a normalized linear frame."""
    det, scale = _normalize_detection_input(linear, AUTOCROP_DETECT_RES)
    det = np.ascontiguousarray(det, dtype=np.float32)
    film_roi, gate = film_and_picture(det)
    # The picture at the frame's own resolution: a highlight narrower than a detection pixel
    # is averaged away at detection size, and it still clips.
    roi = _scale_roi(gate, scale, linear.shape[0], linear.shape[1])
    return det, film_roi, gate, negpy_core.meter_picture(linear, tuple(roi), _SATURATION, _BLOWN_SHARE)


def picture_clipping(img: np.ndarray) -> float | None:
    """Share of a decoded linear frame's picture (`HxWx3`, counts to `CLIP_CEILING`) at saturation,
    when it clips as Meter judges a probe; None when it does not."""
    linear = negpy_core.normalized(np.ascontiguousarray(img), float(CLIP_CEILING))
    _, _, _, (fraction, blown, thin) = _measure_picture(linear)
    return float(fraction) if blown or max(thin) >= _SATURATION else None


def meter_frame(
    img: np.ndarray, shutter: str, candidates: tuple[str, ...] = (), medium: str | None = None, wb: list | None = None
) -> MeterReading:
    """Meter one decoded linear probe (`HxWx3`, counts to `CLIP_CEILING`) shot at `shutter`.
    `medium` is "negative" or "positive" from the preset, or None for the classifier's call.
    `wb` is the as-shot white balance the import applies to a white-light frame: the target
    holds after its gain, while the clip test reads the sensor itself."""
    candidates = tuple(candidates) or SHUTTER_CANDIDATES
    gain = np.asarray(wb, np.float32) / min(wb) if wb else np.ones(3, np.float32)
    linear = negpy_core.normalized(np.ascontiguousarray(img), float(CLIP_CEILING))
    if medium is None:
        medium = "positive" if detect_process_mode(linear) == ProcessMode.E6 else "negative"
    det, film_roi, gate, (clipped_fraction, blown, thin) = _measure_picture(linear)
    gy1, gy2, gx1, gx2 = gate
    picture_det = det[gy1:gy2, gx1:gx2].reshape(-1, 3)
    thinnest = np.array(thin, dtype=np.float32)  # the picture's thin end, per channel

    ring = det[rebate_mask(det.shape[:2], film_roi, gate)]
    # Only unclipped ring pixels are film: a gap, a sprocket hole and a clipped base all saturate.
    film_ring = ring[ring.max(axis=1) < _SATURATION] if ring.size else ring
    ring_saturated = medium == "negative" and bool(ring.size) and film_ring.shape[0] < 0.5 * ring.shape[0]
    # A negative's base is its thinnest film, so a ring darker than the picture is picture.
    base_visible = (
        medium == "negative"
        and not ring_saturated
        and bool(film_ring.size)
        and np.median(film_ring.mean(axis=1)) > np.median(picture_det.mean(axis=1))
    )
    if base_visible:
        region = "base"
        # A ring that is half holder reads under the base, so the base never sets a level under the thin end.
        levels = np.maximum(np.median(film_ring, axis=0), thinnest)
    else:
        region = "highlights"
        levels = thinnest
    clipped = blown or float(levels.max()) >= _SATURATION
    measured = float((levels * gain).max()) * CLIP_CEILING

    probe_seconds = true_seconds(shutter, candidates)
    needed = CLIPPED_STEP_STOPS if clipped else float(np.log2(TARGET_FRACTION * CLIP_CEILING / max(measured, 1.0)))
    recommended = shutter_at_most(probe_seconds * 2.0**needed, candidates)
    if recommended is None and clipped and true_seconds(candidates[0], candidates) < probe_seconds:
        # The step down passes the fastest rung, which is still a step down: take it and meter again.
        recommended = candidates[0]
    stops = float(np.log2(true_seconds(recommended, candidates) / probe_seconds)) if recommended else needed
    peaks = _peak_map(linear)
    thumb = ranked_thumb(gray_of(linear))
    shape = (int(linear.shape[0]), int(linear.shape[1]))
    return MeterReading(
        medium, region, measured, clipped, clipped_fraction, shutter, recommended, stops, peaks, thumb, shape, ring_saturated=ring_saturated
    )
