"""Per-channel ETTR exposure auto-calibration (ported from TriRGB's approach).

For a film stock, meter the clear film base inside a user-drawn ROI and solve **one camera
shutter shared across R/G/B** plus a per-channel Scanlight LED level so each channel's base
sits just below clipping ("expose to the right"). One probe per channel estimates its
response; the shared shutter is chosen so the *dimmest* channel reaches target within the LED
range (the RGB channels differ by ~1 stop — orange mask minus the Scanlight's per-channel LED
balance — well inside the LED's ~2.7-stop range). Then each channel's LED is solved + verified
+ one proportional trim, with a clip guard (pull the LED down) and, if the whole set doesn't
fit the LED window, a shared-shutter escalation. Keeping the shutter constant makes presets a
single shutter + three LED levels. The black level is the dark-frame median (robust to hot
pixels). Hardware-free: light, camera, and a `demosaic` callable are injected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from negpy.infrastructure.capture.base import CAPTURE_ORDER, Camera, LightSource
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# 16-bit demosaiced range; PWM range and probe match TriRGB's calibrate_exposure.
CLIP_CEILING = 65535
SATURATION_VALUE = 65400  # counts; a demosaiced pixel at/above this is treated as clipped
MAX_CLIP_FRACTION = 0.0001  # >0.01% of base pixels saturated → the base is clipping
PWM_MIN = 40
PWM_MAX = 255
PROBE_LEVEL = 200
PROBE_SHUTTER = "1/15"
TARGET_FRACTION = 0.9  # aim the film base at 90% of usable range
MIN_SIGNAL = 10.0  # counts; below this the channel read no real signal
# A clean clip-guard adjustment can intentionally land a little below the ETTR target.
# Beyond this margin the preset is materially mis-exposed and must not be saved.
MAX_TARGET_UNDER_FRACTION = 0.2
# There is much less safe headroom above a 90% ETTR target; 10% is already near clipping.
MAX_TARGET_OVER_FRACTION = 0.1

# Fallback shutter ladder, fastest first (third-stops). The body's own ladder is
# preferred when live view has published it.
SHUTTER_CANDIDATES: tuple[str, ...] = (
    "1/250",
    "1/200",
    "1/160",
    "1/125",
    "1/100",
    "1/80",
    "1/60",
    "1/50",
    "1/40",
    "1/30",
    "1/25",
    "1/20",
    "1/15",
    "1/13",
    "1/10",
    "1/8",
    "1/6",
    "1/5",
    "1/4",
    "1/3",
    "0.4",
    "1/2",
    "0.6",
    "0.8",
    "1",
)

DemosaicFn = Callable[[str], np.ndarray]  # path -> HxWx3 linear array (0..CLIP_CEILING)
ProgressCb = Callable[[float, str], None]


def shutter_seconds(label: str) -> float:
    """Parse a shutter label ('1/100', '0.4', '1') into seconds."""
    label = label.strip()
    if "/" in label:
        num, den = label.split("/", 1)
        return float(num) / float(den)
    return float(label)


@dataclass(frozen=True)
class Roi:
    """Base-region crop in fractions of the frame (0..1), resolution-independent."""

    x: float
    y: float
    w: float
    h: float

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0 = int(round(self.x * width))
        y0 = int(round(self.y * height))
        x1 = int(round((self.x + self.w) * width))
        y1 = int(round((self.y + self.h) * height))
        x0, x1 = sorted((max(0, min(x0, width)), max(0, min(x1, width))))
        y0, y1 = sorted((max(0, min(y0, height)), max(0, min(y1, height))))
        if x1 <= x0:
            x1 = min(width, x0 + 1)
        if y1 <= y0:
            y1 = min(height, y0 + 1)
        return x0, y0, x1, y1


@dataclass(frozen=True)
class ChannelCalibration:
    channel: str  # "R" / "G" / "B"
    level: int  # solved LED level 0-255
    shutter: str  # solved camera shutter label
    signal: float  # measured base p99.9 at the solved settings
    target: int  # target signal
    clip_fraction: float = 0.0  # fraction of base pixels at/above saturation (ETTR keeps this ~0)


@dataclass(frozen=True)
class CalibrationResult:
    channels: dict[str, ChannelCalibration]
    black_levels: dict[str, float]

    @property
    def levels(self) -> tuple[int, int, int]:
        return (self.channels["R"].level, self.channels["G"].level, self.channels["B"].level)

    @property
    def shutters(self) -> tuple[str, str, str]:
        return (self.channels["R"].shutter, self.channels["G"].shutter, self.channels["B"].shutter)


def target_for_black_level(black_level: float, fraction: float = TARGET_FRACTION) -> int:
    usable = max(1.0, float(CLIP_CEILING) - float(black_level))
    return int(round(fraction * usable))


def meter_base(plane: np.ndarray, roi: Roi, black_level: float) -> float:
    """Black-subtracted p99.9 of the ROI on one demosaiced channel plane."""
    h, w = plane.shape[:2]
    x0, y0, x1, y1 = roi.pixels(w, h)
    patch = plane[y0:y1, x0:x1].astype(np.float64) - float(black_level)
    np.clip(patch, 0, None, out=patch)
    if patch.size == 0:
        return 0.0
    return float(np.percentile(patch, 99.9))


def meter_black(plane: np.ndarray, roi: Roi) -> float:
    """Median of the ROI on a dark-frame channel plane → the per-channel black level.

    Median (not p99.9) is robust to hot pixels / stuck columns in the dark frame — matching
    TriRGB's calibrate_exposure. p99.9 would let a few hot pixels inflate the black estimate,
    which then under-states the signal (T% = 10^-A references this black floor)."""
    h, w = plane.shape[:2]
    x0, y0, x1, y1 = roi.pixels(w, h)
    patch = plane[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.median(patch))


def clip_fraction(plane: np.ndarray, roi: Roi, saturation: float = SATURATION_VALUE) -> float:
    """Fraction of ROI pixels at/above saturation on one demosaiced channel plane.

    ETTR meters p99.9, which ignores the top 0.1% by design — so a base can read on-target
    while a sliver saturates. The clear base is the whitepoint reference (blackpoint after
    inversion), so it must stay just *below* clipping; this metric catches what p99.9 hides."""
    h, w = plane.shape[:2]
    x0, y0, x1, y1 = roi.pixels(w, h)
    patch = plane[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch >= saturation))


def faster_shutter(label: str, candidates: tuple[str, ...] = SHUTTER_CANDIDATES) -> Optional[str]:
    """The next-faster shutter in the ladder (candidates are fastest-first), or None."""
    try:
        idx = candidates.index(label)
    except ValueError:
        return None
    return candidates[idx - 1] if idx > 0 else None


def nearest_shutter(label: str, candidates: tuple[str, ...]) -> str:
    """The candidate closest (in seconds) to `label`, snapping onto the camera's own ladder.

    Different bodies expose different writable shutter sets, so the probe/solve must pick
    from *this* camera's list (passed by the caller), not a hardcoded one."""
    if not candidates or label in candidates:
        return label
    target = shutter_seconds(label)
    return min(candidates, key=lambda c: abs(shutter_seconds(c) - target))


def slower_shutter(label: str, candidates: tuple[str, ...] = SHUTTER_CANDIDATES) -> Optional[str]:
    """The next-slower shutter in the ladder (candidates are fastest-first), or None."""
    try:
        idx = candidates.index(label)
    except ValueError:
        return None
    return candidates[idx + 1] if idx + 1 < len(candidates) else None


def shutter_at_least(seconds: float, candidates: tuple[str, ...] = SHUTTER_CANDIDATES) -> str:
    """The fastest candidate whose exposure time is ≥ `seconds` (candidates are fastest-first).

    Picks the shared shutter: as fast as possible while still letting the dimmest channel reach
    target within the LED range. Falls back to the slowest candidate if none is slow enough."""
    for c in candidates:
        if shutter_seconds(c) >= seconds:
            return c
    return candidates[-1]


def correct_led_level(current_level: int, signal: float, target: int) -> int:
    """One proportional LED trim toward target (shutter held)."""
    if signal < MIN_SIGNAL:
        return PWM_MAX
    corrected = round(current_level * (float(target) / float(signal)))
    return int(np.clip(corrected, PWM_MIN, PWM_MAX))


class CalibrationService:
    """Drives a dark frame + per-channel probe/verify to solve ETTR exposures."""

    def __init__(
        self,
        light: LightSource,
        camera: Camera,
        demosaic: DemosaicFn,
        *,
        source_clip: Optional[Callable[[str, int, Roi], float]] = None,
        sleep: Callable[[float], None] = time.sleep,
        settle_s: float = 0.4,
    ) -> None:
        self._light = light
        self._camera = camera
        self._demosaic = demosaic
        # (path, channel_index, roi) → raw-Bayer source-clip fraction. None → the rawpy default;
        # tests inject a stub so the hardware-free path never touches rawpy.
        self._source_clip = source_clip
        self._sleep = sleep
        self._settle_s = settle_s

    def _source_clip_fraction(self, path: str, channel_index: int, roi: Roi) -> float:
        """Raw-Bayer source clip for one channel (catches clipped photosites the demosaic hides)."""
        if self._source_clip is not None:
            measured = float(self._source_clip(path, channel_index, roi))
        else:
            from negpy.infrastructure.capture.raw_demosaic import raw_channel_clip_fraction

            try:
                measured = raw_channel_clip_fraction(path, channel_index, roi)
            except Exception as exc:
                raise RuntimeError(f"calibration failed: raw source-clip check failed for {path}") from exc
        if not np.isfinite(measured):
            raise RuntimeError(f"calibration failed: non-finite raw source-clip measurement for {path}")
        return measured

    def calibrate(
        self,
        roi: Roi,
        scratch_path: str,
        *,
        probe_level: int = PROBE_LEVEL,
        probe_shutter: str = PROBE_SHUTTER,
        target_fraction: float = TARGET_FRACTION,
        candidates: tuple[str, ...] = SHUTTER_CANDIDATES,
        progress: Optional[ProgressCb] = None,
        cancel=None,
    ) -> CalibrationResult:
        # Solve/verify on THIS camera's writable shutter ladder (multi-camera), snapping the
        # probe onto it too; falls back to the built-in ladder when the caller has no list.
        candidates = tuple(candidates) or SHUTTER_CANDIDATES
        probe_shutter = nearest_shutter(probe_shutter, candidates)

        def _check_cancel():
            if cancel is not None and cancel.is_set():
                raise RuntimeError("calibration cancelled")

        def _report(frac: float, msg: str):
            if progress is not None:
                progress(frac, msg)

        try:
            # 1) Dark frame → per-channel black level inside the ROI.
            _check_cancel()
            _report(0.05, "Dark frame…")
            self._light.off()
            self._sleep(self._settle_s)
            dark, _ = self._capture(scratch_path, shutter=probe_shutter)
            black = {ch.letter: meter_black(dark[..., i], roi) for i, ch in enumerate(CAPTURE_ORDER)}

            def _shoot(ch, i: int, base: float, level_: int, shutter_: str) -> tuple[float, float]:
                """Light the channel, capture, and meter it → (base p99.9, max clip fraction).

                Clip is the worse of the demosaiced output and the raw-Bayer source photosites."""
                self._light.set_color(*ch.rgb(level_))
                self._sleep(self._settle_s)
                img, written = self._capture(scratch_path, shutter=shutter_)
                clip = max(clip_fraction(img[..., i], roi), self._source_clip_fraction(written, i, roi))
                return meter_base(img[..., i], roi, base), clip

            targets = {ch.letter: target_for_black_level(black[ch.letter], target_fraction) for ch in CAPTURE_ORDER}

            # 2) Probe each channel once → its response k (counts per LED-level per second).
            k: dict[str, float] = {}
            for i, ch in enumerate(CAPTURE_ORDER):
                _check_cancel()
                _report(0.1 + 0.1 * i, f"Probing {ch.letter}…")
                signal, _ = _shoot(ch, i, black[ch.letter], probe_level, probe_shutter)
                if not np.isfinite(signal) or signal < MIN_SIGNAL:
                    raise RuntimeError(f"calibration failed: no signal from {ch.letter} channel")
                k[ch.letter] = signal / (probe_level * shutter_seconds(probe_shutter))

            # 3) One shutter, shared across R/G/B: slow enough that the dimmest channel reaches
            #    its target at LED ≤ PWM_MAX; the LED alone then balances each channel.
            shutter = shutter_at_least(max(targets[c] / (k[c] * PWM_MAX) for c in "RGB"), candidates)

            # 4) Solve each channel's LED at the shared shutter (verify + one trim + clip guard).
            #    If the set can't fit the LED window, step the *shared* shutter and re-solve all.
            channels: dict[str, ChannelCalibration] = {}
            for _attempt in range(4):
                _check_cancel()
                channels = {}
                too_dim = too_bright = False
                for i, ch in enumerate(CAPTURE_ORDER):
                    base, target = black[ch.letter], targets[ch.letter]
                    tol = 0.05 * target
                    _report(0.4 + 0.2 * i, f"Setting {ch.letter}…")
                    level = int(np.clip(round(target / max(k[ch.letter] * shutter_seconds(shutter), 1e-9)), PWM_MIN, PWM_MAX))
                    measured, clip = _shoot(ch, i, base, level, shutter)
                    # One proportional LED trim toward target (LED only — the shutter is shared).
                    if clip <= MAX_CLIP_FRACTION and abs(measured - target) > tol:
                        level = correct_led_level(level, measured, target)
                        measured, clip = _shoot(ch, i, base, level, shutter)
                    # Clip guard: p99.9 can read on-target while the top 0.1% saturates. The clear
                    # base is the whitepoint (blackpoint after inversion) and must stay below
                    # clipping — pull the LED down (the shared shutter can't move per channel).
                    if clip > MAX_CLIP_FRACTION:
                        level = int(np.clip(round(level * 0.85), PWM_MIN, PWM_MAX))
                        measured, clip = _shoot(ch, i, base, level, shutter)
                    if clip > MAX_CLIP_FRACTION:  # still clipping at reduced LED → shared shutter too slow
                        too_bright = True
                    if level >= PWM_MAX and measured < target - tol:  # maxed LED, still under → too fast
                        too_dim = True
                    channels[ch.letter] = ChannelCalibration(
                        channel=ch.letter, level=level, shutter=shutter, signal=measured, target=target, clip_fraction=clip
                    )
                    logger.info(
                        "calibrated %s → level %d, shutter %s (base %.0f, target %d, got %.0f, clip %.3f%%)",
                        ch.letter,
                        level,
                        shutter,
                        base,
                        target,
                        measured,
                        clip * 100,
                    )
                # A single shutter can't span > ~2.7 stops; nudge it toward whichever rail was hit.
                if too_dim and not too_bright:
                    step = slower_shutter(shutter, candidates)
                elif too_bright and not too_dim:
                    step = faster_shutter(shutter, candidates)
                else:
                    step = None
                if step is None:
                    break
                shutter = step

            for c in channels.values():
                if not np.isfinite(c.signal) or not np.isfinite(c.clip_fraction):
                    raise RuntimeError(f"calibration failed: {c.channel} channel produced a non-finite final measurement")
                if c.clip_fraction > MAX_CLIP_FRACTION:
                    raise RuntimeError(f"calibration failed: {c.channel} channel is still clipping at shared shutter {c.shutter}")
                if c.signal < (1.0 - MAX_TARGET_UNDER_FRACTION) * c.target:
                    raise RuntimeError(
                        f"calibration failed: {c.channel} channel remains materially below target ({c.signal:.0f} vs {c.target})"
                    )
                if c.signal > (1.0 + MAX_TARGET_OVER_FRACTION) * c.target:
                    raise RuntimeError(
                        f"calibration failed: {c.channel} channel remains materially above target ({c.signal:.0f} vs {c.target})"
                    )

            _report(1.0, "Calibration done")
            return CalibrationResult(channels=channels, black_levels=black)
        finally:
            try:
                self._light.off()
            except Exception:
                logger.exception("failed to turn the Scanlight off after calibration")

    def _capture(self, path: str, shutter: Optional[str]) -> tuple[np.ndarray, str]:
        """Capture, decode, and report *where the file landed*.

        The camera names it after its own RAW format, so the path handed in is only a
        stem: the raw-Bayer clip check must read the file that exists, not the one asked
        for. Returning both is the only way the caller can.
        """
        written = self._camera.capture(path, shutter=shutter)
        return self._demosaic(written), written
