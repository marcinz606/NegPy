"""Per-channel ETTR exposure auto-calibration for RGB narrowband film scanning.

The film base is metered inside a user ROI and each of R/G/B is exposed just below clipping
("expose to the right"). The key idea is a **linear model**:

    Signal_c = k_c · Level_c · t

where `k_c` (the channel response) is *measured*, never assumed, so any sensor (Sony, Fuji
X-Trans, …) is handled automatically. With one **shared** shutter `t` and three per-channel LED
levels there are 4 knobs and 3 targets, leaving one degree of freedom (the shutter). It is fixed
uniquely by putting the **dimmest** channel near PWM_MAX_SAFE (fastest shutter + highest levels at
once — no quality/speed trade-off; the gap to PWM_MAX is the verify trim's headroom). Everything
is then solved in one shot instead of searched.

Two representations of a shutter coexist deliberately. Labels are rounded display names for a
geometric ladder ("1/3" exposes 0.315 s, not 0.333 s): ordering/snapping read the label literally
(`shutter_seconds`), anything multiplied into the physics uses the rung's true time
(`true_seconds`). The model above only holds in the second representation.

No dark frame: rawpy already subtracts the sensor bias, so `black = 0` (and the injected demosaic
must scale by the camera's white level, never per-frame — see `linear_demosaic`). Physical limits
degrade gracefully (best result + per-channel status) rather than raising. Hardware-free: light,
camera and a `demosaic` callable are injected.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import numpy as np

from negpy.infrastructure.capture.base import CAPTURE_ORDER, Camera, LightSource
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# The decode normalises the camera's white level to full 16-bit, so the *demosaiced* saturation
# point is camera-independent (65535). This only holds because `linear_demosaic` pins
# adjust_maximum_thr=0.0 — on LibRaw's default the reference silently becomes the frame's own
# brightest pixel, every frame gets its own scale, and nothing below is comparable between shots
# (that bug read as an LED plateau for a whole rig session). The camera-SPECIFIC saturation — the
# raw sensor white level, which differs per body (full-well capacity + ADC bit depth) — is handled
# separately by the raw-Bayer check (raw_channel_clip_fraction, which reads the real per-camera
# white_level and also catches clipped photosites the demosaic averages away). This demosaiced
# ceiling is a fast secondary guard on the normalised scale.
CLIP_CEILING = 65535
# A demosaiced pixel this close to the normalised ceiling counts as clipped; the ~0.2 % margin
# absorbs demosaic interpolation + read noise just below saturation.
SATURATION_VALUE = int(CLIP_CEILING * 0.998)  # ≈ 65404
PWM_MIN = 40
PWM_MAX = 255
# Aim the dimmest channel here, not at 255 — this is the phase-4 trim's headroom, not an LED limit.
# A measured level ramp (40→255 per channel, on the fixed decode — see raw_demosaic) shows the LEDs
# do not saturate: they are gently concave, red losing only ~2 % of k_eff between levels 200 and 250.
# So `k`, measured at the probe level, slightly over-states the light at the solved level and the
# first shot lands ~2 % under target; the trim corrects that by raising the level, which needs room
# above. 250 leaves ~3 % (a solved ~247 can still reach the 255 clamp) — enough for the measured
# curvature — and it is the lowest ceiling that still buys a full shutter step: on a C-41 base at
# ISO 100 / f8 the solved shutter snaps to 1/3 rather than 4/10. Above 250 nothing more is gained
# (the level follows from the snapped shutter), and the LED itself is near its limit there: level
# 255 yields only ~21 % more light than the ~206 a 4/10 solve lands on.
PWM_MAX_SAFE = 250
TARGET_FRACTION = 0.9  # expose the film base to 90 % of the usable range
MIN_SIGNAL = 10.0  # counts; below this the channel read no real signal
# ETTR meters p99.9 (ignores the top 0.1 %), so the base can read on-target while a sliver clips.
# The base is the whitepoint (blackpoint after inversion) and must stay just below clipping.
MAX_CLIP_FRACTION = 0.002
# A channel this far under target is materially under-exposed, from any cause — maxed LED at the
# slowest shutter, or a clip-guard that pulled the LED down hard. Reported as status "under"; a
# small clip-guard undershoot within this margin still counts as "target".
MAX_TARGET_UNDER_FRACTION = 0.2
# Probe budget = the whole reachable range, so the loop can only end by resolving (in-range return,
# graceful-over return, or the dark-side break) — never by exhaustion, which would mislabel a
# blinding over-exposure as "no signal". Worst case is a deeply-over scene from the slowest start:
# ~9 shutter halvings (2 s → 1/250) + 3 LED halvings (255 → 40) + the final in-range measurement.
_MAX_PROBE_STEPS = 14
_MAX_CLIP_GUARD_STEPS = 12  # LED-down steps (PWM_MAX→PWM_MIN at 0.85×) — keeps captures hard-bounded

# Shutter ladder, fastest first (third-stops). Extends to 2 s (up from 1 s) so a closed-down
# aperture can still reach target on the dim channel instead of failing (dark current at ISO 100
# / ≤2 s is negligible). Faster than 1/250 s is avoided (PWM-LED banding). The body's own writable
# ladder is preferred when live view has published it.
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
    "1.3",
    "1.6",
    "2",
)

# --- Fixed start point (Phase 1) ------------------------------------------------------------
# A neutral reference the calibration always starts from, normalized to the live ISO/aperture.
# Rig-measured on a Portra 400 clear base at ISO 100 / f8 (two runs, kR 694…716, solving to levels
# 206…213 / 94…96 / 82…83 at 4/10) and rounded to tidy numbers for the UI. The exact values matter
# little — the probe only measures `k` from here and the phase-4 trim absorbs the rest; a few levels
# either way is well inside the ~3 % run-to-run spread of k itself. What matters is that the start
# point sits close to where the solve lands, so `k` is measured at roughly the level the channel
# ends up at and the LEDs' gentle concavity cancels instead of biasing the solve. Never a previous
# preset — a badly placed ROI in a past run must not poison the next calibration.
REFERENCE_ISO = 100.0
REFERENCE_APERTURE = 8.0
# 0.4 s, spelled in SHUTTER_CANDIDATES' vocabulary — bodies name this speed differently (the a7C II
# publishes "4/10"), and normalize_start_point re-snaps onto whatever ladder the body reports.
REFERENCE_SHUTTER = "0.4"
REFERENCE_LEVELS = (210, 95, 80)  # (R, G, B); R needs the most drive (665 nm, low sensor QE)

DemosaicFn = Callable[[str], np.ndarray]  # path -> HxWx3 linear array (0..CLIP_CEILING)
ProgressCb = Callable[[float, str], None]


def shutter_seconds(label: str) -> float:
    """Parse a shutter label ('1/100', '0.4', '1') into its *nominal* seconds.

    This is the label read literally — the value to sort, snap and filter by, since the ladder is
    monotonic whether or not the labels are rounded. It is NOT the exposure time: anywhere the
    number is multiplied into the physics (k, the level solve, shutter_at_least's ≥-comparison)
    must use `true_seconds` instead.
    """
    label = label.strip()
    if "/" in label:
        num, den = label.split("/", 1)
        denominator = float(den)
        if denominator == 0:
            # Some bodies (e.g. the a7 IV) publish a bulb-like shutter label with a zero
            # denominator. Raise ValueError (not a bare ZeroDivisionError) so the shutter-ladder
            # filter in _available_shutters drops the label instead of crashing calibration.
            raise ValueError(f"invalid shutter label {label!r}: zero denominator")
        return float(num) / denominator
    return float(label)


def usable_ladder(candidates: tuple[str, ...]) -> tuple[str, ...]:
    """The parseable, ascending subset of a body's shutter labels; empty if none are usable.

    A body's ladder is untrusted input — the a7 IV publishes a bulb-like "1/0", others publish
    "Bulb" or "" — and one such label used to kill calibration before it started (#478). The UI
    filters them out before handing the ladder over, but every function below indexes and parses
    this tuple, so it is cleaned once here as well: a second caller passing a raw ladder must not
    reopen that bug.
    """
    parsed: dict[str, float] = {}
    for label in candidates:
        try:
            seconds = shutter_seconds(label)
        except (TypeError, ValueError):
            continue  # bulb-like or non-numeric — not an exposure this solver can reason about
        if seconds > 0:
            parsed[label] = seconds
    return tuple(sorted(parsed, key=parsed.__getitem__))


@lru_cache(maxsize=8)
def _ladder_stops(candidates: tuple[str, ...]) -> float:
    """The ladder's spacing in stops, measured from the body's own labels — never assumed.

    Bodies step the shutter in thirds (the common default) or halves, and which one it is decides
    what a rounded label actually means. Reading it off the ladder keeps `true_seconds` camera-
    agnostic. Labels are individually rounded by up to ~7 %, so the *median* neighbour ratio is
    taken — it ignores that jitter. Falls back to thirds when the ladder is too short to read.
    """
    secs = [shutter_seconds(c) for c in usable_ladder(candidates)]  # ascending, unparseables dropped
    ratios = [b / a for a, b in zip(secs, secs[1:]) if b > a]
    if len(ratios) < 3:
        return 1.0 / 3.0
    step = np.log2(float(np.median(ratios)))
    return min((1.0 / 3.0, 1.0 / 2.0), key=lambda s: abs(s - step))


def true_seconds(label: str, candidates: tuple[str, ...] = SHUTTER_CANDIDATES) -> float:
    """The label's TRUE exposure time, undoing the display rounding.

    Shutter labels are rounded names for a geometric ladder, not exact times: on a third-stop
    ladder "1/6" means 2^(-8/3) = 0.157 s (the fraction says 0.167) and "1/3" means 0.315 s (the
    fraction says 0.333) — up to 6.7 % off. The solver multiplies `k` by this number, so the
    rounding lands straight in the exposure: a rig run that probed at "0.4" (0.8 % off) and solved
    at "1/3" (5.8 % off) came out ~4 % under target with the LED already at the 255 clamp.

    Only the physics needs this. Ordering/snapping keep using the nominal `shutter_seconds`.
    """
    nominal = shutter_seconds(label)
    if nominal <= 0:
        return nominal
    stops = _ladder_stops(tuple(candidates) or SHUTTER_CANDIDATES)
    exact = float(2.0 ** (round(np.log2(nominal) / stops) * stops))
    # Only correct what is plausibly the same rung. A label rounds by ≲7 %; a wider gap means this
    # value isn't on this ladder (a bulb entry, an oddly labelled body), and then the label itself
    # is the better guess — never invent a correction larger than the rounding it undoes.
    return exact if abs(exact / nominal - 1.0) <= 0.08 else nominal


def aperture_fnumber(label: str) -> Optional[float]:
    """Parse an aperture label ('f/8', 'F8', '8', 'f/5.6') into an f-number, or None.

    None when the lens has no electronic aperture (manual enlarging glass) — the caller then
    skips the aperture term of the start-point normalization and lets the probe adjust."""
    if not label:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(label))
    if not m:
        return None
    try:
        f = float(m.group(1))
    except ValueError:
        return None
    return f if f > 0 else None


def normalize_start_point(
    iso: str,
    aperture: str,
    *,
    levels: tuple[int, int, int] = REFERENCE_LEVELS,
    shutter: str = REFERENCE_SHUTTER,
    candidates: tuple[str, ...] = SHUTTER_CANDIDATES,
) -> tuple[tuple[int, int, int], str]:
    """Scale the fixed reference start point to the live ISO/aperture (Phase 1, no capture).

    Exposure ∝ ISO · t / f², so to keep the same sensor exposure the shutter scales by
    (ISO_ref/ISO)·(f/f_ref)². Levels stay fixed (same Scanlight for everyone); only the shutter
    moves (the camera-side difference). ISO or aperture unreadable → that term is skipped and the
    probe absorbs the rest. Returns (levels, shutter snapped to the ladder)."""
    t = shutter_seconds(shutter)
    try:
        iso_now = float(re.sub(r"[^0-9.]", "", str(iso)) or REFERENCE_ISO)
    except ValueError:
        iso_now = REFERENCE_ISO
    if iso_now > 0:
        t *= REFERENCE_ISO / iso_now
    f_now = aperture_fnumber(aperture)
    if f_now is not None:
        t *= (f_now / REFERENCE_APERTURE) ** 2
    # Snap the raw seconds value straight onto the ladder (no label round-trip, which mislabels
    # e.g. 0.8 s as "1/1"). Like calibrate(), clean the body's ladder on the way in — this is a
    # public entry point receiving camera-reported labels, and one bulb-like "1/0" must degrade to
    # a dropped entry, not a ValueError (#478).
    return levels, _nearest_by_seconds(t, usable_ladder(tuple(candidates)) or SHUTTER_CANDIDATES)


def _nearest_by_seconds(seconds: float, candidates: tuple[str, ...]) -> str:
    """The candidate closest to a raw seconds value (no shutter-label round-trip)."""
    ladder = candidates or SHUTTER_CANDIDATES
    return min(ladder, key=lambda c: abs(shutter_seconds(c) - seconds))


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
    shutter: str  # solved camera shutter label (shared across channels)
    signal: float  # measured base p99.9 at the solved settings
    target: int  # target signal
    clip_fraction: float = 0.0  # fraction of base pixels at/above saturation (ETTR keeps this ~0)
    status: str = "target"  # "target" | "under" | "over" — for graceful UI messaging


@dataclass(frozen=True)
class CalibrationResult:
    channels: dict[str, ChannelCalibration]
    spread_stops: float = 0.0  # measured k spread in stops (confirms the shared-shutter assumption)

    @property
    def levels(self) -> tuple[int, int, int]:
        return (self.channels["R"].level, self.channels["G"].level, self.channels["B"].level)

    @property
    def shutters(self) -> tuple[str, str, str]:
        return (self.channels["R"].shutter, self.channels["G"].shutter, self.channels["B"].shutter)

    @property
    def status(self) -> str:
        """Worst channel status, for a single headline message."""
        for s in ("over", "under"):
            if any(c.status == s for c in self.channels.values()):
                return s
        return "target"


def target_signal(target_fraction: float = TARGET_FRACTION) -> int:
    """ETTR target in counts. black = 0 (rawpy already removed the sensor bias)."""
    return int(round(target_fraction * CLIP_CEILING))


def meter_base(plane: np.ndarray, roi: Roi) -> float:
    """p99.9 of the ROI on one demosaiced channel plane (black already 0 after rawpy)."""
    h, w = plane.shape[:2]
    x0, y0, x1, y1 = roi.pixels(w, h)
    patch = plane[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.percentile(patch, 99.9))


def clip_fraction(plane: np.ndarray, roi: Roi, saturation: float = SATURATION_VALUE) -> float:
    """Fraction of ROI pixels at/above saturation on one demosaiced channel plane.

    ETTR meters p99.9, which ignores the top 0.1 % by design — so a base can read on-target while
    a sliver saturates. The clear base is the whitepoint (blackpoint after inversion), so it must
    stay just *below* clipping; this metric catches what p99.9 hides."""
    h, w = plane.shape[:2]
    x0, y0, x1, y1 = roi.pixels(w, h)
    patch = plane[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch >= saturation))


def nearest_shutter(label: str, candidates: tuple[str, ...]) -> str:
    """The candidate closest (in seconds) to `label`, snapping onto the camera's own ladder."""
    if not candidates or label in candidates:
        return label
    target = shutter_seconds(label)
    return min(candidates, key=lambda c: abs(shutter_seconds(c) - target))


def shutter_at_least(seconds: float, candidates: tuple[str, ...] = SHUTTER_CANDIDATES) -> str:
    """The fastest candidate whose exposure time is ≥ `seconds` (candidates are fastest-first).

    This is the shared-shutter pick: as fast as possible while the dimmest channel still reaches
    target within the LED range. Falls back to the slowest candidate if none is slow enough.

    Compares TRUE exposure, not the label: "≥ seconds" is a claim about light. A body's "1/3"
    reads as 0.333 s but exposes 0.315 s, so trusting the label picks a rung that is really too
    fast, and the channel then cannot reach target however far the LED is pushed."""
    for c in candidates:
        if true_seconds(c, candidates) >= seconds:
            return c
    return candidates[-1]


def correct_led_level(current_level: int, signal: float, target: int) -> int:
    """One proportional LED trim toward target (shutter held)."""
    if signal < MIN_SIGNAL:
        return PWM_MAX
    corrected = round(current_level * (float(target) / float(signal)))
    return int(np.clip(corrected, PWM_MIN, PWM_MAX))


class CalibrationService:
    """Solves ETTR exposures: measure per-channel response, solve analytically, verify."""

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

    def calibrate(
        self,
        roi: Roi,
        scratch_path: str,
        *,
        start_levels: tuple[int, int, int] = REFERENCE_LEVELS,
        start_shutter: str = REFERENCE_SHUTTER,
        target_fraction: float = TARGET_FRACTION,
        candidates: tuple[str, ...] = SHUTTER_CANDIDATES,
        progress: Optional[ProgressCb] = None,
        cancel=None,
    ) -> CalibrationResult:
        # Clean once, here: everything downstream parses and indexes this ladder, and an unparseable
        # label from the body would otherwise surface as a crash mid-run (#478) rather than a dropped
        # entry. Empty (or entirely unusable) → the built-in ladder.
        candidates = usable_ladder(tuple(candidates)) or SHUTTER_CANDIDATES
        start_shutter = nearest_shutter(start_shutter, candidates)
        T = target_signal(target_fraction)

        def _check_cancel():
            if cancel is not None and cancel.is_set():
                raise RuntimeError("calibration cancelled")

        _floor = [0.0]

        def _report(frac: float, msg: str):
            _floor[0] = max(_floor[0], frac)
            if progress is not None:
                progress(_floor[0], msg)

        def _shoot(i: int, ch, level: int, shutter: str) -> tuple[float, float]:
            """Light channel `ch` at `level`, capture at `shutter`, meter → (base p99.9, clip)."""
            self._light.set_color(*ch.rgb(level))
            self._sleep(self._settle_s)
            img, written = self._capture(scratch_path, shutter=shutter)
            clip = max(clip_fraction(img[..., i], roi), self._source_clip_fraction(written, i, roi))
            return meter_base(img[..., i], roi), clip

        try:
            # --- Phase 2: measure the response k per channel (adaptive probe) ------------------
            k: dict[str, float] = {}
            for i, ch in enumerate(CAPTURE_ORDER):
                _check_cancel()
                _report(0.1 + 0.2 * i, f"Probing {ch.letter}…")
                k[ch.letter] = self._measure_response(i, ch, start_levels[i], start_shutter, candidates, _shoot)

            spread = _spread_stops(k)
            logger.info("calibration response: kR=%.1f kG=%.1f kB=%.1f (spread %.2f stops)", k["R"], k["G"], k["B"], spread)

            # --- Phase 3: solve the shared shutter + per-channel levels analytically -----------
            _report(0.7, "Solving…")
            shutter, levels = _solve_shared(k, T, candidates)

            # --- Phase 4: verify + one trim + clip guard --------------------------------------
            channels: dict[str, ChannelCalibration] = {}
            for i, ch in enumerate(CAPTURE_ORDER):
                _check_cancel()
                _report(0.75 + 0.08 * i, f"Setting {ch.letter}…")
                channels[ch.letter] = self._verify_channel(i, ch, levels[ch.letter], shutter, T, _shoot)

            _report(1.0, "Calibration done")
            return CalibrationResult(channels=channels, spread_stops=spread)
        finally:
            try:
                self._light.off()
            except Exception:
                logger.exception("failed to turn the Scanlight off after calibration")

    def _measure_response(self, i, ch, start_level, start_shutter, candidates, shoot) -> float:
        """Bring a probe into the measurable range (not clipped, above noise), then return
        k = signal / (level · seconds). k is exposure-normalized, so a faster/slower probe gives the
        same k, just clean. A clipped read is used only as a lower bound when even minimum exposure
        clips (over-exposure → graceful "over"); no signal at maximum exposure raises."""
        level, shutter = start_level, start_shutter
        for _ in range(_MAX_PROBE_STEPS):
            signal, clip = shoot(i, ch, level, shutter)
            # Halve/double the exposure per step (log-convergence: 8 steps span 8 stops), moving the
            # shutter first and only dropping to the LED when the ladder end is reached. A 1-stop
            # step can't jump the ~12.6-stop measurable window, so it never overshoots clip↔no-signal.
            if clip > MAX_CLIP_FRACTION or signal >= SATURATION_VALUE:  # too bright → halve exposure
                faster = _nearest_by_seconds(shutter_seconds(shutter) * 0.5, candidates)
                if shutter_seconds(faster) < shutter_seconds(shutter):
                    shutter = faster
                    continue
                if level > PWM_MIN:
                    level = max(PWM_MIN, level // 2)
                    continue
                # Minimum exposure (fastest shutter + lowest LED) still clips → the aperture is too
                # open. Return the clipped (lower-bound) response so the solver seats this channel at
                # minimum exposure and _verify_channel reports "over" (the UI tells the user to stop
                # down). Graceful, like under-exposure — not a hard error.
                return signal / (level * true_seconds(shutter, candidates))
            if signal < MIN_SIGNAL:  # too dark → double exposure
                slower = _nearest_by_seconds(shutter_seconds(shutter) * 2.0, candidates)
                if shutter_seconds(slower) > shutter_seconds(shutter):
                    shutter = slower
                    continue
                if level < PWM_MAX:
                    level = PWM_MAX
                    continue
                break  # slowest shutter + max LED and still no signal — dead LED / ROI off the base
            # k is exposure-normalised, so it must divide by the TRUE exposure, not the rounded
            # label — otherwise k inherits the label's error and the solver spends it elsewhere.
            return signal / (level * true_seconds(shutter, candidates))
        # Only "no signal even at maximum exposure" reaches here (over-exposure returns gracefully
        # above, under-exposure is handled by the solver + _verify_channel's "under" status).
        raise RuntimeError(
            f"calibration failed: no signal from the {ch.letter} channel even at maximum exposure "
            f"(check the ROI is on the clear film base and the Scanlight is on)"
        )

    def _verify_channel(self, i, ch, level, shutter, T, shoot) -> ChannelCalibration:
        """Capture at the solved settings, one proportional trim, then a clip guard. Returns the
        channel calibration with a graceful status (never raises on a physical limit)."""
        tol = 0.05 * T
        measured, clip = shoot(i, ch, level, shutter)
        if clip <= MAX_CLIP_FRACTION and abs(measured - T) > tol:
            level = correct_led_level(level, measured, T)
            measured, clip = shoot(i, ch, level, shutter)
        # Clip guard: pull the LED down until the base sits below clipping (p99.9 can read on-target
        # while the top 0.1 % saturates). Iterate — a dense base overshoots — re-measuring each time.
        # Bounded by _MAX_CLIP_GUARD_STEPS so the capture budget stays hard-bounded even in the worst
        # case (a level solved near PWM_MAX that still clips).
        for _ in range(_MAX_CLIP_GUARD_STEPS):
            if clip <= MAX_CLIP_FRACTION or level <= PWM_MIN:
                break
            level = max(PWM_MIN, int(round(level * 0.85)))
            measured, clip = shoot(i, ch, level, shutter)

        status = _channel_status(measured, clip, T)
        logger.info(
            "calibrated %s → level %d, shutter %s (target %d, got %.0f, clip %.3f%%, %s)",
            ch.letter,
            level,
            shutter,
            T,
            measured,
            clip * 100,
            status,
        )
        return ChannelCalibration(
            channel=ch.letter,
            level=level,
            shutter=shutter,
            signal=measured,
            target=T,
            clip_fraction=clip,
            status=status,
        )

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

    def _capture(self, path: str, shutter: Optional[str]) -> tuple[np.ndarray, str]:
        """Capture, decode, and report *where the file landed* (the camera names it after its own
        RAW format, so the handed-in path is only a stem; the clip check must read what exists)."""
        written = self._camera.capture(path, shutter=shutter)
        return self._demosaic(written), written


def _channel_status(measured: float, clip: float, target: int) -> str:
    """Graceful per-channel status from the *measured signal*, not the level. 'over' if the base
    still clips; 'under' if it is materially below target from any cause — a maxed LED at the
    slowest shutter, OR a clip-guard that pulled the LED well below PWM_MAX; else 'target' (a small
    clip-guard undershoot within the margin stays 'target')."""
    if clip > MAX_CLIP_FRACTION:
        return "over"
    if measured < (1.0 - MAX_TARGET_UNDER_FRACTION) * target:
        return "under"
    return "target"


def _spread_stops(k: dict[str, float]) -> float:
    """Channel response spread in stops = log2(max k / min k). Confirms one shutter can serve all
    three (must stay below the ~2.7-stop LED window)."""
    vals = [v for v in k.values() if v > 0]
    if len(vals) < 2:
        return 0.0
    return float(np.log2(max(vals) / min(vals)))


def _solve_shared(k: dict[str, float], T: int, candidates: tuple[str, ...]) -> tuple[str, dict[str, int]]:
    """Analytic core: pick the fastest shutter that keeps the dimmest channel at ≤ PWM_MAX_SAFE,
    then set each level = T / (k·t). One shot, no search."""
    k_min = min(k.values())
    t_ideal = T / (k_min * PWM_MAX_SAFE)  # dimmest channel sits at ~PWM_MAX_SAFE here
    shutter = shutter_at_least(t_ideal, candidates)
    # Solve the levels against the shutter's TRUE time: the ladder is snapped by nominal label, but
    # what the sensor actually receives is the rounded-off time (a "1/3" pick exposes 0.315 s, not
    # 0.333 s — 5.8 % less light than the label promises).
    secs = true_seconds(shutter, candidates)
    levels = {c: int(np.clip(round(T / (k[c] * secs)), PWM_MIN, PWM_MAX)) for c in k}
    return shutter, levels
