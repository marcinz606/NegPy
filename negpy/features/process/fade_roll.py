"""Roll-wide Fade Restoration: one set of fade values pooled from the frames of a roll.

The slides of a roll faded alike, but a frame's reading also holds the light of its scene, so pooling
drops the frames that read far from the rest. Averaged crossover tables blur the film's lean with each
scene's, so the roll takes the table of one frame.
"""

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np

from negpy.features.process.fade import apply_crossover, crossover_amount, crossover_bands, crossover_offsets, measure_crossover

#: Fewest frames a roll value is pooled from.
ROLL_MIN_FRAMES = 3
#: A frame further than this many robust standard deviations from the roll median is left out.
ROLL_TRIM_MADS = 3.0
#: The smallest robust deviation, so a roll of near-equal readings keeps a frame a hair off.
ROLL_MIN_DEVIATION = 0.02


class FrameReading(NamedTuple):
    """One frame's reading: its key and the green and blue survivals Estimate found for it."""

    key: str
    green: float
    blue: float


class PooledSurvivals(NamedTuple):
    green: float
    blue: float
    #: The keys left out as reading far from the roll, in the order given.
    outliers: tuple[str, ...]


def _kept(values: np.ndarray) -> np.ndarray:
    med = float(np.median(values))
    dev = max(float(np.median(np.abs(values - med))) * 1.4826, ROLL_MIN_DEVIATION)
    return np.abs(values - med) <= ROLL_TRIM_MADS * dev


def pool_survivals(readings: Sequence[FrameReading]) -> Optional[PooledSurvivals]:
    """The mean green and blue of the frames inside ROLL_TRIM_MADS of the roll median on both; None with
    fewer than ROLL_MIN_FRAMES readings or kept frames. Tone-band grays report no real confidence, so a
    confidence weight would count the least sure frames most."""
    if len(readings) < ROLL_MIN_FRAMES:
        return None
    g = np.array([r.green for r in readings], dtype=np.float64)
    b = np.array([r.blue for r in readings], dtype=np.float64)
    keep = _kept(g) & _kept(b)
    if int(keep.sum()) < ROLL_MIN_FRAMES:
        return None
    outliers = tuple(r.key for r, k in zip(readings, keep) if not k)
    return PooledSurvivals(float(g[keep].mean()), float(b[keep].mean()), outliers)


def pool_red(reds: Sequence[float], shift_at: Sequence[Callable[[float], float]]) -> Optional[tuple[float, float]]:
    """(red, density shift): the median of each frame's Restore Range red inside ROLL_TRIM_MADS, and the
    median of the same frames' shifts at that red, as a shift holds a frame's median only at the red it
    was found for; None with fewer than ROLL_MIN_FRAMES. *shift_at* gives a frame's shift at a red."""
    if len(reds) < ROLL_MIN_FRAMES:
        return None
    r = np.array(reds, dtype=np.float64)
    keep = _kept(r)
    if int(keep.sum()) < min(ROLL_MIN_FRAMES, len(r)):
        keep = np.ones_like(keep)
    red = round(float(np.median(r[keep])), 2)
    return red, round(float(np.median([shift_at[i](red) for i in np.flatnonzero(keep)])), 4)


#: Crossover strengths the best table is judged at: Crossover is rarely right at full strength.
ROLL_CROSSOVER_STRENGTHS = (0.25, 0.5, 0.75, 1.0)
#: The best table fits the roll clearly when it leaves at most this share of the lean the frames keep with none.
ROLL_CROSSOVER_CLEAR = 0.9
#: A table's grays span at least this share of the roll's median span to be picked: a frame whose grays sit
#: in a few tones holds its scene's colors there.
ROLL_CROSSOVER_MIN_SPAN = 0.8
#: The pick tries tables on every this-many cells of each frame's grid: it ranks frames and stores no table.
ROLL_CROSSOVER_STEP = 2
#: Frames the pick tries tables on at once.
ROLL_CROSSOVER_THREADS = 4


class CrossoverPick(NamedTuple):
    """The frame whose crossover table leaves the other frames of the roll the least lean."""

    key: str
    table: tuple
    #: Frames whose grays lean less with it than with none, at the strength from ROLL_CROSSOVER_STRENGTHS
    #: that leaves the least lean, and the frames it was tried on.
    helped: int
    tried: int
    #: Whether it leaves at most ROLL_CROSSOVER_CLEAR of the lean the frames keep with none.
    clear: bool


def _gray_span(table: Sequence[float]) -> float:
    """The density span from the darkest to the lightest band of a table's grays."""
    return float(np.ptp(crossover_bands(table)[:, 0]))


def _lean_left(faded: np.ndarray, zero: float, table: Optional[Sequence[float]], strength: float = 1.0) -> float:
    """The lean the grays of *faded* keep relative to its midtone *zero* after *table* at *strength*."""
    offsets = crossover_offsets(table, strength)
    after = measure_crossover(np.asarray(apply_crossover(faded, offsets)) if offsets is not None else faded, zero)
    return crossover_amount(after) if after else float("nan")


def pick_crossover(frames: Sequence[tuple[str, np.ndarray, float, Optional[Sequence[float]]]]) -> Optional[CrossoverPick]:
    """Tries the table of each (key, faded grid, zero, table) frame whose grays span ROLL_CROSSOVER_MIN_SPAN
    of the roll at full strength on every other frame and picks the one that leaves the least median lean,
    then the strength that leaves the least; None with no such table or fewer than two frames. Frames run
    on threads, as negpy_core releases the GIL for the bands."""
    from concurrent.futures import ThreadPoolExecutor

    step = ROLL_CROSSOVER_STEP
    usable = [(k, np.ascontiguousarray(np.asarray(f)[::step, ::step]), z, t) for k, f, z, t in frames if t]
    if len(usable) < 2:
        return None
    spans = [_gray_span(f[3]) for f in usable]
    least = ROLL_CROSSOVER_MIN_SPAN * float(np.median(spans))

    def left(i: int, strength: float) -> list[float]:
        table = usable[i][3]
        return [_lean_left(f, z, table, strength) for j, (_, f, z, _) in enumerate(usable) if j != i]

    def median_left(i: int) -> float:
        return float(np.nanmedian(left(i, 1.0))) if spans[i] >= least else float("nan")

    with ThreadPoolExecutor(ROLL_CROSSOVER_THREADS) as pool:
        alone = list(pool.map(lambda f: _lean_left(f[1], f[2], None), usable))
        medians = list(pool.map(median_left, range(len(usable))))
    if np.isnan(medians).all():
        return None
    best = int(np.nanargmin(medians))
    with ThreadPoolExecutor(ROLL_CROSSOVER_THREADS) as pool:
        tries = dict(zip(ROLL_CROSSOVER_STRENGTHS, pool.map(lambda s: left(best, s), ROLL_CROSSOVER_STRENGTHS)))
    strength = min(ROLL_CROSSOVER_STRENGTHS, key=lambda s: float(np.nanmedian(tries[s])))
    before = [alone[j] for j in range(len(usable)) if j != best]
    after = tries[strength]
    helped = sum(1 for a, b in zip(after, before) if a < b)
    clear = float(np.nanmedian(after)) <= ROLL_CROSSOVER_CLEAR * float(np.nanmedian(before))
    key, _, _, table = usable[best]
    return CrossoverPick(key, tuple(table), helped, len(before), clear)
