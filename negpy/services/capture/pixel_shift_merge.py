"""Merge and check a Sony Pixel Shift Multi Shooting set, on raw mosaics. The kernels are in `negpy_core`.

A shot's position is the (row, col) of its samples on the output grid, in sensor pixels:
mosaic[y, x] is the sample of output site (y + row, x + col). The 16-shot positions add
half-pixel steps, which are whole sites on the doubled output grid.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import negpy_core
import numpy as np

Position = tuple[float, float]

#: Reject a set above either limit. Sensor pixels, deviation from the position table per row band.
WOBBLE_RMS_LIMIT = 0.25
WOBBLE_WORST_LIMIT = 0.5
#: Fewer textured row bands than this cannot be judged.
MIN_TEXTURED_BANDS = 4
#: A shot this far under the set's median detail was blurred while it was exposed.
SOFT_SHOT_LIMIT = 0.03
#: A shot this far from the set's median level had another exposure: the merge would show it as a pattern.
UNEVEN_SHOT_LIMIT = 0.01

_SONY_FOUR = ((0, 1), (1, 1), (1, 0), (0, 0))
#: Half-pixel step of each group of four in Sony's 16-shot set.
_SONY_HALF_STEPS = ((0.0, 0.5), (0.0, 0.0), (0.5, 0.5), (0.5, 0.0))
#: The position of shots 1..count for each maker and set size. A maker is offered only with a table
# confirmed on a real body: a wrong one merges every set into a pattern.
_LAYOUTS: dict[str, dict[int, tuple[Position, ...]]] = {
    "sony": {
        4: _SONY_FOUR,
        16: tuple((row + half_row, col + half_col) for half_row, half_col in _SONY_HALF_STEPS for row, col in _SONY_FOUR),
    },
}

_BAND_ROWS = 256
_BAND_COLS = 2048
#: A patch is textured above both limits. The contrast limit lies above sensor noise and pixel response non-uniformity.
_MIN_RESPONSE = 0.3
_MIN_CONTRAST = 0.005


@dataclass(frozen=True)
class WobbleReport:
    rms: float  # worst consecutive move, sensor pixels
    worst: float  # worst single row band, sensor pixels
    bands: int  # textured row bands the result rests on
    soft_shot: int = 0  # the shot with less detail than the others, 1-based; 0 for none
    uneven_shot: int = 0  # the shot whose level differs from the others, 1-based; 0 for none
    sharpest_shot: int = 1  # the shot with the most detail, 1-based

    @property
    def judged(self) -> bool:
        return self.bands >= MIN_TEXTURED_BANDS

    @property
    def steady(self) -> bool:
        """An unjudged set (blank film) cannot show movement or blur. A level difference shows on any film."""
        if self.uneven_shot:
            return False
        return not self.judged or (self.rms <= WOBBLE_RMS_LIMIT and self.worst <= WOBBLE_WORST_LIMIT and not self.soft_shot)


def shot_positions(count: int, maker: str = "sony") -> tuple[Position, ...]:
    """The (row, col) position of shots 1..count on `maker`'s bodies."""
    sizes = _LAYOUTS.get(maker, {})
    if count not in sizes:
        raise ValueError(f"no Pixel Shift set of {count} shots is known for {maker}")
    return sizes[count]


def _grid(positions: Sequence[Position]) -> tuple[int, list[tuple[int, int, int, int]]]:
    """The output scale and each shot's (whole row, whole col, half row, half col). A set must put
    every CFA site on every output site once, or the merge would leave sites without a color."""
    scale = 1 if all(float(v).is_integer() for position in positions for v in position) else 2
    steps = []
    for row, col in positions:
        if not (float(row * scale).is_integer() and float(col * scale).is_integer() and 0 <= row < 2 and 0 <= col < 2):
            raise ValueError("a Pixel Shift position must be a whole or half pixel inside one CFA cell")
        whole_row, half_row = divmod(int(row * scale), scale)
        whole_col, half_col = divmod(int(col * scale), scale)
        steps.append((whole_row, whole_col, half_row, half_col))
    if len(set(steps)) != len(steps) or len(steps) != 4 * scale * scale:
        raise ValueError("the Pixel Shift positions do not cover every site once")
    return scale, steps


def measure_wobble(mosaics: Iterable[np.ndarray], positions: Sequence[Position], black: float = 0.0) -> WobbleReport:
    """Movement between consecutive shots against `positions`, per row band, on the lit CFA color.

    The electronic shutter reads row by row, so vibration bends rows; a whole-frame average hides it.
    `mosaics` is consumed once, two frames held at a time. `black` is the sensor's black level: the
    shots' levels are compared above it.
    """
    count = len(positions)
    frames = iter(mosaics)
    previous = next(frames, None)
    if previous is None:
        raise ValueError(f"expected {count} shots, got 0")
    height, width = previous.shape
    site = negpy_core.pixel_shift_brightest_site(_mosaic(previous))
    band_cols = min(_BAND_COLS, width // 3) & ~1
    lefts = [((2 * k + 1) * width // 6 - band_cols // 2) & ~1 for k in range(3)]

    textured: set[int] = set()
    rms = worst = 0.0
    details, levels = [], []
    _measure(previous, site, details, levels)
    shots = 1
    for current in frames:
        if shots == count:
            raise ValueError(f"expected {count} shots, got more")
        (row_a, col_a), (row_b, col_b) = positions[shots - 1], positions[shots]
        expected = (col_a - col_b, row_a - row_b)  # (dx, dy) of the later shot's content
        moves = negpy_core.pixel_shift_band_moves(
            _mosaic(previous), _mosaic(current), site, (_BAND_ROWS, band_cols), lefts, expected, _MIN_CONTRAST, _MIN_RESPONSE
        )
        if moves:
            textured.update(top for top, _, _ in moves)
            deviations = np.array([(dx, dy) for _, dx, dy in moves])
            rms = max(rms, float(np.sqrt(np.mean(np.square(deviations)))))
            worst = max(worst, float(np.max(np.abs(deviations))))
        _measure(current, site, details, levels)
        previous = current
        shots += 1
    if shots != count:
        raise ValueError(f"expected {count} shots, got {shots}")
    softest = int(np.argmin(details))
    soft = details[softest] < (1.0 - SOFT_SHOT_LIMIT) * float(np.median(details))
    signal = np.array(levels) - black
    middle = float(np.median(signal))
    # The floor keeps a frame with no light steady: its level is only noise around black.
    offsets = np.abs(signal - middle) / max(middle, black / 10, 1.0)
    uneven = int(np.argmax(offsets))
    return WobbleReport(
        rms=rms,
        worst=worst,
        bands=len(textured),
        soft_shot=softest + 1 if soft else 0,
        uneven_shot=uneven + 1 if offsets[uneven] > UNEVEN_SHOT_LIMIT else 0,
        sharpest_shot=int(np.argmax(details)) + 1,
    )


def _measure(shot: np.ndarray, site: tuple[int, int], details: list[float], levels: list[float]) -> None:
    """Add one shot's detail (steps per unit of level) and its mean level of the site."""
    steps, total = negpy_core.pixel_shift_site_detail(_mosaic(shot), site)
    details.append(steps / max(total, 1))
    levels.append(total / max(shot.size // 4, 1))


def _mosaic(shot: np.ndarray) -> np.ndarray:
    if shot.ndim != 2 or shot.dtype != np.uint16:
        raise ValueError("every shot must be a uint16 mosaic of one size")
    return np.ascontiguousarray(shot)


def merge_set(
    mosaics: Iterable[np.ndarray],
    colors: np.ndarray,
    positions: Sequence[Position],
    allocate: Callable[[Sequence[int]], Sequence[np.ndarray]] = lambda sizes: [np.empty(n, np.uint16) for n in sizes],
) -> np.ndarray:
    """One uint16 RGB image from one mosaic per position: every site takes its own measured sample,
    the two greens averaged, no interpolation. Whole-pixel positions keep the sensor's size, half
    pixels double it; only sites with every sample are kept. Black level is untouched. `mosaics` is consumed once, one frame held at a time.
    `allocate` gives one uint16 buffer per size, in samples: the result, and the second green the merge works with.
    """
    count = len(positions)
    scale, steps = _grid(positions)
    cfa = [int(color) for color in colors[:2, :2].ravel()]
    if sorted(cfa) != [0, 1, 2, 3]:
        raise ValueError("Pixel Shift needs a 2x2 Bayer pattern")
    frames = iter(mosaics)
    out = second_green = np.empty((0, 0), np.uint16)
    size = None
    for shot, (whole_row, whole_col, half_row, half_col) in enumerate(steps):
        mosaic = next(frames, None)
        if mosaic is None:
            raise ValueError(f"expected {count} shots, got {shot}")
        if size is None:
            size = height, width = mosaic.shape
            rows, cols = scale * (height - 1), scale * (width - 1)
            image, plane = allocate((3 * rows * cols, rows * cols))
            out, second_green = image.reshape(rows, cols, 3), plane.reshape(rows, cols)
        if mosaic.shape != size:
            raise ValueError("every shot must be a uint16 mosaic of one size")
        negpy_core.pixel_shift_place(out, second_green, _mosaic(mosaic), cfa, scale, (1 - whole_row, 1 - whole_col), (half_row, half_col))
        del mosaic
    if next(frames, None) is not None:
        raise ValueError(f"expected {count} shots, got more")
    negpy_core.pixel_shift_average_greens(out, second_green)
    return out
