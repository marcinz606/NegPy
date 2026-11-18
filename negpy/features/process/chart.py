"""Fit the density-domain crosstalk matrix to a color chart photographed on the roll.

The matrix acts on the raw negative before the meters, so the fit runs the real render in the
loop: each trial matrix is rendered, the chart's patches are read back and compared with the
chart's published colors. Lightness counts at half weight, because the print curve decides
tone and the matrix should not bend to it.
"""

from negpy.kernel.system.i18n import mark
from dataclasses import dataclass
from typing import Callable, Optional

import negpy_core
import numpy as np

from negpy.kernel.image.logic import _WORKING_GAMMA, rgb_to_lab_working


@dataclass(frozen=True)
class ChartSpec:
    """A card whose patches sit on a grid: even columns, rows at `row_centers`. The side that
    carries the columns is the longer one between the corner-patch centers."""

    name: str
    patch_names: tuple[str, ...]
    lab_d50: tuple[tuple[float, float, float], ...]  # row-major, as the patches sit on the card
    exposure_patches: tuple[int, ...]  # mid grays that set the exposure match
    cols: int = 6
    # Each row's center, as a fraction of the way from the first row's center to the last's.
    row_centers: tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)

    @property
    def rows(self) -> int:
        return len(self.row_centers)

    @property
    def patch_count(self) -> int:
        return self.rows * self.cols


# Datacolor's published values (D50) for columns E-H of the SpyderCheckr 48, which the 24 carries.
# fmt: off
SPYDERCHECKR_24 = ChartSpec(
    "SpyderCheckr 24",
    (
        mark("Card White"), mark("20% Gray"), mark("40% Gray"), mark("60% Gray"), mark("80% Gray"), mark("Card Black"),
        mark("Primary Cyan"), mark("Primary Magenta"), mark("Primary Yellow"), mark("Primary Red"), mark("Primary Green"), mark("Primary Blue"),
        mark("Primary Orange"), mark("Blueprint"), mark("Pink"), mark("Violet"), mark("Apple Green"), mark("Sunflower"),
        mark("Aqua"), mark("Lavender"), mark("Evergreen"), mark("Steel Blue"), mark("Classic Light Skin"), mark("Classic Dark Skin"),
    ),
    (
        (96.04, 2.16, 2.60), (80.44, 1.17, 2.05), (65.52, 0.69, 1.86), (49.62, 0.58, 1.56), (33.55, 0.35, 1.40), (16.91, 1.43, -0.81),
        (47.12, -32.50, -28.75), (50.49, 53.45, -13.55), (83.61, 3.36, 87.02), (41.05, 60.75, 31.17), (54.14, -40.80, 34.75), (24.75, 13.78, -49.48),
        (60.94, 38.21, 61.31), (37.80, 7.30, -43.04), (49.81, 48.50, 15.76), (28.88, 19.36, -24.48), (72.45, -23.60, 60.47), (71.65, 23.74, 72.28),
        (70.19, -31.90, 1.98), (54.38, 8.84, -25.71), (42.03, -15.80, 22.93), (48.82, -5.11, -23.08), (65.10, 18.14, 18.68), (36.13, 14.15, 15.78),
    ),
    (1, 2, 3, 4),
)

# X-Rite's values (D50) for charts made after November 2014.
COLORCHECKER_24 = ChartSpec(
    "ColorChecker 24",
    (
        mark("Dark Skin"), mark("Light Skin"), mark("Blue Sky"), mark("Foliage"), mark("Blue Flower"), mark("Bluish Green"),
        mark("Orange"), mark("Purplish Blue"), mark("Moderate Red"), mark("Purple"), mark("Yellow Green"), mark("Orange Yellow"),
        mark("Blue"), mark("Green"), mark("Red"), mark("Yellow"), mark("Magenta"), mark("Cyan"),
        mark("White"), mark("Neutral 8"), mark("Neutral 6.5"), mark("Neutral 5"), mark("Neutral 3.5"), mark("Black"),
    ),
    (
        (37.54, 14.37, 14.92), (64.66, 19.27, 17.50), (49.32, -3.82, -22.54), (43.46, -12.74, 22.72), (54.94, 9.61, -24.79), (70.48, -32.26, -0.37),
        (62.73, 35.83, 56.50), (39.43, 10.75, -45.17), (50.57, 48.64, 16.67), (30.10, 22.54, -20.87), (71.77, -24.13, 58.19), (71.51, 18.24, 67.37),
        (28.37, 15.42, -49.80), (54.38, -39.72, 32.27), (42.43, 51.05, 28.62), (81.80, 2.67, 80.41), (50.63, 51.28, -14.12), (49.57, -29.71, -28.32),
        (95.19, -1.03, 2.93), (81.29, -0.57, 0.44), (66.89, -0.75, -0.06), (50.76, -0.13, 0.14), (35.63, -0.46, -0.48), (20.64, 0.07, -0.46),
    ),
    (19, 20, 21, 22),
)

# DGK Color Tools' published values for the DKC-Pro, read as D50. The card has a row of tall grays
# over a caption strip and two rows of colors.
DKC_PRO = ChartSpec(
    "DKC-Pro",
    (
        mark("White"), mark("Gray 73"), mark("Gray 62"), mark("Gray 50"), mark("Gray 38"), mark("Black"),
        mark("Red"), mark("Yellow"), mark("Green"), mark("Cyan"), mark("Blue"), mark("Magenta"),
        mark("CIE TSC 01"), mark("CIE TSC 02"), mark("CIE TSC 06"), mark("CIE TSC 08"), mark("CIE TSC 09"), mark("CIE TSC 10"),
    ),
    (
        (97.0, 0.0, 1.0), (73.0, 0.0, 0.0), (62.0, 0.0, 0.0), (50.0, 0.0, 0.0), (38.0, 0.0, 0.0), (23.0, 0.0, 0.0),
        (48.0, 59.0, 39.0), (92.0, 1.0, 95.0), (64.0, -40.0, 54.0), (57.0, -41.0, -42.0), (18.0, -3.0, -25.0), (49.0, 60.0, -3.0),
        (41.0, 51.0, 26.0), (61.0, 29.0, 57.0), (52.0, -24.0, -24.0), (52.0, 47.0, -14.0), (69.0, 14.0, 17.0), (64.0, 12.0, 17.0),
    ),
    (1, 2, 3, 4),
    row_centers=(0.0, 0.62, 1.0),
)
# fmt: on

CHARTS: tuple[ChartSpec, ...] = (SPYDERCHECKR_24, COLORCHECKER_24, DKC_PRO)

_D50 = np.array([0.96422, 1.0, 0.82521])
_D65 = np.array([0.95047, 1.0, 1.08883])
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
_D50_TO_D65 = np.linalg.inv(_BRADFORD) @ np.diag((_BRADFORD @ _D65) / (_BRADFORD @ _D50)) @ _BRADFORD

# The four ways a grid can be read from its corners: as printed, mirrored (a negative scanned
# through the base), upside down, and both.
ORIENTATIONS = range(4)

# A patch is read over this fraction of the grid pitch, clear of the card's frame and patch edges.
_SAMPLE_FRACTION = 0.4
# Lightness weight in the fit's ΔE00: the print curve, not the matrix, owns tone.
_FIT_KL = 2.0
# Pull toward identity, per off-diagonal term, in ΔE00 units per unit of the term.
_RIDGE = 0.5
_FD_STEP = 0.01
_MAX_ITERATIONS = 48
_MIN_STEP = 2e-3


def reference_lab(chart: ChartSpec) -> np.ndarray:
    """The chart's colors as CIELAB D65, the render's Lab, through a Bradford adaptation."""
    lab = np.asarray(chart.lab_d50, dtype=np.float64)
    fy = (lab[:, 0] + 16.0) / 116.0
    f = np.stack([fy + lab[:, 1] / 500.0, fy, fy - lab[:, 2] / 200.0], axis=1)
    xyz = np.where(f > 6.0 / 29.0, f**3, (f - 16.0 / 116.0) / 7.787) * _D50
    t = (xyz @ _D50_TO_D65.T) / _D65
    g = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
    return np.stack([116.0 * g[:, 1] - 16.0, 500.0 * (g[:, 0] - g[:, 1]), 200.0 * (g[:, 1] - g[:, 2])], axis=1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray, k_l: float = 1.0) -> np.ndarray:
    """CIEDE2000 between rows of two (N, 3) Lab arrays (Sharma, Wu & Dalal 2005)."""
    l1, a1, b1 = np.asarray(lab1, dtype=np.float64).T
    l2, a2, b2 = np.asarray(lab2, dtype=np.float64).T
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2.0
    g = 0.5 * (1.0 - np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2 = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dh = h2 - h1
    dh = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dh = np.where(c1p * c2p == 0.0, 0.0, dh)
    d_hp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2.0))
    l_bar, c_bar_p = (l1 + l2) / 2.0, (c1p + c2p) / 2.0
    h_sum = h1 + h2
    h_bar = np.where(
        c1p * c2p == 0.0, h_sum, np.where(np.abs(h1 - h2) > 180.0, np.where(h_sum < 360.0, h_sum + 360.0, h_sum - 360.0), h_sum) / 2.0
    )
    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar))
        + 0.32 * np.cos(np.radians(3.0 * h_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar - 63.0))
    )
    s_l = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -2.0 * np.sqrt(c_bar_p**7 / (c_bar_p**7 + 25.0**7)) * np.sin(np.radians(60.0 * np.exp(-(((h_bar - 275.0) / 25.0) ** 2))))
    dl, dc, dhh = (l2 - l1) / (k_l * s_l), (c2p - c1p) / s_c, d_hp / s_h
    return np.sqrt(dl**2 + dc**2 + dhh**2 + r_t * dc * dhh)


def delta_e_color(ref: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """CIEDE2000 with the lightness difference left out: the hue and chroma error. The print
    curve, not the matrix, sets a patch's lightness."""
    lab = np.array(lab, dtype=np.float64)
    lab[:, 0] = np.asarray(ref, dtype=np.float64)[:, 0]
    return delta_e2000(ref, lab)


def reading_order(chart: ChartSpec, orientation: int) -> np.ndarray:
    """The card patch that each sample box holds, in the boxes' order, for one of ORIENTATIONS."""
    grid = np.arange(chart.patch_count).reshape(chart.rows, chart.cols)
    return grid[:: -1 if orientation & 2 else 1, :: -1 if orientation & 1 else 1].ravel()


def _row_centers(chart: ChartSpec, orientation: int) -> np.ndarray:
    rows = np.asarray(chart.row_centers, dtype=np.float64)
    return 1.0 - rows[::-1] if orientation & 2 else rows


def grid_corners(corners: np.ndarray) -> np.ndarray:
    """The four corner-patch centers in any order, as clockwise from the end of a long (column) side.

    ORIENTATIONS then covers every way the chart can sit in the frame, sideways included."""
    c = np.asarray(corners, dtype=np.float64)
    d = c - c.mean(axis=0)
    c = c[np.argsort(np.arctan2(d[:, 1], d[:, 0]))]
    return np.roll(c, -1, axis=0) if np.linalg.norm(c[1] - c[0]) < np.linalg.norm(c[2] - c[1]) else c


def patch_centers(corners: np.ndarray, chart: ChartSpec, orientation: int = 0) -> np.ndarray:
    """(N, 2) sample box centers, row-major, from the four corner-patch centers, bilinear.

    Uneven rows sit at other places on a card that is upside down, so the boxes take the orientation."""
    tl, tr, br, bl = grid_corners(corners)
    u = np.linspace(0.0, 1.0, chart.cols)[None, :, None]
    v = _row_centers(chart, orientation)[:, None, None]
    top = tl + (tr - tl) * u
    bottom = bl + (br - bl) * u
    return (top + (bottom - top) * v).reshape(-1, 2)


def sample_radius(corners: np.ndarray, chart: ChartSpec) -> float:
    """Half-width of each patch's sample box, in the corners' units, from the tightest pitch."""
    tl, tr, br, bl = grid_corners(corners)
    pitch_x = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / (2.0 * (chart.cols - 1))
    pitch_y = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0 * float(np.min(np.diff(chart.row_centers)))
    return float(0.5 * _SAMPLE_FRACTION * min(pitch_x, pitch_y))


def _patch_medians(img: np.ndarray, corners_px: np.ndarray, chart: ChartSpec, orientation: int, gamma: Optional[float]) -> np.ndarray:
    """(N, 3) median of each patch box; with `gamma`, of the clipped image decoded from the working encoding."""
    h, w = img.shape[:2]
    r = max(1, int(round(sample_radius(corners_px, chart))))
    boxes = []
    for x, y in patch_centers(corners_px, chart, orientation):
        cx, cy = int(round(x)), int(round(y))
        boxes.append((min(max(cx - r, 0), w - 1), min(max(cx + r + 1, 1), w), min(max(cy - r, 0), h - 1), min(max(cy + r + 1, 1), h)))
    return np.array(negpy_core.chart_patches(np.ascontiguousarray(img, dtype=np.float32), boxes, gamma), dtype=np.float64)


def sample_patches(img: np.ndarray, corners_px: np.ndarray, chart: ChartSpec, orientation: int = 0) -> np.ndarray:
    """(N, 3) median of each patch box. Median, so dust on the card does not move it."""
    return _patch_medians(img, corners_px, chart, orientation, None)


def measure_lab(encoded: np.ndarray, corners_px: np.ndarray, chart: ChartSpec, orientation: int) -> np.ndarray:
    """The chart's patches in card order, as Lab, after one exposure gain that matches its mid grays.

    Exposure is the user's choice and the matrix keeps grays gray, so the gain is not a fit term."""
    linear = _patch_medians(encoded, corners_px, chart, orientation, float(np.float32(_WORKING_GAMMA)))
    rgb = np.empty_like(linear)
    rgb[reading_order(chart, orientation)] = linear
    ref = reference_lab(chart)
    idx = list(chart.exposure_patches)
    y_ref = ((ref[idx, 0] + 16.0) / 116.0) ** 3
    y = np.maximum(rgb[idx] @ np.array([0.2973769, 0.6273491, 0.0752741]), 1e-6)
    gain = float(np.exp(np.mean(np.log(y_ref / y))))
    return np.asarray(rgb_to_lab_working((rgb * gain).astype(np.float32)), dtype=np.float64)


def detect_orientation(encoded: np.ndarray, corners_px: np.ndarray, chart: ChartSpec) -> int:
    """The reading of the grid whose colors are nearest the chart: corner order then does not matter."""
    ref = reference_lab(chart)
    errors = [float(np.mean(delta_e2000(ref, measure_lab(encoded, corners_px, chart, o)))) for o in ORIENTATIONS]
    return int(np.argmin(errors))


def matrix_from_offdiagonal(p: np.ndarray) -> np.ndarray:
    """Row-major 3x3 with a unit diagonal and the six mixing terms, the editor's parametrization."""
    m = np.eye(3)
    m[~np.eye(3, dtype=bool)] = p
    return m


@dataclass(frozen=True)
class ChartFit:
    matrix: tuple[float, ...]  # row-major 3x3, unit diagonal
    before: np.ndarray  # ΔE00 per patch, card order, identity matrix
    after: np.ndarray  # ΔE00 per patch with the fitted matrix
    orientation: int  # grid reading used, one of ORIENTATIONS
    before_color: np.ndarray  # delta_e_color per patch, identity matrix
    after_color: np.ndarray  # delta_e_color per patch with the fitted matrix


class FitCancelled(Exception):
    pass


def fit_chart_matrix(
    render: Callable[[np.ndarray], np.ndarray],
    corners_px: np.ndarray,
    chart: ChartSpec,
    progress: Optional[Callable[[float], None]] = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> ChartFit:
    """Levenberg-Marquardt over the six mixing terms, from identity.

    ``render(matrix)`` returns the frame's working-encoded render with that crosstalk matrix at full
    strength. The Jacobian is by finite differences through it, because normalization and Cast
    Removal re-meter the frame for every matrix."""
    ref = reference_lab(chart)
    identity = render(np.eye(3))
    orientation = detect_orientation(identity, corners_px, chart)

    def lab_for(p: np.ndarray) -> np.ndarray:
        if cancelled():
            raise FitCancelled()
        return measure_lab(render(matrix_from_offdiagonal(p)), corners_px, chart, orientation)

    def residual(lab: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.concatenate([delta_e2000(ref, lab, _FIT_KL), _RIDGE * p])

    p = np.zeros(6)
    lab0 = measure_lab(identity, corners_px, chart, orientation)
    r = residual(lab0, p)
    mu = 1.0
    for it in range(_MAX_ITERATIONS):
        jac = np.empty((r.size, 6))
        for j in range(6):
            q = p.copy()
            q[j] += _FD_STEP
            jac[:, j] = (residual(lab_for(q), q) - r) / _FD_STEP
        jtj, jtr = jac.T @ jac, jac.T @ r
        step = np.zeros(6)
        while mu < 1e6:
            step = np.linalg.solve(jtj + mu * np.diag(np.diag(jtj) + 1e-9), -jtr)
            r_new = residual(lab_for(p + step), p + step)
            if r_new @ r_new < r @ r:
                p, r, mu = p + step, r_new, mu * 0.3
                break
            mu *= 4.0
        if progress is not None:
            progress((it + 1) / _MAX_ITERATIONS)
        if mu >= 1e6 or np.max(np.abs(step)) < _MIN_STEP:
            break
    lab1 = lab_for(p)
    return ChartFit(
        tuple(float(v) for v in matrix_from_offdiagonal(p).ravel()),
        delta_e2000(ref, lab0),
        delta_e2000(ref, lab1),
        orientation,
        delta_e_color(ref, lab0),
        delta_e_color(ref, lab1),
    )
