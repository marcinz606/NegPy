import numpy as np
from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image

# Fixed seed: one filed carrier per darkroom, identical every frame and on both paths.
_CARRIER_SEED = 1898
# All CARRIER_* below are fractions of the rebate width and mirrored as WGSL literals in
# finish.wgsl. The picture-side boundary is the camera's film gate, machine-cut, so it
# gets JITTER * INNER_ROUGH and no slider. The paper-side one is the hand-filed aperture.
CARRIER_SAMPLES = 2048
CARRIER_JITTER = 0.24
CARRIER_INNER_ROUGH = 0.2
CARRIER_OUTER_JITTER = 0.4
CARRIER_CORNER = 1.4
# A camera gate's corners are machined with a small fixed radius; Corners rounds the filed ones.
CARRIER_GATE_CORNER = 0.2
CARRIER_MARGIN = 0.7
# The gate prints soft; filed metal is a hard stop, which is what reads as filed.
CARRIER_SOFT = 0.22
CARRIER_FILED_SOFT = 0.06
# Flare is light off the filed bevel onto the paper: an exposure fraction added outside
# the aperture, so it prints through the tone table like the penumbra.
CARRIER_FLARE_DEPTH = 0.25
CARRIER_FLARE_GAIN = 0.3
CARRIER_FLARE_BASE = 0.35
# A 1-D profile is a height field, so it cannot overhang or shed a fleck. The 2-D field
# displacing the distance field is what makes the edge read as torn metal. Hash noise
# rather than a library, because WGSL has to reproduce it bit for bit.
CARRIER_NOISE_SEED = 0x51ED270B
CARRIER_NOISE_CELL = 0.35
CARRIER_NOISE_OCTAVES = 4
CARRIER_NOISE_OUTER = 0.12
CARRIER_NOISE_INNER = 0.02
# Rebate tone table: print color over the exposure fraction t that reaches the paper,
# sampled at t = u**POWER so the toe, where the fringe hue lives, gets most entries.
CARRIER_TONE_SAMPLES = 64
CARRIER_TONE_POWER = 3.0
_CARRIER_BLOCK_ROWS = 64
_carrier_cache: np.ndarray | None = None


def _blur(rows: np.ndarray, sigma: float) -> np.ndarray:
    taps = int(np.ceil(sigma * 3.0))
    k = np.exp(-0.5 * (np.arange(-taps, taps + 1, dtype=np.float32) / sigma) ** 2)
    k /= k.sum()
    # Circular smoothing so a profile has no seam where an edge wraps.
    tiled = np.concatenate([rows, rows, rows], axis=1)
    return np.stack([np.convolve(row, k, mode="same") for row in tiled])[:, CARRIER_SAMPLES : 2 * CARRIER_SAMPLES]


def _norm(rows: np.ndarray) -> np.ndarray:
    return rows / np.max(np.abs(rows), axis=1, keepdims=True)


def _filed_edge(rng: np.random.Generator) -> np.ndarray:
    """
    One hand-filed edge: straight file strokes meeting at small angles, sparse nicks and
    spurs where the file slipped or stopped short, most of them near the corners, where
    a file cannot reach squarely. Nicks open the aperture (black bulges into the paper)
    more often than spurs close it.
    """
    i = np.arange(CARRIER_SAMPLES, dtype=np.float32)
    knots = np.concatenate([[0.0], np.cumsum(rng.uniform(80.0, 320.0, 24))])
    knots = knots[knots < CARRIER_SAMPLES]
    strokes = np.interp(i, np.append(knots, CARRIER_SAMPLES), rng.normal(0.0, 0.25, len(knots) + 1))
    ends = np.minimum(i, CARRIER_SAMPLES - 1.0 - i) / CARRIER_SAMPLES
    corner = 1.0 + 2.0 * np.exp(-ends / 0.05)
    marks = np.zeros(CARRIER_SAMPLES, dtype=np.float32)
    for _ in range(14):
        # Drawn toward the ends as often as along the run.
        at = rng.uniform(0.0, 0.12) if rng.random() < 0.5 else rng.uniform(0.0, 0.5)
        centre = (at if rng.random() < 0.5 else 1.0 - at) * CARRIER_SAMPLES
        half = rng.uniform(10.0, 26.0)
        depth = rng.uniform(0.4, 1.0) * (-1.0 if rng.random() < 0.7 else 0.6)
        x = np.clip(np.abs(i - centre) / half, 0.0, 1.0)
        marks += depth * 0.5 * (1.0 + np.cos(np.pi * x))
    return ((strokes + marks) * corner).astype(np.float32)


def carrier_profiles() -> np.ndarray:
    """
    (8, CARRIER_SAMPLES) float32 profiles in [-1, 1]: rows 0-3 per-edge film-gate
    wobble, rows 4-7 filed-edge roughness, both ordered top, bottom, left, right.
    Sigmas are in samples, i.e. fractions of an edge, so a mark holds its size on the
    print at any resolution; much finer than the grit term reads as a machine-cut comb.
    """
    global _carrier_cache
    if _carrier_cache is None:
        rng = np.random.default_rng(_CARRIER_SEED)
        raw = rng.standard_normal((8, CARRIER_SAMPLES)).astype(np.float32)
        gate = _blur(raw[:4], 60.0)
        filed = np.stack([_filed_edge(rng) for _ in range(4)])
        _carrier_cache = np.ascontiguousarray(_norm(np.concatenate([gate, filed])), dtype=np.float32)
    return _carrier_cache


def _hash_lattice(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Lattice value in [-1, 1). u32 wrap-around only, so WGSL reproduces it exactly."""
    h = (ix * np.uint32(0x27D4EB2D)) ^ (iy * np.uint32(0x165667B1)) ^ np.uint32(CARRIER_NOISE_SEED)
    h ^= h >> np.uint32(15)
    h = h * np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(13)
    h = h * np.uint32(0x297A2D39)
    h ^= h >> np.uint32(16)
    return (h >> np.uint32(8)).astype(np.float32) * np.float32(2.0 / 16777216.0) - np.float32(1.0)


def _value_noise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    fx, fy = np.floor(x), np.floor(y)
    ux, uy = (x - fx), (y - fy)
    ux = ux * ux * (3.0 - 2.0 * ux)
    uy = uy * uy * (3.0 - 2.0 * uy)
    i0, j0 = fx.astype(np.int64).astype(np.uint32), fy.astype(np.int64).astype(np.uint32)
    i1, j1 = i0 + np.uint32(1), j0 + np.uint32(1)
    lo = _hash_lattice(i0, j0)
    lo = lo + (_hash_lattice(i1, j0) - lo) * ux
    hi = _hash_lattice(i0, j1)
    hi = hi + (_hash_lattice(i1, j1) - hi) * ux
    return lo + (hi - lo) * uy


def carrier_noise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """fBm value noise in [-1, 1]; x/y in noise cells. Mirrored in finish.wgsl."""
    total = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=np.float32)
    amp, freq, norm = np.float32(1.0), np.float32(1.0), np.float32(0.0)
    for _ in range(CARRIER_NOISE_OCTAVES):
        total += amp * _value_noise(x * freq, y * freq)
        norm += amp
        amp *= np.float32(0.5)
        freq *= np.float32(2.0)
    return total / norm


def carrier_tone_exposures() -> np.ndarray:
    """Exposure fraction t of each tone-table entry, 0 (bare paper) to 1 (full rebate)."""
    return np.linspace(0.0, 1.0, CARRIER_TONE_SAMPLES, dtype=np.float32) ** np.float32(CARRIER_TONE_POWER)


def linear_carrier_tone() -> np.ndarray:
    """Tone table with no print model behind it: plain light, paper to black."""
    return np.repeat((1.0 - carrier_tone_exposures())[:, None], 3, axis=1).astype(np.float32)


def carrier_tone_lookup(tone: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(..., 3) paper-relative color at exposure fraction t. Linear between entries, as the shader."""
    u = np.power(np.clip(t, 0.0, 1.0), np.float32(1.0 / CARRIER_TONE_POWER)) * np.float32(CARRIER_TONE_SAMPLES - 1)
    i0 = np.minimum(u.astype(np.int32), CARRIER_TONE_SAMPLES - 2)
    f = (u - i0)[..., None]
    return tone[i0] * (1.0 - f) + tone[i0 + 1] * f


def apply_carrier(
    img: ImageBuffer,
    width_px: float,
    rough: float,
    flare: float = 0.0,
    corner: float = 0.0,
    paper: tuple[float, float, float] = (1.0, 1.0, 1.0),
    tone: np.ndarray | None = None,
) -> ImageBuffer:
    """
    Filed-out negative carrier: the clear rebate prints between the film gate and the
    filed aperture, with a margin of unexposed paper outside it.

    rough ragges the aperture, corner rounds it, flare exposes the paper just outside it.
    paper is the bare-paper color in scene-linear, so the margin meets the mat with no
    seam. tone is the rebate tone table (rebate_tone); the filed edge's penumbra and the
    flare are exposure, read through it.
    Evaluated per pixel over the border frame, as finish.wgsl does.
    """
    if width_px <= 0.0:
        return img

    h, w = img.shape[:2]
    tone = linear_carrier_tone() if tone is None else np.asarray(tone, dtype=np.float32)
    band = (
        int(
            np.ceil(
                width_px * (CARRIER_MARGIN + CARRIER_CORNER * corner + 1.0 + CARRIER_JITTER + CARRIER_NOISE_OUTER + CARRIER_NOISE_INNER)
                + max(1.0, width_px * CARRIER_SOFT)
            )
        )
        + 1
    )
    out = img.copy()
    if 2 * band >= h or 2 * band >= w:
        regions = [(0, h, 0, w)]
    else:
        regions = [(0, band, 0, w), (h - band, h, 0, w), (band, h - band, 0, band), (band, h - band, w - band, w)]
    for y0, y1, x0, x1 in regions:
        for b0 in range(y0, y1, _CARRIER_BLOCK_ROWS):
            b1 = min(b0 + _CARRIER_BLOCK_ROWS, y1)
            out[b0:b1, x0:x1] = _carrier_block(img[b0:b1, x0:x1], b0, x0, h, w, width_px, rough, flare, corner, paper, tone)
    return ensure_image(np.clip(out, 0.0, 1.0))


def _carrier_block(
    img: np.ndarray,
    y0: int,
    x0: int,
    h: int,
    w: int,
    width_px: float,
    rough: float,
    flare: float,
    corner: float,
    paper: tuple[float, float, float],
    tone: np.ndarray,
) -> np.ndarray:
    profiles = carrier_profiles()
    soft = max(1.0, width_px * CARRIER_SOFT)
    soft_filed = max(1.0, width_px * CARRIER_FILED_SOFT)
    margin = width_px * CARRIER_MARGIN
    radius = width_px * CARRIER_CORNER * corner
    gate_radius = width_px * CARRIER_GATE_CORNER
    cell = max(1.0, width_px * CARRIER_NOISE_CELL)
    py = np.arange(y0, y0 + img.shape[0], dtype=np.float32)[:, None]
    px = np.arange(x0, x0 + img.shape[1], dtype=np.float32)[None, :]
    sx = (px + 0.5) / np.float32(w)
    sy = (py + 0.5) / np.float32(h)
    end_x = np.minimum(px, (w - 1.0) - px)
    end_y = np.minimum(py, (h - 1.0) - py)
    n2 = carrier_noise(px / cell, py / cell)

    def prof(row: int, s: np.ndarray) -> np.ndarray:
        return profiles[row, np.minimum((s * CARRIER_SAMPLES).astype(np.int32), CARRIER_SAMPLES - 1)]

    def arc(r: float, at: float, end: np.ndarray) -> np.ndarray | np.float32:
        """Corner retreat of a boundary `at` px from the print edge. Measured from that
        boundary's own corner, or most of the arc is spent outside it."""
        if r <= 0.0:
            return np.float32(0.0)
        x = np.clip(r - (end - at), 0.0, r)
        return r - np.sqrt(np.maximum(r * r - x * x, 0.0))

    def bounds(edge: int, s: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(filed, gate) boundary, px from the print edge."""
        outer = margin + width_px * rough * (CARRIER_OUTER_JITTER * prof(edge + 4, s) + CARRIER_NOISE_OUTER * n2)
        wobble = CARRIER_JITTER * CARRIER_INNER_ROUGH * prof(edge, s) + CARRIER_NOISE_INNER * n2
        inner = margin + width_px * (1.0 + wobble) + arc(gate_radius, margin + width_px, end)
        return outer + arc(radius, margin, end), inner

    edges = ((0, sx, end_x, py), (1, sx, end_x, (h - 1.0) - py), (2, sy, end_y, px), (3, sy, end_y, (w - 1.0) - px))
    reach = max(1.0, width_px * CARRIER_FLARE_DEPTH)
    # Floored |noise|, not a one-sided gate: that left whole edges with no flare.
    flare_amp = flare * CARRIER_FLARE_GAIN * (CARRIER_FLARE_BASE + (1.0 - CARRIER_FLARE_BASE) * np.abs(n2))
    a_in = np.ones(img.shape[:2], dtype=np.float32)
    a_out = np.ones(img.shape[:2], dtype=np.float32)
    peaks, gates = [], []
    for e, s, end, d in edges:
        outer, inner = bounds(e, s, end)
        a_in = a_in * np.clip((d - inner) / soft + 0.5, 0.0, 1.0)
        a_out = a_out * np.clip((d - outer) / soft_filed + 0.5, 0.0, 1.0)
        peaks.append(np.clip(1.0 - np.abs(d - outer) / reach, 0.0, 1.0) ** 2)
        gates.append(np.clip(1.0 + (d - outer) / reach, 0.0, 1.0))
    lit = np.float32(0.0)
    if flare > 0.0:
        # An edge's bevel spans only the aperture, so the other three edges gate its flare.
        for e in range(4):
            lit = lit + peaks[e] * np.prod([gates[k] for k in range(4) if k != e], axis=0)
        lit = flare_amp * lit

    rebate = np.asarray(paper, dtype=np.float32) * carrier_tone_lookup(tone, a_out + lit)
    a_in = a_in[..., None]
    return img * a_in + rebate * (1.0 - a_in)


def apply_vignette(img: ImageBuffer, stops: float, size: float, roundness: float = 0.0) -> ImageBuffer:
    """
    Edge burn / hold-back as a true exposure change in scene-linear.

    Args:
        img: Float32 RGB image [0, 1], scene-linear.
        stops: [-2, 2]. Positive = burn (darken edges, more exposure),
            negative = dodge (hold back, lighten edges), 0 = no effect.
        size: [0, 1]. 0 = effect barely visible at extreme corners, 1 = covers entire image from center.
        roundness: [0, 1]. 0 = radial falloff (lens-like), 1 = rectangular
            following the print edges (card-like burn).

    Returns:
        Modified ImageBuffer with the burn applied.
    """
    if stops == 0.0:
        return img

    h, w = img.shape[:2]
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5

    y_coords = np.arange(h, dtype=np.float32)
    x_coords = np.arange(w, dtype=np.float32)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    max_dist = float(np.sqrt(cx * cx + cy * cy))
    d_radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(max_dist, 1.0)
    d_rect = np.maximum(np.abs(xx - cx) / max(cx, 1.0), np.abs(yy - cy) / max(cy, 1.0))
    dist = d_radial * (1.0 - roundness) + d_rect * roundness

    # Remap: size=0 → vignette barely at corners, size=1 → covers entire image
    midpoint = 1.0 - size
    t = (dist - midpoint) / max(1.0 - midpoint, 1e-6)
    t = np.clip(t, 0.0, 1.0)

    # Smooth cosine falloff
    factor = 0.5 * (1.0 - np.cos(t * np.pi))

    result = img * np.exp2(-stops * factor[:, :, np.newaxis])

    return ensure_image(np.clip(result, 0.0, 1.0))
