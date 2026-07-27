import numpy as np
from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image

# Fixed seed: one filed carrier per darkroom, identical every frame and on both paths.
_CARRIER_SEED = 1898
# All CARRIER_* below are fractions of the rebate width and mirrored as WGSL literals
# in finish.wgsl. The picture-side boundary is the camera's film gate (machine-cut, so
# JITTER * INNER_ROUGH only, no slider); the paper-side one is the hand-filed aperture.
CARRIER_SAMPLES = 2048
CARRIER_JITTER = 0.24
CARRIER_INNER_ROUGH = 0.2
CARRIER_OUTER_JITTER = 0.225
CARRIER_CORNER = 1.4
CARRIER_MARGIN = 0.7
# The gate prints soft; filed metal is a hard stop, which is what reads as filed.
CARRIER_SOFT = 0.22
CARRIER_FILED_SOFT = 0.06
CARRIER_FLARE_DEPTH = 0.35
CARRIER_FLARE_SPILL = 0.7
CARRIER_FLARE_GAIN = 0.55
CARRIER_FLARE_BASE = 0.35
CARRIER_FLARE_HUE = 1.0
# A 1-D profile is a height field, so it cannot overhang or shed a fleck; the 2-D field
# displacing the distance field is what makes the edge read as torn metal. Hash noise
# rather than a library because WGSL has to reproduce it bit for bit.
CARRIER_NOISE_SEED = 0x51ED270B
CARRIER_NOISE_CELL = 1.5
CARRIER_NOISE_OCTAVES = 4
CARRIER_NOISE_OUTER = 0.275
CARRIER_NOISE_INNER = 0.07
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
        bite = _norm(_blur(raw[4:], 6.0))
        # Grit + sparse gouges (squared) + uneven overall cut.
        filed = 0.3 * bite + 0.4 * np.sign(bite) * bite**2 + 0.3 * _norm(_blur(raw[4:], 100.0))
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


def flare_tint(theta: np.ndarray) -> np.ndarray:
    """Hue drift of the bevel reflection: (..., 3) channel gains in [0, 1]."""
    phase = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0], dtype=np.float32)
    return 0.5 + 0.5 * np.cos(theta[..., None] + phase)


def apply_carrier(
    img: ImageBuffer,
    width_px: float,
    rough: float,
    flare: float = 0.0,
    bw: bool = False,
    corner: float = 0.0,
    paper: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ImageBuffer:
    """
    Filed-out negative carrier: the clear rebate prints max black between the film gate
    and the filed aperture, with a margin of unexposed paper outside it.

    rough ragges the aperture, corner rounds it. flare lerps both sides of the filed
    edge toward the bevel's reflection (neutral when bw). paper is the bare-paper colour
    in scene-linear, so the margin meets the mat with no seam.
    """
    if width_px <= 0.0:
        return img

    h, w = img.shape[:2]
    profiles = carrier_profiles()
    soft = max(1.0, width_px * CARRIER_SOFT)
    soft_filed = max(1.0, width_px * CARRIER_FILED_SOFT)
    margin = width_px * CARRIER_MARGIN
    radius = width_px * CARRIER_CORNER * corner
    cell = max(1.0, width_px * CARRIER_NOISE_CELL)
    paper_rgb = np.asarray(paper, dtype=np.float32)
    band = min(
        int(np.ceil(margin + radius + width_px * (1.0 + CARRIER_JITTER + CARRIER_NOISE_OUTER + CARRIER_NOISE_INNER) + soft)) + 1,
        h,
        w,
    )
    d = np.arange(band, dtype=np.float32)[:, None]

    def edge_idx(count: int) -> np.ndarray:
        s = ((np.arange(count, dtype=np.float32) + 0.5) / np.float32(count)).astype(np.float32)
        return np.minimum((s * CARRIER_SAMPLES).astype(np.int32), CARRIER_SAMPLES - 1)

    def corner_cut(count: int) -> np.ndarray:
        """Aperture retreat near an edge's ends. Arc measured from the aperture corner,
        not the print edge, or most of it is spent inside the paper margin."""
        if radius <= 0.0:
            return np.zeros(count, dtype=np.float32)
        i = np.arange(count, dtype=np.float32)
        x = np.clip(radius - (np.minimum(i, count - 1.0 - i) - margin), 0.0, radius)
        return radius - np.sqrt(np.maximum(radius * radius - x * x, 0.0))

    def edge_alphas(edge: int, idx: np.ndarray, count: int, gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, ...]:
        """(a_in, a_out, outer, n2), (band, count) each; axis 0 runs inward from the print edge.
        Both boundaries take the same arc, so the band keeps its width around a corner."""
        cut = corner_cut(count)
        n2 = carrier_noise(gx / cell, gy / cell)
        jitter = CARRIER_OUTER_JITTER * profiles[edge + 4, idx] + CARRIER_NOISE_OUTER * n2
        outer = margin + width_px * rough * jitter + cut
        wobble = CARRIER_JITTER * CARRIER_INNER_ROUGH * profiles[edge, idx] + CARRIER_NOISE_INNER * n2
        inner = margin + width_px * (1.0 + wobble) + cut
        a_in = np.clip((d - inner) / soft + 0.5, 0.0, 1.0)
        a_out = np.clip((d - outer) / soft_filed + 0.5, 0.0, 1.0)
        return a_in, a_out, outer, n2

    def edge_flare(edge: int, idx: np.ndarray, a_in: np.ndarray, outer: np.ndarray, n2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(weight, weight*tint) as (band, count, 1|3); peaks on the filed edge, falls off both ways."""
        reach = max(1.0, width_px * CARRIER_FLARE_DEPTH)
        off = d - outer
        t = np.clip(1.0 - np.maximum(off, 0.0) / reach + np.minimum(off, 0.0) / (reach * CARRIER_FLARE_SPILL), 0.0, 1.0)
        # Floored |noise|, not a one-sided gate: that left whole edges with no flare.
        n = CARRIER_FLARE_BASE + (1.0 - CARRIER_FLARE_BASE) * np.abs(n2)
        amp = flare * CARRIER_FLARE_GAIN * t * t * n * (1.0 - a_in)
        tint = np.ones(3, dtype=np.float32) if bw else flare_tint(CARRIER_FLARE_HUE * profiles[edge, idx])
        return amp[..., None], amp[..., None] * tint

    ix_w, ix_h = edge_idx(w), edge_idx(h)
    cols, rows = np.arange(w, dtype=np.float32)[None, :], np.arange(h, dtype=np.float32)[None, :]
    # (edge, profile index, edge length, slab, orient, noise x, noise y). Noise coords are
    # global, so a corner pixel gets one field value from either slab, as the shader does.
    edges = (
        (0, ix_w, w, np.s_[:band], lambda a: a, cols, d),
        (1, ix_w, w, np.s_[h - band :], lambda a: a[::-1], cols, (h - 1.0) - d),
        (2, ix_h, h, np.s_[:, :band], lambda a: a.swapaxes(0, 1), d, rows),
        (3, ix_h, h, np.s_[:, w - band :], lambda a: a.swapaxes(0, 1)[:, ::-1], (w - 1.0) - d, rows),
    )
    alphas = [(sl, orient, *edge_alphas(e, idx, count, gx, gy)) for e, idx, count, sl, orient, gx, gy in edges]

    out = img.copy()
    # Sequential mixes compose to out*A_in*A_out + paper*(1 - A_out) whatever the order,
    # so corners land on the value the shader computes from the products.
    for sl, orient, a_in, *_ in alphas:
        out[sl] *= orient(a_in)[..., None]
    for sl, orient, _, a_out, *_ in alphas:
        a = orient(a_out)[..., None]
        out[sl] *= a
        out[sl] += (1.0 - a) * paper_rgb

    if flare > 0.0:
        # Lerp, not add: glows on the black, stains the paper. Order-dependent, unlike
        # the mixes above — the shader must walk the edges in this same order.
        for (sl, orient, a_in, _, outer, n2), (e, idx, *_) in zip(alphas, edges):
            amp, lift = edge_flare(e, idx, a_in, outer, n2)
            out[sl] *= 1.0 - orient(amp)
            out[sl] += orient(lift)

    return ensure_image(np.clip(out, 0.0, 1.0))


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
