import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image

# Fixed seed: one filed carrier per darkroom, identical every frame and on both paths.
_CARRIER_SEED = 1898
# Profile samples per edge; the other carrier constants live in negpy_core/src/finish.rs,
# mirrored as WGSL literals in finish.wgsl.
CARRIER_SAMPLES = 2048
# Tone table sampled at t = u**POWER, dense in the toe.
CARRIER_TONE_SAMPLES = 64
CARRIER_TONE_POWER = 3.0
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
    """Straight file strokes plus nicks and spurs, crowded toward the corners."""
    i = np.arange(CARRIER_SAMPLES, dtype=np.float32)
    knots = np.concatenate([[0.0], np.cumsum(rng.uniform(80.0, 320.0, 24))])
    knots = knots[knots < CARRIER_SAMPLES]
    strokes = np.interp(i, np.append(knots, CARRIER_SAMPLES), rng.normal(0.0, 0.25, len(knots) + 1))
    ends = np.minimum(i, CARRIER_SAMPLES - 1.0 - i) / CARRIER_SAMPLES
    corner = 1.0 + 2.0 * np.exp(-ends / 0.05)
    marks = np.zeros(CARRIER_SAMPLES, dtype=np.float32)
    for _ in range(14):
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


def carrier_tone_exposures() -> np.ndarray:
    """Exposure fraction of each tone-table entry, 0 (bare paper) to 1 (full rebate)."""
    return np.linspace(0.0, 1.0, CARRIER_TONE_SAMPLES, dtype=np.float32) ** np.float32(CARRIER_TONE_POWER)


def linear_carrier_tone() -> np.ndarray:
    """Plain light, paper to black."""
    return np.repeat((1.0 - carrier_tone_exposures())[:, None], 3, axis=1).astype(np.float32)


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
    Filed-out negative carrier: the rebate prints between the film gate and the filed
    aperture, inside a margin of bare paper (scene-linear, matching the mat). The filed
    edge's penumbra and the flare are exposure, read through tone (rebate_tone).
    """
    if width_px <= 0.0:
        return img
    tone = linear_carrier_tone() if tone is None else np.ascontiguousarray(tone, dtype=np.float32)
    paper32 = tuple(np.float32(v) for v in paper)
    return negpy_core.apply_carrier(
        np.ascontiguousarray(img, dtype=np.float32), width_px, rough, flare, corner, paper32, tone, carrier_profiles()
    )


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
    return ensure_image(negpy_core.apply_vignette(np.ascontiguousarray(img, dtype=np.float32), float(stops), float(size), float(roundness)))
