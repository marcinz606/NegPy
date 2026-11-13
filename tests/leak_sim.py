"""A synthetic negative with a known light leak, for the leak tests.

Three emulsion layers, each with its own curve D = gamma log10(1 + E) (or a shoulder at D_s), and
an optional crosstalk mix of layer density into the scanner's channels, on an orange base; or a slide, whose density falls from its black with exposure toward clear film. A leak
adds exposure to every layer in one ratio, its color: from the back or through a gap at the gate it
reaches the rebate as well as the picture, through the lens the gate masks the rebate. Edge fog lights the rebate alone."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from negpy.features.geometry.models import GeometryConfig

BASE = np.array([0.8, 0.5, 0.3], dtype=np.float32)
GAMMA = 0.6
# Light through the base reaches the red-sensitive layer first.
BACK = np.array([1.0, 0.2, 0.05], dtype=np.float32)
FRONT = np.array([1.0, 1.0, 1.0], dtype=np.float32)
# Light on the emulsion through a gap at the gate reaches the blue-sensitive layer first, and lights
# the rebate as well as the picture.
GATE = np.array([0.45, 0.8, 1.0], dtype=np.float32)
# Rows mix layer density into channel density and sum to 1, as a crosstalk profile does.
MIX = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.2, 0.75]])
# A slide's slope, its density from black to clear film, and clear film's transmittance.
SLIDE_GAMMA = 1.8
SLIDE_RANGE = 3.0
CLEAR = np.array([0.85, 0.85, 0.85], dtype=np.float32)
H, W = 600, 900
REBATE = H // 12


@dataclass(frozen=True)
class Frame:
    neg: np.ndarray
    clean: np.ndarray
    leak: np.ndarray
    color: np.ndarray
    geometry: GeometryConfig


def crop() -> GeometryConfig:
    """The picture between the rebate along top and bottom."""
    edge = REBATE / H
    return GeometryConfig(crop_rect=(0.0, edge, 1.0, 1.0 - edge))


def gray_scene(seed: int = 0) -> np.ndarray:
    """A neutral picture of soft tones over scattered deep shadows, rebate unexposed."""
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(0.0, 1.0, (12, 18)).astype(np.float32) ** 3 * 8.0
    tone = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_CUBIC).clip(0.0, None)
    e = np.repeat(tone[..., None], 3, axis=2)
    e[:REBATE] = 0.0
    e[-REBATE:] = 0.0
    return e


def sky_scene(seed: int = 0) -> np.ndarray:
    """A bright blue sky with no shadows over the top half, ground with shadows below, and a dark
    figure standing across both at x = 0.6."""
    rng = np.random.default_rng(seed)
    e = np.zeros((H, W, 3), np.float32)
    horizon = H // 2
    rows = np.linspace(1.0, 0.6, horizon, dtype=np.float32)[:, None, None]
    e[:horizon] = rows * np.array([4.0, 6.0, 9.0], np.float32)
    coarse = rng.uniform(0.0, 1.0, (6, 18)).astype(np.float32) ** 3 * 6.0
    ground = cv2.resize(coarse, (W, H - horizon), interpolation=cv2.INTER_CUBIC).clip(0.0, None)
    e[horizon:] = ground[..., None] * np.array([1.2, 1.0, 0.7], np.float32)
    ys, xs = np.mgrid[0:H, 0:W]
    figure = (np.abs(xs - 0.6 * W) < 0.025 * W) & (ys > 0.2 * H) & (ys < 0.85 * H)
    head = (xs - 0.6 * W) ** 2 + (ys - 0.18 * H) ** 2 < (0.03 * W) ** 2
    e[figure | head] = 0.03
    e[:REBATE] = 0.0
    e[-REBATE:] = 0.0
    return e


def band(peak: float, center: float = 0.6, width: float = 0.05) -> np.ndarray:
    """A vertical leak band's strength, (H, W)."""
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)
    return np.broadcast_to(peak * np.exp(-0.5 * ((x - center) / (width * W / max(H, W))) ** 2), (H, W)).copy()


def negative(
    exposure: np.ndarray,
    shoulder: Optional[float] = None,
    mix: Optional[np.ndarray] = None,
    grain: float = 0.01,
    seed: int = 1,
) -> np.ndarray:
    """The scan of a negative given each layer's exposure above the base."""
    logs = np.log10(1.0 + np.maximum(exposure, 0.0))
    d = GAMMA * logs if shoulder is None else shoulder * np.tanh(GAMMA * logs / shoulder)
    if mix is not None:
        d = d @ mix.T
    noise = np.exp(np.random.default_rng(seed).normal(0.0, grain, exposure.shape))
    return (BASE * 10.0 ** (-d) * noise).astype(np.float32)


def slide(exposure: np.ndarray, grain: float = 0.01, seed: int = 1) -> np.ndarray:
    """The scan of a slide given each layer's exposure: its density falls from SLIDE_RANGE over clear
    film by SLIDE_GAMMA log10(1 + E) and flattens at clear film."""
    logs = np.log10(1.0 + np.maximum(exposure, 0.0))
    drop = SLIDE_RANGE * np.tanh(SLIDE_GAMMA * logs / SLIDE_RANGE)
    noise = np.exp(np.random.default_rng(seed).normal(0.0, grain, exposure.shape))
    return (CLEAR * 10.0 ** (drop - SLIDE_RANGE) * noise).astype(np.float32)


def leaked(scene: np.ndarray, strength: np.ndarray, color: np.ndarray, front: bool = False, on_slide: bool = False, **film) -> Frame:
    """The scene with a leak of *strength* (H, W) and *color*, on a negative or *on_slide*. A front
    leak stops at the gate."""
    leak = strength[..., None] * color
    if front:
        leak = leak.copy()
        leak[:REBATE] = 0.0
        leak[-REBATE:] = 0.0
    scan = slide if on_slide else negative
    return Frame(scan(scene + leak, **film), scan(scene, **film), leak, color, crop())


def edge_fog(scene: np.ndarray, level: float, color: np.ndarray = BACK) -> np.ndarray:
    """The scene with fog over the top rebate alone, as light along the film edge leaves it."""
    out = scene.copy()
    out[:REBATE] += level * color
    return out


def edge_leak(peak: float, reach: float = 0.25, tongues: int = 3) -> np.ndarray:
    """A leak's strength (H, W) in from the top edge of the picture, as light through a gap in a back
    leaves it: its strength and reach swing along the edge into tongues, and it lights the rebate
    above the edge at full strength."""
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)
    swing = np.sin(2.0 * np.pi * tongues * x)
    level = peak * (0.75 + 0.25 * swing)
    depth = reach * H * (1.0 + 0.5 * swing)
    t = np.arange(H, dtype=np.float32)[:, None] - REBATE
    return np.where(t <= 0.0, level[None, :], level[None, :] * np.exp(-3.0 * t / depth[None, :])).astype(np.float32)


def gap_leak(peak: float, reach: float = 0.25) -> np.ndarray:
    """A leak's strength (H, W) centered on the top frame line, as light through a gap at the gate
    leaves it: brightest at the line, fading into the picture and out across the rebate alike."""
    t = np.abs(np.arange(H, dtype=np.float32) - REBATE)[:, None]
    return np.broadcast_to(peak * np.exp(-3.0 * t / (reach * H)), (H, W)).astype(np.float32).copy()


def gate_shadow(peak: float, reach: float = 0.25) -> np.ndarray:
    """A leak's strength (H, W) about the top frame line, as the soft shadow of a gap at the gate
    leaves it: held near the line, then falling as a Gaussian into the picture and out across the
    rebate."""
    t = (np.arange(H, dtype=np.float32) - REBATE)[:, None] / (reach * H / 3.0)
    return np.broadcast_to(peak * np.exp(-0.5 * t * t), (H, W)).astype(np.float32).copy()


def edge_tip(peak: float, decay: float = 4.0, tongues: int = 8, ceiling: float = 200.0) -> np.ndarray:
    """A leak's strength (H, W) across the top edge of the picture, as light spreading in the base
    from the film's cut end leaves it: one exponential over *decay* px on both sides of the frame
    line, rising through the rebate to *ceiling*, its level swinging along the edge into tongues."""
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)
    level = peak * (1.0 + 0.5 * np.sin(2.0 * np.pi * tongues * x))
    t = np.arange(H, dtype=np.float32)[:, None] - REBATE
    return np.minimum(level[None, :] * np.exp(-t / decay), ceiling).astype(np.float32)


def spot(peak: float, center=(0.4, 0.45), radius: float = 0.1) -> np.ndarray:
    """A round leak's strength (H, W), as light through a pinhole spreads: brightest at the center,
    fading over *radius* (a share of the width); the gate keeps it off the rebate."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot(xs - center[0] * W, ys - center[1] * H) / (radius * W)
    out = peak * np.exp(-0.5 * (2.0 * r) ** 2)
    out[:REBATE] = 0.0
    out[-REBATE:] = 0.0
    return out.astype(np.float32)


def blotch(peak: float, seed: int = 4) -> np.ndarray:
    """An irregular leak's strength (H, W), as a toy camera throws: overlapping glows of different
    sizes around (0.35, 0.5), with no line or center to it; the gate keeps it off the rebate."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.zeros((H, W), np.float32)
    for _ in range(5):
        cx, cy = (0.35 + rng.uniform(-0.08, 0.08)) * W, (0.5 + rng.uniform(-0.12, 0.12)) * H
        r = rng.uniform(0.03, 0.07) * W
        out += rng.uniform(0.4, 1.0) * np.exp(-0.5 * ((xs - cx) ** 2 + (ys - cy) ** 2) / r**2)
    out *= peak / out.max()
    out[:REBATE] = 0.0
    out[-REBATE:] = 0.0
    return out
