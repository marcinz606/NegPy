"""The negative's film curve and what the bake reads off a frame before any leak: base, film,
picture box and rebate.

A leak adds exposure to each emulsion layer, so it comes off in each layer's exposure above the
base, u = 10^(D / gamma) - 1 for a layer density D over the base: the toe form of
D = D_min + gamma log10(1 + E). The scanner's channels see the dye layers mixed; the crosstalk
unmix, when set, turns channel density into layer density first."""

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional

import negpy_core
import numpy as np

from negpy.features.process.sensor import film_base

# Straight-line slope of a camera negative. One value for every channel and stock.
NEG_GAMMA = 0.6
# A slide's slope where its density falls from its black, and the transmittance it is read down to.
SLIDE_GAMMA = 1.8
_SLIDE_FLOOR = 1e-5

# The estimate runs on a copy at this long edge, so preview and export find the same field.
WORK_SIZE = 1024
# A pixel is film only between these densities over the base: thinner is scanner bed or sprocket
# hole, denser is holder or a blocked shoulder. The low bound is loose because film_base can read
# a little denser than the true base.
_MIN_REL_DENSITY = -0.1
_MAX_REL_DENSITY = 3.0
# Near the shoulder a little density is a lot of exposure: the inverse stops at this share of D_s.
SHOULDER_LIMIT = 0.995
# The frame's densest film, read past its brightest specks.
SHOULDER_CEILING_PERCENTILE = 99.9
RATIO_FLOOR = 1e-4


def film_shoulder(dmax: float = 0.0, densest: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """D_s per channel from Film Dmax, or None for a straight line. Not read from the frame: a frame's
    densest film is as often a bright subject as the film's ceiling, and taking it for the ceiling
    reads a leak over dense film as mostly subject. D_s never sits under *densest*, the frame's own
    densest film per channel: the curve cannot give back a density above D_s."""
    if dmax <= 0.0:
        return None
    shoulder = np.full(3, float(dmax), dtype=np.float32)
    if densest is not None:
        shoulder = np.maximum(shoulder, np.asarray(densest, dtype=np.float32) / SHOULDER_LIMIT)
    return shoulder


def slide_to_negative(t: np.ndarray) -> np.ndarray:
    """A slide's transmittance as the negative the leak tools read. A slide thins with exposure from
    its black, so its density drop below the black, scaled by NEG_GAMMA / SLIDE_GAMMA, is that
    negative's density over its base: the black reads as base and clear film as the densest film."""
    return (_SLIDE_FLOOR / np.maximum(t, _SLIDE_FLOOR)) ** np.float32(NEG_GAMMA / SLIDE_GAMMA)


def negative_to_slide(t: np.ndarray) -> np.ndarray:
    return _SLIDE_FLOOR * np.maximum(t, RATIO_FLOOR * _SLIDE_FLOOR) ** np.float32(-SLIDE_GAMMA / NEG_GAMMA)


def work_copy(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = WORK_SIZE / max(h, w)
    rgb = np.ascontiguousarray(img[..., :3], dtype=np.float32)
    if scale >= 1.0:
        return rgb
    return negpy_core.resize_area(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))


def film_mask(small: np.ndarray, base: np.ndarray) -> np.ndarray:
    rel_density = -np.log10(np.maximum(small / base, RATIO_FLOOR)).mean(axis=-1)
    return (rel_density >= _MIN_REL_DENSITY) & (rel_density <= _MAX_REL_DENSITY)


@dataclass(frozen=True)
class FilmResponse:
    """How a scan maps to each emulsion layer's exposure above the base, and back. *base* is the
    scan's transmittance at the film base; *shoulder* D_s per layer, or None for a straight line;
    *unmix* the channel-to-layer density matrix, or None when the channels are the layers."""

    base: np.ndarray
    gamma: float = NEG_GAMMA
    shoulder: Optional[np.ndarray] = None
    unmix: Optional[np.ndarray] = None

    def density(self, t: np.ndarray) -> np.ndarray:
        """Layer density over the base."""
        d = -np.log10(np.maximum(t / self.base, RATIO_FLOOR))
        return d if self.unmix is None else d @ self.unmix.T.astype(np.float32)

    def excess(self, t: np.ndarray) -> np.ndarray:
        return self.excess_from_density(self.density(t))

    def excess_from_density(self, d: np.ndarray) -> np.ndarray:
        """The inverse of D = D_s tanh(gamma log10(1 + E) / D_s), the toe form in shadows and midtones
        flattening to D_s; near D_s a little density is a lot of exposure, so it stops at SHOULDER_LIMIT."""
        if self.shoulder is None:
            return 10.0 ** (d / self.gamma) - 1.0
        ratio = np.clip(d / self.shoulder, -SHOULDER_LIMIT, SHOULDER_LIMIT)
        return 10.0 ** (self.shoulder / self.gamma * np.arctanh(ratio)) - 1.0

    def density_from_excess(self, u: np.ndarray) -> np.ndarray:
        logs = np.log10(np.maximum(1.0 + u, RATIO_FLOOR))
        if self.shoulder is None:
            return self.gamma * logs
        return self.shoulder * np.tanh(self.gamma * logs / self.shoulder)

    def transmittance_from_density(self, d: np.ndarray) -> np.ndarray:
        if self.unmix is not None:
            d = d @ np.linalg.inv(self.unmix).T.astype(np.float32)
        return self.base * 10.0 ** (-d)

    def transmittance(self, u: np.ndarray) -> np.ndarray:
        return self.transmittance_from_density(self.density_from_excess(u))

    def past_limit(self, d: np.ndarray) -> np.ndarray:
        """Layers at or past the shoulder's limit: their exposure cannot be read back."""
        if self.shoulder is None:
            return np.zeros(d.shape, dtype=bool)
        return d >= SHOULDER_LIMIT * self.shoulder


Rect = tuple[int, int, int, int]


class FilmReading(NamedTuple):
    small: np.ndarray
    film: np.ndarray
    response: FilmResponse
    rect: Optional[Rect]
    profiles: Dict[str, np.ndarray]
    # The rebate just outside the frame line, where a crossing is read.
    lines: Dict[str, np.ndarray]


def read_film(img: np.ndarray, geometry=None, dmax: float = 0.0, unmix: Optional[np.ndarray] = None) -> Optional[FilmReading]:
    """The frame at the work size with its film response, picture box and rebate profiles; None
    when the base cannot be read."""
    from negpy.features.leak.rebate import picture_rect, read_profiles, refine_base

    small = work_copy(img)
    base = film_base(small)
    if base is None or np.any(base <= 0):
        return None
    film = film_mask(small, base)
    rect = picture_rect(small, geometry)
    profiles: Dict[str, np.ndarray] = {}
    lines: Dict[str, np.ndarray] = {}
    if rect is not None:
        profiles = read_profiles(small, film, rect)
        base = refine_base(profiles, base)
        film = film_mask(small, base)
        lines = read_profiles(small, film, rect, line=True)
    layers = FilmResponse(base, unmix=unmix)
    shoulder = film_shoulder(dmax, layers_densest(small, film, layers))
    return FilmReading(small, film, FilmResponse(base, shoulder=shoulder, unmix=unmix), rect, profiles, lines)


def layers_densest(small: np.ndarray, film: np.ndarray, response: FilmResponse) -> Optional[np.ndarray]:
    if not film.any():
        return None
    return np.percentile(response.density(small[film][:, :3]), SHOULDER_CEILING_PERCENTILE, axis=0)
