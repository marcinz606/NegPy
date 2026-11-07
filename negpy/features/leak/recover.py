"""After the leak comes off: the layers it saturated rebuilt from the others, clean-up of what
removal stretched, and the film it burned out in every layer."""

from typing import Optional

import negpy_core
import numpy as np

from negpy.features.leak.film import RATIO_FLOOR, SHOULDER_CEILING_PERCENTILE, SHOULDER_LIMIT, FilmResponse, film_mask, work_copy

# Clean-up after removal: full weight where _CLEAN_FULL D came off, the removal map blurred over
# _CLEAN_WEIGHT_BLUR of the long side; grain is read over _GRAIN_RADIUS of it and flattened below
# _GRAIN_EPS of its spread; contrast lifts detail against a blur over _CONTRAST_SCALE of it, by up
# to _CONTRAST_GAIN.
_CLEAN_FULL = 0.05
_CLEAN_WEIGHT_BLUR = 0.002
_GRAIN_RADIUS = 0.0006
_GRAIN_MIN_RADIUS = 2
_GRAIN_EPS = 2.0
_CONTRAST_SCALE = 0.02
_CONTRAST_GAIN = 1.5
_BROAD_SAMPLES = 8.0
# A layer is saturated inside the leak's area (where it is over _BURNED_LEAK_SHARE of its peak) within
# _BURNED_MARGIN of its ceiling: its shoulder, or its densest film with none. A layer the leak saturated
# is rebuilt from the others by a local fit over the clean film around it, in the smallest window of
# _RECOVER_RADII (shares of the long side) where the clean film weighs at least _RECOVER_MIN_WEIGHT; the
# fit's slope flattens below _RECOVER_EPS of the guide's spread, so grain does not set it.
_BURNED_LEAK_SHARE = 0.1
_BURNED_MARGIN = 0.05
_DRIVEN_SHARE = 0.5
_RECOVER_RADII = (0.03, 0.06, 0.12, 0.24)
_RECOVER_MIN_WEIGHT = 0.05
_RECOVER_EPS = 0.01
# The film a fit reads is where the leak adds under this share to each layer's exposure (plus one).
_CLEAN_LEAK_SHARE = 0.05
# A pixel a rebuilt layer reaches by less than this share keeps its value.
_RECOVER_TOUCH = 1e-3
# Full-size work runs over row blocks of this height, to bound temporary storage.
_ROW_BLOCK = 512


def layer_saturation(small: np.ndarray, response: FilmResponse, leak: np.ndarray) -> Optional[np.ndarray]:
    """(h, w, 3) at the work size: the layers the leak drove to their ceiling, where the curve gives
    no exposure back; None when the frame has no film. The ceiling is the shoulder Film Dmax sets,
    else the layer's densest film. A leak saturates the layers its color drives, those it exposes
    with at least _DRIVEN_SHARE of its strongest layer's exposure: a layer it barely exposes holds
    picture, however dense."""
    film = film_mask(small, response.base)
    if not film.any():
        return None
    density = response.density(small)
    if response.shoulder is not None:
        ceiling = SHOULDER_LIMIT * np.asarray(response.shoulder, dtype=np.float32)
    else:
        ceiling = np.percentile(density[film], SHOULDER_CEILING_PERCENTILE, axis=0)
    strength = leak.max(axis=-1)
    inside = film & (strength > _BURNED_LEAK_SHARE * float(strength.max()))
    driven = leak >= _DRIVEN_SHARE * strength[..., None]
    saturated = inside[..., None] & driven & (density >= (1.0 - _BURNED_MARGIN) * ceiling)
    kernel = np.ones((3, 3), np.uint8)
    return np.stack(
        [
            negpy_core.morphology_ex(np.ascontiguousarray(saturated[..., c], dtype=np.uint8), negpy_core.MORPH_OPEN, kernel).astype(bool)
            for c in range(3)
        ],
        axis=-1,
    )


def burned_mask(saturated: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Film the leak saturated in every layer, at the work size: no layer holds picture to rebuild
    it from."""
    return None if saturated is None else np.all(saturated, axis=-1)


def _weighted_box(values: np.ndarray, weight: np.ndarray, size: int) -> tuple:
    """(weighted local mean, local weight) over a box of *size*."""
    w = negpy_core.box_filter(np.ascontiguousarray(weight, dtype=np.float32), (size, size))
    return negpy_core.box_filter(np.ascontiguousarray(values * weight, dtype=np.float32), (size, size)) / np.maximum(w, 1e-6), w


def recover_layers(out: np.ndarray, response: FilmResponse, saturated: np.ndarray, leak: np.ndarray) -> None:
    """Rebuild, in place on *out*, each layer the leak saturated where another still holds picture.
    Its exposure is fitted, locally, as a straight line in log exposure of the mean of the other two
    layers, over the film around it the leak never reached (He et al.'s guided
    filter, weighted to that film), and the fit carries the other layers' detail in at full size. A layer at its
    ceiling has lost its picture; the scene's own color, read around it, gives it back."""
    partial = saturated & ~np.all(saturated, axis=-1, keepdims=True)
    if not partial.any():
        return
    small = work_copy(out)
    h, w = small.shape[:2]
    full_h, full_w = out.shape[:2]
    u = np.maximum(response.excess(small), 0.0)
    untouched = np.all(leak <= _CLEAN_LEAK_SHARE * (1.0 + u), axis=-1)
    clean = (untouched & film_mask(small, response.base)).astype(np.float32)
    logs = np.log1p(u).astype(np.float32)
    fits = []
    for c in range(3):
        if not partial[..., c].any():
            continue
        others = [o for o in range(3) if o != c]
        guide = logs[..., others].mean(axis=-1)
        slope = np.zeros((h, w), np.float32)
        offset = np.zeros((h, w), np.float32)
        reach = np.zeros((h, w), dtype=bool)
        # The smallest window that holds enough clean film fits each pixel: as local as the gap allows.
        for radius in _RECOVER_RADII:
            size = 2 * max(2, round(radius * max(h, w))) + 1
            mean_g, weight = _weighted_box(guide, clean, size)
            mean_y, _ = _weighted_box(logs[..., c], clean, size)
            var_g = _weighted_box(guide * guide, clean, size)[0] - mean_g * mean_g
            cov = _weighted_box(guide * logs[..., c], clean, size)[0] - mean_g * mean_y
            fits_here = (weight >= _RECOVER_MIN_WEIGHT) & ~reach
            a = cov / (var_g + _RECOVER_EPS)
            slope = np.where(fits_here, a, slope)
            offset = np.where(fits_here, mean_y - a * mean_g, offset)
            reach |= fits_here
        reach &= partial[..., c]
        if reach.any():
            fits.append((c, others, slope, offset, negpy_core.gaussian_blur(reach.astype(np.float32), (0, 0), 1.0)))
    if not fits:
        return
    # Only the rows a rebuilt layer reaches go to full size.
    ys = np.flatnonzero(np.any([blend > 0.0 for *_, blend in fits], axis=(0, 2)))
    y0 = max(0, int(np.floor(ys[0] * full_h / h)) - 2)
    y1 = min(full_h, int(np.ceil((ys[-1] + 1) * full_h / h)) + 2)
    full_logs = np.empty((y1 - y0, full_w, 3), np.float32)
    for b0 in range(0, y1 - y0, _ROW_BLOCK):
        rows = slice(b0, min(y1 - y0, b0 + _ROW_BLOCK))
        full_logs[rows] = np.log1p(np.maximum(response.excess(out[y0 + rows.start : y0 + rows.stop, :, :3]), 0.0))
    up = lambda a: negpy_core.resize_linear(np.ascontiguousarray(a, dtype=np.float32), (full_w, full_h))[y0:y1]  # noqa: E731
    touched = np.zeros((y1 - y0, full_w), dtype=bool)
    for c, others, slope, offset, blend in fits:
        share = up(blend)
        rebuilt = up(slope) * full_logs[..., others].mean(axis=-1) + up(offset)
        full_logs[..., c] = full_logs[..., c] + share * (rebuilt - full_logs[..., c])
        touched |= share > _RECOVER_TOUCH
    if not touched.any():
        return
    # Only rebuilt pixels go through the curve and back; the rest keep their exact value.
    density = response.density_from_excess(np.expm1(full_logs[touched]))
    out[y0:y1, :, :3][touched] = response.transmittance_from_density(density).astype(np.float32)


def _guided(guide: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al.'s guided filter of an image by itself: edges stay, grain flattens."""
    size = (2 * radius + 1, 2 * radius + 1)
    mean = negpy_core.box_filter(guide, size)
    var = negpy_core.box_filter(guide * guide, size) - mean * mean
    a = var / (var + eps)
    b = mean - a * mean
    return negpy_core.box_filter(a, size) * guide + negpy_core.box_filter(b, size)


def clean_up(before: np.ndarray, after: np.ndarray, base: np.ndarray, grain: float, contrast: float) -> None:
    """Grain reduction and a local contrast lift on *after*, in place, in luminance density and
    weighted by how much leak came off each pixel, so film the leak never reached is untouched.
    Removal stretches what the leak left, grain with it, and leaves the contrast it squashed."""
    h, w = after.shape[:2]
    long_side = float(max(h, w))
    # Full-size buffers are worked in place: a frame's density plane is a large share of the bake's storage.
    d_after = -np.log10(np.maximum(after[..., :3].mean(axis=-1), RATIO_FLOOR)).astype(np.float32)
    removed = -np.log10(np.maximum(before[..., :3].mean(axis=-1), RATIO_FLOOR)).astype(np.float32)
    removed -= d_after
    np.maximum(removed, 0.0, out=removed)
    weight = negpy_core.gaussian_blur(removed, (0, 0), max(1.0, long_side * _CLEAN_WEIGHT_BLUR))
    del removed
    weight /= _CLEAN_FULL
    np.clip(weight, 0.0, 1.0, out=weight)
    weight *= weight * (3.0 - 2.0 * weight)
    for y0 in range(0, h, _ROW_BLOCK):
        rows = slice(y0, min(h, y0 + _ROW_BLOCK))
        # Thinner than the base is bed or sprocket hole, never film.
        weight[rows] *= np.all(before[rows, :, :3] <= base, axis=-1)
    if not float(weight.max()) > 0.0:
        return
    d = d_after.copy()
    if grain > 0.0:
        radius = max(_GRAIN_MIN_RADIUS, round(long_side * _GRAIN_RADIUS))
        detail = negpy_core.box_filter(d, (2 * radius + 1, 2 * radius + 1))
        np.subtract(d, detail, out=detail)
        detail -= np.median(detail)
        np.abs(detail, out=detail)
        sigma = 1.4826 * float(np.median(detail))
        del detail
        eps = max((_GRAIN_EPS * sigma) ** 2, 1e-8)
        # Two box passes reach 2 radius: row blocks with that halo filter exactly as the whole frame.
        halo = 2 * radius
        change = np.empty_like(d)
        for y0 in range(0, h, _ROW_BLOCK):
            y1 = min(h, y0 + _ROW_BLOCK)
            a, b = max(0, y0 - halo), min(h, y1 + halo)
            change[y0:y1] = _guided(d[a:b], radius, eps)[y0 - a : y0 - a + (y1 - y0)]
        change -= d
        change *= weight
        change *= grain
        d += change
        del change
    if contrast > 0.0:
        lift = _broad_blur(d, long_side * _CONTRAST_SCALE)
        np.subtract(d, lift, out=lift)
        lift *= weight
        lift *= contrast * _CONTRAST_GAIN
        d += lift
        del lift
    for y0 in range(0, h, _ROW_BLOCK):
        rows = slice(y0, min(h, y0 + _ROW_BLOCK))
        after[rows, :, :3] *= np.power(np.float32(10.0), d_after[rows] - d[rows])[..., None]


def _broad_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur over *sigma*, run on a copy reduced to _BROAD_SAMPLES samples per sigma: a blur this
    wide holds nothing finer, and the full-size kernel is the slowest step of a full-size bake. The copy
    carries 4 sigma of reflected film past each side, so the reduced blur reflects at the true border."""
    f = int(sigma // _BROAD_SAMPLES)
    if f <= 1:
        return negpy_core.gaussian_blur(img, (0, 0), sigma)
    h, w = img.shape[:2]
    r = int(np.ceil(4.0 * sigma))
    padded = np.pad(img, ((r, r + (-(h + 2 * r) % f)), (r, r + (-(w + 2 * r) % f))), mode="reflect")
    ph, pw = padded.shape[:2]
    small = padded.reshape(ph // f, f, pw // f, f).mean(axis=(1, 3), dtype=np.float32)
    blurred = negpy_core.resize_linear(negpy_core.gaussian_blur(small, (0, 0), sigma / f), (pw, ph))
    return blurred[r : r + h, r : r + w]
