"""Light leak reduction (WorkspaceConfig.leak), a bake on the linear negative or slide after flat-field
and unmix and before every dust repair, so each repair copies leak-free film. Removal happens only
under the shapes the user placed or Auto proposed."""

import hashlib
from typing import Optional, Tuple

import negpy_core
import numpy as np

from negpy.features.leak.estimate import estimate_leak
from negpy.features.leak.film import NEG_GAMMA, SLIDE_GAMMA, negative_to_slide, slide_to_negative, work_copy
from negpy.features.leak.recover import burned_mask, clean_up, layer_saturation, recover_layers

# Width of the softplus floor under the subtraction, as a share of the leak.
_FLOOR_SOFTNESS = 0.1
_ROW_BLOCK = 512


def leak_active(config) -> bool:
    """Whether the bake runs on a WorkspaceConfig. A Positive frame holds no film densities, and a
    composite canvas holds more than one frame's film."""
    from negpy.features.hdr.models import hdr_active
    from negpy.features.process.path import RenderPath, render_path

    return (
        bool(config.leak.leak_remove)
        and render_path(config.process) is not RenderPath.POSITIVE
        and not config.stitch.stitch_enabled
        and not hdr_active(config.hdr)
    )


def is_slide(config) -> bool:
    from negpy.features.process.path import RenderPath, render_path

    return render_path(config.process) is RenderPath.TRANSFER


def leak_scan(img: np.ndarray, config) -> np.ndarray:
    """The frame as the leak tools read it: a negative as it is, a slide as slide_to_negative turns it."""
    if not is_slide(config):
        return img
    scan = np.array(img, dtype=np.float32, copy=True)
    scan[..., :3] = slide_to_negative(scan[..., :3])
    return scan


def film_dmax(config) -> float:
    """Film Dmax on the leak tools' negative scale. On a slide it is the density from its black to clear film."""
    dmax = float(config.leak.leak_dmax)
    return dmax * NEG_GAMMA / SLIDE_GAMMA if is_slide(config) else dmax


def color_film(config) -> bool:
    """Whether the film has color layers, which a leak exposes each in its own share."""
    from negpy.features.process.models import ProcessMode

    return config.process.process_mode != ProcessMode.BW


def leak_unmix(config) -> Optional[np.ndarray]:
    """The crosstalk unmix the render applies, channel to layer density, or None."""
    from negpy.features.exposure.normalization import effective_crosstalk_matrix

    return effective_crosstalk_matrix(config.process, config.process.process_mode)


def leak_token(config) -> str:
    """Config identity of the bake, folded into source_hash. Empty when inactive. The crop (or the auto
    crop that stands in for it) and the transforms before it place the picture, the shapes place the
    leak, and the film type and unmix separate the layers, so they belong to it."""
    if not leak_active(config):
        return ""
    leak, g = config.leak, config.geometry
    unmix = leak_unmix(config)
    place = (g.crop_rect, g.rotation, g.flip_horizontal, g.flip_vertical, g.fine_rotation, g.distortion_k1, g.converge_v, g.converge_h)
    if g.crop_rect is None:
        place += (g.autocrop_mode, g.autocrop_ratio, g.autocrop_rebate_trim)
    place += (is_slide(config), color_film(config), leak.leak_shapes, None if unmix is None else np.round(unmix, 6).tolist())
    digest = hashlib.sha1(repr(place).encode()).hexdigest()[:10]
    return f"|leak{leak.leak_dmax:.3f}:{int(leak.leak_recover)}:{leak.leak_grain:.3f}:{leak.leak_contrast:.3f}:{digest}"


def remove_leak(img: np.ndarray, config) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(the frame with the leak removed, the burned film at the work size or None). Returns *img*
    itself when there is nothing to remove."""
    scan = leak_scan(img, config)
    out, burned = _remove(scan, config)
    if out is scan or scan is img:
        return (img if out is scan else out), burned
    # Only pixels the removal moved go back through the conversion; the rest keep their exact value.
    moved = np.any(out[..., :3] != scan[..., :3], axis=-1)
    result = img.copy()
    result[..., :3][moved] = negative_to_slide(out[..., :3][moved])
    return result, burned


def _remove(img: np.ndarray, config) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    leak_cfg = config.leak
    found = estimate_leak(img, leak_cfg.leak_shapes, config.geometry, film_dmax(config), leak_unmix(config), color_film(config))
    if found is None:
        return img, None
    response, leak, mean = found.response, found.field, found.mean
    # The leak comes off the local mean and the grain around it is scaled, not cut through: a
    # per-pixel subtraction sends every grain dip near the leak level to bare base. The softplus
    # floor eases the mean to zero instead of clipping it.
    soft = np.maximum(_FLOOR_SOFTNESS * leak, 1e-6)
    lifted = soft * np.logaddexp(0.0, (mean - leak) / soft)
    gain = np.where(mean > 1e-6, np.minimum(lifted / np.maximum(mean, 1e-6), 1.0), 1.0).astype(np.float32)
    h, w = img.shape[:2]
    gain = negpy_core.resize_linear(gain, (w, h))
    out = img.copy()
    for y0 in range(0, h, _ROW_BLOCK):
        rows = slice(y0, min(h, y0 + _ROW_BLOCK))
        t = out[rows, :, :3]
        d = response.density(t)
        u = response.excess_from_density(d)
        # A layer thinner than the base is bed or sprocket hole, and one past the shoulder's limit
        # has no exposure to read: both keep their density. A pixel nothing came off keeps its value.
        keep = (u <= 0) | response.past_limit(d) | (gain[rows] >= 1.0)
        moved = ~np.all(keep, axis=-1)
        if not moved.any():
            continue
        d_new = np.where(keep, d, response.density_from_excess(u * gain[rows]))
        out[rows, :, :3] = np.where(moved[..., None], response.transmittance_from_density(d_new), t)
    del gain
    saturated = layer_saturation(work_copy(img), response, leak)
    if leak_cfg.leak_recover and saturated is not None:
        recover_layers(out, response, saturated, leak)
    if leak_cfg.leak_grain > 0.0 or leak_cfg.leak_contrast > 0.0:
        clean_up(img, out, response.base, leak_cfg.leak_grain, leak_cfg.leak_contrast)
    return out, burned_mask(saturated)
