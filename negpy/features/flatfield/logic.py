"""Flat-field / vignetting correction for uneven scanner backlight."""

import numpy as np
import cv2


EPSILON = 1e-6


def normalize_flatfield(flat: np.ndarray) -> np.ndarray:
    """Normalizes a raw flat-field frame into a relative illumination map.

    Divides each channel by its mean so the map is centered around 1.0:
    values >1.0 are brighter-than-average, <1.0 are dimmer. The minimum
    is clamped to epsilon to prevent division by zero when used as a divisor.
    """
    flat = flat.astype(np.float32)
    flat = np.maximum(flat, EPSILON)
    for ch in range(flat.shape[2]):
        ch_mean = flat[:, :, ch].mean()
        if ch_mean > EPSILON:
            flat[:, :, ch] /= ch_mean
    flat = np.maximum(flat, EPSILON)
    return flat


def apply_flatfield(image: np.ndarray, flatfield: np.ndarray) -> np.ndarray:
    """Divides image by flat-field map to correct uneven illumination.

    If shapes differ, the flat-field is resized (bilinear) to match.
    Output is clipped to [0, 1].
    """
    h, w = image.shape[:2]
    fh, fw = flatfield.shape[:2]
    if fh != h or fw != w:
        flatfield = cv2.resize(flatfield, (w, h), interpolation=cv2.INTER_LINEAR)
    result = image / flatfield
    clipped: np.ndarray = np.clip(result, 0.0, 1.0).astype(np.float32)
    return clipped


def load_flatfield(path: str) -> np.ndarray:
    """Loads a flat-field reference frame and returns a normalized illumination map.

    Uses the same loader infrastructure as the main pipeline. The result is
    a float32 array with per-channel mean of ~1.0, ready for use with
    apply_flatfield().
    """
    from negpy.infrastructure.loaders.factory import loader_factory
    from negpy.infrastructure.loaders.helpers import get_best_demosaic_algorithm
    from negpy.kernel.image.logic import uint16_to_float32, ensure_rgb

    ctx_mgr, _metadata = loader_factory.get_loader(path)
    with ctx_mgr as raw:
        algo = get_best_demosaic_algorithm(raw)
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=False,
            user_wb=[1, 1, 1, 1],
            output_bps=16,
            demosaic_algorithm=algo,
        )
        rgb = ensure_rgb(rgb)

    flat = uint16_to_float32(np.ascontiguousarray(rgb))
    return normalize_flatfield(flat)


def load_raw_to_float32(path: str) -> np.ndarray:
    """Loads any supported image file as linear float32 [0-1].

    Mirrors the loading portion of ImageProcessor.process_export() so the
    CLI can load files independently when flat-field correction is needed.
    """
    from negpy.infrastructure.loaders.factory import loader_factory
    from negpy.infrastructure.loaders.helpers import get_best_demosaic_algorithm
    from negpy.kernel.image.logic import uint16_to_float32, ensure_rgb

    ctx_mgr, _metadata = loader_factory.get_loader(path)
    with ctx_mgr as raw:
        algo = get_best_demosaic_algorithm(raw)
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=False,
            user_wb=[1, 1, 1, 1],
            output_bps=16,
            demosaic_algorithm=algo,
        )
        rgb = ensure_rgb(rgb)

    f32: np.ndarray = uint16_to_float32(np.ascontiguousarray(rgb))
    return f32
