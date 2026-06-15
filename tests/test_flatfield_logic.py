import os

import numpy as np

from negpy.features.flatfield import logic as ff
from negpy.features.flatfield.models import FlatFieldConfig


def _radial_falloff(h: int, w: int) -> np.ndarray:
    """Smooth center-bright / edge-dark illumination map, 3 channels."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    falloff = 1.0 - 0.4 * np.clip(r, 0.0, 1.0)  # 1.0 center → 0.6 corner
    return np.repeat(falloff[:, :, None], 3, axis=2).astype(np.float32)


def test_disabled_is_noop():
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = ff.apply_flatfield(img, FlatFieldConfig(enabled=False, reference_path="/nope.dng"))
    assert out is img


def test_empty_path_is_noop():
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = ff.apply_flatfield(img, FlatFieldConfig(enabled=True, reference_path=""))
    assert out is img


def test_correction_flattens_uneven_illumination(tmp_path):
    h, w = 128, 192
    falloff = _radial_falloff(h, w)

    # A uniform scene captured under this illumination is just the falloff map.
    captured = falloff.copy()

    # Seed the gain cache as if `falloff` had been decoded from a reference file,
    # so apply_flatfield runs end-to-end without a real RAW decode.
    ref_file = tmp_path / "ref.dng"
    ref_file.write_bytes(b"x")
    path = str(ref_file)
    ff._GAIN_CACHE[(path, os.path.getmtime(path))] = ff._compute_gain(falloff)

    cfg = FlatFieldConfig(enabled=True, reference_path=path)
    corrected = ff.apply_flatfield(captured, cfg)

    # Before: clearly uneven. After: near-flat across the field.
    assert captured.std() > 0.05
    assert corrected.std() < 0.02
    assert corrected.dtype == np.float32


def test_gain_resized_to_image(tmp_path):
    # Gain computed at one size must resize to a differently-sized working image.
    falloff = _radial_falloff(64, 64)
    ref_file = tmp_path / "ref2.dng"
    ref_file.write_bytes(b"x")
    path = str(ref_file)
    ff._GAIN_CACHE[(path, os.path.getmtime(path))] = ff._compute_gain(falloff)

    img = np.full((100, 140, 3), 0.5, dtype=np.float32)
    out = ff.apply_flatfield(img, FlatFieldConfig(enabled=True, reference_path=path))
    assert out.shape == img.shape
