import numpy as np
import pytest

from negpy.features.flatfield import logic as ff
from negpy.features.flatfield.models import FlatFieldConfig


def _radial_falloff(h: int, w: int) -> np.ndarray:
    """Smooth center-bright / edge-dark illumination map, 3 channels."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    falloff = 1.0 - 0.4 * np.clip(r, 0.0, 1.0)  # 1.0 center → 0.6 corner
    return np.repeat(falloff[:, :, None], 3, axis=2).astype(np.float32)


@pytest.fixture
def gain_store():
    """Back the provider with an in-memory {id: (gain, token)} map; restore on exit."""
    store: dict[str, tuple[np.ndarray, str]] = {}

    def register(profile_id: str, reference: np.ndarray) -> None:
        gain = ff.compute_gain(reference)
        store[profile_id] = (gain, ff.gain_token(gain))

    ff.set_gain_provider(lambda pid: store.get(pid))
    try:
        yield register
    finally:
        ff.set_gain_provider(None)


def test_disabled_is_noop():
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = ff.apply_flatfield(img, FlatFieldConfig(apply=False, profile_id="anything"))
    assert out is img


def test_empty_profile_is_noop():
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = ff.apply_flatfield(img, FlatFieldConfig(apply=True, profile_id=""))
    assert out is img


def test_unknown_profile_is_noop(gain_store):
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = ff.apply_flatfield(img, FlatFieldConfig(apply=True, profile_id="ghost"))
    assert out is img
    assert ff.flatfield_token(FlatFieldConfig(apply=True, profile_id="ghost")) == ""


def test_correction_flattens_uneven_illumination(gain_store):
    h, w = 128, 192
    falloff = _radial_falloff(h, w)
    gain_store("rig", falloff)

    # A uniform scene captured under this illumination is just the falloff map.
    captured = falloff.copy()
    corrected = ff.apply_flatfield(captured, FlatFieldConfig(apply=True, profile_id="rig"))

    # Before: clearly uneven. After: near-flat across the field.
    assert captured.std() > 0.05
    assert corrected.std() < 0.02
    assert corrected.dtype == np.float32


def test_gain_resized_to_image(gain_store):
    # Gain baked at one size must resize to a differently-sized working image.
    gain_store("rig", _radial_falloff(64, 64))
    img = np.full((100, 140, 3), 0.5, dtype=np.float32)
    out = ff.apply_flatfield(img, FlatFieldConfig(apply=True, profile_id="rig"))
    assert out.shape == img.shape


def test_token_is_stable_and_profile_scoped(gain_store):
    gain_store("a", _radial_falloff(64, 64))
    gain_store("b", _radial_falloff(48, 72))
    cfg_a = FlatFieldConfig(apply=True, profile_id="a")

    tok = ff.flatfield_token(cfg_a)
    assert tok.startswith("|ff:a:")
    assert tok == ff.flatfield_token(cfg_a)  # deterministic
    assert tok != ff.flatfield_token(FlatFieldConfig(apply=True, profile_id="b"))
    assert ff.flatfield_token(FlatFieldConfig(apply=False, profile_id="a")) == ""


def test_invalidate_drops_cache(gain_store):
    gain_store("rig", _radial_falloff(64, 64))
    cfg = FlatFieldConfig(apply=True, profile_id="rig")
    assert ff.flatfield_token(cfg) != ""  # populates the cache

    ff.set_gain_provider(lambda pid: None)  # provider now yields nothing
    ff.invalidate_gain("rig")
    assert ff.flatfield_token(cfg) == ""
