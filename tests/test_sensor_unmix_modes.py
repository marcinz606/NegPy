from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import ProcessConfig, SensorUnmix
from negpy.features.process.sensor import (
    ADAPTIVE_FULL,
    ADAPTIVE_NONE,
    SOFT_FLOOR,
    apply_sensor_correction,
    density_unmix_matrix,
    film_base,
    sensor_token,
)

# A real single-shot calibration (A7R4 + Big Scanlight): green leaks ~25% into blue.
_M = (1.0133, -0.0848, -0.0108, -0.1535, 1.1168, -0.4224, -0.0011, -0.2706, 1.104)
_BASE = np.array([0.23, 0.56, 0.33], dtype=np.float32)


def _negative(rng, h=256, w=384):
    """Orange-base negative: per-pixel densities above the base, with mild grain, and a
    patch of clear base in the interior, where film_base reads it."""
    density = rng.uniform(0.1, 1.2, (h, w, 1)) + rng.normal(0.0, 0.03, (h, w, 3))
    img = _BASE * 10.0 ** (-density)
    img[96:160, 160:224] = _BASE
    return img.astype(np.float32)


def _linear(img):
    return np.clip(np.einsum("ck,hwk->hwc", np.asarray(_M, np.float32).reshape(3, 3), img), 0.0, None)


def test_linear_mode_is_the_plain_unmix():
    img = _negative(np.random.default_rng(1))
    assert np.array_equal(apply_sensor_correction(img, _M), _linear(img))
    assert np.array_equal(apply_sensor_correction(img, _M, SensorUnmix.LINEAR), _linear(img))


def test_adaptive_matches_linear_where_the_signal_survives():
    img = _negative(np.random.default_rng(2))
    out = apply_sensor_correction(img, _M, SensorUnmix.ADAPTIVE)
    assert np.allclose(out, _linear(img), rtol=1e-5, atol=1e-7)


def _neon_pixels():
    """Blue light through dense yellow dye: almost all of the blue record is green leak."""
    g = np.float32(0.29)
    blues = np.linspace(0.055, 0.15, 16, dtype=np.float32)  # unmixed blue from below zero to healthy
    img = np.empty((16, 16, 3), np.float32)
    img[:, :, 0] = 0.23
    img[:, :, 1] = g
    img[:, :, 2] = blues[None, :]
    return img


def test_adaptive_never_drops_a_channel_below_its_floor():
    img = _neon_pixels()
    assert (_linear(img)[:, :, 2] == 0).any()  # the case Linear clips to infinite density
    out = apply_sensor_correction(img, _M, SensorUnmix.ADAPTIVE)
    raw = img * np.diag(np.asarray(_M).reshape(3, 3))
    assert (out >= ADAPTIVE_NONE * raw - 1e-7).all()


def test_adaptive_fades_every_channel_together():
    img = _neon_pixels()
    out = apply_sensor_correction(img, _M, SensorUnmix.ADAPTIVE)
    raw = img * np.diag(np.asarray(_M).reshape(3, 3))
    full = np.einsum("ck,hwk->hwc", np.asarray(_M, np.float32).reshape(3, 3), img)
    moved = (out - raw) / np.where(np.abs(full - raw) > 1e-6, full - raw, np.nan)
    spread = np.nanmax(moved, axis=2) - np.nanmin(moved, axis=2)
    assert np.nanmax(spread) < 1e-4


def test_adaptive_strength_follows_the_surviving_fraction():
    img = _neon_pixels()
    out = apply_sensor_correction(img, _M, SensorUnmix.ADAPTIVE)
    m = np.asarray(_M, np.float32).reshape(3, 3)
    surviving = (np.einsum("ck,hwk->hwc", m, img) / (img * np.diag(m)))[:, :, 2]
    healthy = surviving > ADAPTIVE_FULL + 0.02
    assert healthy.any()
    assert np.allclose(out[healthy], _linear(img)[healthy], rtol=1e-5)


def test_adaptive_guide_handles_a_frame_above_the_guide_size():
    img = np.tile(_neon_pixels(), (2, 140, 1))  # 32 x 2240
    out = apply_sensor_correction(img, _M, SensorUnmix.ADAPTIVE)
    assert out.shape == img.shape and np.isfinite(out).all() and (out > 0).all()


def test_density_matrix_rows_sum_to_one():
    c = density_unmix_matrix(_M, _BASE)
    assert np.allclose(c.sum(axis=1), 1.0, atol=1e-12)


def test_density_is_exact_at_the_film_base_and_close_near_it():
    img = _negative(np.random.default_rng(3))
    base = film_base(img)
    out = apply_sensor_correction(img, _M, SensorUnmix.DENSITY)
    m = np.asarray(_M, np.float64).reshape(3, 3)
    assert np.allclose(base, _BASE, rtol=1e-6)
    assert np.allclose(np.asarray(out[96:160, 160:224], np.float64), (m @ base)[None, None, :], rtol=1e-4)
    near = np.log10(out) - np.log10(np.maximum(_linear(img), 1e-6))
    near -= np.median(near, axis=(0, 1))
    assert np.median(np.abs(near)) < 0.01


def test_density_never_reaches_zero():
    img = _neon_pixels()
    img[:2, :2] = _BASE
    out = apply_sensor_correction(img, _M, SensorUnmix.DENSITY)
    assert (out > 0).all() and np.isfinite(out).all()


def test_density_falls_back_to_linear_without_a_usable_base():
    img = np.zeros((8, 8, 3), np.float32)
    assert np.array_equal(apply_sensor_correction(img, _M, SensorUnmix.DENSITY), _linear(img))


def test_sensor_token_names_a_non_linear_mode_only():
    process = replace(ProcessConfig(), linear_raw=True, sensor_matrix=_M)
    linear = sensor_token(process)
    assert "|adaptive" not in linear and "|density" not in linear
    tokens = {sensor_token(replace(process, sensor_unmix=mode)) for mode in SensorUnmix}
    assert len(tokens) == len(SensorUnmix)


def test_mode_roundtrips_and_an_unknown_value_reads_linear():
    cfg = replace(WorkspaceConfig(), process=replace(ProcessConfig(), sensor_unmix=SensorUnmix.DENSITY))
    assert WorkspaceConfig.from_flat_dict(cfg.to_dict()).process.sensor_unmix == SensorUnmix.DENSITY
    assert ProcessConfig(sensor_unmix="bogus").sensor_unmix == SensorUnmix.LINEAR


def test_soft_floor_holds_every_channel_above_its_floor():
    img = _neon_pixels()
    out = apply_sensor_correction(img, _M, SensorUnmix.SOFT_FLOOR)
    raw = img * np.diag(np.asarray(_M).reshape(3, 3))
    assert (out >= SOFT_FLOOR * raw - 1e-7).all()


def test_soft_floor_stays_close_to_linear_where_the_signal_is_strong():
    img = _negative(np.random.default_rng(4))
    lin = _linear(img)
    out = apply_sensor_correction(img, _M, SensorUnmix.SOFT_FLOOR)
    assert (out >= lin - 1e-7).all()
    assert np.median(np.log10(out) - np.log10(np.maximum(lin, 1e-6))) < 0.02
