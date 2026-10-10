"""The Sabattier effect: light tones fold back at the re-exposure density, dense tones hold,
and a light Mackie line runs along every dark/light edge."""

import numpy as np

from negpy.features.sabattier.logic import SABATTIER_CONSTANTS, apply_sabattier, line_sigma_px

D_MAX = 2.1


def _flat(value: float, h: int = 16, w: int = 24) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.float32)


def _density(img: np.ndarray) -> np.ndarray:
    return -np.log10(np.clip(np.asarray(img)[:, :, 0], 1e-6, 1.0))


def test_disabled_returns_the_same_object():
    img = _flat(0.5)
    assert apply_sabattier(img, D_MAX, enabled=False) is img


def test_zero_strength_leaves_the_print_alone():
    img = _flat(0.5)
    np.testing.assert_allclose(apply_sabattier(img, D_MAX, enabled=True, strength=0.0), img, atol=1e-6)


def test_paper_white_comes_back_at_the_strength_times_the_re_exposure():
    out = apply_sabattier(_flat(1.0), D_MAX, enabled=True, strength=0.8, reexposure=0.45, line_sigma_px=0.0)
    re = 0.45 * D_MAX
    # The softplus fold adds w·ln(1 + e^{-R/w}) above the straight line, which is nothing at R ≫ w.
    np.testing.assert_allclose(_density(out), 0.8 * re, atol=1e-3)


def test_dense_tones_hold():
    dense = 10.0 ** -(0.45 * D_MAX + 0.5)
    out = apply_sabattier(_flat(dense), D_MAX, enabled=True, strength=1.5, reexposure=0.45, line_sigma_px=0.0)
    np.testing.assert_allclose(_density(out), 0.45 * D_MAX + 0.5, atol=2e-3)


def test_the_fold_reverses_the_light_tones_in_order():
    """The lighter the tone, the more it reverses: the output is non-monotone below R."""
    tones = np.array([1.0, 0.7, 0.4, 0.2], dtype=np.float32)
    img = np.ascontiguousarray(np.repeat(np.repeat(tones[None, :, None], 8, axis=0), 3, axis=2))
    out = _density(apply_sabattier(img, D_MAX, enabled=True, strength=1.3, reexposure=0.6, line_sigma_px=0.0))[0]
    assert out[0] > out[1] > out[2], "paper white prints darkest after the flash"
    flat = _density(apply_sabattier(img, D_MAX, enabled=True, strength=1.0, reexposure=0.6, line_sigma_px=0.0))[0]
    np.testing.assert_allclose(flat[:3], 0.6 * D_MAX, atol=0.06)  # strength 1 lays them on the fold


def test_the_reversal_never_passes_paper_black():
    out = apply_sabattier(_flat(1.0), D_MAX, enabled=True, strength=2.0, reexposure=0.9, line_sigma_px=0.0)
    np.testing.assert_allclose(_density(out), D_MAX, atol=1e-3)


def test_the_mackie_line_is_a_light_band_on_the_light_side_of_an_edge():
    h, w = 32, 96
    img = _flat(1.0, h, w)
    img[:, w // 2 :] = 10.0 ** -(0.45 * D_MAX + 0.8)  # a dense right half
    flat = _density(apply_sabattier(img, D_MAX, enabled=True, strength=0.8, reexposure=0.45, line_sigma_px=0.0))
    lined = _density(apply_sabattier(img, D_MAX, enabled=True, strength=0.8, reexposure=0.45, line_sigma_px=3.0))
    mid = h // 2
    # Next to the edge the light side reverses less, so it prints lighter than away from the edge.
    assert lined[mid, w // 2 - 1] < lined[mid, 4] - 0.1
    assert lined[mid, w // 2 - 1] < flat[mid, w // 2 - 1] - 0.1
    np.testing.assert_allclose(lined[mid, :8], flat[mid, :8], atol=1e-3)
    np.testing.assert_allclose(lined[mid, -8:], flat[mid, -8:], atol=1e-3)


def test_agitation_shortens_the_line_on_the_short_side():
    assert line_sigma_px(0.0, (1000, 2000)) == 10.0
    assert line_sigma_px(0.5, (1000, 2000)) == 5.0
    assert line_sigma_px(0.5, (250, 2000)) == 1.25
    assert line_sigma_px(1.0, (1000, 2000)) == 0.0


def test_the_output_is_neutral_silver():
    img = _flat(0.6)
    img[:, :, 0] *= 1.05  # a trace of halation red
    out = np.asarray(apply_sabattier(img, D_MAX, enabled=True))
    np.testing.assert_array_equal(out[:, :, 0], out[:, :, 1])
    np.testing.assert_array_equal(out[:, :, 1], out[:, :, 2])
    assert SABATTIER_CONSTANTS["fold_width"] > 0


def test_a_sabattier_print_keeps_its_carrier_ramp_like_a_plain_print():
    """Plain silver: the rebate prints through the kernel and the toners; lith and cyanotype
    fall back to the flat carrier tone because their color is not in the ramp."""
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.features.altprocess.models import AltProcess
    from negpy.features.exposure.normalization import LogNegativeBounds
    from negpy.features.finish.processor import linear_carrier_tone, rebate_tone
    from negpy.features.process.models import ProcessMode

    base = WorkspaceConfig()
    base = replace(base, process=replace(base.process, process_mode=ProcessMode.BW))
    metrics = {"final_bounds": LogNegativeBounds((-2.0, -2.0, -2.0), (-0.2, -0.2, -0.2))}
    sab = rebate_tone(replace(base, altproc=replace(base.altproc, alt_process=AltProcess.SABATTIER)), metrics)
    lith = rebate_tone(replace(base, altproc=replace(base.altproc, alt_process=AltProcess.LITH)), metrics)
    np.testing.assert_array_equal(lith, linear_carrier_tone())
    assert not np.array_equal(sab, linear_carrier_tone())
    np.testing.assert_array_equal(sab, rebate_tone(base, metrics))
