"""Dye-fade restoration: a diagonal restoration matrix composed into the crosstalk unmix
slot rather than run as its own stage.
"""

import numpy as np

from negpy.features.exposure.normalization import (
    effective_crosstalk_matrix,
    resolve_crosstalk_matrix,
    resolve_fade_matrix,
)
from negpy.features.process.models import ProcessConfig, ProcessMode

_CROSSTALK = (1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0)
_RATIO_G, _RATIO_B = 0.75, 1.125


def _forward_fade_matrix(ratio_r, ratio_g, ratio_b):
    """The forward fade on densities, built here and not reused from production so a form
    error in resolve_fade_matrix cannot cancel against this test."""
    return np.diag([ratio_r, ratio_r * ratio_g, ratio_r * ratio_b])


def test_unit_survivals_are_the_identity():
    np.testing.assert_allclose(resolve_fade_matrix(1.0, 1.0, 1.0), np.eye(3), atol=1e-12)


def test_ratio_r_is_an_overall_scalar_on_the_restoration_matrix():
    """Red's absolute survival scales the whole restoration: it is the ratio_r == 1 matrix divided by ratio_r."""
    pinned = resolve_fade_matrix(1.0, _RATIO_G, _RATIO_B)
    for ratio_r in (0.1, 0.25, 0.5, 0.9, 1.0, 2.0):
        np.testing.assert_allclose(resolve_fade_matrix(ratio_r, _RATIO_G, _RATIO_B), pinned / ratio_r, atol=1e-10)


def test_fade_matrix_round_trips_density():
    """Forward-fade then restore recovers the original density."""
    ratio_r = 0.8
    restore = resolve_fade_matrix(ratio_r, _RATIO_G, _RATIO_B)
    density = np.array([0.3, 0.9, 1.4])
    faded = _forward_fade_matrix(ratio_r, _RATIO_G, _RATIO_B) @ density
    np.testing.assert_allclose(restore @ faded, density, atol=1e-10)


def test_fade_matrix_is_diagonal_and_not_row_normalized():
    """Fade changes each layer's neutral density, and normalizing that away would make the feature a no-op."""
    m = resolve_fade_matrix(1.0, _RATIO_G, _RATIO_B)
    np.testing.assert_array_equal(m, np.diag(np.diag(m)))
    assert np.max(np.abs(np.sum(m, axis=1) - 1.0)) > 1e-6


def test_fade_off_leaves_crosstalk_matrix_unchanged():
    """An unfaded slide must not perturb the crosstalk result, including when the matrix is off."""
    assert effective_crosstalk_matrix(ProcessConfig(), ProcessMode.E6) is None

    process = ProcessConfig(crosstalk_strength=0.6, crosstalk_matrix=_CROSSTALK, crosstalk_process=ProcessMode.C41)
    unmix_alone = resolve_crosstalk_matrix(0.6, _CROSSTALK)
    np.testing.assert_array_equal(effective_crosstalk_matrix(process, ProcessMode.C41), unmix_alone)


def test_unit_survivals_compose_to_the_crosstalk_matrix_exactly():
    process = ProcessConfig(crosstalk_strength=0.6, crosstalk_matrix=_CROSSTALK, crosstalk_process=ProcessMode.E6)
    np.testing.assert_array_equal(effective_crosstalk_matrix(process, ProcessMode.E6), resolve_crosstalk_matrix(0.6, _CROSSTALK))


def test_a_faded_slide_composes_the_fade_after_the_unmix():
    process = ProcessConfig(
        crosstalk_strength=0.6,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.E6,
        fade_ratio_g=_RATIO_G,
        fade_ratio_b=_RATIO_B,
    )
    expected = resolve_fade_matrix(1.0, _RATIO_G, _RATIO_B) @ resolve_crosstalk_matrix(0.6, _CROSSTALK)
    np.testing.assert_allclose(effective_crosstalk_matrix(process, ProcessMode.E6), expected, atol=1e-12)


def test_a_faded_slide_without_crosstalk_gets_the_fade_alone():
    process = ProcessConfig(fade_ratio_g=_RATIO_G, fade_ratio_b=_RATIO_B)
    np.testing.assert_allclose(
        effective_crosstalk_matrix(process, ProcessMode.E6), resolve_fade_matrix(1.0, _RATIO_G, _RATIO_B), atol=1e-12
    )


def test_fade_applies_to_slides_only():
    """A color negative with fade ratios set keeps only its mode-matched crosstalk unmix."""
    process = ProcessConfig(
        crosstalk_strength=0.6,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.C41,
        fade_ratio_g=_RATIO_G,
        fade_ratio_b=_RATIO_B,
    )
    np.testing.assert_array_equal(effective_crosstalk_matrix(process, ProcessMode.C41), resolve_crosstalk_matrix(0.6, _CROSSTALK))
    assert effective_crosstalk_matrix(ProcessConfig(fade_ratio_g=_RATIO_G), ProcessMode.C41) is None


def test_the_restoration_stays_well_conditioned_across_the_slider_range():
    for ratio_g in (0.2, 0.5, 1.0, 2.0, 5.0):
        for ratio_b in (0.2, 0.5, 1.0, 2.0, 5.0):
            assert np.linalg.cond(resolve_fade_matrix(1.0, ratio_g, ratio_b)) <= 25.0 + 1e-9


def test_effective_crosstalk_matrix_is_a_pure_function_of_config():
    """CPU and GPU both call it with (process, mode) and no cached state, so the engines cannot disagree."""
    process = ProcessConfig(
        crosstalk_strength=0.6,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.E6,
        fade_ratio_g=_RATIO_G,
        fade_ratio_b=_RATIO_B,
    )
    np.testing.assert_array_equal(effective_crosstalk_matrix(process, ProcessMode.E6), effective_crosstalk_matrix(process, ProcessMode.E6))
