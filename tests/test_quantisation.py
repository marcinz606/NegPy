"""Quantisation rounds to nearest rather than truncating.

A bare cast to uint truncates toward zero, so every sample lands half a level low: the
error runs [-1, 0] instead of [-0.5, +0.5]. That is a 0.196% darkening of full scale at
8 bits, on every pixel of every export and of the canvas itself. The two greyscale kernels
in the same module have always rounded — these two had not, so a B&W export was correct
while a colour one was not.
"""

import numpy as np
import pytest

from negpy.kernel.image.logic import float_to_uint8, float_to_uint16, float_to_uint_luma

RNG = np.random.default_rng(0)
SAMPLE = RNG.random((64, 64, 3)).astype(np.float32)


@pytest.mark.parametrize("fn,scale", [(float_to_uint8, 255.0), (float_to_uint16, 65535.0)])
def test_matches_round_to_nearest_exactly(fn, scale):
    got = fn(SAMPLE).astype(np.float64)
    assert np.array_equal(got, np.rint(SAMPLE.astype(np.float64) * scale))


@pytest.mark.parametrize("fn,scale", [(float_to_uint8, 255.0), (float_to_uint16, 65535.0)])
def test_no_systematic_bias(fn, scale):
    """The regression this guards: a -0.5 level DC offset across the whole image."""
    err = fn(SAMPLE).astype(np.float64) - SAMPLE.astype(np.float64) * scale
    assert abs(err.mean()) < 0.01, f"mean quantisation error {err.mean():+.4f} levels — truncating?"
    assert np.abs(err).max() <= 0.5 + 1e-6, "error must stay within half a level"


@pytest.mark.parametrize("fn,scale", [(float_to_uint8, 255.0), (float_to_uint16, 65535.0)])
def test_endpoints_and_midpoint(fn, scale):
    out = fn(np.array([[[0.0, 1.0, 0.5]]], dtype=np.float32)).ravel()
    assert out[0] == 0
    assert out[1] == scale, "1.0 must reach full scale, not fall a level short"
    assert out[2] == round(scale * 0.5)


@pytest.mark.parametrize("fn", [float_to_uint8, float_to_uint16])
def test_nan_and_out_of_range_are_still_clamped(fn):
    """Rounding must not reintroduce wraparound at the ends."""
    out = fn(np.array([[[np.nan, -1.0, 2.0]]], dtype=np.float32)).ravel()
    assert out[0] == 0 and out[1] == 0
    assert out[2] == np.iinfo(fn(np.zeros((1, 1, 3), np.float32)).dtype).max


def test_every_quantiser_in_the_module_rounds():
    """All four, held to the same rule. 0.25 is exact in float32 and 0.25*255 = 63.75, so
    rounding gives 64 and truncating gives 63 — the two are distinguishable.

    The greyscale kernels are fed a 2-D luminance plane directly: computing luma from three
    channels first accumulates its own float32 error, which is a separate question from how
    the result is quantised (a 0.5 grey lands a hair under 0.5 that way, and lands on 127).
    """
    rgb = np.full((4, 4, 3), 0.25, dtype=np.float32)
    plane = np.full((4, 4), 0.25, dtype=np.float32)
    assert int(float_to_uint8(rgb)[0, 0, 0]) == 64
    assert int(float_to_uint16(rgb)[0, 0, 0]) == round(0.25 * 65535)
    assert int(float_to_uint_luma(plane, 8)[0, 0]) == 64
    assert int(float_to_uint_luma(plane, 16)[0, 0]) == round(0.25 * 65535)


def test_the_print_layout_path_uses_the_shared_quantiser():
    """It had its own bare cast, which is how one path drifted from the other."""
    import inspect

    from negpy.services.export import print as print_service

    src = inspect.getsource(print_service)
    assert "astype(np.uint8)" not in src, "a second quantiser here will drift from the shared one"
