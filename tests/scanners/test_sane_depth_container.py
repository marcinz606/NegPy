"""A Coolscan takes its native depth on the `depth` option (an LS-50 offers 8
and 14) but reports a 16-bit container back and rescales the samples to fill it.
Driving the dtype or the readback check off the requested value instead of the
container mis-types every scan deeper than 8 bits that is not exactly 16."""

import numpy as np
import pytest

from negpy.infrastructure.scanners.sane_backend import (
    _sane_container_depth,
    _validate_inline_rgbi_parameters,
)


@pytest.mark.parametrize(
    "requested, container",
    [(8, 8), (10, 16), (12, 16), (14, 16), (16, 16)],
)
def test_container_depth(requested: int, container: int) -> None:
    assert _sane_container_depth(requested) == container


@pytest.mark.parametrize("requested", [10, 12, 14, 16])
def test_deeper_than_8_bits_is_uint16(requested: int) -> None:
    dtype = np.uint16 if _sane_container_depth(requested) == 16 else np.uint8
    assert dtype is np.uint16


def test_8_bit_is_uint8() -> None:
    dtype = np.uint16 if _sane_container_depth(8) == 16 else np.uint8
    assert dtype is np.uint8


def _validate(requested_depth: int, returned_depth: int) -> None:
    px = 100
    _validate_inline_rgbi_parameters(
        context="test",
        frame_format="color",
        last_frame=1,
        returned_depth=returned_depth,
        requested_depth=requested_depth,
        pixels_per_line=px,
        lines=10,
        bytes_per_line=px * 4 * (returned_depth // 8),
    )


def test_14_bit_request_accepts_a_16_bit_container() -> None:
    """The LS-50 case: requesting 14 and being handed 16 is correct, not a fault."""
    _validate(requested_depth=14, returned_depth=16)


def test_16_bit_request_accepts_a_16_bit_container() -> None:
    _validate(requested_depth=16, returned_depth=16)


def test_8_bit_request_accepts_an_8_bit_container() -> None:
    _validate(requested_depth=8, returned_depth=8)


def test_8_bit_request_rejects_a_16_bit_container() -> None:
    with pytest.raises(RuntimeError, match="8-bit container"):
        _validate(requested_depth=8, returned_depth=16)


def test_deep_request_rejects_an_8_bit_container() -> None:
    """A silent drop to 8 bits would throw away half the scan's information."""
    with pytest.raises(RuntimeError, match="16-bit container"):
        _validate(requested_depth=14, returned_depth=8)
