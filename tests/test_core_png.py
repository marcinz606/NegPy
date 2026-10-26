import io

import imagecodecs
import negpy_core
import numpy as np
import pytest
from PIL import Image


@pytest.mark.parametrize("shape", [(1, 1), (1, 1, 3), (7, 5, 3), (130, 97, 3), (200, 31), (65, 3, 4), (129, 17, 2)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("level", [0, 1, 6, 9])
def test_png_round_trips_through_libpng(shape, dtype, level):
    rng = np.random.default_rng(sum(shape) + level)
    arr = rng.integers(0, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
    arr[: shape[0] // 2] //= 7  # smooth rows the filters and the deflate can win on
    data = bytes(negpy_core.encode_png(arr, level))
    assert np.array_equal(imagecodecs.png_decode(data), arr)


def test_png_reads_back_in_pil():
    arr = np.random.default_rng(1).integers(0, 256, size=(300, 211, 3), dtype=np.uint8)
    img = Image.open(io.BytesIO(bytes(negpy_core.encode_png(arr, 6))))
    img.load()
    assert np.array_equal(np.asarray(img), arr)


def test_png_level_zero_is_larger():
    arr = np.tile(np.arange(256, dtype=np.uint8), (300, 2))
    assert len(negpy_core.encode_png(arr, 0)) > len(negpy_core.encode_png(arr, 9))


def test_png_rejects_other_shapes():
    with pytest.raises(ValueError):
        negpy_core.encode_png(np.zeros((2, 2, 5), np.uint8), 6)
