from unittest.mock import patch

import imagecodecs
import numpy as np
import pytest
import tifffile

from negpy.infrastructure.loaders.fff_loader import FffLoader
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper
from negpy.infrastructure.loaders.jxl_loader import JxlLoader
from negpy.infrastructure.loaders.nef_loader import NefLoader
from negpy.infrastructure.loaders.noritsu_loader import NoritsuLoader
from negpy.kernel.image.logic import uint16_to_float32


def _source() -> np.ndarray:
    rng = np.random.default_rng(7)
    src = rng.integers(0, 65536, (6, 10, 3), dtype=np.uint16)
    src[0, 0] = (0, 1, 65535)
    return src


def _first_page(tif):
    return tif.pages[0]


def _load_nef(tmp_path, src):
    path = str(tmp_path / "scan.nef")
    tifffile.imwrite(path, src, photometric="rgb")
    with patch("negpy.infrastructure.loaders.nef_loader._find_rgb_subifd", _first_page):
        return NefLoader().load(path)[0]


def _load_fff(tmp_path, src):
    path = str(tmp_path / "scan.fff")
    tifffile.imwrite(path, src, photometric="rgb")
    with patch("negpy.infrastructure.loaders.fff_loader._find_full_res_ifd", _first_page):
        return FffLoader().load(path)[0]


def _load_noritsu(tmp_path, src):
    path = str(tmp_path / "scan.raw")
    np.ascontiguousarray(src[..., ::-1]).astype("<u2").tofile(path)
    with patch("negpy.infrastructure.loaders.noritsu_loader.detect_noritsu_dims", return_value=(src.shape[1], src.shape[0])):
        return NoritsuLoader().load(path)[0]


def _load_jxl(tmp_path, src):
    path = tmp_path / "scan.jxl"
    path.write_bytes(imagecodecs.jpegxl_encode(src, lossless=True))
    return JxlLoader().load(str(path))[0]


@pytest.mark.parametrize("load", [_load_nef, _load_fff, _load_noritsu, _load_jxl])
def test_a_16_bit_source_matches_the_float32_wrapper_bit_for_bit(tmp_path, load):
    src = _source()
    wrapper = load(tmp_path, src)
    assert wrapper._source_u16 is not None
    old = NonStandardFileWrapper(uint16_to_float32(src))

    for half_size in (False, True):
        kwargs = {"output_bps": 16, "gamma": (1, 1), "half_size": half_size}
        new_out = wrapper.postprocess(**kwargs)
        assert new_out.dtype == np.uint16
        np.testing.assert_array_equal(new_out, old.postprocess(**kwargs))
        np.testing.assert_array_equal(new_out, src[::2, ::2] if half_size else src)
    assert wrapper.data.dtype == np.float32
    np.testing.assert_array_equal(wrapper.data, old.data)
    np.testing.assert_array_equal(wrapper.postprocess(output_bps=8), old.postprocess(output_bps=8))
