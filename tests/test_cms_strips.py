import numpy as np
import pytest

from negpy.services.rendering import image_processor as ip


def _icc(name):
    from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry

    path = ColorSpaceRegistry.get_icc_path(name)
    if not path:
        pytest.skip(f"no ICC for {name}")
    with open(path, "rb") as f:
        return f.read()


@pytest.mark.parametrize("dst", ["sRGB", "ProPhoto RGB"])
def test_strips_equal_single_transform(dst):
    imagecodecs = pytest.importorskip("imagecodecs")
    src_bytes, dst_bytes = _icc("Adobe RGB"), _icc(dst)
    img = (np.random.default_rng(3).random((97, 41, 3)) * 65535).astype(np.uint16)
    try:
        ref = imagecodecs.cms_transform(img, src_bytes, dst_bytes, colorspace="RGB", outcolorspace="RGB", intent=1, flags=0x2000)
    except ImportError:
        pytest.skip("cms codec unavailable")
    assert np.array_equal(ip._cms_transform_strips(img, src_bytes, dst_bytes), ref)
