import imagecodecs
import numpy as np

from negpy.infrastructure.loaders.jxl_loader import JxlLoader
from negpy.infrastructure.loaders.factory import LoaderFactory
from negpy.infrastructure.loaders.constants import SUPPORTED_JXL_EXTENSIONS, SUPPORTED_RAW_EXTENSIONS


def _write_jxl(path: str, img: np.ndarray) -> None:
    data = imagecodecs.jpegxl_encode(img, lossless=True)
    with open(path, "wb") as f:
        f.write(data)


class TestJxlLoader:
    def test_uint8_round_trip(self, tmp_path):
        src = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        p = str(tmp_path / "test.jxl")
        _write_jxl(p, src)

        loader = JxlLoader()
        wrapper, meta = loader.load(p)
        img = wrapper.data

        assert img.dtype == np.float32
        assert img.shape == (8, 8, 3)
        np.testing.assert_allclose(img, src.astype(np.float32) / 255.0, atol=1e-3)

    def test_uint16_round_trip(self, tmp_path):
        src = np.random.randint(0, 65535, (8, 8, 3), dtype=np.uint16)
        p = str(tmp_path / "test.jxl")
        _write_jxl(p, src)

        loader = JxlLoader()
        wrapper, _ = loader.load(p)
        img = wrapper.data

        assert img.dtype == np.float32
        np.testing.assert_allclose(img, src.astype(np.float32) / 65535.0, atol=1e-4)

    def test_float32_round_trip(self, tmp_path):
        src = np.random.rand(8, 8, 3).astype(np.float32)
        p = str(tmp_path / "test.jxl")
        _write_jxl(p, src)

        loader = JxlLoader()
        wrapper, _ = loader.load(p)
        img = wrapper.data

        assert img.dtype == np.float32
        np.testing.assert_allclose(img, src, atol=1e-5)

    def test_grayscale_promoted_to_rgb(self, tmp_path):
        src = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
        p = str(tmp_path / "gray.jxl")
        _write_jxl(p, src)

        loader = JxlLoader()
        wrapper, _ = loader.load(p)
        img = wrapper.data

        assert img.shape == (8, 8, 3)

    def test_metadata_defaults(self, tmp_path):
        src = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        p = str(tmp_path / "test.jxl")
        _write_jxl(p, src)

        loader = JxlLoader()
        _, meta = loader.load(p)

        assert meta["orientation"] == 1
        assert meta["color_space"] is None
        assert meta["icc_profile"] is None
        assert meta["ir"] is None

    def test_no_srgb_linearization(self, tmp_path):
        src = np.full((4, 4, 3), 128, dtype=np.uint8)
        p = str(tmp_path / "test.jxl")
        _write_jxl(p, src)

        loader = JxlLoader()
        wrapper, _ = loader.load(p)
        img = wrapper.data

        expected = 128.0 / 255.0
        np.testing.assert_allclose(img[0, 0, 0], expected, atol=1e-3)


class TestJxlExtensions:
    def test_jxl_in_supported_extensions(self):
        assert ".jxl" in SUPPORTED_JXL_EXTENSIONS
        assert ".jxl" in SUPPORTED_RAW_EXTENSIONS


class TestFactoryDispatch:
    def test_factory_routes_jxl(self, tmp_path):
        src = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        p = str(tmp_path / "photo.jxl")
        _write_jxl(p, src)

        factory = LoaderFactory()
        wrapper, meta = factory.get_loader(p)

        assert wrapper.data.dtype == np.float32
        assert wrapper.data.shape == (4, 4, 3)
