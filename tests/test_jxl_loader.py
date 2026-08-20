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


class TestJxlSourceMetadata:
    """A JPEG XL scan is a supported source, so its EXIF has to be readable. PIL
    cannot open the format and piexif cannot parse it, leaving the container boxes
    as the only way in."""

    def test_exif_is_read_out_of_the_container(self, tmp_path):
        import piexif

        from negpy.features.metadata.models import MetadataConfig
        from negpy.features.metadata.writer import embed_metadata
        from negpy.infrastructure.loaders.helpers import read_exif_from_file

        source_exif = {
            "0th": {piexif.ImageIFD.Make: b"Plustek", piexif.ImageIFD.Orientation: 6},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }
        scan = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        path = str(tmp_path / "scan.jxl")
        with open(path, "wb") as f:
            f.write(embed_metadata(bytes(imagecodecs.jpegxl_encode(scan, lossless=True)), MetadataConfig(), source_exif))

        exif = read_exif_from_file(path)

        assert exif is not None
        assert exif["0th"][piexif.ImageIFD.Make] == b"Plustek"

    def test_brotli_compressed_exif_box_is_decompressed(self, tmp_path):
        """cjxl writes metadata into a 'brob' box, which names its inner type in the
        first four payload bytes."""
        import struct

        import piexif

        from negpy.infrastructure.loaders.jxl_boxes import read_jxl_exif

        exif = piexif.dump({"0th": {piexif.ImageIFD.Make: b"Nikon"}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}})
        payload = b"Exif" + bytes(imagecodecs.brotli_encode(b"\x00\x00\x00\x00" + exif[6:]))
        codestream = bytes(imagecodecs.jpegxl_encode(np.zeros((8, 8, 3), dtype=np.uint8), lossless=True, usecontainer=False))
        container = (
            b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a"
            + struct.pack(">I", 20)
            + b"ftypjxl \x00\x00\x00\x00jxl "
            + struct.pack(">I", len(payload) + 8)
            + b"brob"
            + payload
            + struct.pack(">I", len(codestream) + 8)
            + b"jxlc"
            + codestream
        )

        assert piexif.load(read_jxl_exif(container))["0th"][piexif.ImageIFD.Make] == b"Nikon"
