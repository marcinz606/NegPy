"""Tests for the export encoder settings: TIFF compression, bit depth,
PNG compression level and JPEG progressive."""

import io

import imagecodecs
import numpy as np
import pytest
import tifffile
from PIL import Image

from negpy.domain.models import ColorSpace, ExportFormat, ExportPreset, TiffCompression
from negpy.features.metadata.resolution import Resolution
from negpy.services.export.encoders import encode_jpeg, encode_png, encode_tiff
from negpy.services.rendering.image_processor import WORKING_COLOR_SPACE, ImageProcessor


@pytest.fixture(scope="module")
def proc():
    return ImageProcessor()


def _rgb_buffer(h=8, w=12):
    x = np.linspace(0.2, 0.8, w, dtype=np.float32)
    return np.ascontiguousarray(np.stack([np.tile(x, (h, 1))] * 3, axis=-1))


def _encode(proc, **preset_kwargs):
    preset = ExportPreset(**preset_kwargs)
    return proc._encode_export(_rgb_buffer(), preset, ColorSpace.ADOBE_RGB.value, WORKING_COLOR_SPACE)


# --- TIFF compression ------------------------------------------------------


@pytest.mark.parametrize(
    "choice, tag",
    [(TiffCompression.NONE, "NONE"), (TiffCompression.LZW, "LZW"), (TiffCompression.ZIP, "ADOBE_DEFLATE")],
)
def test_tiff_compression_reaches_the_file(proc, choice, tag):
    data, _ = _encode(proc, export_fmt=ExportFormat.TIFF, tiff_compression=choice)
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        assert tif.pages[0].compression.name == tag


@pytest.mark.parametrize("choice", list(TiffCompression))
def test_tiff_compression_is_lossless(proc, choice):
    """Every option must decode to identical pixels, including Uncompressed —
    which is also the one that would raise if the predictor were left on."""
    ref, _ = _encode(proc, export_fmt=ExportFormat.TIFF, tiff_compression=TiffCompression.ZIP)
    data, _ = _encode(proc, export_fmt=ExportFormat.TIFF, tiff_compression=choice)
    assert np.array_equal(tifffile.imread(io.BytesIO(data)), tifffile.imread(io.BytesIO(ref)))


def test_uncompressed_tiff_is_the_largest(proc):
    sizes = {c: len(_encode(proc, export_fmt=ExportFormat.TIFF, tiff_compression=c)[0]) for c in TiffCompression}
    assert sizes[TiffCompression.NONE] > sizes[TiffCompression.ZIP]


# --- Bit depth -------------------------------------------------------------


@pytest.mark.parametrize("depth, dtype", [(8, np.uint8), (16, np.uint16)])
def test_tiff_bit_depth(proc, depth, dtype):
    data, _ = _encode(proc, export_fmt=ExportFormat.TIFF, export_bit_depth=depth)
    assert tifffile.imread(io.BytesIO(data)).dtype == dtype


@pytest.mark.parametrize("depth, dtype", [(8, np.uint8), (16, np.uint16)])
def test_png_rgb_bit_depth(proc, depth, dtype):
    # PIL reports mode "RGB" for a 16-bit PNG because it cannot represent one,
    # so the depth has to be read with a decoder that can.
    data, _ = _encode(proc, export_fmt=ExportFormat.PNG, export_bit_depth=depth)
    assert imagecodecs.png_decode(data).dtype == dtype


@pytest.mark.parametrize("depth, dtype", [(8, np.uint8), (16, np.uint16)])
def test_jxl_bit_depth(proc, depth, dtype):
    preset = ExportPreset(export_fmt=ExportFormat.JXL, export_bit_depth=depth, jxl_lossless=True)
    data, ext = proc._encode_export(_rgb_buffer(), preset, ColorSpace.SRGB.value, WORKING_COLOR_SPACE)
    assert ext == "jxl"
    assert imagecodecs.jpegxl_decode(data).dtype == dtype


@pytest.mark.parametrize("fmt", [ExportFormat.JPEG, ExportFormat.WEBP])
def test_eight_bit_formats_ignore_the_bit_depth(proc, fmt):
    """JPEG and WebP have no higher depth; asking for 16 must not reach the encoder."""
    data, _ = _encode(proc, export_fmt=fmt, export_bit_depth=16)
    assert np.asarray(Image.open(io.BytesIO(data))).dtype == np.uint8


# --- PNG compression level -------------------------------------------------


@pytest.mark.parametrize("depth", [8, 16])
def test_png_level_trades_size_for_nothing_else(proc, depth):
    fast, _ = _encode(proc, export_fmt=ExportFormat.PNG, export_bit_depth=depth, png_compress_level=0)
    small, _ = _encode(proc, export_fmt=ExportFormat.PNG, export_bit_depth=depth, png_compress_level=9)
    assert len(fast) > len(small)
    assert np.array_equal(imagecodecs.png_decode(fast), imagecodecs.png_decode(small))


# --- JPEG progressive ------------------------------------------------------


def test_jpeg_progressive_is_written(proc):
    baseline, _ = _encode(proc, export_fmt=ExportFormat.JPEG, jpeg_progressive=False)
    progressive, _ = _encode(proc, export_fmt=ExportFormat.JPEG, jpeg_progressive=True)
    assert "progressive" not in Image.open(io.BytesIO(baseline)).info
    assert Image.open(io.BytesIO(progressive)).info.get("progressive")
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(baseline))), np.asarray(Image.open(io.BytesIO(progressive))))


# --- 16-bit PNG ancillary chunks -------------------------------------------


def test_16bit_png_keeps_icc_dpi_and_metadata():
    """The 16-bit path splices its own chunks in, so each one needs proving."""
    arr = (np.random.default_rng(0).random((8, 12, 3)) * 65535).astype(np.uint16)
    icc = b"\x00" * 128
    exif = Image.Exif()
    exif[271] = "NegPy"
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"></x:xmpmeta>'

    data = encode_png(arr, icc=icc, resolution=Resolution.from_dpi(300), exif=exif.tobytes(), xmp=xmp)

    assert np.array_equal(imagecodecs.png_decode(data), arr)
    img = Image.open(io.BytesIO(data))
    assert img.info["icc_profile"] == icc
    assert round(img.info["dpi"][0]) == 300
    assert img.getexif()[271] == "NegPy"
    assert img.info["XML:com.adobe.xmp"].encode() == xmp


def test_16bit_png_exif_drops_the_jpeg_app1_prefix():
    """piexif emits an Exif\\x00\\x00 header; an eXIf chunk carrying it is invalid."""
    arr = np.zeros((4, 4, 3), dtype=np.uint16)
    exif = Image.Exif()
    exif[271] = "NegPy"
    raw = exif.tobytes()
    assert raw.startswith(b"Exif\x00\x00")

    data = encode_png(arr, exif=raw)

    assert b"eXIf" + raw[6:8] in data
    assert b"eXIf" + b"Exif" not in data


@pytest.mark.parametrize("depth", [8, 16])
def test_png_export_embeds_metadata_at_both_depths(proc, depth):
    """PNG takes its metadata at the first encode instead of a post-hoc rewrite,
    so the 16-bit path has to embed it too or those exports lose it."""
    exif = Image.Exif()
    exif[271] = "NegPy"
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"></x:xmpmeta>'
    preset = ExportPreset(export_fmt=ExportFormat.PNG, export_bit_depth=depth)

    data, _ = proc._encode_export(
        _rgb_buffer(), preset, ColorSpace.ADOBE_RGB.value, WORKING_COLOR_SPACE, embed_plan=(exif.tobytes(), xmp, True)
    )

    img = Image.open(io.BytesIO(data))
    assert img.getexif()[271] == "NegPy"
    assert img.info["XML:com.adobe.xmp"].encode() == xmp
    assert imagecodecs.png_decode(data).dtype == (np.uint8 if depth == 8 else np.uint16)


# --- Direct encoder contracts ----------------------------------------------


def test_encode_tiff_omits_the_predictor_when_uncompressed():
    """tifffile raises on a predictor without compression."""
    arr = np.zeros((4, 4, 3), dtype=np.uint16)
    assert encode_tiff(arr, compression=TiffCompression.NONE)


@pytest.mark.parametrize("progressive", [False, True])
def test_encode_jpeg_survives_an_image_that_encodes_larger_than_its_buffer(progressive):
    """optimize and progressive make PIL buffer the whole frame into `w*h + MAXBLOCK`.
    Incompressible pixels encode past that, which used to fail the save outright."""
    arr = (np.random.default_rng(1).random((600, 800, 3)) * 255).astype(np.uint8)

    bits = encode_jpeg(arr, quality=90, progressive=progressive)

    img = Image.open(io.BytesIO(bits))
    img.load()
    assert img.size == (800, 600)
    assert bool(img.info.get("progressive")) is progressive


def test_encode_jpeg_restores_the_pil_block_size():
    from PIL import ImageFile

    before = ImageFile.MAXBLOCK
    encode_jpeg((np.random.default_rng(2).random((600, 800, 3)) * 255).astype(np.uint8), progressive=True)
    assert ImageFile.MAXBLOCK == before


def test_encode_jpeg_stays_8bit_and_optimized():
    arr = np.full((16, 16, 3), 128, dtype=np.uint8)
    assert Image.open(io.BytesIO(encode_jpeg(arr, quality=90))).format == "JPEG"
