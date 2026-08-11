"""Tests for the Linear Output export feature."""

import io
import os
from typing import Optional
from unittest import mock

import imagecodecs
import numpy as np
import pytest
import tifffile

from negpy.features.geometry.models import GeometryConfig
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.features.stitch.models import StitchConfig
from negpy.kernel.image.logic import apply_exif_orientation
from negpy.infrastructure.loaders.fff_loader import is_flextight_fff
from negpy.infrastructure.loaders.nef_loader import is_coolscan_nef
from negpy.infrastructure.loaders.noritsu_loader import is_noritsu_raw, KNOWN_NORITSU_DIMS, KNOWN_NORITSU_HEIGHTS, detect_noritsu_dims
from negpy.services.export.linear_output import (
    TIFF_GAMMA_OPTIONS,
    _CameraWB,
    _SourceMeta,
    _apply_white_balance,
    _build_xmp,
    _decode_dng,
    _default_pakon_expansion,
    _effective_expansion,
    _is_camera_raw,
    _is_tiff,
    _linearize,
    _normalize_wb_rgb,
    _source_format_label,
    _write_tiff,
    export_linear_output,
    export_linear_output_bytes,
    is_linear_output_supported,
    linear_output_source_type,
)


_LINEAR_RAW = 34892


def _make_linearraw_dng_4ch(tmp_dir: str, h: int = 100, w: int = 150) -> str:
    """Create a synthetic 4-channel LinearRaw DNG (RGB + IR)."""
    rng = np.random.RandomState(99)
    data = rng.randint(0, 40000, size=(h, w, 4), dtype=np.uint16)
    path = os.path.join(tmp_dir, "scan_4ch.dng")
    with tifffile.TiffWriter(path) as tw:
        tw.write(data, photometric=_LINEAR_RAW, planarconfig="contig")
    return path


def _make_linearraw_dng_3ch(tmp_dir: str, h: int = 100, w: int = 150) -> str:
    """Create a synthetic 3-channel LinearRaw DNG (RGB, no IR)."""
    rng = np.random.RandomState(77)
    data = rng.randint(0, 40000, size=(h, w, 3), dtype=np.uint16)
    path = os.path.join(tmp_dir, "scan_3ch.dng")
    with tifffile.TiffWriter(path) as tw:
        tw.write(data, photometric=_LINEAR_RAW, planarconfig="contig")
    return path


# DNG calibration tag codes (TIFF/EP + DNG spec).
_TAG_LINEARIZATION_TABLE = 50712
_TAG_BLACK_LEVEL = 50714
_TAG_WHITE_LEVEL = 50717
_TAG_DEFAULT_CROP_ORIGIN = 50719
_TAG_DEFAULT_CROP_SIZE = 50720


def _make_linearraw_dng_3ch_tagged(
    tmp_dir: str,
    h: int = 100,
    w: int = 150,
    max_val: int = 40000,
    black: Optional[int] = None,
    white: Optional[int] = None,
    lin_table: Optional[np.ndarray] = None,
    crop_origin: Optional[tuple] = None,
    crop_size: Optional[tuple] = None,
    with_ir_page: bool = False,
    seed: int = 77,
) -> tuple:
    """Create a synthetic 3-channel LinearRaw DNG with optional DNG calibration tags.

    Returns (path, raw_source_array). `with_ir_page` writes a second, uncropped
    grayscale page at the source dims, mimicking a SilverFast HDRi IR plane.
    """
    rng = np.random.RandomState(seed)
    data = rng.randint(0, max_val, size=(h, w, 3), dtype=np.uint16)
    path = os.path.join(tmp_dir, "scan_3ch_tagged.dng")
    extratags = []
    if black is not None:
        extratags.append((_TAG_BLACK_LEVEL, 4, 3, (black, black, black), False))
    if white is not None:
        extratags.append((_TAG_WHITE_LEVEL, 4, 3, (white, white, white), False))
    if lin_table is not None:
        extratags.append((_TAG_LINEARIZATION_TABLE, 3, len(lin_table), tuple(int(v) for v in lin_table), False))
    if crop_origin is not None:
        extratags.append((_TAG_DEFAULT_CROP_ORIGIN, 4, 2, crop_origin, False))
    if crop_size is not None:
        extratags.append((_TAG_DEFAULT_CROP_SIZE, 4, 2, crop_size, False))
    with tifffile.TiffWriter(path) as tw:
        tw.write(data, photometric=_LINEAR_RAW, planarconfig="contig", extratags=extratags or None)
        if with_ir_page:
            ir = np.random.RandomState(seed + 1).randint(0, max_val, size=(h, w), dtype=np.uint16)
            tw.write(ir, photometric="minisblack")
    return path, data


def _make_pakon_raw(tmp_dir: str, h: int = 1000, w: int = 1500) -> str:
    """Create a minimal synthetic Pakon RAW file (F135 Plus Low Res, 9 MB)."""
    data = np.random.RandomState(42).randint(0, 32768, size=(h, w, 3), dtype=np.uint16)
    path = os.path.join(tmp_dir, "test_scan.raw")
    data.tofile(path)
    assert os.path.getsize(path) == h * w * 3 * 2  # 9000000
    return path


def _make_pakon_f335_raw(tmp_dir: str) -> str:
    """Create a synthetic F335 RAW file (4000×3000, 72 MB)."""
    data = np.random.RandomState(55).randint(0, 65535, size=(4000, 3000, 3), dtype=np.uint16)
    path = os.path.join(tmp_dir, "f335_scan.raw")
    data.tofile(path)
    assert os.path.getsize(path) == 72000000
    return path


class TestIsLinearOutputSupported:
    def test_pakon_raw_supported(self, tmp_path: str) -> None:
        path = _make_pakon_raw(str(tmp_path))
        assert is_linear_output_supported(path)

    def test_regular_tiff_supported(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "photo.tiff")
        arr = np.zeros((10, 10, 3), dtype=np.uint16)
        tifffile.imwrite(path, arr)
        assert is_linear_output_supported(path)

    def test_nonexistent_raw_supported_by_extension(self) -> None:
        """A .raw extension is in SUPPORTED_RAW_EXTENSIONS; support is a format check."""
        assert is_linear_output_supported("/nonexistent/file.raw")

    def test_nonexistent_unknown_ext(self) -> None:
        assert not is_linear_output_supported("/nonexistent/file.xyz")


class TestExportLinearOutput:
    def test_basic_roundtrip(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")

        export_linear_output(raw_path, out_path)

        assert os.path.exists(out_path)
        with tifffile.TiffFile(out_path) as tf:
            page = tf.pages[0]
            arr = page.asarray()
            assert arr.dtype == np.uint16
            assert arr.shape == (1000, 1500, 3)
            assert page.photometric.name == "RGB"
            assert page.iccprofile is None

    def test_no_icc_profile(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")

        export_linear_output(raw_path, out_path)

        with tifffile.TiffFile(out_path) as tf:
            assert tf.pages[0].iccprofile is None

    def test_image_description(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")

        export_linear_output(raw_path, out_path)

        with tifffile.TiffFile(out_path) as tf:
            desc = tf.pages[0].description
            assert "NegPy Linear Output" in desc
            assert "no color management" in desc
            assert "no WB applied" in desc
            assert "Pakon" in desc
            assert "F135" in desc
            assert "x4" in desc

    def test_pixel_values_roundtrip(self, tmp_path: str) -> None:
        """The output uint16 values should be the expanded float32 loader output * 65535, rounded."""
        raw_path = _make_pakon_raw(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")

        from negpy.infrastructure.loaders.pakon_loader import PakonLoader
        from negpy.services.export.linear_output import PAKON_EXPANSION

        loader = PakonLoader()
        ctx_mgr, _meta = loader.load(raw_path)
        with ctx_mgr as wrapper:
            expected_f32 = wrapper.data.copy()

        export_linear_output(raw_path, out_path)

        with tifffile.TiffFile(out_path) as tf:
            actual_u16 = tf.pages[0].asarray()

        expected_u16 = np.clip(expected_f32 * PAKON_EXPANSION * 65535.0, 0, 65535).astype(np.uint16)
        np.testing.assert_allclose(actual_u16.astype(np.int32), expected_u16.astype(np.int32), atol=1)

    def test_rejects_unsupported_file(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "photo.jpeg")
        with open(path, "wb") as fh:
            fh.write(b"\xff\xd8\xff\xe0")
        out = os.path.join(str(tmp_path), "out.tiff")
        with pytest.raises(ValueError, match="not supported"):
            export_linear_output(path, out)

    def test_creates_output_directory(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "subdir", "nested", "output.tiff")

        export_linear_output(raw_path, out_path)
        assert os.path.exists(out_path)


class TestExportLinearOutputJxl:
    def test_jxl_roundtrip(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.jxl")

        export_linear_output(raw_path, out_path, output_format="jxl")

        assert os.path.exists(out_path)
        import imagecodecs

        with open(out_path, "rb") as f:
            data = f.read()
        decoded = imagecodecs.jpegxl_decode(data)
        assert decoded.dtype == np.uint16
        assert decoded.shape == (1000, 1500, 3)

    def test_jxl_pixel_values_match_tiff(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        tiff_path = os.path.join(str(tmp_path), "output.tiff")
        jxl_path = os.path.join(str(tmp_path), "output.jxl")

        export_linear_output(raw_path, tiff_path, output_format="tiff")
        export_linear_output(raw_path, jxl_path, output_format="jxl")

        with tifffile.TiffFile(tiff_path) as tf:
            tiff_u16 = tf.pages[0].asarray()

        import imagecodecs

        with open(jxl_path, "rb") as f:
            jxl_u16 = imagecodecs.jpegxl_decode(f.read())

        np.testing.assert_array_equal(tiff_u16, jxl_u16)

    def test_default_format_is_tiff(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(raw_path, out_path)
        with tifffile.TiffFile(out_path) as tf:
            assert tf.pages[0].asarray().dtype == np.uint16

    def test_ir_sidecar_follows_main_format(self, tmp_path: str) -> None:
        dng_path = _make_linearraw_dng_4ch(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.jxl")
        export_linear_output(dng_path, out_path, output_format="jxl")
        ir_path = os.path.join(str(tmp_path), "output_ir.jxl")
        assert os.path.exists(ir_path)
        assert not os.path.exists(os.path.join(str(tmp_path), "output_ir.tiff"))
        decoded = imagecodecs.jpegxl_decode(open(ir_path, "rb").read())
        assert decoded.dtype == np.uint16

    def test_ir_sidecar_tiff_when_main_is_tiff(self, tmp_path: str) -> None:
        dng_path = _make_linearraw_dng_4ch(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(dng_path, out_path, output_format="tiff")
        ir_path = os.path.join(str(tmp_path), "output_ir.tiff")
        assert os.path.exists(ir_path)
        with tifffile.TiffFile(ir_path) as tf:
            assert tf.pages[0].asarray().dtype == np.uint16


class TestExportLinearOutputBytes:
    def test_returns_valid_tiff(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))

        tiff_bytes, stem = export_linear_output_bytes(raw_path)

        assert stem == "test_scan"
        assert len(tiff_bytes) > 0
        with tifffile.TiffFile(io.BytesIO(tiff_bytes)) as tf:
            arr = tf.pages[0].asarray()
            assert arr.dtype == np.uint16
            assert arr.shape == (1000, 1500, 3)


class TestOrientationHandling:
    """Verify that apply_exif_orientation is applied correctly in the export path.

    Pakon always reports orientation=0 (no-op), but the code should handle
    nonzero values correctly if a future source provides them.
    """

    def test_orientation_zero_is_identity(self) -> None:
        arr = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        result = apply_exif_orientation(arr, 0)
        np.testing.assert_array_equal(result, arr)

    def test_orientation_one_is_identity(self) -> None:
        arr = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        result = apply_exif_orientation(arr, 1)
        np.testing.assert_array_equal(result, arr)

    def test_orientation_6_rotates_cw(self) -> None:
        arr = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        result = apply_exif_orientation(arr, 6)
        assert result.shape == (4, 2, 3)
        expected = np.rot90(arr, 3)
        np.testing.assert_array_equal(result, expected)

    def test_orientation_8_rotates_ccw(self) -> None:
        arr = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        result = apply_exif_orientation(arr, 8)
        assert result.shape == (4, 2, 3)
        expected = np.rot90(arr, 1)
        np.testing.assert_array_equal(result, expected)

    def test_orientation_3_rotates_180(self) -> None:
        arr = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        result = apply_exif_orientation(arr, 3)
        assert result.shape == (2, 4, 3)
        expected = np.rot90(arr, 2)
        np.testing.assert_array_equal(result, expected)


class TestGeometryHandling:
    """Verify that user rotation/flip from GeometryConfig is applied."""

    def test_rotation_90cw(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")
        geo = GeometryConfig(rotation=1)

        export_linear_output(raw_path, out_path, geometry=geo)

        with tifffile.TiffFile(out_path) as tf:
            arr = tf.pages[0].asarray()
            assert arr.shape == (1500, 1000, 3)

    def test_rotation_180(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")
        geo = GeometryConfig(rotation=2)

        export_linear_output(raw_path, out_path, geometry=geo)

        with tifffile.TiffFile(out_path) as tf:
            arr = tf.pages[0].asarray()
            assert arr.shape == (1000, 1500, 3)

    def test_flip_horizontal(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        out_no_flip = os.path.join(str(tmp_path), "no_flip.tiff")
        out_flip = os.path.join(str(tmp_path), "flip.tiff")

        export_linear_output(raw_path, out_no_flip)
        export_linear_output(raw_path, out_flip, geometry=GeometryConfig(flip_horizontal=True))

        with tifffile.TiffFile(out_no_flip) as tf:
            arr_orig = tf.pages[0].asarray()
        with tifffile.TiffFile(out_flip) as tf:
            arr_flip = tf.pages[0].asarray()

        np.testing.assert_array_equal(arr_flip, arr_orig[:, ::-1, :])

    def test_flip_vertical(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        out_no_flip = os.path.join(str(tmp_path), "no_flip.tiff")
        out_flip = os.path.join(str(tmp_path), "flip.tiff")

        export_linear_output(raw_path, out_no_flip)
        export_linear_output(raw_path, out_flip, geometry=GeometryConfig(flip_vertical=True))

        with tifffile.TiffFile(out_no_flip) as tf:
            arr_orig = tf.pages[0].asarray()
        with tifffile.TiffFile(out_flip) as tf:
            arr_flip = tf.pages[0].asarray()

        np.testing.assert_array_equal(arr_flip, arr_orig[::-1, :, :])

    def test_fine_rotation_ignored(self, tmp_path: str) -> None:
        """Fine rotation involves resampling and should be skipped."""
        raw_path = _make_pakon_raw(str(tmp_path))
        out_plain = os.path.join(str(tmp_path), "plain.tiff")
        out_fine = os.path.join(str(tmp_path), "fine.tiff")

        export_linear_output(raw_path, out_plain)
        export_linear_output(raw_path, out_fine, geometry=GeometryConfig(fine_rotation=5.0))

        with tifffile.TiffFile(out_plain) as tf:
            arr_plain = tf.pages[0].asarray()
        with tifffile.TiffFile(out_fine) as tf:
            arr_fine = tf.pages[0].asarray()

        np.testing.assert_array_equal(arr_plain, arr_fine)

    def test_no_geometry_is_identity(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        out_none = os.path.join(str(tmp_path), "none.tiff")
        out_default = os.path.join(str(tmp_path), "default.tiff")

        export_linear_output(raw_path, out_none)
        export_linear_output(raw_path, out_default, geometry=GeometryConfig())

        with tifffile.TiffFile(out_none) as tf:
            arr_none = tf.pages[0].asarray()
        with tifffile.TiffFile(out_default) as tf:
            arr_default = tf.pages[0].asarray()

        np.testing.assert_array_equal(arr_none, arr_default)


class TestDngSupport:
    def test_4ch_dng_supported(self, tmp_path: str) -> None:
        path = _make_linearraw_dng_4ch(str(tmp_path))
        assert is_linear_output_supported(path)

    def test_3ch_dng_supported(self, tmp_path: str) -> None:
        path = _make_linearraw_dng_3ch(str(tmp_path))
        assert is_linear_output_supported(path)

    def test_non_linearraw_dng_supported_as_camera(self, tmp_path: str) -> None:
        """A camera DNG (no LinearRaw IFD) is supported via the rawpy path."""
        path = os.path.join(str(tmp_path), "camera.dng")
        tifffile.imwrite(path, np.zeros((10, 10, 3), dtype=np.uint16), photometric="rgb")
        assert is_linear_output_supported(path)

    def test_4ch_dng_roundtrip(self, tmp_path: str) -> None:
        dng_path = _make_linearraw_dng_4ch(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")

        export_linear_output(dng_path, out_path)

        assert os.path.exists(out_path)
        with tifffile.TiffFile(out_path) as tf:
            page = tf.pages[0]
            arr = page.asarray()
            assert arr.dtype == np.uint16
            assert arr.shape == (100, 150, 3)
            assert page.photometric.name == "RGB"
            assert page.iccprofile is None

    def test_4ch_dng_ir_written(self, tmp_path: str) -> None:
        dng_path = _make_linearraw_dng_4ch(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")

        export_linear_output(dng_path, out_path)

        ir_path = os.path.join(str(tmp_path), "output_ir.tiff")
        assert os.path.exists(ir_path)
        with tifffile.TiffFile(ir_path) as tf:
            ir_arr = tf.pages[0].asarray()
            assert ir_arr.dtype == np.uint16
            assert ir_arr.shape == (100, 150)
            assert "infrared" in tf.pages[0].description

    def test_4ch_dng_pixel_values(self, tmp_path: str) -> None:
        dng_path = _make_linearraw_dng_4ch(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")

        source = tifffile.imread(dng_path)
        expected_rgb = np.clip(source[:, :, :3].astype(np.float32) / 65535.0 * 65535.0, 0, 65535).astype(np.uint16)

        export_linear_output(dng_path, out_path)

        with tifffile.TiffFile(out_path) as tf:
            actual = tf.pages[0].asarray()
        np.testing.assert_allclose(actual.astype(np.int32), expected_rgb.astype(np.int32), atol=1)

    def test_3ch_dng_roundtrip(self, tmp_path: str) -> None:
        dng_path = _make_linearraw_dng_3ch(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")

        export_linear_output(dng_path, out_path)

        assert os.path.exists(out_path)
        with tifffile.TiffFile(out_path) as tf:
            arr = tf.pages[0].asarray()
            assert arr.dtype == np.uint16
            assert arr.shape == (100, 150, 3)
            assert tf.pages[0].iccprofile is None

    def test_3ch_dng_no_ir_file(self, tmp_path: str) -> None:
        dng_path = _make_linearraw_dng_3ch(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")

        export_linear_output(dng_path, out_path)

        ir_path = os.path.join(str(tmp_path), "output_ir.tiff")
        assert not os.path.exists(ir_path)

    def test_dng_geometry_applied(self, tmp_path: str) -> None:
        dng_path = _make_linearraw_dng_4ch(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")
        geo = GeometryConfig(rotation=1)

        export_linear_output(dng_path, out_path, geometry=geo)

        with tifffile.TiffFile(out_path) as tf:
            arr = tf.pages[0].asarray()
            assert arr.shape == (150, 100, 3)

        ir_path = os.path.join(str(tmp_path), "output_ir.tiff")
        with tifffile.TiffFile(ir_path) as tf:
            ir_arr = tf.pages[0].asarray()
            assert ir_arr.shape == (150, 100)


class TestDng3ChTagAwareDecode:
    """_decode_dng's 3-channel LinearRaw path (SilverFast HDRi and DxO/Lightroom
    Enhance-style DNGs) reads BlackLevel/WhiteLevel/LinearizationTable/DefaultCrop
    via _peek_linear_dng_rgb instead of a naive divide-by-container-max."""

    def test_black_white_level_applied(self, tmp_path: str) -> None:
        """A file with real BlackLevel/WhiteLevel (as DxO/Lightroom Enhance write)
        gets rescaled to those, not the container max — this is the ~20x bug."""
        black, white = 1000, 12000
        path, source = _make_linearraw_dng_3ch_tagged(str(tmp_path), black=black, white=white)
        out_path = os.path.join(str(tmp_path), "output.tiff")

        export_linear_output(path, out_path)

        expected = np.clip((source.astype(np.float64) - black) / (white - black), 0.0, 1.0)
        expected_u16 = (expected * 65535.0 + 0.5).astype(np.uint16)
        with tifffile.TiffFile(out_path) as tf:
            actual = tf.pages[0].asarray()
        np.testing.assert_allclose(actual.astype(np.int32), expected_u16.astype(np.int32), atol=1)
        # sanity: the old naive /65535 scale (~= source unchanged, since the
        # container max equals the final quantization range) is far darker.
        assert actual.astype(np.int32).mean() > source.astype(np.int32).mean() * 2

    def test_no_tags_matches_untagged_scale(self, tmp_path: str) -> None:
        """A file with no calibration tags at all (as SilverFast HDRi writes) is
        unaffected by the tag-aware decode — same output as the plain /65535 scale."""
        path, source = _make_linearraw_dng_3ch_tagged(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")

        export_linear_output(path, out_path)

        expected_u16 = np.clip(source.astype(np.float64), 0, 65535).astype(np.uint16)
        with tifffile.TiffFile(out_path) as tf:
            actual = tf.pages[0].asarray()
        # float64 (tag-aware path) vs float32 (old inline path) intermediate math can
        # differ by the last bit at quantization boundaries — not a behavior change.
        np.testing.assert_allclose(actual.astype(np.int32), expected_u16.astype(np.int32), atol=1)

    def test_default_crop_applied(self, tmp_path: str) -> None:
        """DefaultCropOrigin/Size trims the exported RGB to the declared active area."""
        path, source = _make_linearraw_dng_3ch_tagged(str(tmp_path), h=100, w=150, crop_origin=(10, 5), crop_size=(120, 80))
        out_path = os.path.join(str(tmp_path), "output.tiff")

        export_linear_output(path, out_path)

        with tifffile.TiffFile(out_path) as tf:
            actual = tf.pages[0].asarray()
        assert actual.shape == (80, 120, 3)
        expected = source[5 : 5 + 80, 10 : 10 + 120, :]
        np.testing.assert_allclose(actual.astype(np.int32), expected.astype(np.int32), atol=1)

    def test_crop_with_mismatched_ir_page_raises(self, tmp_path: str) -> None:
        """A DefaultCrop* that trims RGB but leaves a same-file HDRi IR page at full
        sensor size must fail loudly, not silently export misaligned planes."""
        path, _source = _make_linearraw_dng_3ch_tagged(
            str(tmp_path), h=100, w=150, crop_origin=(10, 5), crop_size=(120, 80), with_ir_page=True
        )
        out_path = os.path.join(str(tmp_path), "output.tiff")

        with pytest.raises(ValueError, match="size"):
            export_linear_output(path, out_path)

    def test_linearization_table_applied(self, tmp_path: str) -> None:
        """A non-identity LinearizationTable (what DxO/Lightroom Enhance populate)
        is looked up before the black/white normalization."""
        max_val = 1000
        table = np.clip(np.arange(max_val + 1, dtype=np.float64) ** 1.5, 0, 60000).astype(np.uint16)
        black, white = 0, 50000
        path, source = _make_linearraw_dng_3ch_tagged(str(tmp_path), max_val=max_val, black=black, white=white, lin_table=table)
        out_path = os.path.join(str(tmp_path), "output.tiff")

        export_linear_output(path, out_path)

        looked_up = table[np.clip(source, 0, len(table) - 1).astype(np.int64)].astype(np.float64)
        expected = np.clip((looked_up - black) / (white - black), 0.0, 1.0)
        expected_u16 = (expected * 65535.0 + 0.5).astype(np.uint16)
        with tifffile.TiffFile(out_path) as tf:
            actual = tf.pages[0].asarray()
        np.testing.assert_allclose(actual.astype(np.int32), expected_u16.astype(np.int32), atol=1)

    def test_peek_failure_wrapped_as_value_error(self, tmp_path: str) -> None:
        """Any exception from _peek_linear_dng_rgb surfaces as ValueError, matching
        the exception type the old inline tifffile read always raised on failure."""
        path = _make_linearraw_dng_3ch(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "output.tiff")

        with mock.patch(
            "negpy.services.export.linear_output._peek_linear_dng_rgb",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(ValueError, match="Failed to read LinearRaw data"):
                export_linear_output(path, out_path)

    def test_decode_dng_no_linearraw_ifd_raises_value_error(self, tmp_path: str) -> None:
        """A DNG with no LinearRaw IFD at all still raises ValueError, not None
        propagating silently (peeked_3ch is None branch)."""
        path = os.path.join(str(tmp_path), "camera.dng")
        tifffile.imwrite(path, np.zeros((10, 10, 3), dtype=np.uint16), photometric="rgb")

        with pytest.raises(ValueError, match="No LinearRaw IFD found"):
            _decode_dng(path)

    def test_dxo_shaped_jpegxl_dng_end_to_end(self, tmp_path: str) -> None:
        """No real DxO PhotoLab/Lightroom Enhance sample is available, so this is the
        closest synthetic stand-in: a genuinely JPEG-XL-compressed (compression=52546)
        LinearRaw SubIFD behind a thumbnail IFD0, with LinearizationTable/BlackLevel/
        WhiteLevel/AsShotNeutral all populated — the same fixture shape
        tests/test_raw_handlers.py uses to verify _peek_linear_dng_rgb and the
        RawpyLoader import fallback directly, exercised here end-to-end through
        export_linear_output instead of calling the peek function in isolation."""
        h, w = 24, 20
        table = np.linspace(0, 65535, 1024).astype(np.uint16)
        codes = np.random.default_rng(0).integers(0, 1024, (h, w, 3)).astype(np.uint16)
        thumb = np.zeros((max(1, h // 8), max(1, w // 8), 3), dtype=np.uint8)
        dng_tags = [
            (50706, 1, 4, (1, 7, 0, 0), True),  # DNGVersion
            (50707, 1, 4, (1, 4, 0, 0), True),  # DNGBackwardVersion
            (50712, 3, len(table), tuple(int(x) for x in table), True),  # LinearizationTable
            (50714, 5, 3, (0, 1, 256, 1, 0, 1), True),  # BlackLevel: R=0, G=256, B=0
            (50717, 3, 3, (65535, 65535, 65535), True),  # WhiteLevel
            (50728, 5, 3, (477612, 1000000, 1000000, 1000000, 474074, 1000000), True),  # AsShotNeutral
        ]
        path = os.path.join(str(tmp_path), "dxo.dng")
        with tifffile.TiffWriter(path) as tw:
            tw.write(thumb, photometric="rgb", subfiletype=1, subifds=1, extratags=dng_tags)
            tw.write(codes, photometric=_LINEAR_RAW, subfiletype=0, planarconfig="contig", compression=52546)
        out_path = os.path.join(str(tmp_path), "output.tiff")

        export_linear_output(path, out_path)

        black = np.array([0.0, 256.0, 0.0])
        white = np.array([65535.0, 65535.0, 65535.0])
        linear = table[codes.astype(np.int64)].astype(np.float64)
        expected = np.clip((linear - black) / (white - black), 0.0, 1.0)
        expected_u16 = (expected * 65535.0 + 0.5).astype(np.uint16)
        with tifffile.TiffFile(out_path) as tf:
            actual = tf.pages[0].asarray()
        np.testing.assert_allclose(actual.astype(np.int32), expected_u16.astype(np.int32), atol=1)


class TestCameraRawSupport:
    def test_is_camera_raw_nef(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "photo.nef")
        open(path, "wb").close()
        assert _is_camera_raw(path)

    def test_is_camera_raw_cr2(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "photo.cr2")
        open(path, "wb").close()
        assert _is_camera_raw(path)

    def test_is_camera_raw_arw(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "photo.arw")
        open(path, "wb").close()
        assert _is_camera_raw(path)

    def test_tiff_not_camera_raw(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "photo.tiff")
        open(path, "wb").close()
        assert not _is_camera_raw(path)

    def test_jpeg_not_camera_raw(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "photo.jpg")
        open(path, "wb").close()
        assert not _is_camera_raw(path)

    def test_pakon_not_camera_raw(self, tmp_path: str) -> None:
        path = _make_pakon_raw(str(tmp_path))
        assert not _is_camera_raw(path)

    def test_dng_is_camera_raw(self, tmp_path: str) -> None:
        """A DNG is in SUPPORTED_RAW_EXTENSIONS and not in TIFF/JPEG sets."""
        path = os.path.join(str(tmp_path), "photo.dng")
        open(path, "wb").close()
        assert _is_camera_raw(path)

    def test_normalize_wb_rgb(self) -> None:
        r, g, b = _normalize_wb_rgb((398.0, 302.0, 873.0, 304.0))
        assert g == 1.0
        g_avg = (302.0 + 304.0) / 2.0
        assert abs(r - 398.0 / g_avg) < 1e-6
        assert abs(b - 873.0 / g_avg) < 1e-6

    def test_build_xmp_maketiff_format(self) -> None:
        wb = _CameraWB(
            as_shot=(398.0, 302.0, 873.0, 304.0),
            daylight=(1.94, 0.94, 1.38, 0.96),
        )
        xmp = _build_xmp("/path/to/DSCF3404.RAF", wb)
        text = xmp.decode("utf-8")
        assert "RAW-WB:" in text
        assert "1.000000" in text
        assert "crs:RawFileName" in text
        assert "DSCF3404.RAF" in text
        assert "dc:description" in text

    def test_camera_raw_supported(self, tmp_path: str) -> None:
        """A .nef file should be supported for linear output."""
        path = os.path.join(str(tmp_path), "photo.nef")
        open(path, "wb").close()
        assert is_linear_output_supported(path)

    def test_write_tiff_with_source_meta(self, tmp_path: str) -> None:
        f32 = np.random.RandomState(0).rand(10, 10, 3).astype(np.float32)
        out = os.path.join(str(tmp_path), "meta.tiff")
        meta = _SourceMeta(make="Plustek", model="OpticFilm 8100", datetime="2025:01:15 12:00:00")
        _write_tiff(f32, out, "test.dng", source_meta=meta)
        with tifffile.TiffFile(out) as tf:
            tags = tf.pages[0].tags
            assert tags["Make"].value == "Plustek"
            assert tags["Model"].value == "OpticFilm 8100"
            assert "2025:01:15" in tags["DateTime"].value
            assert tags["Software"].value == "NegPy"

    def test_write_tiff_software_always_set(self, tmp_path: str) -> None:
        f32 = np.random.RandomState(0).rand(10, 10, 3).astype(np.float32)
        out = os.path.join(str(tmp_path), "sw.tiff")
        _write_tiff(f32, out, "test.raw")
        with tifffile.TiffFile(out) as tf:
            assert tf.pages[0].tags["Software"].value == "NegPy"


class TestF335Detection:
    def test_f335_detected_by_size(self, tmp_path: str) -> None:
        path = _make_pakon_f335_raw(str(tmp_path))
        assert _default_pakon_expansion(path) == 1.0

    def test_f135_gets_4x(self, tmp_path: str) -> None:
        path = _make_pakon_raw(str(tmp_path))
        assert _default_pakon_expansion(path) == 4.0

    def test_source_type_f335(self, tmp_path: str) -> None:
        path = _make_pakon_f335_raw(str(tmp_path))
        assert linear_output_source_type(path) == "pakon_f335"

    def test_source_type_f135(self, tmp_path: str) -> None:
        path = _make_pakon_raw(str(tmp_path))
        assert linear_output_source_type(path) == "pakon"

    def test_f335_export_no_expansion(self, tmp_path: str) -> None:
        path = _make_pakon_f335_raw(str(tmp_path))
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(path, out)
        with tifffile.TiffFile(out) as tf:
            arr = tf.pages[0].asarray()
            assert arr.dtype == np.uint16
            assert arr.shape == (4000, 3000, 3)
            assert arr.max() > 0
            desc = tf.pages[0].description
            assert "no scaling" in desc
            assert "F335" in desc

    def test_f135_description_records_expansion(self, tmp_path: str) -> None:
        path = _make_pakon_raw(str(tmp_path))
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(path, out)
        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "x4" in desc
            assert "F135" in desc

    def test_effective_expansion_camera_raw(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "photo.nef")
        open(path, "wb").close()
        assert _effective_expansion(path, None) == 1.0
        assert _effective_expansion(path, 2.0) == 1.0

    def test_pakon_make_model_tags(self, tmp_path: str) -> None:
        path = _make_pakon_raw(str(tmp_path))
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(path, out)
        with tifffile.TiffFile(out) as tf:
            tags = tf.pages[0].tags
            assert tags["Make"].value == "Pakon"
            assert "F135 Plus Low Res" in tags["Model"].value

    def test_f335_make_model_tags(self, tmp_path: str) -> None:
        path = _make_pakon_f335_raw(str(tmp_path))
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(path, out)
        with tifffile.TiffFile(out) as tf:
            tags = tf.pages[0].tags
            assert tags["Make"].value == "Pakon"
            assert "F335" in tags["Model"].value


def _make_fake_camera_raws(tmp_dir: str, h: int = 40, w: int = 60) -> tuple[str, str, str]:
    """Create three empty .nef files to act as triplet paths."""
    paths = []
    for name in ("red.nef", "green.nef", "blue.nef"):
        p = os.path.join(tmp_dir, name)
        open(p, "wb").close()
        paths.append(p)
    return tuple(paths)  # type: ignore[return-value]


def _triplet_buffers(h: int = 40, w: int = 60) -> dict[str, np.ndarray]:
    """Synthetic RGB buffers where each exposure is bright in its own channel."""
    r = np.full((h, w, 3), 0.1, dtype=np.float32)
    r[..., 0] = 0.8
    g = np.full((h, w, 3), 0.1, dtype=np.float32)
    g[..., 1] = 0.7
    b = np.full((h, w, 3), 0.1, dtype=np.float32)
    b[..., 2] = 0.9
    return {"r": r, "g": g, "b": b}


_MOCK_WB = _CameraWB(as_shot=(1.5, 1.0, 2.0, 1.0), daylight=(2.0, 1.0, 1.5, 1.0))
_MOCK_META = _SourceMeta(make="Nikon", model="D850", datetime="2026:01:01 12:00:00")


class TestTripletExport:
    """Linear Output with RGB-scan triplet merge."""

    def _patch_decode(self, paths: tuple[str, str, str], bufs: dict[str, np.ndarray]):
        mapping = {paths[0]: bufs["r"], paths[1]: bufs["g"], paths[2]: bufs["b"]}

        def fake_decode(path: str):
            return mapping[path], _MOCK_WB, _MOCK_META

        return mock.patch(
            "negpy.services.export.linear_output._decode_camera_raw_buffer",
            side_effect=fake_decode,
        )

    def test_triplet_produces_merged_tiff(self, tmp_path: str) -> None:
        paths = _make_fake_camera_raws(str(tmp_path))
        bufs = _triplet_buffers()
        rgbscan = RgbScanConfig(enabled=True, green_path=paths[1], blue_path=paths[2], align=False)
        out = os.path.join(str(tmp_path), "triplet_linear.tiff")

        with self._patch_decode(paths, bufs):
            export_linear_output(paths[0], out, rgbscan=rgbscan)

        assert os.path.exists(out)
        with tifffile.TiffFile(out) as tf:
            arr = tf.pages[0].asarray()
            assert arr.dtype == np.uint16
            assert arr.shape == (40, 60, 3)
            f32 = arr.astype(np.float32) / 65535.0
            assert f32[0, 0, 0] == pytest.approx(0.8, abs=0.01)
            assert f32[0, 0, 1] == pytest.approx(0.7, abs=0.01)
            assert f32[0, 0, 2] == pytest.approx(0.9, abs=0.01)

    def test_triplet_channels_from_correct_exposures(self, tmp_path: str) -> None:
        """Red channel from red exposure, green from green, blue from blue."""
        paths = _make_fake_camera_raws(str(tmp_path))
        bufs = _triplet_buffers()
        rgbscan = RgbScanConfig(enabled=True, green_path=paths[1], blue_path=paths[2], align=False)
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode(paths, bufs):
            export_linear_output(paths[0], out, rgbscan=rgbscan)

        with tifffile.TiffFile(out) as tf:
            f32 = tf.pages[0].asarray().astype(np.float32) / 65535.0
            assert f32[..., 0].mean() == pytest.approx(0.8, abs=0.01)
            assert f32[..., 1].mean() == pytest.approx(0.7, abs=0.01)
            assert f32[..., 2].mean() == pytest.approx(0.9, abs=0.01)

    def test_triplet_description_mentions_triplet(self, tmp_path: str) -> None:
        paths = _make_fake_camera_raws(str(tmp_path))
        bufs = _triplet_buffers()
        rgbscan = RgbScanConfig(enabled=True, green_path=paths[1], blue_path=paths[2], align=False)
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode(paths, bufs):
            export_linear_output(paths[0], out, rgbscan=rgbscan)

        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "RGB triplet" in desc

    def test_triplet_preserves_wb_metadata(self, tmp_path: str) -> None:
        paths = _make_fake_camera_raws(str(tmp_path))
        bufs = _triplet_buffers()
        rgbscan = RgbScanConfig(enabled=True, green_path=paths[1], blue_path=paths[2], align=False)
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode(paths, bufs):
            export_linear_output(paths[0], out, rgbscan=rgbscan)

        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "no WB applied" in desc
            assert "as-shot:" in desc

    def test_triplet_preserves_make_model(self, tmp_path: str) -> None:
        paths = _make_fake_camera_raws(str(tmp_path))
        bufs = _triplet_buffers()
        rgbscan = RgbScanConfig(enabled=True, green_path=paths[1], blue_path=paths[2], align=False)
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode(paths, bufs):
            export_linear_output(paths[0], out, rgbscan=rgbscan)

        with tifffile.TiffFile(out) as tf:
            tags = tf.pages[0].tags
            assert tags["Make"].value == "Nikon"
            assert tags["Model"].value == "D850"

    def test_triplet_with_geometry(self, tmp_path: str) -> None:
        paths = _make_fake_camera_raws(str(tmp_path))
        bufs = _triplet_buffers()
        rgbscan = RgbScanConfig(enabled=True, green_path=paths[1], blue_path=paths[2], align=False)
        geo = GeometryConfig(rotation=1)
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode(paths, bufs):
            export_linear_output(paths[0], out, geometry=geo, rgbscan=rgbscan)

        with tifffile.TiffFile(out) as tf:
            arr = tf.pages[0].asarray()
            assert arr.shape == (60, 40, 3)

    def test_no_triplet_without_rgbscan(self, tmp_path: str) -> None:
        """Without rgbscan config, camera RAW goes through the normal single-file path."""
        paths = _make_fake_camera_raws(str(tmp_path))
        bufs = _triplet_buffers()
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode(paths, bufs):
            export_linear_output(paths[0], out)

        with tifffile.TiffFile(out) as tf:
            arr = tf.pages[0].asarray()
            assert arr.shape == (40, 60, 3)
            f32 = arr.astype(np.float32) / 65535.0
            assert f32[0, 0, 0] == pytest.approx(0.8, abs=0.01)
            assert f32[0, 0, 1] == pytest.approx(0.1, abs=0.01)

    def test_source_format_label_triplet(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "test.nef")
        open(path, "wb").close()
        rgbscan = RgbScanConfig(enabled=True, green_path="g.nef", blue_path="b.nef")
        assert _source_format_label(path, rgbscan) == "camera RAW (RGB triplet)"
        assert _source_format_label(path) == "camera RAW"


def _make_stitch_config(
    part1_path: str,
    w: int = 60,
    h: int = 40,
    triplets: tuple[tuple[str, str], ...] = (),
) -> StitchConfig:
    """Two side-by-side parts with 10px overlap, identity + offset transforms."""
    offset = w - 10
    return StitchConfig(
        stitch_enabled=True,
        stitch_paths=(part1_path,),
        stitch_transforms=(
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            (1.0, 0.0, float(offset), 0.0, 1.0, 0.0),
        ),
        stitch_canvas=(w + offset, h),
        stitch_sizes=((w, h), (w, h)),
        stitch_triplets=triplets,
    )


class TestStitchExport:
    """Linear Output with stitch composites."""

    def _patch_decode(self, path_to_buf: dict[str, np.ndarray]):
        def fake_decode(path: str):
            return path_to_buf[path], _MOCK_WB, _MOCK_META

        return mock.patch(
            "negpy.services.export.linear_output._decode_camera_raw_buffer",
            side_effect=fake_decode,
        )

    def test_stitch_produces_composite_tiff(self, tmp_path: str) -> None:
        p0 = os.path.join(str(tmp_path), "part0.nef")
        p1 = os.path.join(str(tmp_path), "part1.nef")
        for p in (p0, p1):
            open(p, "wb").close()

        h, w = 40, 60
        buf0 = np.full((h, w, 3), 0.4, dtype=np.float32)
        buf1 = np.full((h, w, 3), 0.6, dtype=np.float32)
        stitch = _make_stitch_config(p1, w=w, h=h)
        out = os.path.join(str(tmp_path), "stitch_linear.tiff")

        with self._patch_decode({p0: buf0, p1: buf1}):
            export_linear_output(p0, out, stitch=stitch)

        assert os.path.exists(out)
        with tifffile.TiffFile(out) as tf:
            arr = tf.pages[0].asarray()
            assert arr.dtype == np.uint16
            expected_w = w + (w - 10)
            assert arr.shape == (h, expected_w, 3)

    def test_stitch_description_mentions_stitch(self, tmp_path: str) -> None:
        p0 = os.path.join(str(tmp_path), "part0.nef")
        p1 = os.path.join(str(tmp_path), "part1.nef")
        for p in (p0, p1):
            open(p, "wb").close()

        h, w = 40, 60
        buf = np.full((h, w, 3), 0.5, dtype=np.float32)
        stitch = _make_stitch_config(p1, w=w, h=h)
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode({p0: buf, p1: buf}):
            export_linear_output(p0, out, stitch=stitch)

        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "stitch 2-part" in desc

    def test_stitch_preserves_make_model(self, tmp_path: str) -> None:
        p0 = os.path.join(str(tmp_path), "part0.nef")
        p1 = os.path.join(str(tmp_path), "part1.nef")
        for p in (p0, p1):
            open(p, "wb").close()

        h, w = 40, 60
        buf = np.full((h, w, 3), 0.5, dtype=np.float32)
        stitch = _make_stitch_config(p1, w=w, h=h)
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode({p0: buf, p1: buf}):
            export_linear_output(p0, out, stitch=stitch)

        with tifffile.TiffFile(out) as tf:
            tags = tf.pages[0].tags
            assert tags["Make"].value == "Nikon"
            assert tags["Model"].value == "D850"

    def test_stitch_with_geometry(self, tmp_path: str) -> None:
        p0 = os.path.join(str(tmp_path), "part0.nef")
        p1 = os.path.join(str(tmp_path), "part1.nef")
        for p in (p0, p1):
            open(p, "wb").close()

        h, w = 40, 60
        buf = np.full((h, w, 3), 0.5, dtype=np.float32)
        stitch = _make_stitch_config(p1, w=w, h=h)
        geo = GeometryConfig(rotation=1)
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode({p0: buf, p1: buf}):
            export_linear_output(p0, out, geometry=geo, stitch=stitch)

        with tifffile.TiffFile(out) as tf:
            arr = tf.pages[0].asarray()
            expected_w = w + (w - 10)
            assert arr.shape == (expected_w, h, 3)

    def test_stitch_with_triplets(self, tmp_path: str) -> None:
        """Stitch where each part is an RGB triplet."""
        p0r = os.path.join(str(tmp_path), "p0_r.nef")
        p0g = os.path.join(str(tmp_path), "p0_g.nef")
        p0b = os.path.join(str(tmp_path), "p0_b.nef")
        p1r = os.path.join(str(tmp_path), "p1_r.nef")
        p1g = os.path.join(str(tmp_path), "p1_g.nef")
        p1b = os.path.join(str(tmp_path), "p1_b.nef")
        for p in (p0r, p0g, p0b, p1r, p1g, p1b):
            open(p, "wb").close()

        h, w = 40, 60
        bufs = {}
        for path, ch in [(p0r, 0), (p0g, 1), (p0b, 2), (p1r, 0), (p1g, 1), (p1b, 2)]:
            arr = np.full((h, w, 3), 0.1, dtype=np.float32)
            arr[..., ch] = 0.7
            bufs[path] = arr

        triplets = ((p0g, p0b), (p1g, p1b))
        stitch = _make_stitch_config(p1r, w=w, h=h, triplets=triplets)
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode(bufs):
            export_linear_output(p0r, out, stitch=stitch)

        assert os.path.exists(out)
        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "stitch 2-part" in desc
            assert "RGB triplet" in desc

    def test_stitch_triplet_no_wb_in_output(self, tmp_path: str) -> None:
        """Triplet composites don't record WB (narrowband captures have no meaningful WB)."""
        p0r = os.path.join(str(tmp_path), "p0_r.nef")
        p0g = os.path.join(str(tmp_path), "p0_g.nef")
        p0b = os.path.join(str(tmp_path), "p0_b.nef")
        p1r = os.path.join(str(tmp_path), "p1_r.nef")
        p1g = os.path.join(str(tmp_path), "p1_g.nef")
        p1b = os.path.join(str(tmp_path), "p1_b.nef")
        for p in (p0r, p0g, p0b, p1r, p1g, p1b):
            open(p, "wb").close()

        h, w = 40, 60
        buf = np.full((h, w, 3), 0.5, dtype=np.float32)
        bufs = {p: buf for p in (p0r, p0g, p0b, p1r, p1g, p1b)}

        triplets = ((p0g, p0b), (p1g, p1b))
        stitch = _make_stitch_config(p1r, w=w, h=h, triplets=triplets)
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode(bufs):
            export_linear_output(p0r, out, stitch=stitch)

        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "as-shot:" not in desc

    def test_source_format_label_stitch(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "test.nef")
        open(path, "wb").close()
        stitch = StitchConfig(stitch_enabled=True, stitch_paths=("/p1.nef",))
        assert "stitch 2-part" in _source_format_label(path, stitch=stitch)

    def test_source_format_label_stitch_triplet(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "test.nef")
        open(path, "wb").close()
        stitch = StitchConfig(
            stitch_enabled=True,
            stitch_paths=("/p1.nef",),
            stitch_triplets=(("g0.nef", "b0.nef"), ("g1.nef", "b1.nef")),
        )
        label = _source_format_label(path, stitch=stitch)
        assert "stitch 2-part" in label
        assert "RGB triplet" in label


class TestLinearCorrections:
    """Tests for optional per-step corrections (WB, flatfield, sensor)."""

    def _patch_decode(self, path_to_buf: dict[str, np.ndarray]):
        def fake_decode(path: str):
            return path_to_buf[path], _MOCK_WB, _MOCK_META

        return mock.patch(
            "negpy.services.export.linear_output._decode_camera_raw_buffer",
            side_effect=fake_decode,
        )

    def test_apply_white_balance_scales_channels(self) -> None:
        f32 = np.full((4, 4, 3), 0.5, dtype=np.float32)
        wb = _CameraWB(as_shot=(2.0, 1.0, 3.0, 1.0), daylight=(1.0, 1.0, 1.0, 1.0))
        result = _apply_white_balance(f32, wb)
        assert result.shape == f32.shape
        np.testing.assert_allclose(result[:, :, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(result[:, :, 1], 0.5, atol=1e-6)
        np.testing.assert_allclose(result[:, :, 2], 1.0, atol=1e-6)

    def test_apply_white_balance_clamps(self) -> None:
        f32 = np.full((4, 4, 3), 0.8, dtype=np.float32)
        wb = _CameraWB(as_shot=(2.0, 1.0, 2.0, 1.0), daylight=(1.0, 1.0, 1.0, 1.0))
        result = _apply_white_balance(f32, wb)
        assert result.max() <= 1.0

    def test_apply_wb_flag_bakes_wb(self, tmp_path: str) -> None:
        p = os.path.join(str(tmp_path), "photo.nef")
        open(p, "wb").close()

        buf = np.full((10, 10, 3), 0.3, dtype=np.float32)
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode({p: buf}):
            export_linear_output(p, out, apply_wb=True)

        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "WB applied" in desc
            assert "no WB applied" not in desc

    def test_no_apply_wb_flag_records_raw(self, tmp_path: str) -> None:
        p = os.path.join(str(tmp_path), "photo.nef")
        open(p, "wb").close()

        buf = np.full((10, 10, 3), 0.3, dtype=np.float32)
        out = os.path.join(str(tmp_path), "out.tiff")

        with self._patch_decode({p: buf}):
            export_linear_output(p, out, apply_wb=False)

        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "no WB applied" in desc

    def test_apply_flatfield_calls_correction(self, tmp_path: str) -> None:
        from negpy.features.flatfield.models import FlatFieldConfig

        p = os.path.join(str(tmp_path), "photo.nef")
        open(p, "wb").close()

        buf = np.full((10, 10, 3), 0.3, dtype=np.float32)
        ff = FlatFieldConfig(apply=True, profile_id="test")
        out = os.path.join(str(tmp_path), "out.tiff")

        with (
            self._patch_decode({p: buf}),
            mock.patch("negpy.services.export.linear_output._apply_flatfield_correction", return_value=buf) as ff_mock,
        ):
            export_linear_output(p, out, flatfield=ff, apply_flatfield=True)

        ff_mock.assert_called_once()
        with tifffile.TiffFile(out) as tf:
            assert "flatfield" in tf.pages[0].description

    def test_no_apply_flatfield_skips(self, tmp_path: str) -> None:
        from negpy.features.flatfield.models import FlatFieldConfig

        p = os.path.join(str(tmp_path), "photo.nef")
        open(p, "wb").close()

        buf = np.full((10, 10, 3), 0.3, dtype=np.float32)
        ff = FlatFieldConfig(apply=True, profile_id="test")
        out = os.path.join(str(tmp_path), "out.tiff")

        with (
            self._patch_decode({p: buf}),
            mock.patch("negpy.services.export.linear_output._apply_flatfield_correction", return_value=buf) as ff_mock,
        ):
            export_linear_output(p, out, flatfield=ff, apply_flatfield=False)

        ff_mock.assert_not_called()

    def test_apply_sensor_calls_correction(self, tmp_path: str) -> None:
        from negpy.features.process.models import ProcessConfig

        p = os.path.join(str(tmp_path), "photo.nef")
        open(p, "wb").close()

        buf = np.full((10, 10, 3), 0.3, dtype=np.float32)
        matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        proc = ProcessConfig(sensor_matrix=matrix)
        out = os.path.join(str(tmp_path), "out.tiff")

        with (
            self._patch_decode({p: buf}),
            mock.patch("negpy.services.export.linear_output.apply_sensor_correction", return_value=buf) as sc_mock,
        ):
            export_linear_output(p, out, process=proc, apply_sensor=True)

        sc_mock.assert_called_once()
        with tifffile.TiffFile(out) as tf:
            assert "sensor" in tf.pages[0].description

    def test_apply_sensor_noop_without_matrix(self, tmp_path: str) -> None:
        from negpy.features.process.models import ProcessConfig

        p = os.path.join(str(tmp_path), "photo.nef")
        open(p, "wb").close()

        buf = np.full((10, 10, 3), 0.3, dtype=np.float32)
        proc = ProcessConfig(sensor_matrix=None)
        out = os.path.join(str(tmp_path), "out.tiff")

        with (
            self._patch_decode({p: buf}),
            mock.patch("negpy.services.export.linear_output.apply_sensor_correction", return_value=buf) as sc_mock,
        ):
            export_linear_output(p, out, process=proc, apply_sensor=True)

        sc_mock.assert_not_called()

    def test_no_apply_sensor_skips(self, tmp_path: str) -> None:
        from negpy.features.process.models import ProcessConfig

        p = os.path.join(str(tmp_path), "photo.nef")
        open(p, "wb").close()

        buf = np.full((10, 10, 3), 0.3, dtype=np.float32)
        proc = ProcessConfig()
        out = os.path.join(str(tmp_path), "out.tiff")

        with (
            self._patch_decode({p: buf}),
            mock.patch("negpy.services.export.linear_output.apply_sensor_correction", return_value=buf) as sc_mock,
        ):
            export_linear_output(p, out, process=proc, apply_sensor=False)

        sc_mock.assert_not_called()

    def test_description_lists_corrections(self, tmp_path: str) -> None:
        f32 = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = os.path.join(str(tmp_path), "out.tiff")
        _write_tiff(f32, out, "test.nef", flatfield_applied=True, sensor_applied=True)
        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "corrections: flatfield, sensor" in desc

    def test_description_no_corrections_by_default(self, tmp_path: str) -> None:
        f32 = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = os.path.join(str(tmp_path), "out.tiff")
        _write_tiff(f32, out, "test.nef")
        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "corrections:" not in desc

    def test_description_includes_gamma_linearization(self, tmp_path: str) -> None:
        f32 = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = os.path.join(str(tmp_path), "out.tiff")
        _write_tiff(f32, out, "test.tif", gamma_key="2.2")
        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "linearized from Gamma 2.2" in desc

    def test_description_no_gamma_for_linear(self, tmp_path: str) -> None:
        f32 = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = os.path.join(str(tmp_path), "out.tiff")
        _write_tiff(f32, out, "test.tif", gamma_key="linear")
        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "linearized" not in desc


class TestTiffLinearOutput:
    def test_is_tiff_extensions(self) -> None:
        assert _is_tiff("scan.tif")
        assert _is_tiff("scan.tiff")
        assert _is_tiff("scan.TIF")
        assert not _is_tiff("scan.dng")
        assert not _is_tiff("scan.nef")

    def test_linearize_identity(self) -> None:
        data = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        result = _linearize(data, "linear")
        np.testing.assert_array_equal(result, data)

    def test_linearize_gamma_22(self) -> None:
        data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = _linearize(data, "2.2")
        np.testing.assert_allclose(result[0], 0.0, atol=1e-7)
        np.testing.assert_allclose(result[1], 0.5**2.2, rtol=1e-5)
        np.testing.assert_allclose(result[2], 1.0, atol=1e-7)

    def test_linearize_srgb(self) -> None:
        result = _linearize(np.array([0.0, 0.04045, 0.5, 1.0], dtype=np.float32), "srgb")
        np.testing.assert_allclose(result[0], 0.0, atol=1e-7)
        np.testing.assert_allclose(result[1], 0.04045 / 12.92, rtol=1e-5)
        np.testing.assert_allclose(result[3], 1.0, atol=1e-7)

    def test_linearize_lstar(self) -> None:
        result = _linearize(np.array([0.0, 1.0], dtype=np.float32), "lstar")
        np.testing.assert_allclose(result[0], 0.0, atol=1e-7)
        np.testing.assert_allclose(result[1], 1.0, atol=1e-7)

    def test_linearize_rec709(self) -> None:
        result = _linearize(np.array([0.0, 0.081, 1.0], dtype=np.float32), "rec709")
        np.testing.assert_allclose(result[0], 0.0, atol=1e-7)
        np.testing.assert_allclose(result[1], 0.081 / 4.5, rtol=1e-5)
        np.testing.assert_allclose(result[2], 1.0, atol=1e-7)

    def test_linearize_clamps_input(self) -> None:
        data = np.array([-0.1, 1.5], dtype=np.float32)
        result = _linearize(data, "2.2")
        assert result[0] >= 0.0
        assert result[1] <= 1.0

    def test_linearize_all_gamma_options_have_keys(self) -> None:
        keys = [k for k, _ in TIFF_GAMMA_OPTIONS]
        data = np.array([0.5], dtype=np.float32)
        for key in keys:
            result = _linearize(data, key)
            assert result.shape == data.shape

    def test_source_type_tiff(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "scan.tif")
        tifffile.imwrite(path, np.zeros((4, 4, 3), dtype=np.uint16))
        assert linear_output_source_type(path) == "tiff"

    def test_tiff_supported(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "scan.tif")
        tifffile.imwrite(path, np.zeros((4, 4, 3), dtype=np.uint16))
        assert is_linear_output_supported(path)

    def test_source_format_label_tiff(self) -> None:
        assert _source_format_label("scan.tif") == "TIFF"
        assert _source_format_label("scan.tiff") == "TIFF"

    def test_decode_tiff_rgb(self, tmp_path: str) -> None:
        from negpy.services.export.linear_output import _decode_tiff

        rng = np.random.RandomState(42)
        data = rng.randint(0, 65535, size=(10, 10, 3), dtype=np.uint16)
        path = os.path.join(str(tmp_path), "rgb.tif")
        tifffile.imwrite(path, data)
        rgb, ir = _decode_tiff(path)
        assert rgb.shape == (10, 10, 3)
        assert rgb.dtype == np.float32
        assert ir is None

    def test_decode_tiff_4ch_splits_ir(self, tmp_path: str) -> None:
        from negpy.services.export.linear_output import _decode_tiff

        data = np.ones((8, 8, 4), dtype=np.uint16) * 32768
        path = os.path.join(str(tmp_path), "4ch.tif")
        tifffile.imwrite(path, data, extrasamples=[0])
        rgb, ir = _decode_tiff(path)
        assert rgb.shape == (8, 8, 3)
        assert ir is not None
        assert ir.shape[:2] == (8, 8)

    def test_decode_tiff_4ch_alpha_not_ir(self, tmp_path: str) -> None:
        from negpy.services.export.linear_output import _decode_tiff

        data = np.ones((8, 8, 4), dtype=np.uint16) * 32768
        path = os.path.join(str(tmp_path), "4ch_alpha.tif")
        tifffile.imwrite(path, data, extrasamples=[2])
        rgb, ir = _decode_tiff(path)
        assert rgb.shape == (8, 8, 3)
        assert ir is None

    def test_decode_tiff_applies_gamma(self, tmp_path: str) -> None:
        from negpy.services.export.linear_output import _decode_tiff

        data = np.full((4, 4, 3), 32768, dtype=np.uint16)
        path = os.path.join(str(tmp_path), "gamma.tif")
        tifffile.imwrite(path, data)
        rgb_lin, _ = _decode_tiff(path, gamma_key="linear")
        rgb_22, _ = _decode_tiff(path, gamma_key="2.2")
        assert np.all(rgb_22 < rgb_lin)

    def test_export_tiff_with_gamma(self, tmp_path: str) -> None:
        data = np.full((4, 4, 3), 32768, dtype=np.uint16)
        src = os.path.join(str(tmp_path), "input.tif")
        tifffile.imwrite(src, data)
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(src, out, gamma_key="2.2")
        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "linearized from Gamma 2.2" in desc
            assert "source: TIFF" in desc

    def test_decode_tiff_with_expansion(self, tmp_path: str) -> None:
        from negpy.services.export.linear_output import _decode_tiff

        data = np.full((4, 4, 3), 16384, dtype=np.uint16)
        path = os.path.join(str(tmp_path), "dim.tif")
        tifffile.imwrite(path, data)
        rgb_no_exp, _ = _decode_tiff(path)
        rgb_2x, _ = _decode_tiff(path, expansion=2.0)
        np.testing.assert_allclose(rgb_2x, np.clip(rgb_no_exp * 2.0, 0.0, 1.0), atol=1e-6)

    def test_export_tiff_with_expansion(self, tmp_path: str) -> None:
        data = np.full((4, 4, 3), 16384, dtype=np.uint16)
        src = os.path.join(str(tmp_path), "input.tif")
        tifffile.imwrite(src, data)
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(src, out, expansion=2.0)
        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "expansion: x2" in desc

    def test_tiff_output_has_no_icc_profile(self, tmp_path: str) -> None:
        data = np.full((4, 4, 3), 32768, dtype=np.uint16)
        src = os.path.join(str(tmp_path), "input.tif")
        tifffile.imwrite(src, data)
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(src, out)
        with tifffile.TiffFile(out) as tf:
            tag_codes = [t.code for t in tf.pages[0].tags.values()]
            assert 34675 not in tag_codes  # no ICC profile
            assert 34665 not in tag_codes  # no EXIF IFD

    def test_tiff_output_keeps_make_model(self, tmp_path: str) -> None:
        f32 = np.full((4, 4, 3), 0.5, dtype=np.float32)
        meta = _SourceMeta(make="Nikon", model="CoolScan 5000")
        out = os.path.join(str(tmp_path), "out.tiff")
        _write_tiff(f32, out, "test.tif", source_meta=meta)
        with tifffile.TiffFile(out) as tf:
            tags = {t.code: t.value for t in tf.pages[0].tags.values()}
            assert tags.get(271) == "Nikon"
            assert tags.get(272) == "CoolScan 5000"


def _make_coolscan_nef(tmp_dir: str, h: int = 200, w: int = 300, channels: int = 3) -> str:
    """Create a synthetic Coolscan-style NEF: thumbnail in IFD0, full-res RGB in SubIFD."""
    thumb = np.zeros((50, 75, 3), dtype=np.uint8)
    rng = np.random.RandomState(42)
    fullres = rng.randint(0, 65535, (h, w, channels), dtype=np.uint16)
    path = os.path.join(tmp_dir, "coolscan.nef")
    with tifffile.TiffWriter(path) as tw:
        tw.write(thumb, photometric="rgb", subifds=1)
        tw.write(fullres, photometric="rgb")
    return path


def _make_camera_nef(tmp_dir: str) -> str:
    """Create a synthetic camera-style NEF: single-channel Bayer, no RGB SubIFD."""
    bayer = np.zeros((200, 300), dtype=np.uint16)
    path = os.path.join(tmp_dir, "camera.nef")
    with tifffile.TiffWriter(path) as tw:
        tw.write(bayer, photometric="minisblack")
    return path


def _make_camera_nef_with_preview(tmp_dir: str) -> str:
    """Camera NEF with both a Bayer SubIFD and an RGB preview SubIFD.

    Real Nikon cameras routinely embed a full-res RGB preview alongside
    the CFA data. This must NOT match the scanner detector.
    """
    thumb = np.zeros((50, 75, 3), dtype=np.uint8)
    bayer = np.zeros((4000, 6000), dtype=np.uint16)
    preview = np.zeros((4000, 6000, 3), dtype=np.uint8)
    path = os.path.join(tmp_dir, "camera_preview.nef")
    with tifffile.TiffWriter(path) as tw:
        tw.write(thumb, photometric="rgb", subifds=2)
        tw.write(bayer, photometric="minisblack")
        tw.write(preview, photometric="rgb")
    return path


class TestCoolscanNef:
    def test_detect_coolscan_nef(self, tmp_path: str) -> None:
        path = _make_coolscan_nef(str(tmp_path))
        assert is_coolscan_nef(path)

    def test_camera_nef_not_detected(self, tmp_path: str) -> None:
        path = _make_camera_nef(str(tmp_path))
        assert not is_coolscan_nef(path)

    def test_camera_nef_with_preview_not_detected(self, tmp_path: str) -> None:
        """Camera NEF with RGB preview SubIFD alongside Bayer data must not match."""
        path = _make_camera_nef_with_preview(str(tmp_path))
        assert not is_coolscan_nef(path)

    def test_camera_nef_with_preview_is_camera_raw(self, tmp_path: str) -> None:
        path = _make_camera_nef_with_preview(str(tmp_path))
        assert _is_camera_raw(path)

    def test_non_nef_not_detected(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "photo.tiff")
        tifffile.imwrite(path, np.zeros((10, 10, 3), dtype=np.uint16))
        assert not is_coolscan_nef(path)

    def test_coolscan_nef_not_camera_raw(self, tmp_path: str) -> None:
        path = _make_coolscan_nef(str(tmp_path))
        assert not _is_camera_raw(path)

    def test_camera_nef_is_camera_raw(self, tmp_path: str) -> None:
        path = _make_camera_nef(str(tmp_path))
        assert _is_camera_raw(path)

    def test_linear_output_supported(self, tmp_path: str) -> None:
        path = _make_coolscan_nef(str(tmp_path))
        assert is_linear_output_supported(path)

    def test_source_type_nef(self, tmp_path: str) -> None:
        path = _make_coolscan_nef(str(tmp_path))
        assert linear_output_source_type(path) == "nef"

    def test_source_format_label(self, tmp_path: str) -> None:
        path = _make_coolscan_nef(str(tmp_path))
        assert _source_format_label(path) == "Coolscan NEF"

    def test_no_expansion(self, tmp_path: str) -> None:
        path = _make_coolscan_nef(str(tmp_path))
        assert _effective_expansion(path, None) == 1.0
        assert _effective_expansion(path, 4.0) == 1.0

    def test_export_roundtrip(self, tmp_path: str) -> None:
        path = _make_coolscan_nef(str(tmp_path))
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(path, out)
        with tifffile.TiffFile(out) as tf:
            arr = tf.pages[0].asarray()
            assert arr.dtype == np.uint16
            assert arr.shape == (200, 300, 3)
            desc = tf.pages[0].description
            assert "Coolscan NEF" in desc
            assert "no scaling" in desc

    def test_export_4ch_drops_extra_channel(self, tmp_path: str) -> None:
        path = _make_coolscan_nef(str(tmp_path), channels=4)
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(path, out)
        ir_path = os.path.join(str(tmp_path), "output_ir.tiff")
        assert not os.path.exists(ir_path)
        with tifffile.TiffFile(out) as tf:
            assert tf.pages[0].asarray().shape == (200, 300, 3)

    def test_loader_returns_float32(self, tmp_path: str) -> None:
        from negpy.infrastructure.loaders.nef_loader import NefLoader

        path = _make_coolscan_nef(str(tmp_path))
        loader = NefLoader()
        wrapper, metadata = loader.load(path)
        with wrapper as w:
            assert w.data.dtype == np.float32
            assert w.data.shape == (200, 300, 3)
            assert w.data.min() >= 0.0
            assert w.data.max() <= 1.0
        assert "orientation" in metadata

    def test_loader_drops_extra_channel(self, tmp_path: str) -> None:
        from negpy.infrastructure.loaders.nef_loader import NefLoader

        path = _make_coolscan_nef(str(tmp_path), channels=4)
        loader = NefLoader()
        wrapper, metadata = loader.load(path)
        with wrapper as w:
            assert w.data.shape == (200, 300, 3)
        assert metadata.get("ir") is None


def _make_flextight_fff(tmp_dir: str, h: int = 400, w: int = 600, channels: int = 3) -> str:
    """Create a synthetic Flextight FFF: big-endian TIFF, full-res RGB in IFD0, small preview in IFD1."""
    rng = np.random.RandomState(42)
    fullres = rng.randint(0, 65535, (h, w, channels), dtype=np.uint16)
    preview = np.zeros((50, 75, 3), dtype=np.uint8)
    path = os.path.join(tmp_dir, "scan.fff")
    with tifffile.TiffWriter(path, byteorder=">") as tw:
        tw.write(fullres, photometric="rgb")
        tw.write(preview, photometric="rgb")
    return path


class TestFlextightFff:
    def test_detect_flextight_fff(self, tmp_path: str) -> None:
        path = _make_flextight_fff(str(tmp_path))
        assert is_flextight_fff(path)

    def test_non_fff_not_detected(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "photo.tiff")
        tifffile.imwrite(path, np.zeros((10, 10, 3), dtype=np.uint16))
        assert not is_flextight_fff(path)

    def test_fff_not_camera_raw(self, tmp_path: str) -> None:
        path = _make_flextight_fff(str(tmp_path))
        assert not _is_camera_raw(path)

    def test_linear_output_supported(self, tmp_path: str) -> None:
        path = _make_flextight_fff(str(tmp_path))
        assert is_linear_output_supported(path)

    def test_source_type_fff(self, tmp_path: str) -> None:
        path = _make_flextight_fff(str(tmp_path))
        assert linear_output_source_type(path) == "fff"

    def test_source_format_label(self, tmp_path: str) -> None:
        path = _make_flextight_fff(str(tmp_path))
        assert _source_format_label(path) == "Flextight FFF"

    def test_no_expansion(self, tmp_path: str) -> None:
        path = _make_flextight_fff(str(tmp_path))
        assert _effective_expansion(path, None) == 1.0
        assert _effective_expansion(path, 4.0) == 1.0

    def test_export_roundtrip(self, tmp_path: str) -> None:
        path = _make_flextight_fff(str(tmp_path))
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(path, out)
        with tifffile.TiffFile(out) as tf:
            arr = tf.pages[0].asarray()
            assert arr.dtype == np.uint16
            assert arr.shape == (400, 600, 3)
            desc = tf.pages[0].description
            assert "Flextight FFF" in desc
            assert "no scaling" in desc

    def test_picks_largest_ifd(self, tmp_path: str) -> None:
        """When multiple IFDs exist, the largest by pixel count is used."""
        rng = np.random.RandomState(42)
        fullres = rng.randint(0, 65535, (500, 750, 3), dtype=np.uint16)
        small = np.zeros((50, 75, 3), dtype=np.uint16)
        path = os.path.join(str(tmp_path), "multi.fff")
        with tifffile.TiffWriter(path, byteorder=">") as tw:
            tw.write(fullres, photometric="rgb")
            tw.write(small, photometric="rgb")
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(path, out)
        with tifffile.TiffFile(out) as tf:
            assert tf.pages[0].asarray().shape == (500, 750, 3)

    def test_loader_returns_float32(self, tmp_path: str) -> None:
        from negpy.infrastructure.loaders.fff_loader import FffLoader

        path = _make_flextight_fff(str(tmp_path))
        loader = FffLoader()
        wrapper, metadata = loader.load(path)
        with wrapper as w:
            assert w.data.dtype == np.float32
            assert w.data.shape == (400, 600, 3)
            assert w.data.min() >= 0.0
            assert w.data.max() <= 1.0
        assert "orientation" in metadata


def _make_logluv_fff(tmp_dir: str, h: int = 4, w: int = 6) -> str:
    """Create a synthetic SGI LogLuv32 FFF file (minimal hand-built TIFF)."""
    import struct

    from negpy.infrastructure.loaders.logluv import (
        COMPRESSION_SGILOG,
    )

    UVSCALE = 410.0
    U_NEU = 0.210526316
    V_NEU = 0.473684211
    PHOTOMETRIC_LOGLUV = 32845

    rng = np.random.RandomState(123)
    luminance = rng.uniform(0.01, 1.0, (h, w))

    def logl16_from_y(y):
        le = np.floor(256.0 * (np.log2(np.abs(y)) + 64.0)).astype(np.int64)
        le = np.clip(le, 0, 0x7FFF)
        le = np.where(y <= 0, 0, le)
        return le.astype(np.uint32)

    le = logl16_from_y(luminance)
    ue = np.clip(np.trunc(UVSCALE * U_NEU), 0, 255).astype(np.uint32)
    ve = np.clip(np.trunc(UVSCALE * V_NEU), 0, 255).astype(np.uint32)
    packed = (le << 16) | (ue << 8) | ve

    pixel_bytes = packed.astype(">u4").tobytes()

    byte_order = b"MM"
    ifd_offset = 8
    n_tags = 8
    ifd_size = 2 + n_tags * 12 + 4
    strip_offset = ifd_offset + ifd_size

    ifd = struct.pack(">H", n_tags)

    def tag(t, typ, cnt, val):
        if typ == 3:
            return struct.pack(">HHIH2x", t, typ, cnt, val)
        return struct.pack(">HHII", t, typ, cnt, val)

    ifd += tag(256, 4, 1, w)  # ImageWidth
    ifd += tag(257, 4, 1, h)  # ImageLength
    ifd += tag(258, 3, 1, 32)  # BitsPerSample
    ifd += tag(259, 3, 1, COMPRESSION_SGILOG)  # Compression
    ifd += tag(262, 3, 1, PHOTOMETRIC_LOGLUV)  # PhotometricInterpretation
    ifd += tag(273, 4, 1, strip_offset)  # StripOffsets
    ifd += tag(277, 3, 1, 3)  # SamplesPerPixel
    ifd += tag(278, 4, 1, h)  # RowsPerStrip
    ifd += struct.pack(">I", 0)  # next IFD

    header = byte_order + struct.pack(">HI", 42, ifd_offset)

    # StripByteCounts: we omit it — the decoder should handle this
    # Actually we need it. Rebuild with 9 tags.
    n_tags = 9
    ifd_size = 2 + n_tags * 12 + 4
    strip_offset = ifd_offset + ifd_size

    ifd = struct.pack(">H", n_tags)
    ifd += tag(256, 4, 1, w)
    ifd += tag(257, 4, 1, h)
    ifd += tag(258, 3, 1, 32)
    ifd += tag(259, 3, 1, COMPRESSION_SGILOG)
    ifd += tag(262, 3, 1, PHOTOMETRIC_LOGLUV)
    ifd += tag(273, 4, 1, strip_offset)
    ifd += tag(277, 3, 1, 3)
    ifd += tag(278, 4, 1, h)
    ifd += tag(279, 4, 1, len(pixel_bytes))  # StripByteCounts
    ifd += struct.pack(">I", 0)

    header = byte_order + struct.pack(">HI", 42, ifd_offset)
    data = header + ifd + pixel_bytes

    path = os.path.join(tmp_dir, "logluv.fff")
    with open(path, "wb") as f:
        f.write(data)
    return path


class TestFlextightLogLuv:
    def test_detect_logluv_fff(self, tmp_path: str) -> None:
        path = _make_logluv_fff(str(tmp_path))
        assert is_flextight_fff(path)

    def test_loader_decodes_logluv(self, tmp_path: str) -> None:
        from negpy.infrastructure.loaders.fff_loader import FffLoader

        path = _make_logluv_fff(str(tmp_path), h=4, w=6)
        loader = FffLoader()
        wrapper, metadata = loader.load(path)
        with wrapper as w:
            assert w.data.dtype == np.float32
            assert w.data.shape == (4, 6, 3)
            assert w.data.min() >= 0.0
            assert w.data.max() <= 1.0
        assert "orientation" in metadata

    def test_logluv_export_roundtrip(self, tmp_path: str) -> None:
        path = _make_logluv_fff(str(tmp_path), h=10, w=15)
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(path, out)
        with tifffile.TiffFile(out) as tf:
            arr = tf.pages[0].asarray()
            assert arr.dtype == np.uint16
            assert arr.shape == (10, 15, 3)
            desc = tf.pages[0].description
            assert "Flextight FFF" in desc

    def test_logluv_produces_nonzero_rgb(self, tmp_path: str) -> None:
        from negpy.infrastructure.loaders.fff_loader import FffLoader

        path = _make_logluv_fff(str(tmp_path), h=4, w=6)
        loader = FffLoader()
        wrapper, _meta = loader.load(path)
        with wrapper as w:
            assert w.data.mean() > 0.01, "LogLuv decode produced near-zero output"

    def test_logluv_source_type(self, tmp_path: str) -> None:
        path = _make_logluv_fff(str(tmp_path))
        assert linear_output_source_type(path) == "fff"


def _make_noritsu_raw(tmp_dir: str, w: int = 4042, h: int = 6391) -> str:
    """Create a synthetic Noritsu RAW file: headerless BGR16 LE, 12-bit data."""
    rng = np.random.RandomState(42)
    bgr = rng.randint(0, 4096, size=(h, w, 3), dtype=np.uint16)
    path = os.path.join(tmp_dir, "FULL000000020000.RAW")
    bgr.astype("<u2").tofile(path)
    assert os.path.getsize(path) == w * h * 3 * 2
    return path


def _make_noritsu_raw_small(tmp_dir: str) -> str:
    """Create a Noritsu RAW with the smallest known dims (3551×4502)."""
    w, h = 3551, 4502
    rng = np.random.RandomState(77)
    bgr = rng.randint(0, 4096, size=(h, w, 3), dtype=np.uint16)
    path = os.path.join(tmp_dir, "FULL_small.raw")
    bgr.astype("<u2").tofile(path)
    return path


class TestNoritsuRaw:
    def test_detect_noritsu_raw(self, tmp_path: str) -> None:
        path = _make_noritsu_raw(str(tmp_path))
        assert is_noritsu_raw(path)

    def test_pakon_not_detected_as_noritsu(self, tmp_path: str) -> None:
        path = _make_pakon_raw(str(tmp_path))
        assert not is_noritsu_raw(path)

    def test_unknown_size_not_detected(self, tmp_path: str) -> None:
        data = np.zeros(12345678, dtype=np.uint8)
        path = os.path.join(str(tmp_path), "mystery.raw")
        data.tofile(path)
        assert not is_noritsu_raw(path)

    def test_non_raw_ext_not_detected(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "photo.tiff")
        tifffile.imwrite(path, np.zeros((10, 10, 3), dtype=np.uint16))
        assert not is_noritsu_raw(path)

    def test_noritsu_not_camera_raw(self, tmp_path: str) -> None:
        path = _make_noritsu_raw(str(tmp_path))
        assert not _is_camera_raw(path)

    def test_linear_output_supported(self, tmp_path: str) -> None:
        path = _make_noritsu_raw(str(tmp_path))
        assert is_linear_output_supported(path)

    def test_source_type_noritsu(self, tmp_path: str) -> None:
        path = _make_noritsu_raw(str(tmp_path))
        assert linear_output_source_type(path) == "noritsu"

    def test_source_format_label(self, tmp_path: str) -> None:
        path = _make_noritsu_raw(str(tmp_path))
        assert _source_format_label(path) == "Noritsu RAW"

    def test_default_expansion(self, tmp_path: str) -> None:
        path = _make_noritsu_raw(str(tmp_path))
        assert _effective_expansion(path, None) == 16.0

    def test_custom_expansion(self, tmp_path: str) -> None:
        path = _make_noritsu_raw(str(tmp_path))
        assert _effective_expansion(path, 8.0) == 8.0

    def test_export_roundtrip(self, tmp_path: str) -> None:
        path = _make_noritsu_raw(str(tmp_path))
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(path, out)
        with tifffile.TiffFile(out) as tf:
            arr = tf.pages[0].asarray()
            assert arr.dtype == np.uint16
            assert arr.shape == (6391, 4042, 3)
            desc = tf.pages[0].description
            assert "Noritsu RAW" in desc
            assert "expansion: x16" in desc

    def test_export_custom_expansion(self, tmp_path: str) -> None:
        path = _make_noritsu_raw(str(tmp_path))
        out = os.path.join(str(tmp_path), "output.tiff")
        export_linear_output(path, out, expansion=8.0)
        with tifffile.TiffFile(out) as tf:
            desc = tf.pages[0].description
            assert "expansion: x8" in desc

    def test_bgr_to_rgb_swap(self, tmp_path: str) -> None:
        """Verify the loader performs BGR to RGB channel swap."""
        w, h = 4042, 6391
        bgr = np.zeros((h, w, 3), dtype=np.uint16)
        bgr[:, :, 0] = 100  # B channel
        bgr[:, :, 1] = 200  # G channel
        bgr[:, :, 2] = 300  # R channel
        path = os.path.join(str(tmp_path), "swap_test.raw")
        bgr.astype("<u2").tofile(path)

        from negpy.infrastructure.loaders.noritsu_loader import NoritsuLoader

        loader = NoritsuLoader()
        wrapper, metadata = loader.load(path)
        with wrapper as w_obj:
            f32 = w_obj.data
            r_val = f32[0, 0, 0]
            g_val = f32[0, 0, 1]
            b_val = f32[0, 0, 2]
            assert r_val > g_val > b_val

    def test_loader_returns_float32(self, tmp_path: str) -> None:
        from negpy.infrastructure.loaders.noritsu_loader import NoritsuLoader

        path = _make_noritsu_raw(str(tmp_path))
        loader = NoritsuLoader()
        wrapper, metadata = loader.load(path)
        with wrapper as w:
            assert w.data.dtype == np.float32
            assert w.data.shape == (6391, 4042, 3)
            assert w.data.min() >= 0.0
            assert w.data.max() <= 1.0
        assert metadata["orientation"] == 0
        assert metadata["ir"] is None

    def test_all_known_dims_unique_sizes(self) -> None:
        """Every known dimension pair must produce a unique file size."""
        sizes = [w * h * 6 for w, h in KNOWN_NORITSU_DIMS]
        assert len(sizes) == len(set(sizes))

    def test_small_dims_detected(self, tmp_path: str) -> None:
        path = _make_noritsu_raw_small(str(tmp_path))
        assert is_noritsu_raw(path)

    def test_novel_width_known_height_detected(self, tmp_path: str) -> None:
        """Tier 2: a new width (not in the table) under a known height resolves."""
        novel_w, h = 5555, 5028
        assert (novel_w, h) not in KNOWN_NORITSU_DIMS
        rng = np.random.RandomState(99)
        bgr = rng.randint(0, 4096, size=(h, novel_w, 3), dtype=np.uint16)
        path = os.path.join(str(tmp_path), "novel.raw")
        bgr.astype("<u2").tofile(path)
        dims = detect_noritsu_dims(path)
        assert dims == (novel_w, h)
        assert is_noritsu_raw(path)

    def test_ambiguous_heights_not_detected(self, tmp_path: str) -> None:
        """If a file size divides evenly by multiple known heights, reject it."""
        from math import lcm

        common = lcm(KNOWN_NORITSU_HEIGHTS[0], KNOWN_NORITSU_HEIGHTS[1])
        size = common * 6
        data = np.zeros(size, dtype=np.uint8)
        path = os.path.join(str(tmp_path), "ambiguous.raw")
        data.tofile(path)
        assert detect_noritsu_dims(path) is None
        assert not is_noritsu_raw(path)

    def test_tier3_novel_height_detected(self, tmp_path: str) -> None:
        """Tier 3: a file whose dimensions match no known height but has
        exactly one film-plausible divisor pair resolves.

        4001 is prime, so 4001×4001 is the only (w,h) where both are in
        the scan-width range — the open divisor search finds it uniquely.
        """
        w, h = 4001, 4001
        assert h not in KNOWN_NORITSU_HEIGHTS
        assert (w, h) not in KNOWN_NORITSU_DIMS
        data = np.zeros(w * h * 3, dtype=np.uint16)
        path = os.path.join(str(tmp_path), "tier3.raw")
        data.astype("<u2").tofile(path)
        dims = detect_noritsu_dims(path)
        assert dims == (w, h)

    def test_tier3_ambiguous_rejected(self, tmp_path: str) -> None:
        """Tier 3: if multiple film-plausible divisor pairs exist, reject."""
        w, h = 4000, 6000
        assert h not in KNOWN_NORITSU_HEIGHTS
        data = np.zeros(w * h * 3, dtype=np.uint16)
        path = os.path.join(str(tmp_path), "ambiguous_t3.raw")
        data.astype("<u2").tofile(path)
        assert detect_noritsu_dims(path) is None

    def test_pakon_noritsu_tier1_sizes_disjoint(self) -> None:
        """No known Noritsu dimension pair produces a file size within Pakon's tolerance."""
        from negpy.infrastructure.loaders.pakon_loader import PakonLoader

        for w, h in KNOWN_NORITSU_DIMS:
            noritsu_size = w * h * 6
            for spec in PakonLoader.PAKON_SPECS:
                assert abs(noritsu_size - spec["size"]) >= 1024, (
                    f"Noritsu {w}x{h} ({noritsu_size}) collides with Pakon {spec['desc']} ({spec['size']})"
                )

    def test_pakon_noritsu_tier2_no_collision_above_min_scan_width(self) -> None:
        """No Noritsu tier-2 file at a real scan width lands in Pakon's tolerance.

        The narrowest known Noritsu scan width is 3551. Checks every known height
        x every width from 3000-15000 (generous margin below the minimum).
        Widths below 3000 are implausible for any real film scan.
        """
        from negpy.infrastructure.loaders.pakon_loader import PakonLoader

        min_scan_width = 3000
        collisions: list[str] = []
        for h in KNOWN_NORITSU_HEIGHTS:
            for w in range(min_scan_width, 15001):
                noritsu_size = w * h * 6
                for spec in PakonLoader.PAKON_SPECS:
                    if abs(noritsu_size - spec["size"]) < 1024:
                        collisions.append(f"{w}x{h} ({noritsu_size}) vs Pakon {spec['desc']} ({spec['size']})")
        assert not collisions, f"Collisions at real scan widths: {collisions}"

    def test_pakon_noritsu_theoretical_collision_documented(self) -> None:
        """Document the known theoretical collision at width 1777 (not a real scan width).

        Height 4502, width 1777 produces 48,000,324 bytes — within Pakon's
        48M ± 1024 window. This is harmless because: (a) 1777 is far below
        any real Noritsu scan width (min known: 3551), and (b) Pakon checks
        first in the factory, so this size would be claimed as Pakon.
        """
        from negpy.infrastructure.loaders.pakon_loader import PakonLoader

        collision_size = 1777 * 4502 * 6
        pakon_48m = next(s for s in PakonLoader.PAKON_SPECS if s["size"] == 48000000)
        assert abs(collision_size - pakon_48m["size"]) < 1024
        assert 1777 < 3000  # well below any real scan width


class TestHdrDecodeIsFormatAgnostic:
    """A bracket must merge whatever the source format.

    The HDR branch used to sit inside the camera-RAW arm of `_decode_linear`, after every
    per-format branch that returns. A bracket of TIFF or LinearRaw DNG scans therefore
    exported as its unmerged reference frame — the canvas showed the merge, the file did
    not, and nothing reported it.
    """

    def _bracket(self, tmp_path: str):
        from negpy.features.hdr.models import HdrConfig

        # The left half is black on the reference and just above it on the 4x frame: the
        # shadow detail only the longer exposure recorded, which is the whole point of the
        # merge and the one thing a passthrough cannot produce. The right half is exactly
        # proportional, so it alone would not tell a merge from an unmerged decode.
        ref = np.zeros((8, 8, 3), dtype=np.uint16)
        ref[:, 4:] = 4000
        long_ = np.full((8, 8, 3), 400, dtype=np.uint16)
        long_[:, 4:] = 16000
        paths = []
        for name, data in (("ref.tif", ref), ("long.tif", long_)):
            path = os.path.join(str(tmp_path), name)
            tifffile.imwrite(path, data)
            paths.append(path)
        return paths, HdrConfig(hdr_enabled=True, hdr_paths=(paths[1],), hdr_ratios=(1.0, 4.0), hdr_align=False)

    def test_a_tiff_bracket_decodes_merged(self, tmp_path: str) -> None:
        from negpy.services.export.linear_output import _decode_linear

        paths, hdr = self._bracket(tmp_path)
        unmerged, _, _, _ = _decode_linear(paths[0])
        merged, ir, _, _ = _decode_linear(paths[0], hdr=hdr)

        assert merged.shape == unmerged.shape
        assert ir is None
        assert float(unmerged[:, :4].max()) == 0.0
        # Inverse-variance weighting (weight ~ ratio^2) puts the long frame 16:1 ahead
        # there, so the shadow arrives at ~15/16 of its converted level.
        assert float(merged[:, :4].min()) > 0.9 * (400.0 / 65535.0) / 4.0, "the bracket decoded unmerged"
        assert float(merged.max()) <= 1.0
