"""Tests for the Linear Output export feature."""

import io
import os

import numpy as np
import pytest
import tifffile

from negpy.features.geometry.models import GeometryConfig
from negpy.kernel.image.logic import apply_exif_orientation
from negpy.services.export.linear_output import (
    _CameraWB,
    _SourceMeta,
    _build_xmp,
    _default_pakon_expansion,
    _effective_expansion,
    _is_camera_raw,
    _normalize_wb_rgb,
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

    def test_regular_tiff_not_supported(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "photo.tiff")
        arr = np.zeros((10, 10, 3), dtype=np.uint16)
        tifffile.imwrite(path, arr)
        assert not is_linear_output_supported(path)

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
        path = os.path.join(str(tmp_path), "photo.tiff")
        tifffile.imwrite(path, np.zeros((10, 10, 3), dtype=np.uint16))
        out = os.path.join(str(tmp_path), "out.tiff")
        with pytest.raises(ValueError, match="not supported"):
            export_linear_output(path, out)

    def test_creates_output_directory(self, tmp_path: str) -> None:
        raw_path = _make_pakon_raw(str(tmp_path))
        out_path = os.path.join(str(tmp_path), "subdir", "nested", "output.tiff")

        export_linear_output(raw_path, out_path)
        assert os.path.exists(out_path)


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
        r, g, b = _normalize_wb_rgb((398.0, 302.0, 873.0, 0.0))
        assert g == 1.0
        assert abs(r - 398.0 / 302.0) < 1e-6
        assert abs(b - 873.0 / 302.0) < 1e-6

    def test_build_xmp_maketiff_format(self) -> None:
        wb = _CameraWB(
            as_shot=(398.0, 302.0, 873.0, 0.0),
            daylight=(1.94, 0.94, 1.38, 0.0),
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
