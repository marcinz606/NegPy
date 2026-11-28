import numpy as np
import pytest
import rawpy
import tifffile

from negpy.infrastructure.capture.raw_demosaic import _user_sat
from negpy.infrastructure.loaders.factory import loader_factory
from negpy.infrastructure.loaders.helpers import camera_wb_multipliers, camera_xyz_matrix, is_scanned_dng
from negpy.services.capture import pixel_shift_dng
from negpy.services.capture.pixel_shift_dng import DngVerifyError, RawColor, raw_color, write_linear_dng

COLOR = RawColor(
    black=(512.0, 512.0, 512.0),
    white=16383,
    color_matrix=(0.82, -0.2976, -0.0719, -0.4296, 1.2053, 0.2532, -0.0429, 0.1282, 0.5774),
    as_shot_neutral=(0.4, 1.0, 0.6),
    make="SONY",
    model="ILCE-7RM5",
)


def _image():
    # Well under the white level: a brightest pixel at white hides a wrong white level.
    return np.random.default_rng(0).integers(600, 8000, (64, 96, 3), np.uint16)


def _decode(raw):
    return raw.postprocess(
        output_color=rawpy.ColorSpace.raw,
        gamma=(1, 1),
        no_auto_bright=True,
        output_bps=16,
        user_wb=[1, 1, 1, 1],
        adjust_maximum_thr=0.0,  # as NegPy decodes: the file's white level, never the frame's maximum
        user_sat=_user_sat(raw),
    )


def _scaled(rgb):
    black = np.array(COLOR.black)
    return (rgb - black) / (COLOR.white - black) * 65535


def test_libraw_reads_pixels_and_color(tmp_path):
    rgb, path = _image(), str(tmp_path / "merged.dng")
    write_linear_dng(rgb, COLOR, path, 4)
    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == 1
        assert int(tif.pages[0].photometric) == 34892
        assert 338 not in tif.pages[0].tags
    np.testing.assert_array_equal(tifffile.imread(path), rgb)
    with rawpy.imread(path) as raw:
        np.testing.assert_allclose(_decode(raw), _scaled(rgb), atol=1)
        np.testing.assert_allclose(camera_xyz_matrix(raw), np.reshape(COLOR.color_matrix, (3, 3)), atol=1e-3)
        np.testing.assert_allclose(camera_wb_multipliers(raw, path), (2.5, 1.0, 1 / 0.6), rtol=1e-3)
    assert list(tmp_path.iterdir()) == [tmp_path / "merged.dng"]


def test_negpy_loader_opens_it_as_a_camera_file(tmp_path):
    rgb, path = _image(), str(tmp_path / "merged.dng")
    write_linear_dng(rgb, COLOR, path, 4)
    assert not is_scanned_dng(path)
    context, _ = loader_factory.get_loader(path)
    with context as raw:
        np.testing.assert_allclose(_decode(raw), _scaled(rgb), atol=1)


def test_failed_read_back_leaves_no_file(tmp_path, monkeypatch):
    rgb = _image()
    monkeypatch.setattr(pixel_shift_dng.tifffile, "memmap", lambda _path, mode: rgb + 1)
    with pytest.raises(DngVerifyError):
        write_linear_dng(rgb, COLOR, str(tmp_path / "merged.dng"), 4)
    assert not list(tmp_path.iterdir())


class _FakeRaw:
    black_level_per_channel = [512, 512, 510, 513]
    white_level = 16383
    camera_white_level_per_channel = None
    camera_whitebalance = [2560.0, 1024.0, 1280.0, 1024.0]
    rgb_xyz_matrix = np.array([*np.reshape(COLOR.color_matrix, (3, 3)), [0, 0, 0]])

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_raw_color_of_a_camera_file(tmp_path, monkeypatch):
    source = tmp_path / "shot.arw"
    tifffile.imwrite(source, np.zeros((8, 8), np.uint16), extratags=[(271, "s", 0, "SONY", True), (272, "s", 0, "ILCE-7RM5", True)])
    monkeypatch.setattr(pixel_shift_dng.rawpy, "imread", lambda _path: _FakeRaw())
    color = raw_color(str(source))
    assert color.black == (512.0, 512.5, 510.0)
    assert color.white == 16383
    assert color.color_matrix == pytest.approx(COLOR.color_matrix)
    assert color.as_shot_neutral == pytest.approx((0.4, 1.0, 0.8))
    assert (color.make, color.model) == ("SONY", "ILCE-7RM5")


def test_the_rgb_frame_probe_reads_a_merged_file(tmp_path):
    """A merged set has no mosaic; the probe that groups R, G and B exposures must still name its color."""
    from negpy.features.rgbscan.logic import classify_channel, probe_frame

    rgb = np.random.default_rng(3).integers(600, 900, (64, 96, 3)).astype(np.uint16)
    rgb[..., 2] += 4000  # a frame lit blue
    path = str(tmp_path / "blue.dng")
    write_linear_dng(rgb, COLOR, path, 4)
    probe = probe_frame(path)
    assert classify_channel(probe.means) == 2
    assert probe.signature.ndim == 2


def test_the_info_panel_reads_what_the_frame_was_shot_with(tmp_path):
    from dataclasses import replace

    from negpy.features.metadata.exif_read import extract_scan_from_exif
    from negpy.infrastructure.loaders.helpers import read_exif_from_file

    shot = replace(
        COLOR,
        exposure_time=(1, 30),
        f_number=(56, 10),
        iso=100,
        focal_length=(90, 1),
        lens_model="FE 90mm F2.8 Macro G OSS",
        datetime="2026:11:27 10:00:00",
    )
    path = str(tmp_path / "frame.dng")
    write_linear_dng(_image(), shot, path, 4)
    scan = extract_scan_from_exif(read_exif_from_file(path))
    assert (scan.exposure, scan.iso, scan.aperture, scan.focal_length_mm) == ("1/30s  f/5.6  ISO 100", 100, 5.6, 90.0)
    assert scan.lens_model == "FE 90mm F2.8 Macro G OSS" and scan.camera_model == "ILCE-7RM5"
    assert scan.datetime_original
    with rawpy.imread(path) as raw:  # the appended block leaves the pixels readable
        np.testing.assert_array_equal(raw.raw_image_visible[..., :3], _image() - 512)  # less the black level


def test_a_manual_lens_leaves_no_aperture_entry(tmp_path):
    from negpy.features.metadata.exif_read import extract_scan_from_exif
    from negpy.infrastructure.loaders.helpers import read_exif_from_file

    path = str(tmp_path / "frame.dng")
    write_linear_dng(_image(), COLOR, path, 4)
    scan = extract_scan_from_exif(read_exif_from_file(path))
    assert scan.aperture is None and scan.exposure == ""


def test_a_merged_file_names_its_shot_count(tmp_path):
    from negpy.services.capture.pixel_shift_dng import merged_shots

    merged, other, plain = str(tmp_path / "m.dng"), str(tmp_path / "o.dng"), str(tmp_path / "p.ARW")
    write_linear_dng(_image(), COLOR, merged, 16)
    tifffile.imwrite(other, _image(), software="a camera")
    open(plain, "w").close()
    assert [merged_shots(p) for p in (merged, other, plain, str(tmp_path / "missing.dng"))] == [16, 0, 0, 0]
