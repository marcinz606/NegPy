from __future__ import annotations

import logging

import numpy as np
import pytest
import tifffile

from negpy.infrastructure.loaders.tiff_loader import TiffLoader


def _write_rgb_ir_pair(tmp_path, *, ir: np.ndarray) -> str:
    rgb_path = tmp_path / "frame003.tif"
    tifffile.imwrite(
        rgb_path,
        np.full((*ir.shape, 3), 30000, dtype=np.uint16),
        photometric="rgb",
    )
    tifffile.imwrite(tmp_path / "frame003_IR.tif", ir, photometric="minisblack")
    return str(rgb_path)


def test_valid_ir_mask_is_exposed_and_invalid_pixels_are_forced_to_sensor_white(tmp_path) -> None:
    ir = np.full((6, 8), 32768, dtype=np.uint16)
    ir[2, 3] = 0
    valid = np.ones(ir.shape, dtype=np.uint8) * 255
    valid[2, 3] = 0
    rgb_path = _write_rgb_ir_pair(tmp_path, ir=ir)
    tifffile.imwrite(
        tmp_path / "frame003_IR_VALID.tif",
        valid,
        photometric="minisblack",
    )

    _context, metadata = TiffLoader().load(rgb_path)

    assert metadata["ir_valid_mask"].dtype == np.bool_
    np.testing.assert_array_equal(metadata["ir_valid_mask"], valid.astype(bool))
    assert metadata["ir"][2, 3] == 1.0
    assert metadata["ir"][0, 0] == np.float32(32768.0 / 65535.0)


def test_ir_sidecar_without_validity_mask_preserves_existing_behavior(tmp_path) -> None:
    ir = np.full((5, 7), 50000, dtype=np.uint16)
    ir[1, 2] = 0
    rgb_path = _write_rgb_ir_pair(tmp_path, ir=ir)

    _context, metadata = TiffLoader().load(rgb_path)

    assert metadata["ir_valid_mask"] is None
    assert metadata["ir"][1, 2] == 0.0
    assert metadata["ir"][0, 0] == np.float32(50000.0 / 65535.0)


def test_lowercase_ir_sidecar_is_detected(tmp_path) -> None:
    """Scanner tools (nkscan) emit lowercase `_ir.tiff`; the loader must still pair it."""
    ir = np.full((5, 7), 50000, dtype=np.uint16)
    ir[1, 2] = 0
    rgb_path = tmp_path / "frame.tif"
    tifffile.imwrite(rgb_path, np.full((*ir.shape, 3), 30000, dtype=np.uint16), photometric="rgb")
    tifffile.imwrite(tmp_path / "frame_ir.tiff", ir, photometric="minisblack")

    _context, metadata = TiffLoader().load(str(rgb_path))

    assert metadata["ir"] is not None
    assert metadata["ir_valid_mask"] is None
    assert metadata["ir"][1, 2] == 0.0
    assert metadata["ir"][0, 0] == np.float32(50000.0 / 65535.0)


def test_lowercase_ir_valid_mask_pairs_with_lowercase_sidecar(tmp_path) -> None:
    ir = np.full((6, 8), 32768, dtype=np.uint16)
    ir[2, 3] = 0
    valid = np.ones(ir.shape, dtype=np.uint8) * 255
    valid[2, 3] = 0
    rgb_path = tmp_path / "frame.tif"
    tifffile.imwrite(rgb_path, np.full((*ir.shape, 3), 30000, dtype=np.uint16), photometric="rgb")
    tifffile.imwrite(tmp_path / "frame_ir.tiff", ir, photometric="minisblack")
    tifffile.imwrite(tmp_path / "frame_ir_valid.tiff", valid, photometric="minisblack")

    _context, metadata = TiffLoader().load(str(rgb_path))

    assert metadata["ir_valid_mask"].dtype == np.bool_
    np.testing.assert_array_equal(metadata["ir_valid_mask"], valid.astype(bool))
    assert metadata["ir"][2, 3] == 1.0


@pytest.mark.parametrize("failure", ("mismatched", "malformed"))
def test_invalid_validity_mask_ignores_ir_sidecar_fail_closed(
    tmp_path,
    caplog,
    failure: str,
) -> None:
    ir = np.full((5, 7), 1200, dtype=np.uint16)
    rgb_path = _write_rgb_ir_pair(tmp_path, ir=ir)
    mask_path = tmp_path / "frame003_IR_VALID.tif"
    if failure == "mismatched":
        tifffile.imwrite(
            mask_path,
            np.ones((4, 7), dtype=np.uint8) * 255,
            photometric="minisblack",
        )
    else:
        mask_path.write_bytes(b"not a TIFF")

    with caplog.at_level(logging.WARNING):
        _context, metadata = TiffLoader().load(rgb_path)

    assert metadata["ir"] is None
    assert metadata["ir_valid_mask"] is None
    assert "ignoring IR sidecar" in caplog.text


@pytest.mark.parametrize(
    "invalid_mask",
    (
        np.ones((5, 7), dtype=np.float32),
        np.ones((5, 7), dtype=np.uint16) * 255,
        np.full((5, 7), 2, dtype=np.uint8),
    ),
)
def test_invalid_validity_mask_dtype_or_domain_ignores_ir_fail_closed(
    tmp_path,
    caplog,
    invalid_mask: np.ndarray,
) -> None:
    ir = np.full((5, 7), 1200, dtype=np.uint16)
    rgb_path = _write_rgb_ir_pair(tmp_path, ir=ir)
    tifffile.imwrite(
        tmp_path / "frame003_IR_VALID.tif",
        invalid_mask,
        photometric="minisblack",
    )

    with caplog.at_level(logging.WARNING):
        _context, metadata = TiffLoader().load(rgb_path)

    assert metadata["ir"] is None
    assert metadata["ir_valid_mask"] is None
    assert "ignoring IR sidecar" in caplog.text


@pytest.mark.parametrize("extrasamples,expect_ir", [(2, False), (0, True)])
def test_a_two_sample_gray_tiff_loads_as_gray(tmp_path, extrasamples, expect_ir) -> None:
    gray = np.full((6, 8), 20000, dtype=np.uint16)
    extra = np.full((6, 8), 40000, dtype=np.uint16)
    path = tmp_path / "gray2.tif"
    tifffile.imwrite(path, np.stack([gray, extra], axis=-1), photometric="minisblack", extrasamples=[extrasamples])

    wrapper, metadata = TiffLoader().load(str(path), linear_raw=True)

    assert wrapper.data.shape == (6, 8, 3)
    np.testing.assert_allclose(wrapper.data, np.float32(20000 / 65535), rtol=1e-6)
    assert (metadata["ir"] is not None) is expect_ir


def test_a_planar_tiff_loads_with_samples_last(tmp_path) -> None:
    rgb = np.zeros((3, 40, 60), dtype=np.uint16)
    rgb[0], rgb[1], rgb[2] = 10000, 20000, 30000
    path = tmp_path / "planar.tif"
    tifffile.imwrite(path, rgb, photometric="rgb", planarconfig="separate")

    wrapper, _ = TiffLoader().load(str(path), linear_raw=True)

    assert wrapper.data.shape == (40, 60, 3)
    np.testing.assert_allclose(wrapper.data[5, 5], np.array([10000, 20000, 30000]) / 65535, rtol=1e-6)


def test_linear_output_reads_a_planar_tiff_with_samples_last(tmp_path) -> None:
    from negpy.services.export.linear_output import _decode_tiff

    rgb = np.full((3, 40, 60), 20000, dtype=np.uint16)
    path = tmp_path / "planar.tif"
    tifffile.imwrite(path, rgb, photometric="rgb", planarconfig="separate")

    out, _ = _decode_tiff(str(path))

    assert out.shape[:2] == (40, 60)


def test_a_miniswhite_tiff_reads_as_intensity(tmp_path) -> None:
    gray = np.full((6, 8), 10000, dtype=np.uint16)
    path = tmp_path / "white.tif"
    tifffile.imwrite(path, gray, photometric="miniswhite")
    wrapper, _ = TiffLoader().load(str(path), linear_raw=True)
    np.testing.assert_allclose(wrapper.data, np.float32((65535 - 10000) / 65535), rtol=1e-6)


def test_a_palette_tiff_expands_its_colormap(tmp_path) -> None:
    index = np.zeros((6, 8), dtype=np.uint8)
    index[:, 4:] = 1
    colormap = np.zeros((3, 256), dtype=np.uint16)
    colormap[:, 1] = (65535, 0, 32768)
    path = tmp_path / "palette.tif"
    tifffile.imwrite(path, index, photometric="palette", colormap=colormap)
    wrapper, _ = TiffLoader().load(str(path), linear_raw=True)
    np.testing.assert_allclose(wrapper.data[0, 5], [1.0, 0.0, 32768 / 65535], rtol=1e-6)
    np.testing.assert_allclose(wrapper.data[0, 0], [0.0, 0.0, 0.0])


def test_a_cmyk_tiff_converts_to_rgb_without_an_ir_plane(tmp_path) -> None:
    cmyk = np.zeros((6, 8, 4), dtype=np.uint8)
    cmyk[..., 0] = 255  # full cyan
    path = tmp_path / "cmyk.tif"
    tifffile.imwrite(path, cmyk, photometric="separated")
    wrapper, metadata = TiffLoader().load(str(path), linear_raw=True)
    np.testing.assert_allclose(wrapper.data[0, 0], [0.0, 1.0, 1.0], atol=1e-6)
    assert metadata["ir"] is None


def test_a_cmyk_jpeg_loads_its_colors(tmp_path) -> None:
    from PIL import Image

    from negpy.infrastructure.loaders.jpeg_loader import JpegLoader

    path = tmp_path / "cmyk.jpg"
    Image.new("CMYK", (8, 8), (0, 255, 255, 0)).save(path, quality=100)  # red
    wrapper, _ = JpegLoader().load(str(path))
    r, g, b = wrapper.data[4, 4]
    assert r > 0.5 and g < 0.1 and b < 0.1
