from __future__ import annotations

from typing import cast

import numpy as np
import tifffile

from negpy.infrastructure.loaders.tiff_loader import TiffLoader
from negpy.infrastructure.scanners.result import ScanResult
from negpy.infrastructure.tiff_contract import (
    LINEAR_SCANNER_RGB_EXTRATAG,
    LINEAR_SCANNER_RGB_MARKER,
    LINEAR_SCANNER_RGB_TAG,
)
from negpy.kernel.image.logic import srgb_to_linear
from negpy.services.scanning.writer import write_full_negative_tiff, write_tiff_16bit


def _load_pixels(path) -> tuple[np.ndarray, dict]:
    context, metadata = TiffLoader().load(str(path))
    with context as image:
        return image.data, metadata


def test_scanner_tiff_round_trip_preserves_all_uint16_codes_as_linear(tmp_path) -> None:
    codes = np.arange(1 << 16, dtype=np.uint16).reshape(256, 256)
    rgb = np.stack((codes, np.flip(codes, axis=0), np.flip(codes, axis=1)), axis=-1)
    path = write_tiff_16bit(
        ScanResult(rgb=rgb, ir=None, dpi=4000, device_model="Nikon LS-5000"),
        str(tmp_path / "frame001.tif"),
    )

    np.testing.assert_array_equal(tifffile.imread(path), rgb)
    with tifffile.TiffFile(path) as tif:
        page = cast(tifffile.TiffPage, tif.pages[0])
        assert page.tags[LINEAR_SCANNER_RGB_TAG].value == LINEAR_SCANNER_RGB_MARKER

    loaded, metadata = _load_pixels(path)

    # TiffLoader's normalized float representation must round-trip every
    # possible source code.  An accidental sRGB decode fails this assertion.
    reconstructed = np.rint(loaded * np.float32(65535.0)).astype(np.uint16)
    np.testing.assert_array_equal(reconstructed, rgb)
    assert metadata["sample_encoding"] == "linear-scanner-rgb"


def test_ordinary_untagged_tiff_keeps_legacy_srgb_decode(tmp_path) -> None:
    rgb = np.array(
        [
            [[0, 1, 1024], [8192, 32768, 65535]],
            [[65535, 40000, 20000], [10000, 5000, 1000]],
        ],
        dtype=np.uint16,
    )
    path = tmp_path / "ordinary.tif"
    tifffile.imwrite(path, rgb, photometric="rgb")

    loaded, metadata = _load_pixels(path)
    normalized = rgb.astype(np.float32) * np.float32(1.0 / 65535.0)

    np.testing.assert_allclose(loaded, srgb_to_linear(normalized), rtol=0, atol=1e-7)
    assert not np.array_equal(loaded, normalized)
    assert metadata["sample_encoding"] == "display-encoded"


def test_unknown_private_tag_payload_does_not_claim_linear_encoding(tmp_path) -> None:
    rgb = np.full((3, 4, 3), 32768, dtype=np.uint16)
    path = tmp_path / "unknown-contract.tif"
    unknown_tag = (
        LINEAR_SCANNER_RGB_TAG,
        "s",
        0,
        "negpy.scanner-rgb.linear.uint16.v999",
        False,
    )
    tifffile.imwrite(path, rgb, photometric="rgb", extratags=[unknown_tag])

    loaded, metadata = _load_pixels(path)
    normalized = rgb.astype(np.float32) * np.float32(1.0 / 65535.0)

    np.testing.assert_allclose(loaded, srgb_to_linear(normalized), rtol=0, atol=1e-7)
    assert metadata["sample_encoding"] == "display-encoded"


def test_uint8_file_with_linear_uint16_marker_fails_closed(tmp_path, caplog) -> None:
    rgb = np.full((3, 4, 3), 128, dtype=np.uint8)
    path = tmp_path / "wrong-sample-format.tif"
    tifffile.imwrite(path, rgb, photometric="rgb", extratags=[LINEAR_SCANNER_RGB_EXTRATAG])

    loaded, metadata = _load_pixels(path)
    normalized = rgb.astype(np.float32) * np.float32(1.0 / 255.0)

    np.testing.assert_allclose(loaded, srgb_to_linear(normalized), rtol=0, atol=1e-7)
    assert metadata["sample_encoding"] == "display-encoded"
    assert "Ignoring incompatible linear scanner RGB marker" in caplog.text


def test_full_negative_linear_rgb_contract_preserves_ir_pairing(tmp_path) -> None:
    rgb = np.arange(6 * 8 * 3, dtype=np.uint16).reshape(6, 8, 3) * np.uint16(313)
    ir = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8) * np.uint16(997)
    valid = np.ones((6, 8), dtype=np.bool_)
    valid[0, 0] = False
    valid[-1, -1] = False
    rgb_path = tmp_path / "frame003.tif"

    write_full_negative_tiff(
        ScanResult(rgb=rgb, ir=ir, dpi=4000, device_model="Nikon LS-5000"),
        ir_valid_mask=valid,
        path=rgb_path,
    )

    loaded, metadata = _load_pixels(rgb_path)
    reconstructed = np.rint(loaded * np.float32(65535.0)).astype(np.uint16)

    np.testing.assert_array_equal(reconstructed, rgb)
    np.testing.assert_array_equal(metadata["ir_valid_mask"], valid)
    expected_ir = ir.astype(np.float32) * np.float32(1.0 / 65535.0)
    expected_ir[~valid] = 1.0
    np.testing.assert_array_equal(metadata["ir"], expected_ir)
    assert metadata["sample_encoding"] == "linear-scanner-rgb"

    with tifffile.TiffFile(rgb_path) as tif:
        page = cast(tifffile.TiffPage, tif.pages[0])
        assert page.tags[LINEAR_SCANNER_RGB_TAG].value == LINEAR_SCANNER_RGB_MARKER
    for sidecar in (tmp_path / "frame003_IR.tif", tmp_path / "frame003_IR_VALID.tif"):
        with tifffile.TiffFile(sidecar) as tif:
            page = cast(tifffile.TiffPage, tif.pages[0])
            assert page.tags.get(LINEAR_SCANNER_RGB_TAG) is None
