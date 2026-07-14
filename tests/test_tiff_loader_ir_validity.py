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
