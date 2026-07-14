"""Tests for TIFF and DNG output writers."""

import json
import os
import tempfile

import numpy as np
import pytest
import tifffile

from negpy.infrastructure.scanners.result import ScanResult, SplitSourceCapture
from negpy.services.scanning.writer import (
    write_dng_linear,
    write_full_negative_tiff,
    write_split_source_bundle,
    write_tiff_16bit,
)


class TestTiffWriter:
    def test_writes_16bit_tiff(self) -> None:
        rgb = np.random.randint(0, 65535, (200, 300, 3), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=None, dpi=3600, device_model="TestScanner")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "test_scan"))
            assert os.path.exists(path)
            assert path.endswith(".tif")

            # Round-trip readback
            readback = tifffile.imread(path)
            assert readback.shape == (200, 300, 3)
            assert readback.dtype == np.uint16

    def test_writes_ir_sidecar(self) -> None:
        rgb = np.random.randint(0, 65535, (100, 150, 3), dtype=np.uint16)
        ir = np.random.randint(0, 65535, (100, 150), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=ir, dpi=3600, device_model="TestScanner")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "test_ir"))
            ir_path = path.replace(".tif", "_IR.tif")
            assert os.path.exists(path)
            assert os.path.exists(ir_path)

            ir_readback = tifffile.imread(ir_path)
            assert ir_readback.shape == (100, 150)

    def test_adds_tif_extension(self) -> None:
        rgb = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = ScanResult(rgb=rgb, ir=None, dpi=300, device_model="T")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "noext"))
            assert path.endswith(".tif")

    def test_converts_non_uint16(self) -> None:
        rgb = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = ScanResult(rgb=rgb, ir=None, dpi=300, device_model="T")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "test8"))
            readback = tifffile.imread(path)
            assert readback.dtype == np.uint16


def test_split_source_bundle_commits_all_planes_with_manifest_evidence(tmp_path) -> None:
    rgb4x = np.arange(7 * 5 * 3, dtype=np.uint16).reshape(7, 5, 3)
    rgb1x_proxy = (rgb4x[:-1] + 100).copy()
    ir1x = np.arange(6 * 5, dtype=np.uint16).reshape(6, 5)
    aligned_ir = np.pad(ir1x, ((0, 1), (0, 0)), constant_values=123)
    ir_valid_mask = np.ones((7, 5), dtype=bool)
    ir_valid_mask[-1] = False

    manifest = write_split_source_bundle(
        SplitSourceCapture(
            rgb4x=rgb4x,
            rgb1x_proxy=rgb1x_proxy,
            ir1x=ir1x,
        ),
        aligned_ir=aligned_ir,
        ir_valid_mask=ir_valid_mask,
        output_dir=tmp_path,
        dpi=4000,
    )

    assert manifest["version"] == 1
    assert manifest["kind"] == "negpy.full-negative-split-source"
    assert len(manifest["content_sha256"]) == 64
    assert set(manifest["artifacts"]) == {
        "rgb4x",
        "rgb1x_proxy",
        "ir1x",
        "aligned_ir",
        "ir_valid_mask",
    }
    bundle_path = tmp_path / manifest["bundle_path"]
    assert (bundle_path / "manifest.json").is_file()
    for role, evidence in manifest["artifacts"].items():
        artifact_path = bundle_path / evidence["path"]
        assert artifact_path.is_file(), role
        assert len(evidence["sha256"]) == 64
        assert evidence["bytes"] == artifact_path.stat().st_size
        assert evidence["page_count"] == 1
        assert evidence["payload_within_file"] is True
        assert evidence["x_resolution"] == [4000, 1]
        assert evidence["y_resolution"] == [4000, 1]
        assert evidence["resolution_unit"] == "INCH"
    assert manifest["artifacts"]["rgb4x"]["shape"] == [7, 5, 3]
    assert manifest["artifacts"]["rgb4x"]["dtype"] == "uint16"
    assert manifest["artifacts"]["ir_valid_mask"]["shape"] == [7, 5]
    assert manifest["artifacts"]["ir_valid_mask"]["dtype"] == "uint8"


def test_split_source_bundle_repeat_is_content_addressed_and_idempotent(tmp_path) -> None:
    source = SplitSourceCapture(
        rgb4x=np.full((7, 5, 3), 1000, dtype=np.uint16),
        rgb1x_proxy=np.full((6, 5, 3), 1100, dtype=np.uint16),
        ir1x=np.full((6, 5), 1200, dtype=np.uint16),
    )
    aligned_ir = np.full((7, 5), 1300, dtype=np.uint16)
    valid = np.ones((7, 5), dtype=bool)

    first = write_split_source_bundle(
        source,
        aligned_ir=aligned_ir,
        ir_valid_mask=valid,
        output_dir=tmp_path,
        dpi=4000,
    )
    bundle_path = tmp_path / first["bundle_path"]
    mtimes = {path.name: path.stat().st_mtime_ns for path in bundle_path.iterdir()}

    second = write_split_source_bundle(
        source,
        aligned_ir=aligned_ir,
        ir_valid_mask=valid,
        output_dir=tmp_path,
        dpi=4000,
    )

    assert second == first
    assert [path.name for path in tmp_path.iterdir()] == [first["bundle_path"]]
    assert {path.name: path.stat().st_mtime_ns for path in bundle_path.iterdir()} == mtimes


def test_split_source_bundle_never_overwrites_a_conflicting_commit(tmp_path) -> None:
    source = SplitSourceCapture(
        rgb4x=np.full((7, 5, 3), 2000, dtype=np.uint16),
        rgb1x_proxy=np.full((6, 5, 3), 2100, dtype=np.uint16),
        ir1x=np.full((6, 5), 2200, dtype=np.uint16),
    )
    aligned_ir = np.full((7, 5), 2300, dtype=np.uint16)
    valid = np.ones((7, 5), dtype=bool)
    committed = write_split_source_bundle(
        source,
        aligned_ir=aligned_ir,
        ir_valid_mask=valid,
        output_dir=tmp_path,
        dpi=4000,
    )
    bundle_path = tmp_path / committed["bundle_path"]
    manifest_path = bundle_path / "manifest.json"
    conflicting = json.loads(manifest_path.read_text(encoding="utf-8"))
    conflicting["content_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(conflicting), encoding="utf-8")
    artifact_bytes = {path.name: path.read_bytes() for path in bundle_path.iterdir() if path.name != "manifest.json"}

    with pytest.raises(RuntimeError, match="conflicting content"):
        write_split_source_bundle(
            source,
            aligned_ir=aligned_ir,
            ir_valid_mask=valid,
            output_dir=tmp_path,
            dpi=4000,
        )

    assert json.loads(manifest_path.read_text(encoding="utf-8"))["content_sha256"] == "0" * 64
    assert {path.name: path.read_bytes() for path in bundle_path.iterdir() if path.name != "manifest.json"} == artifact_bytes


def test_split_source_bundle_refuses_a_tampered_committed_artifact(tmp_path) -> None:
    source = SplitSourceCapture(
        rgb4x=np.full((7, 5, 3), 3000, dtype=np.uint16),
        rgb1x_proxy=np.full((6, 5, 3), 3100, dtype=np.uint16),
        ir1x=np.full((6, 5), 3200, dtype=np.uint16),
    )
    aligned_ir = np.full((7, 5), 3300, dtype=np.uint16)
    valid = np.ones((7, 5), dtype=bool)
    committed = write_split_source_bundle(
        source,
        aligned_ir=aligned_ir,
        ir_valid_mask=valid,
        output_dir=tmp_path,
        dpi=4000,
    )
    bundle_path = tmp_path / committed["bundle_path"]
    rgb_path = bundle_path / committed["artifacts"]["rgb4x"]["path"]
    tifffile.imwrite(
        rgb_path,
        np.zeros((7, 5, 3), dtype=np.uint16),
        photometric="rgb",
        resolution=(4000, 4000),
        resolutionunit="INCH",
    )
    tampered_bytes = rgb_path.read_bytes()

    with pytest.raises(RuntimeError, match="artifact evidence"):
        write_split_source_bundle(
            source,
            aligned_ir=aligned_ir,
            ir_valid_mask=valid,
            output_dir=tmp_path,
            dpi=4000,
        )

    assert rgb_path.read_bytes() == tampered_bytes


def test_full_negative_writer_commits_rgb_sanitized_ir_and_validity_evidence(tmp_path) -> None:
    rgb = np.arange(7 * 5 * 3, dtype=np.uint16).reshape(7, 5, 3)
    ir = np.arange(7 * 5, dtype=np.uint16).reshape(7, 5)
    valid = np.ones((7, 5), dtype=bool)
    valid[0, 0] = False
    valid[-1] = False

    evidence = write_full_negative_tiff(
        ScanResult(rgb=rgb, ir=ir, dpi=4000, device_model="Nikon LS-5000"),
        ir_valid_mask=valid,
        path=tmp_path / "frame003.tif",
    )

    assert set(evidence) == {"rgb", "ir", "ir_valid_mask"}
    for role, artifact in evidence.items():
        artifact_path = tmp_path / artifact["path"]
        assert artifact_path.is_file(), role
        assert len(artifact["sha256"]) == 64
        assert artifact["bytes"] == artifact_path.stat().st_size
        assert artifact["x_resolution"] == [4000, 1]
        assert artifact["y_resolution"] == [4000, 1]
        assert artifact["resolution_unit"] == "INCH"
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "frame003.tif"), rgb)
    saved_ir = tifffile.imread(tmp_path / "frame003_IR.tif")
    np.testing.assert_array_equal(saved_ir[valid], ir[valid])
    assert np.all(saved_ir[~valid] == np.iinfo(np.uint16).max)
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "frame003_IR_VALID.tif"),
        valid.astype(np.uint8) * 255,
    )
    assert evidence["rgb"]["shape"] == [7, 5, 3]
    assert evidence["ir"]["shape"] == [7, 5]
    assert evidence["ir_valid_mask"]["dtype"] == "uint8"


class TestDngWriter:
    def test_writes_linear_dng(self) -> None:
        rgb = np.random.randint(0, 65535, (200, 300, 3), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=None, dpi=3600, device_model="TestScanner")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dng_linear(result, os.path.join(tmpdir, "test_scan"))
            assert os.path.exists(path)
            assert path.endswith(".dng")

            readback = tifffile.imread(path)
            assert readback.shape == (200, 300, 3)
            assert readback.dtype == np.uint16
            np.testing.assert_array_equal(readback, rgb)

            with tifffile.TiffFile(path) as tf:
                tags = tf.pages[0].tags
                assert int(tags["PhotometricInterpretation"].value) == 34892  # LinearRaw
                assert tuple(tags["DNGVersion"].value) == (1, 4, 0, 0)
                assert int(tags["SamplesPerPixel"].value) == 3
                # 3 plain colour samples, no ExtraSamples (matches pidng); marking colour
                # planes as extra makes some raw processors mis-demosaic the file.
                assert tags.get("ExtraSamples") is None

    def test_writes_dng_with_ir(self) -> None:
        rgb = np.random.randint(0, 65535, (100, 150, 3), dtype=np.uint16)
        ir = np.random.randint(0, 65535, (100, 150), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=ir, dpi=3600, device_model="TestScanner")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dng_linear(result, os.path.join(tmpdir, "test_ir"))
            assert os.path.exists(path)

            readback = tifffile.imread(path)
            assert readback.shape == (100, 150, 4)
            np.testing.assert_array_equal(readback[:, :, :3], rgb)
            np.testing.assert_array_equal(readback[:, :, 3], ir)
            with tifffile.TiffFile(path) as tf:
                assert int(tf.pages[0].tags["SamplesPerPixel"].value) == 4

    def test_adds_dng_extension(self) -> None:
        rgb = np.random.randint(0, 65535, (50, 50, 3), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=None, dpi=300, device_model="T")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dng_linear(result, os.path.join(tmpdir, "noext"))
            assert path.endswith(".dng")
