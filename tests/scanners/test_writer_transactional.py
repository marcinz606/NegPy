"""Transactional-write tests for write_tiff_16bit: RGB+IR must land as a unit.

A failure writing the IR sidecar after the RGB payload already succeeded must
not leave an orphan RGB file with no indication that its IR channel is
missing. IR exists for dust detection (see TiffLoader._read_sidecar_ir); a
silently-orphaned RGB later paired with an unrelated, stale `_IR.tif`
sidecar would misattribute someone else's dust map to this frame — purely by
filename convention, with no shape/content cross-check.
"""

import os
import tempfile

import numpy as np
import pytest
import tifffile

from negpy.infrastructure.scanners.result import ScanResult
from negpy.services.scanning import writer
from negpy.services.scanning.writer import write_full_negative_tiff, write_tiff_16bit


def _result(with_ir: bool = True) -> ScanResult:
    rgb = np.random.randint(0, 65535, (20, 30, 3), dtype=np.uint16)
    ir = np.random.randint(0, 65535, (20, 30), dtype=np.uint16) if with_ir else None
    return ScanResult(rgb=rgb, ir=ir, dpi=1200, device_model="TestScanner")


class TestIrFailureAfterRgbSucceeds:
    """monkeypatch tifffile.imwrite to fail on the 2nd call (the IR write),
    simulating a disk/codec failure after the RGB payload already succeeded."""

    def test_no_orphan_rgb_left_when_ir_write_fails(self, monkeypatch) -> None:
        real_imwrite = tifffile.imwrite
        calls = {"n": 0}

        def flaky_imwrite(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("simulated disk failure writing IR sidecar")
            return real_imwrite(*args, **kwargs)

        monkeypatch.setattr(tifffile, "imwrite", flaky_imwrite)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "frame001")
            result = _result(with_ir=True)

            with pytest.raises(Exception):
                write_tiff_16bit(result, path)

            # Archival invariant: no silent RGB-without-IR. The API already
            # surfaces a clear failure (the raise above); on top of that, no
            # orphan RGB may be left looking like a complete, valid scan.
            rgb_path = path + ".tif"
            ir_path = path + "_IR.tif"
            assert not os.path.exists(ir_path)
            assert not os.path.exists(rgb_path), (
                "IR write failed after RGB succeeded but the orphan RGB file was left on disk with no indication its IR channel is missing"
            )
            # No temp files left behind either.
            assert os.listdir(tmpdir) == []

    def test_existing_pair_survives_ir_commit_failure(self, monkeypatch, tmp_path) -> None:
        """Replacing a complete scan must not sacrifice it for a partial update."""
        rgb_path = tmp_path / "frame.tif"
        ir_path = tmp_path / "frame_IR.tif"
        old_rgb = np.full((7, 9, 3), 1111, dtype=np.uint16)
        old_ir = np.full((7, 9), 2222, dtype=np.uint16)
        tifffile.imwrite(rgb_path, old_rgb, photometric="rgb")
        tifffile.imwrite(ir_path, old_ir, photometric="minisblack")

        real_replace = writer.os.replace
        failed = False

        def fail_first_ir_commit(src, dst) -> None:
            nonlocal failed
            if os.fspath(dst) == os.fspath(ir_path) and not failed:
                failed = True
                raise OSError("simulated IR commit failure")
            real_replace(src, dst)

        monkeypatch.setattr(writer.os, "replace", fail_first_ir_commit)

        with pytest.raises(OSError, match="IR commit failure"):
            write_tiff_16bit(_result(with_ir=True), str(rgb_path))

        assert np.array_equal(tifffile.imread(rgb_path), old_rgb)
        assert np.array_equal(tifffile.imread(ir_path), old_ir)
        assert sorted(path.name for path in tmp_path.iterdir()) == ["frame.tif", "frame_IR.tif"]

    @pytest.mark.parametrize("failed_replace", (1, 2, 3, 4))
    def test_existing_pair_survives_failure_at_each_commit_rename(
        self,
        monkeypatch,
        tmp_path,
        failed_replace: int,
    ) -> None:
        rgb_path = tmp_path / "frame.tif"
        ir_path = tmp_path / "frame_IR.tif"
        old_rgb = np.full((7, 9, 3), 3333, dtype=np.uint16)
        old_ir = np.full((7, 9), 4444, dtype=np.uint16)
        tifffile.imwrite(rgb_path, old_rgb, photometric="rgb")
        tifffile.imwrite(ir_path, old_ir, photometric="minisblack")

        real_replace = writer.os.replace
        calls = 0

        def fail_selected_replace(src, dst) -> None:
            nonlocal calls
            calls += 1
            if calls == failed_replace:
                raise OSError(f"simulated commit rename {failed_replace} failure")
            real_replace(src, dst)

        monkeypatch.setattr(writer.os, "replace", fail_selected_replace)

        with pytest.raises(OSError, match=f"commit rename {failed_replace} failure"):
            write_tiff_16bit(_result(with_ir=True), str(rgb_path))

        assert np.array_equal(tifffile.imread(rgb_path), old_rgb)
        assert np.array_equal(tifffile.imread(ir_path), old_ir)
        assert sorted(path.name for path in tmp_path.iterdir()) == ["frame.tif", "frame_IR.tif"]

    def test_rgb_write_failure_leaves_nothing(self, monkeypatch) -> None:
        """Sanity companion: a failure on the *first* (RGB) write must also
        leave no partial file — the baseline the IR-failure case above is
        held to."""

        def failing_imwrite(*args, **kwargs):
            raise OSError("simulated disk failure writing RGB")

        monkeypatch.setattr(tifffile, "imwrite", failing_imwrite)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "frame002")
            result = _result(with_ir=True)

            with pytest.raises(Exception):
                write_tiff_16bit(result, path)

            assert os.listdir(tmpdir) == []


class TestStaleIrSidecar:
    """A stale `_IR.tif` from a previous IR-enabled write of the same target
    must not be silently left next to a fresh IR-less RGB write."""

    def test_stale_sidecar_removed_when_new_write_has_no_ir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "frame003")
            rgb_path = path + ".tif"
            ir_path = path + "_IR.tif"

            # Leftover IR sidecar from an earlier, unrelated write.
            stale_ir = np.random.randint(0, 65535, (99, 77), dtype=np.uint16)
            tifffile.imwrite(ir_path, stale_ir, photometric="minisblack")
            assert os.path.exists(ir_path)

            result = _result(with_ir=False)
            returned_path = write_tiff_16bit(result, path)

            assert returned_path == rgb_path
            assert os.path.exists(rgb_path)
            assert not os.path.exists(ir_path), (
                "stale _IR.tif sidecar survived a fresh IR-less write of the same target — a downstream loader would misattribute it"
            )

    def test_fresh_ir_overwrites_stale_sidecar(self) -> None:
        """When the new write DOES have IR, it should simply overwrite the
        stale sidecar with its own (correctly paired) data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "frame004")
            ir_path = path + "_IR.tif"

            stale_ir = np.zeros((5, 5), dtype=np.uint16)
            tifffile.imwrite(ir_path, stale_ir, photometric="minisblack")

            result = _result(with_ir=True)
            write_tiff_16bit(result, path)

            readback = tifffile.imread(ir_path)
            assert readback.shape == result.ir.shape
            assert np.array_equal(readback, result.ir)


def test_dpi_metadata_written(tmp_path):
    import numpy as np
    import tifffile

    from negpy.infrastructure.scanners.result import ScanResult
    from negpy.services.scanning.writer import write_tiff_16bit

    rgb = np.zeros((8, 6, 3), dtype=np.uint16)
    ir = np.zeros((8, 6), dtype=np.uint16)
    out = write_tiff_16bit(ScanResult(rgb=rgb, ir=ir, dpi=4000, device_model="t"), str(tmp_path / "f.tif"))
    for f in (out, str(tmp_path / "f_IR.tif")):
        with tifffile.TiffFile(f) as t:
            xres = t.pages[0].tags["XResolution"].value
            assert xres[0] / xres[1] == 4000, f"{f}: XResolution {xres}"
            assert t.pages[0].tags["ResolutionUnit"].value == 2  # INCH


def test_full_negative_evidence_failure_preserves_existing_triplet(monkeypatch, tmp_path) -> None:
    rgb_path = tmp_path / "frame003.tif"
    ir_path = tmp_path / "frame003_IR.tif"
    valid_path = tmp_path / "frame003_IR_VALID.tif"
    old_rgb = np.full((7, 5, 3), 1111, dtype=np.uint16)
    old_ir = np.full((7, 5), 2222, dtype=np.uint16)
    old_valid = np.full((7, 5), 255, dtype=np.uint8)
    tifffile.imwrite(rgb_path, old_rgb, photometric="rgb")
    tifffile.imwrite(ir_path, old_ir, photometric="minisblack")
    tifffile.imwrite(valid_path, old_valid, photometric="minisblack")

    real_inspect = writer.inspect_tiff_payload
    calls = 0

    def fail_third_inspection(path):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated validity evidence failure")
        return real_inspect(path)

    monkeypatch.setattr(writer, "inspect_tiff_payload", fail_third_inspection)
    new_rgb = np.full((7, 5, 3), 3333, dtype=np.uint16)
    new_ir = np.full((7, 5), 4444, dtype=np.uint16)
    result = ScanResult(rgb=new_rgb, ir=new_ir, dpi=4000, device_model="Nikon LS-5000")

    with pytest.raises(OSError, match="validity evidence failure"):
        write_full_negative_tiff(
            result,
            ir_valid_mask=np.ones((7, 5), dtype=bool),
            path=rgb_path,
        )

    np.testing.assert_array_equal(tifffile.imread(rgb_path), old_rgb)
    np.testing.assert_array_equal(tifffile.imread(ir_path), old_ir)
    np.testing.assert_array_equal(tifffile.imread(valid_path), old_valid)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "frame003.tif",
        "frame003_IR.tif",
        "frame003_IR_VALID.tif",
    ]


@pytest.mark.parametrize("failed_replace", range(1, 7))
def test_full_negative_triplet_survives_each_commit_rename_failure(
    monkeypatch,
    tmp_path,
    failed_replace: int,
) -> None:
    rgb_path = tmp_path / "frame003.tif"
    ir_path = tmp_path / "frame003_IR.tif"
    valid_path = tmp_path / "frame003_IR_VALID.tif"
    old_rgb = np.full((7, 5, 3), 1111, dtype=np.uint16)
    old_ir = np.full((7, 5), 2222, dtype=np.uint16)
    old_valid = np.full((7, 5), 255, dtype=np.uint8)
    tifffile.imwrite(rgb_path, old_rgb, photometric="rgb")
    tifffile.imwrite(ir_path, old_ir, photometric="minisblack")
    tifffile.imwrite(valid_path, old_valid, photometric="minisblack")

    real_replace = writer.os.replace
    calls = 0

    def fail_selected_replace(src, dst) -> None:
        nonlocal calls
        calls += 1
        if calls == failed_replace:
            raise OSError(f"simulated triplet commit rename {failed_replace} failure")
        real_replace(src, dst)

    monkeypatch.setattr(writer.os, "replace", fail_selected_replace)
    result = ScanResult(
        rgb=np.full((7, 5, 3), 3333, dtype=np.uint16),
        ir=np.full((7, 5), 4444, dtype=np.uint16),
        dpi=4000,
        device_model="Nikon LS-5000",
    )

    with pytest.raises(OSError, match=f"triplet commit rename {failed_replace} failure"):
        write_full_negative_tiff(
            result,
            ir_valid_mask=np.ones((7, 5), dtype=bool),
            path=rgb_path,
        )

    np.testing.assert_array_equal(tifffile.imread(rgb_path), old_rgb)
    np.testing.assert_array_equal(tifffile.imread(ir_path), old_ir)
    np.testing.assert_array_equal(tifffile.imread(valid_path), old_valid)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "frame003.tif",
        "frame003_IR.tif",
        "frame003_IR_VALID.tif",
    ]
