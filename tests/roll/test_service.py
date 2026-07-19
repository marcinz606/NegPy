"""Tests for RollScanningService: lifecycle orchestration and output writing.

Lifecycle tests use `fake_coolscanpy` (see tests/roll/conftest.py) the
same way test_coolscanpy_roll.py does. write_frame() tests construct a fake
Frame/Receipt directly -- writing to disk never touches coolscanpy itself,
so no module injection is needed for those.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import tifffile

from negpy.services.roll import service as roll_service
from negpy.services.roll.service import RollScanningError, RollScanningService


class TestAvailable:
    def test_reexports_backend_availability(self, fake_coolscanpy) -> None:
        assert roll_service.available() is True


class TestRollLifecycle:
    def _open_service(self, fake_coolscanpy, roll=None):
        roll = roll if roll is not None else fake_coolscanpy.Roll()
        device = fake_coolscanpy.Device(roll)
        fake_coolscanpy.state["open_device"] = device
        service = RollScanningService()
        service.open_roll("ls5000-usb-001")
        return service, roll, device

    def test_open_then_close(self, fake_coolscanpy) -> None:
        service, roll, device = self._open_service(fake_coolscanpy)
        service.close()
        assert roll.closed is True
        assert device.closed is True

    def test_double_open_raises(self, fake_coolscanpy) -> None:
        service, _roll, _device = self._open_service(fake_coolscanpy)
        with pytest.raises(RollScanningError, match="already open"):
            service.open_roll()

    def test_methods_before_open_raise(self) -> None:
        service = RollScanningService()
        with pytest.raises(RollScanningError, match="no roll is open"):
            service.preview()

    def test_close_without_open_is_a_no_op(self) -> None:
        RollScanningService().close()  # must not raise

    def test_preview_and_approve_delegate_to_handle(self, fake_coolscanpy) -> None:
        thumb = fake_coolscanpy.Thumbnail(slot=1, image=np.zeros((2, 2, 3)), boundary_rows=(0, 2), spacing_offset=0, needs_approval=True)
        service, roll, _device = self._open_service(fake_coolscanpy, fake_coolscanpy.Roll(thumbnails=[thumb]))

        assert service.preview() == [thumb]
        service.approve(1)
        assert roll.approved == [1]

    def test_scan_many_delegates_to_handle(self, fake_coolscanpy) -> None:
        frame = fake_coolscanpy.Frame(slot=1, rgb=np.zeros((2, 2, 3), dtype=np.uint16), ir=None, ir_validity=None, receipt=None)
        service, _roll, _device = self._open_service(fake_coolscanpy, fake_coolscanpy.Roll(frames=[frame]))

        assert list(service.scan_many([1])) == [frame]

    def test_safe_stop_before_open_is_a_no_op(self) -> None:
        RollScanningService().safe_stop()  # must not raise

    def test_context_manager_closes(self, fake_coolscanpy) -> None:
        roll = fake_coolscanpy.Roll()
        device = fake_coolscanpy.Device(roll)
        fake_coolscanpy.state["open_device"] = device

        with RollScanningService() as service:
            service.open_roll()
        assert roll.closed is True


class TestWriteFrame:
    def _frame(self, fake_coolscanpy, *, slot=7, ir=None):
        rgb = np.random.randint(0, 65535, (40, 60, 3), dtype=np.uint16)
        receipt = fake_coolscanpy.Receipt(version=1, slot=slot, dpi=4000, depth=16, device_id="usb:1:2", transport_smear_verdict="clean")
        return fake_coolscanpy.Frame(slot=slot, rgb=rgb, ir=ir, ir_validity=None, receipt=receipt)

    def test_writes_rgb_tiff(self, fake_coolscanpy, tmp_path) -> None:
        frame = self._frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ date }}_slot{{ "%02d" % seq }}')

        assert os.path.exists(output.rgb_path)
        assert output.rgb_path.endswith(".tif")
        readback = tifffile.imread(output.rgb_path)
        assert readback.shape == (40, 60, 3)
        assert readback.dtype == np.uint16
        np.testing.assert_array_equal(readback, frame.rgb)

    def test_seq_seeded_from_slot_number(self, fake_coolscanpy, tmp_path) -> None:
        frame = self._frame(fake_coolscanpy, slot=23)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}')

        assert "023" in os.path.basename(output.rgb_path)

    def test_writes_ir_sidecar_when_present(self, fake_coolscanpy, tmp_path) -> None:
        ir = np.random.randint(0, 65535, (40, 60), dtype=np.uint16)
        frame = self._frame(fake_coolscanpy, ir=ir)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}')

        assert output.ir_path is not None
        assert output.ir_path.endswith("_IR.tif")
        assert os.path.exists(output.ir_path)
        readback = tifffile.imread(output.ir_path)
        np.testing.assert_array_equal(readback, ir)

    def test_no_ir_sidecar_when_absent(self, fake_coolscanpy, tmp_path) -> None:
        frame = self._frame(fake_coolscanpy, ir=None)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}')

        assert output.ir_path is None
        assert not any(name.endswith("_IR.tif") for name in os.listdir(tmp_path))

    def test_writes_receipt_json_sidecar(self, fake_coolscanpy, tmp_path) -> None:
        frame = self._frame(fake_coolscanpy, slot=9)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}')

        assert output.receipt_path.endswith("_receipt.json")
        with open(output.receipt_path) as fh:
            payload = json.load(fh)
        assert payload["slot"] == 9
        assert payload["dpi"] == 4000
        assert payload["transport_smear_verdict"] == "clean"

    def test_rescanning_same_slot_overwrites(self, fake_coolscanpy, tmp_path) -> None:
        service = RollScanningService()
        pattern = '{{ "%03d" % seq }}'

        first = service.write_frame(self._frame(fake_coolscanpy, slot=4), str(tmp_path), pattern)
        second_frame = self._frame(fake_coolscanpy, slot=4)
        second = service.write_frame(second_frame, str(tmp_path), pattern)

        assert first.rgb_path == second.rgb_path
        readback = tifffile.imread(second.rgb_path)
        np.testing.assert_array_equal(readback, second_frame.rgb)

    def test_creates_output_folder(self, fake_coolscanpy, tmp_path) -> None:
        nested = tmp_path / "does" / "not" / "exist" / "yet"
        frame = self._frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(nested), '{{ "%03d" % seq }}')

        assert os.path.exists(output.rgb_path)


def _tier_frame(fake_coolscanpy, *, slot=11, ir=True, seed=0):
    """A frame with real per-pixel variance (not a flat fill) so a rendered
    Tier-3 positive has something non-degenerate to measure -- red-biased
    like a plausible C41 negative, matching tests/roll/test_positive.py's
    own synthetic data."""
    rng = np.random.default_rng(seed)
    shape = (24, 32)
    rgb = np.zeros((*shape, 3), dtype=np.uint16)
    rgb[..., 0] = rng.integers(40000, 60000, size=shape)
    rgb[..., 1] = rng.integers(25000, 45000, size=shape)
    rgb[..., 2] = rng.integers(15000, 35000, size=shape)
    ir_plane = rng.integers(0, 65535, size=shape, dtype=np.uint16) if ir else None
    receipt = fake_coolscanpy.Receipt(version=1, slot=slot, dpi=4000, depth=16, device_id="usb:1:2", transport_smear_verdict="clean")
    return fake_coolscanpy.Frame(slot=slot, rgb=rgb, ir=ir_plane, ir_validity=None, receipt=receipt)


class TestThreeTierWriting:
    """`write_frame`'s three independently-selectable output tiers: unrepaired
    (Tier 1), repaired (Tier 2, needs a registered repair engine), and
    positive (Tier 3, always derived from Tier 2's in-memory result). Naming,
    receipt provenance, and every degrade path get covered here; the plain
    default-settings behavior (Tier 1 only) is already covered above by
    TestWriteFrame, unchanged."""

    def _receipt(self, path: str) -> dict:
        with open(path) as fh:
            return json.load(fh)

    # -- selecting nothing --------------------------------------------------

    def test_no_tier_selected_writes_only_the_receipt(self, fake_coolscanpy, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=False, write_repaired=False, write_positive=False
        )

        assert (output.rgb_path, output.ir_path, output.repaired_rgb_path, output.repaired_ir_path, output.positive_path) == (
            None,
            None,
            None,
            None,
            None,
        )
        assert os.path.exists(output.receipt_path)
        payload = self._receipt(output.receipt_path)
        assert payload["outputs"]["unrepaired"] == {"written": False, "status": "not selected"}
        assert payload["outputs"]["repaired"] == {"written": False, "status": "not selected"}
        assert payload["outputs"]["positive"] == {"written": False, "status": "not selected"}

    # -- Tier 2 without a registered engine ----------------------------------

    def test_repaired_without_an_engine_degrades_but_unrepaired_still_writes(self, fake_coolscanpy, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True)

        assert output.rgb_path is not None and os.path.exists(output.rgb_path)
        assert output.repaired_rgb_path is None
        assert output.repaired_ir_path is None
        payload = self._receipt(output.receipt_path)
        assert payload["outputs"]["unrepaired"]["written"] is True
        assert payload["outputs"]["repaired"] == {"written": False, "status": "unavailable: no dust-repair engine registered"}

    def test_positive_without_an_engine_degrades_too_since_it_needs_tier_2(self, fake_coolscanpy, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_positive=True)

        assert output.rgb_path is not None and os.path.exists(output.rgb_path)
        assert output.positive_path is None
        payload = self._receipt(output.receipt_path)
        assert payload["outputs"]["unrepaired"]["written"] is True
        assert payload["outputs"]["positive"]["written"] is False
        assert "Tier 2" in payload["outputs"]["positive"]["status"]
        assert "no dust-repair engine registered" in payload["outputs"]["positive"]["status"]

    # -- Tier 2 with a registered engine -------------------------------------

    def test_repaired_writes_rgb_and_retains_original_ir(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        fake_repair_engine.transform = lambda rgb: np.clip(rgb.astype(np.int32) + 1000, 0, 65535).astype(np.uint16)
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True)

        assert output.repaired_rgb_path is not None
        assert output.repaired_rgb_path.endswith("_repaired.tif")
        readback = tifffile.imread(output.repaired_rgb_path)
        np.testing.assert_array_equal(readback, fake_repair_engine.transform(frame.rgb))
        assert not np.array_equal(readback, frame.rgb)  # actually repaired, not a copy

        assert output.repaired_ir_path is not None
        assert output.repaired_ir_path.endswith("_repaired_IR.tif")
        ir_readback = tifffile.imread(output.repaired_ir_path)
        np.testing.assert_array_equal(ir_readback, frame.ir)  # Tier 1's own IR, unchanged

    def test_repaired_naming_does_not_collide_with_unrepaired(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True)

        paths = {output.rgb_path, output.ir_path, output.repaired_rgb_path, output.repaired_ir_path}
        assert len(paths) == 4  # four distinct files
        assert all(os.path.exists(p) for p in paths)

    def test_repaired_receipt_records_engine_provenance(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=False, write_repaired=True, repair_mode="hybrid"
        )

        payload = self._receipt(output.receipt_path)
        entry = payload["outputs"]["repaired"]
        assert entry["written"] is True
        assert entry["engine"] == "test-repair-engine"
        assert entry["engine_version"] == "0.0.1-test"
        assert entry["mode"] == "hybrid"
        assert entry["rgb_path"] == output.repaired_rgb_path

    def test_repair_engine_failure_degrades_without_losing_unrepaired(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        fake_repair_engine.raise_error = RuntimeError("inpainting model unavailable")
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True, write_positive=True
        )

        assert output.rgb_path is not None and os.path.exists(output.rgb_path)
        assert output.repaired_rgb_path is None
        assert output.positive_path is None
        payload = self._receipt(output.receipt_path)
        assert payload["outputs"]["unrepaired"]["written"] is True
        assert "repair failed" in payload["outputs"]["repaired"]["status"]
        assert "inpainting model unavailable" in payload["outputs"]["repaired"]["status"]
        assert "Tier 2" in payload["outputs"]["positive"]["status"]

    def test_frame_without_infrared_degrades_repaired_and_positive(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy, ir=False)
        service = RollScanningService()

        output = service.write_frame(
            frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True, write_positive=True
        )

        assert fake_repair_engine.calls == []  # never even attempted
        assert output.repaired_rgb_path is None
        assert output.positive_path is None
        payload = self._receipt(output.receipt_path)
        assert "no infrared plane" in payload["outputs"]["repaired"]["status"]

    def test_invalid_repair_mode_string_falls_back_to_exact(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_repaired=True, repair_mode="bogus-mode")

        assert len(fake_repair_engine.calls) == 1
        assert fake_repair_engine.calls[0][2] == roll_service.RepairMode.EXACT

    def test_positive_requested_without_write_repaired_still_repairs_in_memory(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=False, write_positive=True)

        assert len(fake_repair_engine.calls) == 1  # repair ran to feed Tier 3...
        assert output.repaired_rgb_path is None  # ...but Tier 2 itself was never written
        assert output.positive_path is not None
        payload = self._receipt(output.receipt_path)
        entry = payload["outputs"]["repaired"]
        assert entry["written"] is False
        assert entry["status"] == "not selected (computed in memory for the positive)"
        assert entry["engine"] == "test-repair-engine"  # still recorded even though unwritten

    # -- Tier 3 -----------------------------------------------------------------

    def test_positive_is_written_and_derived_from_repaired_not_raw_rgb(
        self, fake_coolscanpy, fake_repair_engine, tmp_path, monkeypatch
    ) -> None:
        fake_repair_engine.transform = lambda rgb: np.clip(rgb.astype(np.int32) + 5000, 0, 65535).astype(np.uint16)
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        from negpy.services.roll import positive as roll_positive_module

        captured = {}
        real_render = roll_positive_module.render_positive

        def _spy(rgb_u16, *, processor):
            captured["rgb_u16"] = rgb_u16
            return real_render(rgb_u16, processor=processor)

        monkeypatch.setattr(roll_positive_module, "render_positive", _spy)

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=False, write_positive=True)

        np.testing.assert_array_equal(captured["rgb_u16"], fake_repair_engine.transform(frame.rgb))
        assert not np.array_equal(captured["rgb_u16"], frame.rgb)
        assert output.positive_path is not None
        readback = tifffile.imread(output.positive_path)
        assert readback.shape == frame.rgb.shape
        assert readback.dtype == np.uint16

    def test_positive_receipt_records_inversion_and_repair_provenance(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=False, write_positive=True)

        payload = self._receipt(output.receipt_path)
        entry = payload["outputs"]["positive"]
        assert entry["written"] is True
        assert entry["rgb_path"] == output.positive_path
        assert entry["inversion_path"] == "negpy.services.rendering.image_processor.ImageProcessor.run_pipeline"
        assert entry["render_intent"] == "print"
        assert entry["process_mode"] == "C41"
        assert entry["auto_exposure"] is True
        assert entry["negpy_version"]
        assert entry["repair_engine"] == "test-repair-engine"
        assert entry["repair_engine_version"] == "0.0.1-test"
        assert entry["repair_mode"] == "exact"

    def test_positive_filename_has_no_infrared_companion(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=False, write_positive=True)

        assert output.positive_path is not None
        assert output.positive_path.endswith("_positive.tif")
        assert os.path.exists(output.positive_path)
        assert not any(name.endswith("_positive_IR.tif") for name in os.listdir(str(tmp_path)))

    def test_inversion_unavailable_degrades_positive_but_keeps_unrepaired_and_repaired(
        self, fake_coolscanpy, fake_repair_engine, tmp_path, monkeypatch
    ) -> None:
        from negpy.services.roll import positive as roll_positive_module

        monkeypatch.setattr(roll_positive_module, "available", lambda: False)
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True, write_positive=True
        )

        assert output.rgb_path is not None and os.path.exists(output.rgb_path)
        assert output.repaired_rgb_path is not None and os.path.exists(output.repaired_rgb_path)
        assert output.positive_path is None
        payload = self._receipt(output.receipt_path)
        assert payload["outputs"]["unrepaired"]["written"] is True
        assert payload["outputs"]["repaired"]["written"] is True
        assert payload["outputs"]["positive"] == {"written": False, "status": "unavailable: inversion path not available"}

    def test_inversion_failure_degrades_positive_only(self, fake_coolscanpy, fake_repair_engine, tmp_path, monkeypatch) -> None:
        from negpy.services.roll import positive as roll_positive_module

        def _boom(rgb_u16, *, processor):
            raise RuntimeError("no CPU render backend available")

        monkeypatch.setattr(roll_positive_module, "render_positive", _boom)
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True, write_positive=True
        )

        assert output.rgb_path is not None and os.path.exists(output.rgb_path)
        assert output.repaired_rgb_path is not None and os.path.exists(output.repaired_rgb_path)
        assert output.positive_path is None
        payload = self._receipt(output.receipt_path)
        assert payload["outputs"]["repaired"]["written"] is True
        assert "inversion failed" in payload["outputs"]["positive"]["status"]
        assert "no CPU render backend available" in payload["outputs"]["positive"]["status"]

    # -- all three together, and receipt backward-compatibility -----------------

    def test_all_three_tiers_together(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        frame = _tier_frame(fake_coolscanpy)
        service = RollScanningService()

        output = service.write_frame(
            frame, str(tmp_path), '{{ "%03d" % seq }}', write_unrepaired=True, write_repaired=True, write_positive=True
        )

        for path in (output.rgb_path, output.ir_path, output.repaired_rgb_path, output.repaired_ir_path, output.positive_path):
            assert path is not None and os.path.exists(path)
        assert len({output.rgb_path, output.ir_path, output.repaired_rgb_path, output.repaired_ir_path, output.positive_path}) == 5
        payload = self._receipt(output.receipt_path)
        assert all(payload["outputs"][tier]["written"] for tier in ("unrepaired", "repaired", "positive"))

    def test_receipt_keeps_the_original_scan_receipt_fields_alongside_outputs(self, fake_coolscanpy, tmp_path) -> None:
        """`outputs` must be additive -- write_frame's pre-tiering callers (and
        TestWriteFrame's tests above) read coolscanpy's own receipt fields
        (slot, dpi, transport_smear_verdict, ...) at the JSON root."""
        frame = _tier_frame(fake_coolscanpy, slot=42)
        service = RollScanningService()

        output = service.write_frame(frame, str(tmp_path), '{{ "%03d" % seq }}')

        payload = self._receipt(output.receipt_path)
        assert payload["slot"] == 42
        assert payload["dpi"] == 4000
        assert payload["transport_smear_verdict"] == "clean"
        assert "outputs" in payload
