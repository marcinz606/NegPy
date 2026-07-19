"""Tests for RollScanningService: lifecycle orchestration and output writing.

Lifecycle tests use `fake_coolscanpy` (see tests/scanners/conftest.py) the
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

from negpy.services.scanning import roll_service
from negpy.services.scanning.roll_service import RollScanningService


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
        with pytest.raises(RuntimeError, match="already open"):
            service.open_roll()

    def test_methods_before_open_raise(self) -> None:
        service = RollScanningService()
        with pytest.raises(RuntimeError, match="no roll is open"):
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
