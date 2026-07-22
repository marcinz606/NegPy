"""Tests for the optional coolscanpy-backed roll-scanning adapter.

No real coolscanpy install is required or assumed: `fake_coolscanpy` (see
tests/scanners/conftest.py) injects a minimal stand-in module, exercising
the same lazy `import coolscanpy` path the production code takes.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from negpy.infrastructure.roll import coolscanpy_roll


class TestAvailable:
    def test_false_when_not_importable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "coolscanpy", raising=False)
        # The optional group may intentionally be installed by this test run,
        # so model an absent package at the availability seam rather than
        # assuming anything about the interpreter's installed extras.
        monkeypatch.setattr(coolscanpy_roll.importlib.util, "find_spec", lambda _name: None)
        assert coolscanpy_roll.available() is False

    def test_true_when_fake_module_present(self, fake_coolscanpy) -> None:
        assert coolscanpy_roll.available() is True


class TestListDevices:
    def test_passthrough(self, fake_coolscanpy) -> None:
        fake_coolscanpy.state["devices"] = ["device-a", "device-b"]
        assert coolscanpy_roll.list_devices() == ["device-a", "device-b"]


class TestOpenRoll:
    def test_opens_named_device_with_default_material(self, fake_coolscanpy) -> None:
        roll = fake_coolscanpy.Roll()
        device = fake_coolscanpy.Device(roll)
        fake_coolscanpy.state["open_device"] = device

        handle = coolscanpy_roll.open_roll("ls5000-usb-001")

        assert device.roll_called_with == fake_coolscanpy.module.Material.COLOR_NEGATIVE
        handle.close()
        assert roll.closed is True
        assert device.closed is True

    def test_explicit_material_forwarded(self, fake_coolscanpy) -> None:
        roll = fake_coolscanpy.Roll()
        device = fake_coolscanpy.Device(roll)
        fake_coolscanpy.state["open_device"] = device

        coolscanpy_roll.open_roll(material=fake_coolscanpy.module.Material.BLACK_AND_WHITE_NEGATIVE)

        assert device.roll_called_with == fake_coolscanpy.module.Material.BLACK_AND_WHITE_NEGATIVE

    def test_device_not_found_translated_to_runtime_error(self, fake_coolscanpy) -> None:
        fake_coolscanpy.state["open_error"] = fake_coolscanpy.module.DeviceNotFound("no Coolscan LS-5000 unit is attached")

        with pytest.raises(RuntimeError, match="no Coolscan LS-5000 unit is attached"):
            coolscanpy_roll.open_roll()

    def test_roll_open_failure_closes_device_and_translates(self, fake_coolscanpy) -> None:
        class ExplodingDevice(fake_coolscanpy.Device):
            def roll(self, *, material=None):
                raise ValueError("no roll adapter is attached/detected on this device")

        device = ExplodingDevice(None)
        fake_coolscanpy.state["open_device"] = device

        with pytest.raises(RuntimeError, match="no roll adapter is attached"):
            coolscanpy_roll.open_roll()
        assert device.closed is True


class TestRollHandle:
    @staticmethod
    def _handle(fake_coolscanpy, roll=None):
        roll = roll if roll is not None else fake_coolscanpy.Roll()
        device = fake_coolscanpy.Device(roll)
        fake_coolscanpy.state["open_device"] = device
        return coolscanpy_roll.open_roll(), roll, device

    def test_preview_passthrough(self, fake_coolscanpy) -> None:
        thumb = fake_coolscanpy.Thumbnail(slot=1, image=np.zeros((4, 4, 3)), boundary_rows=(0, 4), spacing_offset=0, needs_approval=False)
        handle, _roll, _device = self._handle(fake_coolscanpy, fake_coolscanpy.Roll(thumbnails=[thumb]))

        assert handle.preview() == [thumb]

    def test_preview_filters_by_requested_slots(self, fake_coolscanpy) -> None:
        thumbs = [
            fake_coolscanpy.Thumbnail(slot=i, image=np.zeros((2, 2, 3)), boundary_rows=(0, 2), spacing_offset=0, needs_approval=False)
            for i in (1, 2, 3)
        ]
        handle, _roll, _device = self._handle(fake_coolscanpy, fake_coolscanpy.Roll(thumbnails=thumbs))

        result = handle.preview([2])

        assert [t.slot for t in result] == [2]

    def test_preview_exception_translated(self, fake_coolscanpy) -> None:
        handle, _roll, _device = self._handle(
            fake_coolscanpy,
            fake_coolscanpy.Roll(raise_on={"preview": fake_coolscanpy.module.PyCoolscanError("fingerprint mismatch")}),
        )
        with pytest.raises(RuntimeError, match="fingerprint mismatch"):
            handle.preview()

    def test_restore_preview_session_passthrough(self, fake_coolscanpy) -> None:
        thumbnails = [
            fake_coolscanpy.Thumbnail(
                slot=slot,
                image=np.zeros((2, 2, 3)),
                boundary_rows=(0, 2),
                spacing_offset=0,
                needs_approval=False,
            )
            for slot in (1, 2, 3)
        ]
        handle, roll, _device = self._handle(
            fake_coolscanpy,
            fake_coolscanpy.Roll(thumbnails=thumbnails),
        )

        result = handle.restore_preview_session("saved-session", [1, 3])

        assert [thumbnail.slot for thumbnail in result] == [1, 3]
        assert roll.restore_preview_session_calls == [
            ("saved-session", (1, 3))
        ]

    def test_restore_preview_session_exception_translated(
        self, fake_coolscanpy
    ) -> None:
        error = fake_coolscanpy.module.PyCoolscanError(
            "saved preview hash mismatch"
        )
        handle, _roll, _device = self._handle(
            fake_coolscanpy,
            fake_coolscanpy.Roll(
                raise_on={"restore_preview_session": error}
            ),
        )

        with pytest.raises(RuntimeError, match="saved preview hash mismatch") as excinfo:
            handle.restore_preview_session("saved-session")

        assert excinfo.value.__cause__ is error

    def test_approve_returns_underlying_content_bound_receipt(
        self, fake_coolscanpy
    ) -> None:
        approval = object()

        class ReturningApprovalRoll(fake_coolscanpy.Roll):
            def approve(self, slot):
                super().approve(slot)
                return approval

        handle, roll, _device = self._handle(
            fake_coolscanpy,
            ReturningApprovalRoll(),
        )

        result = handle.approve(3)

        assert result is approval
        assert roll.approved == [3]

    def test_manual_review_required_translated(self, fake_coolscanpy) -> None:
        handle, _roll, _device = self._handle(
            fake_coolscanpy,
            fake_coolscanpy.Roll(raise_on={"approve": fake_coolscanpy.module.ManualReviewRequired("slot 3 needs review", slot=3)}),
        )
        with pytest.raises(RuntimeError, match="slot 3 needs review"):
            handle.approve(3)

    def test_set_spacing_offset_records_value(self, fake_coolscanpy) -> None:
        handle, roll, _device = self._handle(fake_coolscanpy)
        handle.set_spacing_offset(5, -12)
        assert roll.spacing_offsets[5] == -12

    def test_scan_many_yields_frames(self, fake_coolscanpy) -> None:
        frame = fake_coolscanpy.Frame(slot=2, rgb=np.zeros((4, 4, 3), dtype=np.uint16), ir=None, ir_validity=None, receipt=None)
        handle, _roll, _device = self._handle(fake_coolscanpy, fake_coolscanpy.Roll(frames=[frame]))

        assert list(handle.scan_many([2])) == [frame]

    def test_scan_many_exception_translated(self, fake_coolscanpy) -> None:
        error = fake_coolscanpy.module.SafeStopRequested("safe stop requested; 0 of 1 requested frames completed")
        handle, _roll, _device = self._handle(fake_coolscanpy, fake_coolscanpy.Roll(raise_on={"scan_many_slots": {5: error}}))

        with pytest.raises(RuntimeError, match="safe stop requested") as excinfo:
            list(handle.scan_many([5]))
        # __cause__ preserves the original typed exception for a caller that
        # wants to distinguish a deliberate stop from a genuine failure.
        assert isinstance(excinfo.value.__cause__, fake_coolscanpy.module.SafeStopRequested)

    def test_safe_stop_forwarded(self, fake_coolscanpy) -> None:
        handle, roll, _device = self._handle(fake_coolscanpy)
        handle.safe_stop()
        assert roll.safe_stop_called is True

    def test_close_releases_roll_then_device(self, fake_coolscanpy) -> None:
        handle, roll, device = self._handle(fake_coolscanpy)
        handle.close()
        assert roll.closed is True
        assert device.closed is True

    def test_close_retains_device_when_roll_ownership_is_uncertain(
        self, fake_coolscanpy
    ) -> None:
        ownership_error = RuntimeError("USB ownership is retained")

        class UncertainRoll(fake_coolscanpy.Roll):
            def close(self) -> None:
                raise ownership_error

        handle, _roll, device = self._handle(fake_coolscanpy, UncertainRoll())

        with pytest.raises(RuntimeError) as raised:
            handle.close()

        assert raised.value is ownership_error
        assert device.closed is False

    def test_context_manager_closes_on_exit(self, fake_coolscanpy) -> None:
        roll = fake_coolscanpy.Roll()
        device = fake_coolscanpy.Device(roll)
        fake_coolscanpy.state["open_device"] = device

        with coolscanpy_roll.open_roll() as handle:
            assert handle is not None
            assert roll.closed is False

        assert roll.closed is True
