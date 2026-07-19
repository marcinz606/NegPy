"""Tests for RollWorker: signal emissions, device-session reuse, and the
safe-stop / cancellation distinction. Mirrors test_capture_worker.py's
shape: fakes wired directly into the worker, signals captured with plain
list-appending slots, no QThread or QApplication involved.
"""

from __future__ import annotations

import os

import numpy as np

from negpy.desktop.workers.roll_worker import RollBatchScanRequest, RollPreviewRequest, RollWorker


def _frame(fake_coolscanpy, *, slot, ir=None):
    rgb = np.random.randint(0, 65535, (4, 6, 3), dtype=np.uint16)
    receipt = fake_coolscanpy.Receipt(version=1, slot=slot, dpi=4000, depth=16, device_id="usb:1:2", transport_smear_verdict="clean")
    return fake_coolscanpy.Frame(slot=slot, rgb=rgb, ir=ir, ir_validity=None, receipt=receipt)


def _open(fake_coolscanpy, roll=None):
    roll = roll if roll is not None else fake_coolscanpy.Roll()
    device = fake_coolscanpy.Device(roll)
    fake_coolscanpy.state["open_device"] = device
    return roll, device


class TestListDevices:
    def test_success_emits_devices(self, fake_coolscanpy) -> None:
        fake_coolscanpy.state["devices"] = ["device-a", "device-b"]
        worker = RollWorker()
        seen = []
        worker.devices_ready.connect(seen.append)

        worker.list_devices()

        assert seen == [["device-a", "device-b"]]

    def test_failure_emits_error_and_empty_list(self, fake_coolscanpy, monkeypatch) -> None:
        worker = RollWorker()

        def boom(local_only: bool = False):
            raise RuntimeError("usb enumeration failed")

        monkeypatch.setattr(fake_coolscanpy.module, "get_devices", boom)
        errors, devices = [], []
        worker.error.connect(errors.append)
        worker.devices_ready.connect(devices.append)

        worker.list_devices()

        assert errors and "usb enumeration failed" in errors[-1]
        assert devices == [[]]


class TestOpenAndPreview:
    def test_preview_opens_then_returns_thumbnails(self, fake_coolscanpy) -> None:
        thumb = fake_coolscanpy.Thumbnail(slot=1, image=np.zeros((2, 2, 3)), boundary_rows=(0, 2), spacing_offset=0, needs_approval=False)
        roll, device = _open(fake_coolscanpy, fake_coolscanpy.Roll(thumbnails=[thumb]))
        worker = RollWorker()
        opened, ready, progress = [], [], []
        worker.opened.connect(opened.append)
        worker.preview_ready.connect(ready.append)
        worker.progress.connect(lambda frac, msg: progress.append((frac, msg)))

        worker.run_preview(RollPreviewRequest(device_id="ls5000-usb-001"))

        assert opened == ["ls5000-usb-001"]
        assert ready == [[thumb]]
        assert device.roll_called_with == fake_coolscanpy.module.Material.COLOR_NEGATIVE
        # the fake preview() reports 0.0 then 1.0, forwarded as (fraction, message) pairs
        assert progress[0][0] == 0.0
        assert progress[-1] == (1.0, "preview complete")

    def test_preview_filters_to_requested_slots(self, fake_coolscanpy) -> None:
        thumbs = [
            fake_coolscanpy.Thumbnail(slot=i, image=np.zeros((2, 2, 3)), boundary_rows=(0, 2), spacing_offset=0, needs_approval=False)
            for i in (1, 2, 3)
        ]
        _open(fake_coolscanpy, fake_coolscanpy.Roll(thumbnails=thumbs))
        worker = RollWorker()
        ready = []
        worker.preview_ready.connect(ready.append)

        worker.run_preview(RollPreviewRequest(device_id="ls5000-usb-001", slots=(2,)))

        assert [t.slot for t in ready[0]] == [2]

    def test_second_preview_of_the_same_device_does_not_reopen(self, fake_coolscanpy) -> None:
        _open(fake_coolscanpy)
        open_calls = []
        real_open = fake_coolscanpy.module.open
        fake_coolscanpy.module.open = lambda devname: (open_calls.append(devname), real_open(devname))[1]
        worker = RollWorker()

        worker.run_preview(RollPreviewRequest(device_id="ls5000-usb-001"))
        worker.run_preview(RollPreviewRequest(device_id="ls5000-usb-001"))

        assert open_calls == ["ls5000-usb-001"]  # opened once, reused the second time

    def test_switching_device_closes_the_previous_roll(self, fake_coolscanpy) -> None:
        roll_a, device_a = _open(fake_coolscanpy)
        worker = RollWorker()
        worker.run_preview(RollPreviewRequest(device_id="device-a"))
        assert roll_a.closed is False

        roll_b, device_b = _open(fake_coolscanpy)
        worker.run_preview(RollPreviewRequest(device_id="device-b"))

        assert roll_a.closed is True
        assert device_a.closed is True
        assert roll_b.closed is False

    def test_preview_failure_emits_error(self, fake_coolscanpy) -> None:
        _open(fake_coolscanpy, fake_coolscanpy.Roll(raise_on={"preview": fake_coolscanpy.module.PyCoolscanError("fingerprint mismatch")}))
        worker = RollWorker()
        errors = []
        worker.error.connect(errors.append)

        worker.run_preview(RollPreviewRequest(device_id="ls5000-usb-001"))

        assert errors and "fingerprint mismatch" in errors[-1]

    def test_close_roll_forgets_the_open_device(self, fake_coolscanpy) -> None:
        roll, device = _open(fake_coolscanpy)
        worker = RollWorker()
        worker.run_preview(RollPreviewRequest(device_id="ls5000-usb-001"))
        closed = []
        worker.closed.connect(lambda: closed.append(True))

        worker.close_roll()

        assert closed == [True]
        assert roll.closed is True
        assert worker._open_device_id is None


class TestSpacingAndApproval:
    def test_set_spacing_offset_echoes_the_applied_value(self, fake_coolscanpy) -> None:
        roll, _device = _open(fake_coolscanpy)
        worker = RollWorker()
        worker.run_preview(RollPreviewRequest(device_id="ls5000-usb-001"))
        seen = []
        worker.spacing_offset_set.connect(lambda slot, offset: seen.append((slot, offset)))

        worker.set_spacing_offset(3, -12)

        assert seen == [(3, -12)]
        assert roll.spacing_offsets[3] == -12

    def test_set_spacing_offset_before_open_emits_error(self, fake_coolscanpy) -> None:
        worker = RollWorker()
        errors = []
        worker.error.connect(errors.append)

        worker.set_spacing_offset(1, 5)

        assert errors and "no roll is open" in errors[-1]

    def test_approve_records_slot(self, fake_coolscanpy) -> None:
        roll, _device = _open(fake_coolscanpy)
        worker = RollWorker()
        worker.run_preview(RollPreviewRequest(device_id="ls5000-usb-001"))
        approved = []
        worker.approved.connect(approved.append)

        worker.approve(4)

        assert approved == [4]
        assert roll.approved == [4]

    def test_approve_failure_emits_error(self, fake_coolscanpy) -> None:
        _open(fake_coolscanpy, fake_coolscanpy.Roll(raise_on={"approve": fake_coolscanpy.module.PyCoolscanError("slot 3 needs review")}))
        worker = RollWorker()
        worker.run_preview(RollPreviewRequest(device_id="ls5000-usb-001"))
        errors = []
        worker.error.connect(errors.append)

        worker.approve(3)

        assert errors and "slot 3 needs review" in errors[-1]


class TestBatchScan:
    def test_writes_every_frame_and_reports_progress(self, fake_coolscanpy, tmp_path) -> None:
        frames = [_frame(fake_coolscanpy, slot=s) for s in (1, 2, 3)]
        _open(fake_coolscanpy, fake_coolscanpy.Roll(frames=frames))
        worker = RollWorker()
        written, finished, progress = [], [], []
        worker.frame_written.connect(written.append)
        worker.finished.connect(finished.append)
        worker.progress.connect(lambda frac, msg: progress.append((frac, msg)))

        worker.run_batch_scan(
            RollBatchScanRequest(device_id="ls5000-usb-001", slots=(1, 2, 3), output_folder=str(tmp_path), filename_pattern='{{ "%03d" % seq }}')
        )

        assert [w.slot for w in written] == [1, 2, 3]
        assert all(os.path.exists(w.rgb_path) for w in written)
        assert len(finished) == 1 and finished[0] == written
        assert [msg for _frac, msg in progress] == ["slot 1 complete", "slot 2 complete", "slot 3 complete"]

    def test_batch_scan_opens_the_requested_device_first(self, fake_coolscanpy, tmp_path) -> None:
        _open(fake_coolscanpy, fake_coolscanpy.Roll(frames=[_frame(fake_coolscanpy, slot=1)]))
        worker = RollWorker()
        opened = []
        worker.opened.connect(opened.append)

        worker.run_batch_scan(
            RollBatchScanRequest(device_id="ls5000-usb-001", slots=(1,), output_folder=str(tmp_path), filename_pattern='{{ "%03d" % seq }}')
        )

        assert opened == ["ls5000-usb-001"]

    def test_safe_stop_mid_batch_is_reported_as_cancelled_not_error(self, fake_coolscanpy, tmp_path) -> None:
        stop_error = fake_coolscanpy.module.SafeStopRequested("safe stop requested; 1 of 2 requested frames completed")
        frames = [_frame(fake_coolscanpy, slot=1)]
        roll = fake_coolscanpy.Roll(frames=frames, raise_on={"scan_many_slots": {2: stop_error}})
        _open(fake_coolscanpy, roll)
        worker = RollWorker()
        written, finished, cancelled, errors = [], [], [], []
        worker.frame_written.connect(written.append)
        worker.finished.connect(finished.append)
        worker.cancelled.connect(lambda: cancelled.append(True))
        worker.error.connect(errors.append)

        worker.run_batch_scan(
            RollBatchScanRequest(device_id="ls5000-usb-001", slots=(1, 2), output_folder=str(tmp_path), filename_pattern='{{ "%03d" % seq }}')
        )

        assert len(written) == 1  # slot 1 finished and was written before the stop landed
        assert cancelled == [True]
        assert finished == []  # a stop is not a completion
        assert errors == []

    def test_genuine_failure_mid_batch_is_reported_as_error_not_cancelled(self, fake_coolscanpy, tmp_path) -> None:
        boom = fake_coolscanpy.module.PyCoolscanError("transport jam")
        frames = [_frame(fake_coolscanpy, slot=1)]
        roll = fake_coolscanpy.Roll(frames=frames, raise_on={"scan_many_slots": {2: boom}})
        _open(fake_coolscanpy, roll)
        worker = RollWorker()
        written, cancelled, errors = [], [], []
        worker.frame_written.connect(written.append)
        worker.cancelled.connect(lambda: cancelled.append(True))
        worker.error.connect(errors.append)

        worker.run_batch_scan(
            RollBatchScanRequest(device_id="ls5000-usb-001", slots=(1, 2), output_folder=str(tmp_path), filename_pattern='{{ "%03d" % seq }}')
        )

        assert len(written) == 1
        assert cancelled == []
        assert errors and "transport jam" in errors[-1]

    def test_writes_ir_sidecar_when_frame_carries_one(self, fake_coolscanpy, tmp_path) -> None:
        ir = np.random.randint(0, 65535, (4, 6), dtype=np.uint16)
        _open(fake_coolscanpy, fake_coolscanpy.Roll(frames=[_frame(fake_coolscanpy, slot=1, ir=ir)]))
        worker = RollWorker()
        written = []
        worker.frame_written.connect(written.append)

        worker.run_batch_scan(
            RollBatchScanRequest(device_id="ls5000-usb-001", slots=(1,), output_folder=str(tmp_path), filename_pattern='{{ "%03d" % seq }}')
        )

        assert written[0].ir_path is not None
        assert written[0].ir_path.endswith("_IR.tif")
        assert os.path.exists(written[0].ir_path)


class TestSafeStopAndShutdown:
    def test_safe_stop_before_any_roll_is_open_is_a_no_op(self, fake_coolscanpy) -> None:
        RollWorker().safe_stop()  # must not raise

    def test_safe_stop_forwards_to_the_open_roll(self, fake_coolscanpy) -> None:
        roll, _device = _open(fake_coolscanpy)
        worker = RollWorker()
        worker.run_preview(RollPreviewRequest(device_id="ls5000-usb-001"))

        worker.safe_stop()

        assert roll.safe_stop_called is True

    def test_shutdown_stops_and_closes(self, fake_coolscanpy) -> None:
        roll, _device = _open(fake_coolscanpy)
        worker = RollWorker()
        worker.run_preview(RollPreviewRequest(device_id="ls5000-usb-001"))

        worker.shutdown()

        assert roll.safe_stop_called is True
        assert roll.closed is True

    def test_shutdown_without_any_roll_open_does_not_raise(self) -> None:
        RollWorker().shutdown()


class TestBatchScanOutputTiers:
    """RollBatchScanRequest's tier fields, forwarded to RollScanningService
    .write_frame -- see tests/roll/test_service.py for the full tier-behavior
    matrix (naming, receipt provenance, every degrade path); these tests only
    pin the plumbing between the request and the service call."""

    def test_request_defaults_write_only_tier_1(self) -> None:
        """A request built without naming the tier fields (every pre-existing
        test in this file builds one this way) matches RollScanSettings
        .defaults() -- unchanged from before Tier 2/3 existed."""
        req = RollBatchScanRequest(device_id="d", slots=(1,), output_folder="/tmp/x", filename_pattern="p")

        assert req.write_unrepaired is True
        assert req.write_repaired is False
        assert req.write_positive is False
        assert req.repair_mode == "exact"

    def test_tier_flags_are_forwarded_to_write_frame(self, fake_coolscanpy, tmp_path, monkeypatch) -> None:
        _open(fake_coolscanpy, fake_coolscanpy.Roll(frames=[_frame(fake_coolscanpy, slot=1)]))
        worker = RollWorker()
        captured = {}
        real_write_frame = worker._service.write_frame

        def _spy(frame, output_folder, filename_pattern, **kwargs):
            captured.update(kwargs)
            return real_write_frame(frame, output_folder, filename_pattern, **kwargs)

        monkeypatch.setattr(worker._service, "write_frame", _spy)

        worker.run_batch_scan(
            RollBatchScanRequest(
                device_id="ls5000-usb-001",
                slots=(1,),
                output_folder=str(tmp_path),
                filename_pattern='{{ "%03d" % seq }}',
                write_unrepaired=False,
                write_repaired=True,
                write_positive=True,
                repair_mode="hybrid",
            )
        )

        # write_repaired/write_positive=True with no repair engine registered degrades
        # gracefully (see test_service.py) rather than raising, so this exercises the
        # real write_frame end to end without needing a fake engine here.
        assert captured == {
            "write_unrepaired": False,
            "write_repaired": True,
            "write_positive": True,
            "repair_mode": "hybrid",
        }

    def test_repaired_tier_writes_through_the_worker_with_a_registered_engine(self, fake_coolscanpy, fake_repair_engine, tmp_path) -> None:
        ir = np.zeros((4, 6), dtype=np.uint16)
        _open(fake_coolscanpy, fake_coolscanpy.Roll(frames=[_frame(fake_coolscanpy, slot=1, ir=ir)]))
        worker = RollWorker()
        written = []
        worker.frame_written.connect(written.append)

        worker.run_batch_scan(
            RollBatchScanRequest(
                device_id="ls5000-usb-001",
                slots=(1,),
                output_folder=str(tmp_path),
                filename_pattern='{{ "%03d" % seq }}',
                write_unrepaired=True,
                write_repaired=True,
            )
        )

        assert len(written) == 1
        assert written[0].repaired_rgb_path is not None
        assert os.path.exists(written[0].repaired_rgb_path)
        assert fake_repair_engine.calls  # the worker's request actually reached the engine
