"""CaptureWorker cancellation behavior at the camera/service boundary."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from negpy.desktop.workers.capture_worker import CalibrationRequest, CaptureRequest, CaptureWorker, SensorResponseRequest
from negpy.services.capture.calibration import Roi


class CancellingCamera:
    def __init__(self, worker: CaptureWorker) -> None:
        self.worker = worker

    def capture(self, out_path: str, shutter=None, iso=None, aperture=None) -> str:
        path = os.path.splitext(out_path)[0] + ".ARW"
        with open(path, "wb") as raw:
            raw.truncate(8 * 1024 * 1024)
        self.worker.cancel()  # Stop pressed while the camera is downloading the RAW.
        return path


class FakeLight:
    def set_color(self, **_channels) -> None:
        pass

    def off(self) -> None:
        pass


class FailingControlCamera:
    def __init__(self) -> None:
        self.closed = False

    def is_open(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True

    def set_focus_magnifier(self, _on: bool) -> None:
        raise RuntimeError("USB disconnected")

    def set_focus_magnifier_at(self, _x: int, _y: int) -> None:
        raise RuntimeError("USB disconnected")

    def set_iso(self, _raw: int) -> None:
        raise RuntimeError("USB disconnected")

    def set_shutter(self, _raw: int) -> None:
        pass

    def set_aperture(self, _raw: int) -> None:
        pass


@pytest.mark.parametrize(
    ("slot", "args"),
    [
        ("set_focus_magnifier", (True,)),
        ("set_focus_magnifier_pos", (320, 240)),
        ("set_camera_setting", ("iso", 1)),
    ],
)
def test_camera_control_slot_failure_is_recoverable(slot, args, caplog):
    worker = CaptureWorker()
    camera = FailingControlCamera()
    worker._camera = camera
    errors = []
    worker.error.connect(errors.append)

    getattr(worker, slot)(*args)

    assert camera.closed
    assert errors and "USB disconnected" in errors[-1]
    assert "Reconnect" in errors[-1]
    assert f"{slot} failed" in caplog.text


def test_normal_capture_cancel_before_promotion_preserves_retake(tmp_path, monkeypatch):
    existing = tmp_path / "Roll01_Frame007.ARW"
    existing.write_bytes(b"existing-good-raw")
    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: CancellingCamera(worker))
    finished = []
    worker.finished.connect(finished.append)

    worker.run_capture(
        CaptureRequest(
            roll_name="Roll01",
            frame_number=7,
            output_folder=str(tmp_path),
            levels=(200, 180, 255),
            rgb_mode=False,
            is_retake=True,
        )
    )

    assert existing.read_bytes() == b"existing-good-raw"
    assert finished == []


def test_scanlight_white_cancel_before_promotion_preserves_retake(tmp_path, monkeypatch):
    existing = tmp_path / "Slide01_Frame003.ARW"
    existing.write_bytes(b"existing-good-raw")
    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: CancellingCamera(worker))
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: FakeLight())
    finished = []
    worker.finished.connect(finished.append)

    worker.run_capture(
        CaptureRequest(
            roll_name="Slide01",
            frame_number=3,
            output_folder=str(tmp_path),
            levels=(200, 180, 255),
            settle_s=0,
            white_mode=True,
            is_retake=True,
        )
    )

    assert existing.read_bytes() == b"existing-good-raw"
    assert finished == []


def test_single_capture_emits_one_file_and_no_triplet(tmp_path, monkeypatch):
    class Camera:
        def capture(self, out_path: str, shutter=None, iso=None, aperture=None) -> str:
            path = os.path.splitext(out_path)[0] + ".ARW"
            with open(path, "wb") as raw:
                raw.truncate(8 * 1024 * 1024)
            return path

    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: Camera())
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: FakeLight())
    finished = []
    worker.finished.connect(finished.append)

    worker.run_capture(
        CaptureRequest(
            roll_name="Roll01", frame_number=2, output_folder=str(tmp_path), levels=(200, 180, 255), settle_s=0, single_capture=True
        )
    )

    assert finished == [[str(tmp_path / "Roll01_Frame002.ARW")]]


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_calibration_uses_disposable_scratch_without_touching_roll(tmp_path, monkeypatch, outcome):
    import negpy.desktop.workers.capture_worker as capture_worker_module

    user_file = tmp_path / "_negpy_calibration.ARW"
    user_file.write_bytes(b"user-owned-raw")
    worker = CaptureWorker()

    class FakeCalibrationService:
        written_path: Path | None = None

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def calibrate(self, _roi, scratch_path, **_kwargs):
            written = Path(scratch_path).with_suffix(".ARW")
            written.write_bytes(b"temporary-calibration-raw")
            FakeCalibrationService.written_path = written
            if outcome == "cancel":
                worker.cancel()
                raise RuntimeError("calibration cancelled")
            if outcome == "error":
                raise RuntimeError("decode failed")
            return SimpleNamespace(levels=(1, 2, 3), shutters=("1/15",) * 3)

    monkeypatch.setattr(capture_worker_module, "CalibrationService", FakeCalibrationService)
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: FakeLight())
    monkeypatch.setattr(worker, "_acquire_camera", lambda: object())

    worker.run_calibration(CalibrationRequest(roi=Roi(0, 0, 1, 1), output_folder=str(tmp_path), settle_s=0))

    assert user_file.read_bytes() == b"user-owned-raw"
    assert FakeCalibrationService.written_path is not None
    assert not FakeCalibrationService.written_path.exists()
    assert not FakeCalibrationService.written_path.parent.exists()


def test_meter_probes_once_in_scratch_and_reports_the_reading(tmp_path, monkeypatch):
    import numpy as np

    import negpy.infrastructure.capture.raw_demosaic as demosaic_module
    from negpy.desktop.workers.capture_worker import ExposureProbeRequest
    from negpy.services.capture.calibration import CLIP_CEILING

    worker = CaptureWorker()
    probes: list[tuple[Path, str | None]] = []

    class Camera:
        def capture(self, path, shutter=None, **_kwargs):
            written = Path(path).with_suffix(".ARW")
            written.write_bytes(b"probe")
            probes.append((written, shutter))
            return str(written)

    monkeypatch.setattr(worker, "_acquire_camera", lambda: Camera())
    monkeypatch.setattr(
        demosaic_module, "linear_demosaic", lambda _path, half_size=False: np.full((60, 90, 3), 0.6 * CLIP_CEILING, np.float32)
    )
    readings = []
    worker.exposure_probe_finished.connect(readings.append)

    worker.run_exposure_probe(ExposureProbeRequest(shutter="1/30", session=7))

    assert len(probes) == 1 and not probes[0][0].exists()
    assert probes[0][1] == "1/30"  # shot at the label the reading is computed against
    session, reading = readings[0]
    assert session == 7  # echoed, so a stopped stream's report can be told apart
    assert reading.shutter == "1/30"
    assert reading.measured == pytest.approx(0.6 * CLIP_CEILING, rel=0.01)


def test_a_white_light_probe_meters_after_the_cameras_white_balance(tmp_path, monkeypatch):
    import numpy as np

    import negpy.infrastructure.capture.raw_demosaic as demosaic_module
    from negpy.desktop.workers.capture_worker import ExposureProbeRequest
    from negpy.services.capture.calibration import CLIP_CEILING

    worker = CaptureWorker()

    class Camera:
        def capture(self, path, shutter=None, **_kwargs):
            written = Path(path).with_suffix(".ARW")
            written.write_bytes(b"probe")
            return str(written)

    monkeypatch.setattr(worker, "_acquire_camera", lambda: Camera())
    monkeypatch.setattr(
        demosaic_module, "linear_demosaic", lambda _path, half_size=False: np.full((60, 90, 3), 0.3 * CLIP_CEILING, np.float32)
    )
    monkeypatch.setattr(demosaic_module, "as_shot_wb", lambda _path: [2.0, 1.0, 1.5])
    readings = []
    worker.exposure_probe_finished.connect(readings.append)

    worker.run_exposure_probe(ExposureProbeRequest(shutter="1/30", wb_decode=True))
    worker.run_exposure_probe(ExposureProbeRequest(shutter="1/30"))

    assert readings[0][1].measured == pytest.approx(0.6 * CLIP_CEILING, rel=0.01)  # red, after its 2x gain
    assert readings[1][1].measured == pytest.approx(0.3 * CLIP_CEILING, rel=0.01)  # a narrowband frame decodes linear


def test_a_probe_that_cannot_be_metered_reports_without_dropping_the_session(tmp_path, monkeypatch):
    import numpy as np

    import negpy.infrastructure.capture.raw_demosaic as demosaic_module
    from negpy.desktop.workers.capture_worker import ExposureProbeRequest

    worker = CaptureWorker()

    class Camera:
        def is_open(self):
            return True

        def close(self):
            pass

        def capture(self, path, shutter=None, **_kwargs):
            written = Path(path).with_suffix(".ARW")
            written.write_bytes(b"probe")
            return str(written)

    worker._camera = Camera()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: worker._camera)
    monkeypatch.setattr(demosaic_module, "linear_demosaic", lambda _path, half_size=False: np.zeros((60, 90, 3), np.float32))
    failures, errors = [], []
    worker.exposure_probe_failed.connect(failures.append)
    worker.error.connect(errors.append)

    worker.run_exposure_probe(ExposureProbeRequest(shutter="Bulb"))

    assert len(failures) == 1 and errors == []
    assert worker._camera is not None  # the session stands; only the math failed


class ClaimedCamera:
    """A body on the bus whose USB claim another program holds (gphoto -53)."""

    def __init__(self) -> None:
        self.open_calls = 0

    def is_open(self) -> bool:
        return False

    def open(self) -> None:
        self.open_calls += 1
        from negpy.infrastructure.capture.gphoto import CameraClaimedError

        raise CameraClaimedError(
            "could not open the camera: [-53] Could not claim the USB device. Close Preview, Photos and Image Capture, then retry."
        )

    def close(self) -> None:
        pass


def test_poll_reports_a_camera_claimed_by_another_app(monkeypatch):
    # macOS hands the body to Preview/Photos/Image Capture the moment one of them opens.
    # Enumeration still succeeds, so the camera dot showed a healthy "connected" while every
    # open failed with -53 — the poll must surface the claim so the UI can say what to do.
    import negpy.desktop.workers.capture_worker as capture_worker_module

    worker = CaptureWorker()
    worker._camera = ClaimedCamera()
    monkeypatch.setattr(capture_worker_module, "list_cameras", lambda: [{"model": "USB PTP Class Camera"}])
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: (_ for _ in ()).throw(RuntimeError("no light")))
    seen: list[dict] = []
    worker.poll_status.connect(seen.append)

    with pytest.raises(Exception):
        worker._acquire_camera()  # as any live-view/scan/calibration attempt would
    # The failed open flips the dot IMMEDIATELY (rig find: waiting for the next 3 s timer
    # tick reads as "I pressed Scan and nothing happened" while the dot stays green).
    assert seen and seen[-1]["usb_ok"] is True
    assert seen[-1]["usb_claimed_elsewhere"] is True

    # A claimed body routinely flaps OFF gphoto's enumeration while the other app holds it
    # (macOS daemons answer the bus in our stead). That is part of the claimed state, not an
    # unplug — the verdict must survive it, or the dot snaps back to green within one tick
    # (the exact regression seen on the rig).
    monkeypatch.setattr(capture_worker_module, "list_cameras", lambda: [])
    worker.poll_connection("")
    assert seen[-1]["usb_ok"] is True
    assert seen[-1]["usb_claimed_elsewhere"] is True

    # Without the claim verdict, an empty bus means what it says: no camera.
    worker._claimed_elsewhere = False
    worker.poll_connection("")
    assert seen[-1]["usb_ok"] is False


def test_poll_self_heals_once_the_other_app_releases_the_camera(monkeypatch):
    # Only an open attempt can clear the verdict, but Scan/Calibrate are gated while it
    # stands — the user cannot trigger one (rig find: dot stayed red after closing Preview).
    # In the claimed state we hold no session, so the poll probes the open itself: the dot
    # turns green by itself, and the probe session is released again immediately.
    import negpy.desktop.workers.capture_worker as capture_worker_module

    class ReleasedCamera:
        model = "ILCE-7CM2"

        def __init__(self) -> None:
            self.opened = False
            self.closed = False

        def is_open(self) -> bool:
            return self.opened and not self.closed

        def open(self) -> None:
            self.opened = True

        def close(self) -> None:
            self.closed = True

    worker = CaptureWorker()
    worker._claimed_elsewhere = True  # Preview held it during the last attempt
    cam = ReleasedCamera()
    worker._camera = cam
    monkeypatch.setattr(capture_worker_module, "list_cameras", lambda: [{"model": "USB PTP Class Camera"}])
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: (_ for _ in ()).throw(RuntimeError("no light")))
    seen: list[dict] = []
    worker.poll_status.connect(seen.append)
    worker.poll_connection("")
    assert seen[-1]["usb_claimed_elsewhere"] is False  # healed without any user action
    assert seen[-1]["usb_model"] == "ILCE-7CM2"  # the probe read the real name off the body
    # The reclaimed session is kept. Handing it back gives the OS camera daemon an opening it
    # holds for minutes, and the next thing the user does is exactly what needs the camera.
    assert cam.opened and not cam.closed
    assert worker._holds_camera()


def test_a_setting_write_takes_the_session_when_live_view_has_released_it(monkeypatch):
    # A closed Live View window releases the session; a card stepper write opens it again.
    class Body:
        def __init__(self) -> None:
            self.writes: list[tuple] = []
            self._open = False

        def is_open(self) -> bool:
            return self._open

        def open(self) -> None:
            self._open = True

        def set_shutter(self, raw, label=None) -> None:
            self.writes.append(("shutter", raw, label))

        def set_iso(self, raw, label=None) -> None:
            self.writes.append(("iso", raw, label))

        set_aperture = set_iso

    worker = CaptureWorker()
    body = Body()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: (body.open(), body)[1])
    applied: list[str] = []
    worker.camera_setting_applied.connect(applied.append)

    worker.set_camera_setting("shutter", 29, "1/20")

    assert body.writes == [("shutter", 29, "1/20")]
    assert applied == ["shutter"]  # the panel's Scan gate counts this report


def test_a_setting_write_with_no_camera_reports_and_still_releases_the_scan_gate(monkeypatch):
    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: (_ for _ in ()).throw(RuntimeError("unplugged")))
    errors: list[str] = []
    applied: list[str] = []
    worker.error.connect(errors.append)
    worker.camera_setting_applied.connect(applied.append)

    worker.set_camera_setting("shutter", 29, "1/20")

    assert applied == ["shutter"]
    assert errors and "Reconnect" in errors[0]


def test_a_setting_write_stays_quiet_while_another_app_holds_the_body(monkeypatch):
    worker = CaptureWorker()
    worker._claimed_elsewhere = True
    monkeypatch.setattr(worker, "_acquire_camera", lambda: (_ for _ in ()).throw(AssertionError("must not probe")))
    errors: list[str] = []
    applied: list[str] = []
    worker.error.connect(errors.append)
    worker.camera_setting_applied.connect(applied.append)

    worker.set_camera_setting("shutter", 29, "1/20")

    assert errors == [] and applied == ["shutter"]


def test_poll_identifies_the_body_on_first_sight(monkeypatch):
    # libgphoto2's database labels post-database bodies "USB PTP Class Camera" (the a7C II is
    # not in it, #431); the real name lives on the device. The first poll after a body appears
    # probes it once (open → read → close), so the dot shows "ILCE-7CM2" from the start instead
    # of the database placeholder — and a foreign claim is detected before the user's first
    # attempt, not only after it.
    import negpy.desktop.workers.capture_worker as capture_worker_module

    class IdentifiableCamera:
        model = "ILCE-7CM2"

        def __init__(self) -> None:
            self.open_calls = 0
            self._open = False

        def is_open(self) -> bool:
            return self._open

        def open(self) -> None:
            self.open_calls += 1
            self._open = True

        def close(self) -> None:
            self._open = False

    worker = CaptureWorker()
    cam = IdentifiableCamera()
    worker._camera = cam
    monkeypatch.setattr(capture_worker_module, "list_cameras", lambda: [{"model": "USB PTP Class Camera"}])
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: (_ for _ in ()).throw(RuntimeError("no light")))
    seen: list[dict] = []
    worker.poll_status.connect(seen.append)
    worker.poll_connection("")
    assert seen[-1]["usb_model"] == "ILCE-7CM2"  # real name, not the database placeholder
    assert worker._holds_camera()  # the probe keeps the claim it won
    worker.poll_connection("")
    assert cam.open_calls == 1  # identified once, not per tick


def test_poll_publishes_the_held_bodys_settings_while_idle(monkeypatch):
    # Only the preview thread publishes the settings file, so a preset picked before Live View
    # opened showed every exposure stepper empty. The poll publishes from the session it holds.
    import negpy.desktop.workers.capture_worker as capture_worker_module

    class HeldCamera:
        model = "ILCE-7RM5"

        def __init__(self) -> None:
            self.published = 0

        def is_open(self) -> bool:
            return True

        def publish_settings(self) -> None:
            self.published += 1

    worker = CaptureWorker()
    worker._camera = HeldCamera()
    worker._identify_attempted = True
    monkeypatch.setattr(capture_worker_module, "list_cameras", lambda: [{"model": "USB PTP Class Camera"}])
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: (_ for _ in ()).throw(RuntimeError("no light")))
    worker.poll_connection("")
    worker.poll_connection("")
    assert worker._camera.published == 2  # every idle tick, so a dial turn shows within one


def test_poll_identify_is_one_shot_for_a_body_that_will_not_open(monkeypatch):
    # A body that fails to open for a non-claim reason must not be hammered with an open
    # attempt every 3 s tick — one try per bus appearance, then the placeholder name stands.
    import negpy.desktop.workers.capture_worker as capture_worker_module

    class BrokenCamera:
        def __init__(self) -> None:
            self.open_calls = 0

        def is_open(self) -> bool:
            return False

        def open(self) -> None:
            self.open_calls += 1
            from negpy.infrastructure.capture.gphoto import GphotoError

            raise GphotoError("could not open the camera: [-1] Unspecified error")

        def close(self) -> None:
            pass

    worker = CaptureWorker()
    cam = BrokenCamera()
    worker._camera = cam
    monkeypatch.setattr(capture_worker_module, "list_cameras", lambda: [{"model": "USB PTP Class Camera"}])
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: (_ for _ in ()).throw(RuntimeError("no light")))
    seen: list[dict] = []
    worker.poll_status.connect(seen.append)
    worker.poll_connection("")
    worker.poll_connection("")
    assert cam.open_calls == 1  # tried once, then left alone
    assert seen[-1]["usb_ok"] is True and seen[-1]["usb_claimed_elsewhere"] is False
    # A re-plug re-arms the identify: the next body may be a different one. (The empty-bus
    # poll drops the camera object, so the fake must be re-injected for the re-appearance.)
    monkeypatch.setattr(capture_worker_module, "list_cameras", lambda: [])
    worker.poll_connection("")
    assert worker._identify_attempted is False  # re-armed by the disappearance
    worker._camera = cam
    monkeypatch.setattr(capture_worker_module, "list_cameras", lambda: [{"model": "USB PTP Class Camera"}])
    worker.poll_connection("")
    assert cam.open_calls == 2


def test_a_non_claim_open_failure_resets_the_claimed_verdict():
    # An unplugged body fails to open with a plain GphotoError. That must clear the claim
    # verdict, or the "in use — close Preview…" advice sticks to a camera that is simply gone.
    class OpenFailsOtherwise:
        def is_open(self) -> bool:
            return False

        def open(self) -> None:
            from negpy.infrastructure.capture.gphoto import GphotoError

            raise GphotoError("could not open the camera: [-105] Unknown model")

        def close(self) -> None:
            pass

    worker = CaptureWorker()
    worker._claimed_elsewhere = True  # left over from the Preview episode
    worker._camera = OpenFailsOtherwise()
    with pytest.raises(Exception):
        worker._acquire_camera()
    assert worker._claimed_elsewhere is False


def test_successful_open_clears_the_claimed_state(monkeypatch):
    class OpensFine:
        model = "ILCE-7CM2"

        def is_open(self) -> bool:
            return False

        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

    worker = CaptureWorker()
    worker._claimed_elsewhere = True  # left over from a failed attempt
    worker._camera = OpensFine()
    worker._acquire_camera()
    assert worker._claimed_elsewhere is False


def _sensor_run(worker, req=None):
    measured, failed = [], []
    worker.sensor_response_measured.connect(measured.append)
    worker.sensor_response_failed.connect(failed.append)
    worker.measure_sensor_response(req or SensorResponseRequest(settle_s=0))
    return measured, failed


def test_sensor_response_measures_on_the_simulated_rig_and_releases_the_camera(monkeypatch):
    monkeypatch.setenv("NEGPY_SIMULATE_HARDWARE", "1")
    worker = CaptureWorker()
    try:
        measured, failed = _sensor_run(worker)
        assert not failed and measured[0].shape == (3, 3)
        assert all(measured[0][i, i] == measured[0][:, i].max() for i in range(3))
        assert not worker._holds_camera()
    finally:
        worker.shutdown()


def test_sensor_response_keeps_a_session_it_did_not_open_and_restores_the_light(monkeypatch):
    monkeypatch.setenv("NEGPY_SIMULATE_HARDWARE", "1")
    from negpy.infrastructure.simulated.scanlight import current_color

    worker = CaptureWorker()
    try:
        worker._acquire_camera()
        worker.set_light(10, 20, 30, 0, "")
        measured, _failed = _sensor_run(worker)
        assert measured and worker._holds_camera()
        assert tuple(current_color()[:3]) == (10, 20, 30)
    finally:
        worker.shutdown()


def test_an_abandoned_sensor_response_never_touches_the_hardware(monkeypatch):
    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: pytest.fail("light used"))
    req = SensorResponseRequest()
    req.cancel.set()
    assert _sensor_run(worker, req) == ([], [])


@pytest.mark.parametrize("broken, drops", [("_ensure_light", False), ("read_settings", True)])
def test_a_failed_sensor_response_drops_a_held_session_only_for_a_camera_fault(monkeypatch, broken, drops):
    monkeypatch.setenv("NEGPY_SIMULATE_HARDWARE", "1")
    worker = CaptureWorker()
    dropped = []
    worker.live_view_failed.connect(dropped.append)

    def _fail(*_args):
        raise RuntimeError("it broke")

    try:
        camera = worker._acquire_camera()
        monkeypatch.setattr(camera if broken == "read_settings" else worker, broken, _fail)
        measured, failed = _sensor_run(worker)
        assert not measured and failed == ["it broke"]
        assert bool(dropped) == drops and worker._holds_camera() != drops
    finally:
        worker.shutdown()


def test_presence_poll_reports_the_simulated_rig_without_opening_the_camera(monkeypatch):
    monkeypatch.setenv("NEGPY_SIMULATE_HARDWARE", "1")
    worker = CaptureWorker()
    seen = []
    worker.presence_polled.connect(lambda camera, light: seen.append((camera, light)))
    try:
        worker.poll_presence("")
        assert seen == [(True, True)] and not worker._holds_camera()
        worker._claimed_elsewhere = True
        worker.poll_presence("")
        assert seen[-1] == (False, True)
    finally:
        worker.shutdown()


def test_presence_poll_reports_a_missing_scanlight(monkeypatch):
    import negpy.desktop.workers.capture_worker as capture_worker_module

    worker = CaptureWorker()
    seen = []
    worker.presence_polled.connect(lambda camera, light: seen.append((camera, light)))
    monkeypatch.setattr(capture_worker_module, "list_cameras", lambda: [])
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: (_ for _ in ()).throw(RuntimeError("No serial ports found.")))
    worker.poll_presence("")
    assert seen == [(False, False)]


def test_a_probe_that_cannot_be_decoded_keeps_the_session(tmp_path, monkeypatch):
    import negpy.infrastructure.capture.raw_demosaic as demosaic_module
    from negpy.desktop.workers.capture_worker import ExposureProbeRequest

    worker = CaptureWorker()

    class Camera:
        def is_open(self):
            return True

        def close(self):
            raise AssertionError("the session must stay open")

        def capture(self, path, shutter=None, **_kwargs):
            written = Path(path).with_suffix(".ARW")
            written.write_bytes(b"probe")
            return str(written)

    def reject(_path, half_size=False):
        raise ValueError("unsupported file format")

    worker._camera = Camera()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: worker._camera)
    monkeypatch.setattr(demosaic_module, "linear_demosaic", reject)
    failures, errors = [], []
    worker.exposure_probe_failed.connect(failures.append)
    worker.error.connect(errors.append)

    worker.run_exposure_probe(ExposureProbeRequest(shutter="1/30"))

    assert failures == [(0, "unsupported file format", False)] and errors == []
    assert worker._camera is not None


def test_a_probe_whose_capture_fails_reports_the_lost_camera_with_its_session(tmp_path, monkeypatch):
    from negpy.desktop.workers.capture_worker import ExposureProbeRequest

    worker = CaptureWorker()

    class Camera:
        def is_open(self):
            return True

        def close(self):
            pass

        def capture(self, path, shutter=None, **_kwargs):
            raise RuntimeError("could not set shutter")

    worker._camera = Camera()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: worker._camera)
    failures, errors = [], []
    worker.exposure_probe_failed.connect(failures.append)
    worker.error.connect(errors.append)

    worker.run_exposure_probe(ExposureProbeRequest(shutter="1/30", session=3))

    assert failures == [(3, "could not set shutter", True)] and errors == []


class _TripletCamera:
    """Writes every exposure it is asked for; the `_IR` stem fails when `ir_fails` is set."""

    def __init__(self, ir_fails: bool = False, cur: int = 7):
        self.ir_fails = ir_fails
        self.cur = cur
        self.shots: list[tuple[str, object]] = []
        self.restored: list[int] = []

    def read_settings(self):
        return {"shutter": {"cur": self.cur, "options": [{"label": "1/60", "raw": 7}]}}

    def set_shutter(self, raw):
        self.restored.append(raw)

    def capture(self, out_path, shutter=None, iso=None, aperture=None):
        path = os.path.splitext(out_path)[0] + ".ARW"
        self.shots.append((os.path.basename(path), shutter))
        if self.ir_fails and path.endswith("_IR.ARW"):
            raise RuntimeError("the body refused the infrared shutter")
        with open(path, "wb") as f:
            f.write(b"\0" * (9 * 1024 * 1024))
        return path

    def close(self):
        pass


def _run(worker, **fields):
    finished, errors, cancelled = [], [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.run_capture(CaptureRequest(**fields))
    return finished, errors, cancelled


def test_a_failed_infrared_exposure_keeps_the_finished_frame(tmp_path, monkeypatch):
    # The visible exposures are promoted before the infrared one runs, so an infrared failure
    # must not fail the frame: that path deletes every file of the frame as a partial capture.
    camera = _TripletCamera(ir_fails=True)
    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: camera)
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: FakeLight())
    statuses = []
    worker.status.connect(statuses.append)

    finished, errors, _ = _run(
        worker, roll_name="Roll", frame_number=5, output_folder=str(tmp_path), levels=(255, 255, 255), infrared=True, shutter_ir="4"
    )

    assert [os.path.basename(p) for p in finished[0]] == ["Roll_Frame005_R.ARW", "Roll_Frame005_G.ARW", "Roll_Frame005_B.ARW"]
    assert sorted(os.listdir(tmp_path)) == ["Roll_Frame005_B.ARW", "Roll_Frame005_G.ARW", "Roll_Frame005_R.ARW"]
    assert errors == [] and any("Infrared exposure failed" in s for s in statuses)
    assert camera.restored == [7]  # the body is back on its own shutter even so


def test_a_subset_scan_shoots_each_selected_channel_at_its_own_level_and_shutter(tmp_path, monkeypatch):
    camera = _TripletCamera()
    light = FakeLight()
    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: camera)
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: light)

    finished, errors, _ = _run(
        worker,
        roll_name="Roll",
        frame_number=2,
        output_folder=str(tmp_path),
        levels=(200, 150, 100),
        shutters=("1/5", "1/10", "1/20"),
        channels=("G", "B"),
    )

    assert [os.path.basename(p) for p in finished[0]] == ["Roll_Frame002_G.ARW", "Roll_Frame002_B.ARW"]
    assert camera.shots == [("Roll_Frame002_G.ARW", "1/10"), ("Roll_Frame002_B.ARW", "1/20")]
    assert errors == []


def test_an_infrared_exposure_leaves_the_body_alone_when_its_shutter_is_unknown(tmp_path, monkeypatch):
    camera = _TripletCamera(cur=-1)  # the body reports no current shutter
    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: camera)
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: FakeLight())

    finished, _, _ = _run(
        worker, roll_name="Roll", frame_number=1, output_folder=str(tmp_path), levels=(255, 255, 255), infrared=True, shutter_ir="4"
    )

    assert len(finished[0]) == 4 and camera.restored == []


def test_an_infrared_preset_ends_the_frame_with_an_ir_exposure(tmp_path, monkeypatch):
    class WritingCamera:
        def __init__(self):
            self.shutters = []
            self.restored = []

        def read_settings(self):
            return {"shutter": {"cur": 7, "options": [{"label": "1/60", "raw": 7}]}}

        def set_shutter(self, raw):
            self.restored.append(raw)

        def capture(self, out_path, shutter=None, iso=None, aperture=None):
            path = os.path.splitext(out_path)[0] + ".ARW"
            self.shutters.append(shutter)
            with open(path, "wb") as f:
                f.write(b"\0" * (9 * 1024 * 1024))
            return path

        def close(self):
            pass

    camera = WritingCamera()
    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: camera)
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: FakeLight())
    finished = []
    worker.finished.connect(finished.append)

    worker.run_capture(
        CaptureRequest(
            roll_name="Slide01",
            frame_number=3,
            output_folder=str(tmp_path),
            levels=(0, 0, 0),
            white_mode=True,
            w_level=255,
            infrared=True,
            shutter_ir="4",
        )
    )

    assert [os.path.basename(p) for p in finished[0]] == ["Slide01_Frame003.ARW", "Slide01_Frame003_IR.ARW"]
    assert camera.shutters == [None, "4"] and camera.restored == [7]  # the body is back on its own shutter


class RecordingLight:
    def __init__(self) -> None:
        self.colors: list[tuple] = []

    def set_color(self, r=0, g=0, b=0, w=0, ir=0, save=False) -> None:
        self.colors.append((r, g, b, w, ir))

    def off(self) -> None:
        self.colors.append((0, 0, 0, 0, 0))


class RawCamera:
    def capture(self, out_path: str, shutter=None, iso=None, aperture=None) -> str:
        path = os.path.splitext(out_path)[0] + ".ARW"
        with open(path, "wb") as raw:
            raw.truncate(8 * 1024 * 1024)
        return path


@pytest.mark.parametrize("infrared", [False, True])
def test_only_the_frames_last_exposure_leaves_the_end_light_lit(tmp_path, monkeypatch, infrared):
    worker = CaptureWorker()
    light = RecordingLight()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: RawCamera())
    monkeypatch.setattr(worker, "_ensure_light", lambda _port: light)

    worker.run_capture(
        CaptureRequest(
            roll_name="Roll01",
            frame_number=1,
            output_folder=str(tmp_path),
            levels=(200, 180, 255),
            settle_s=0,
            infrared=infrared,
            end_light=(25, 12, 11, 0),
        )
    )

    assert light.colors[-1] == (25, 12, 11, 0, 0)
    assert light.colors.count((25, 12, 11, 0, 0)) == 1
    # The visible exposures hand to the infrared one through off; with no infrared, nothing turns it off.
    assert ((0, 0, 0, 0, 0) in light.colors) is infrared


@pytest.mark.parametrize(("share", "emitted"), [(0.36, [0.36]), (None, [])])
def test_a_single_file_capture_reports_clipping_after_it_finishes(tmp_path, monkeypatch, share, emitted):
    class Camera:
        def capture(self, out_path: str, shutter=None, iso=None, aperture=None) -> str:
            path = os.path.splitext(out_path)[0] + ".ARW"
            with open(path, "wb") as raw:
                raw.truncate(8 * 1024 * 1024)
            return path

    worker = CaptureWorker()
    monkeypatch.setattr(worker, "_acquire_camera", lambda: Camera())
    monkeypatch.setattr("negpy.infrastructure.capture.raw_demosaic.linear_demosaic", lambda _p, half_size=False: None)
    monkeypatch.setattr("negpy.services.capture.meter.picture_clipping", lambda _img: share)
    events = []
    worker.finished.connect(lambda paths: events.append("finished"))
    worker.clipped.connect(lambda _path, s: events.append(s))

    worker.run_capture(
        CaptureRequest(roll_name="Roll01", frame_number=1, output_folder=str(tmp_path), levels=(0, 0, 0), settle_s=0, rgb_mode=False)
    )

    assert events == ["finished", *emitted]


def test_a_failed_clipping_check_keeps_the_frame(tmp_path, monkeypatch):
    def broken(_path, half_size=False):
        raise ValueError("not a raw")

    monkeypatch.setattr("negpy.infrastructure.capture.raw_demosaic.linear_demosaic", broken)
    worker = CaptureWorker()
    errors, clipped = [], []
    worker.error.connect(errors.append)
    worker.clipped.connect(lambda *a: clipped.append(a))

    worker._report_clipping(str(tmp_path / "Roll01_Frame001.ARW"))

    assert errors == [] and clipped == []
