"""The simulated camera, Scanlight and scanners that `make run-sim` starts the app with."""

import importlib
import os
import threading
import time

import numpy as np
import pytest

from negpy.infrastructure.capture import gphoto
from negpy.infrastructure.capture.gphoto import GphotoCamera
from negpy.infrastructure.capture.raw_demosaic import linear_demosaic
from negpy.infrastructure.capture.scanlight import Scanlight
from negpy.infrastructure.scanners import registry
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.simulated.gphoto import MODEL, SimGphoto
from negpy.infrastructure.simulated.scanner import SimulatedBackend
from negpy.services.capture.calibration import CalibrationService, Roi, meter_base

# Inside the clear-base band above the picture.
_REBATE = Roi(0.3, 0.07, 0.4, 0.04)


@pytest.fixture
def sim_env(monkeypatch):
    monkeypatch.setenv("NEGPY_SIMULATE_HARDWARE", "1")


@pytest.fixture
def camera(tmp_path):
    cam = GphotoCamera(gp_module=SimGphoto(), jpeg_path=str(tmp_path / "live.jpg"), settings_path=str(tmp_path / "live.json"))
    cam.open()
    yield cam
    cam.close()


def test_camera_opens_with_settings_and_no_aperture(camera):
    assert camera.model == MODEL
    settings = camera.read_settings()
    assert [o["label"] for o in settings["iso"]["options"]] == ["100", "200", "400", "800"]
    assert settings["shutter"]["options"]
    assert "aperture" not in settings


def test_camera_streams_a_jpeg_preview(camera):
    camera.start()
    deadline = time.monotonic() + 5
    while not os.path.exists(camera.jpeg_path) and time.monotonic() < deadline:
        time.sleep(0.05)
    with open(camera.jpeg_path, "rb") as f:
        assert f.read(2) == b"\xff\xd8"


def _base_signal(path):
    img = linear_demosaic(path, half_size=True)
    return [meter_base(img[..., c], _REBATE) for c in range(3)]


def test_still_is_a_raw_exposed_by_the_lit_channel(camera, tmp_path, sim_env):
    light = Scanlight()
    try:
        light.set_color(100, 0, 0)
        dim = camera.capture(str(tmp_path / "dim.raw"))
        light.set_color(200, 0, 0)
        bright = camera.capture(str(tmp_path / "bright.raw"))
    finally:
        light.close()

    assert dim.endswith("dim.DNG") and os.path.getsize(dim) >= 8 * 1024 * 1024
    r_dim, g_dim, b_dim = _base_signal(dim)
    r_bright, _, _ = _base_signal(bright)
    assert r_dim > 10 * max(g_dim, b_dim)
    assert r_bright / r_dim == pytest.approx(2.0, rel=0.05)


def test_calibration_reaches_target_on_the_simulated_rig(camera, tmp_path, sim_env):
    light = Scanlight()
    try:
        service = CalibrationService(light, camera, lambda p: linear_demosaic(p, half_size=True), settle_s=0.0)
        result = service.calibrate(_REBATE, str(tmp_path / "cal"))
    finally:
        light.close()
    assert {c.channel for c in result.channels.values()} == {"R", "G", "B"}


def test_flag_selects_the_simulated_gphoto_module(sim_env):
    assert isinstance(gphoto._gp(), SimGphoto)
    assert gphoto.list_cameras() == [{"model": MODEL, "port": "usb:sim"}]


def test_scanlight_answers_and_reports_temperature(sim_env):
    light = Scanlight()
    try:
        assert light.port == "sim"
        assert light.get_fw_version() == (7, 3)
        light.set_color(10, 20, 30)
        deadline = time.monotonic() + 2
        while light.last_temp_c is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert light.last_temp_c is not None
    finally:
        light.close()


def test_scanner_lists_one_device_per_panel_shape():
    devices = SimulatedBackend().list_devices()
    caps = {d.id: d.capabilities for d in devices}
    assert all(c.sources for c in caps.values())
    assert caps["sim:feeder"].adapter_frame_capacity == 6
    assert caps["sim:prescan"].prescan
    assert caps["sim:roll"].roll_discovery


def test_scan_returns_a_negative_with_ir():
    backend = SimulatedBackend()
    result = backend.scan("sim:feeder", ScanParams(dpi=1000, depth=16, capture_ir=True, frame=2), lambda *_: None, threading.Event())
    assert result.rgb.dtype == np.uint16 and result.rgb.shape[2] == 3
    assert result.ir is not None and result.ir.shape == result.rgb.shape[:2]
    # Orange mask: blue is the densest channel inside the frame.
    h, w = result.ir.shape
    centre = result.rgb[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3].reshape(-1, 3).mean(axis=0)
    assert centre[0] > centre[1] > centre[2]


def test_scan_stops_when_cancelled():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        SimulatedBackend().scan("sim:feeder", ScanParams(dpi=1000, depth=8, capture_ir=False), lambda *_: None, cancel)


def test_roll_previews_every_slot_and_asks_to_confirm_the_last():
    backend = SimulatedBackend()
    device = backend.list_devices()[2]
    roll = backend.open_roll(device, dpi=500)
    previews = list(roll.preview(range(1, backend.detect_frames(device.id) + 1), cancel=threading.Event()))
    assert [p.slot for p in previews] == [1, 2, 3, 4, 5, 6]
    assert [p.needs_approval for p in previews] == [False] * 5 + [True]
    roll.approve(6)
    assert not next(roll.preview([6], cancel=threading.Event())).needs_approval


def test_flag_registers_the_simulated_backend_as_default(monkeypatch):
    monkeypatch.setenv("NEGPY_SIMULATE_HARDWARE", "1")
    try:
        importlib.reload(registry)
        assert registry.backend_choices()[0] == ("sim", "Simulated")
        assert registry.DEFAULT_BACKEND_ID == "sim"
        assert isinstance(registry.create_backend("sim"), SimulatedBackend)
    finally:
        monkeypatch.delenv("NEGPY_SIMULATE_HARDWARE")
        importlib.reload(registry)
    assert "sim" not in registry.BACKENDS
