"""Adapter tests for the fuji_ble connector — a fake `fujitrigger` module is injected, so
no Bluetooth hardware and no real fuji-ble-negpy-trigger package are needed (the same
approach #615 uses for coolscanpy). The adapter is the only module under test; it must
never let a fujitrigger type escape and must translate vendor errors to plain RuntimeError.
"""

import importlib.machinery
import sys
import types

import pytest

from negpy.infrastructure.capture import fuji_ble
from negpy.infrastructure.capture.base import Camera


def _fake_fujitrigger(discover_exc=None):
    mod = types.ModuleType("fujitrigger")
    mod.__spec__ = importlib.machinery.ModuleSpec("fujitrigger", loader=None)

    class Flavor:
        def __init__(self, value="secure"):
            self.value = value

    class Pairing:
        def __init__(self, address="", name="", flavor=None, token="", serial=""):
            self.address = address
            self.name = name
            self.flavor = flavor or Flavor()
            self.token = token
            self.serial = serial

    class FujiBleTrigger:
        def __init__(self, pairing, **_kw):
            self.pairing = pairing
            self._connected = False

        def connect(self, timeout=60.0):
            self._connected = True

        def is_connected(self):
            return self._connected

        def fire(self, **_kw):
            pass

        def close(self):
            self._connected = False

    class FolderWatchAcquirer:
        def __init__(self, drop_dir, **_kw):
            self.drop_dir = drop_dir

        def arm(self):
            pass

        def acquire(self, out_path, timeout=60.0, cancel=None):
            import os

            final = os.path.splitext(out_path)[0] + ".RAF"
            os.makedirs(os.path.dirname(final) or ".", exist_ok=True)
            with open(final, "wb") as fh:
                fh.truncate(8 * 1024 * 1024)
            return final

        def close(self):
            pass

    class FujiCamera:
        def __init__(self, trigger, acquirer, cancel=None):
            self.trigger = trigger
            self.acquirer = acquirer

        def is_open(self):
            return self.trigger.is_connected()

        def capture(self, out_path, shutter=None, iso=None, aperture=None):
            self.trigger.connect()
            self.acquirer.arm()
            self.trigger.fire()
            return self.acquirer.acquire(out_path)

        def close(self):
            self.trigger.close()
            self.acquirer.close()

    def discover(timeout=8.0):
        if discover_exc is not None:
            raise discover_exc
        return [
            types.SimpleNamespace(
                name="GFX100II",
                address="AA-BB",
                flavor=Flavor("secure"),
                serial=b"\x01\x02\x03\x04\x05",
                token=b"",
                rssi=-50,
            )
        ]

    mod.Flavor = Flavor
    mod.Pairing = Pairing
    mod.FujiBleTrigger = FujiBleTrigger
    mod.FujiCamera = FujiCamera
    mod.FolderWatchAcquirer = FolderWatchAcquirer
    mod.discover = discover
    return mod


@pytest.fixture
def fake_ft(monkeypatch):
    mod = _fake_fujitrigger()
    monkeypatch.setitem(sys.modules, "fujitrigger", mod)
    return mod


def test_available_true_when_package_present(fake_ft):
    assert fuji_ble.available() is True


def test_available_false_when_package_absent(monkeypatch):
    monkeypatch.setattr(fuji_ble.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setitem(sys.modules, "fujitrigger", None)  # force `import fujitrigger` to fail
    assert fuji_ble.available() is False
    with pytest.raises(RuntimeError, match="fuji-ble-negpy-trigger"):
        fuji_ble._ft()


def test_make_camera_returns_camera_and_captures(fake_ft, tmp_path):
    cfg = {"address": "AA-BB", "name": "GFX100II", "flavor": "secure", "serial": "0102030405", "drop_folder": str(tmp_path)}
    camera = fuji_ble.make_camera(cfg)
    assert isinstance(camera, Camera)
    path = camera.capture(str(tmp_path / "stage" / "Roll001_Frame001_R.raw"))
    assert path.endswith("Roll001_Frame001_R.RAF")
    camera.close()


def test_make_camera_requires_drop_folder(fake_ft):
    with pytest.raises(RuntimeError, match="drop folder"):
        fuji_ble.make_camera({"address": "AA-BB", "flavor": "secure"})


def test_discover_returns_plain_dicts(fake_ft):
    found = fuji_ble.discover()
    assert found == [{"name": "GFX100II", "address": "AA-BB", "flavor": "secure", "serial": "0102030405", "token": "", "rssi": -50}]


def test_pair_returns_pairing_dict(fake_ft):
    ok, message, pairing = fuji_ble.pair()
    assert ok is True
    assert message == "GFX100II"
    assert pairing["address"] == "AA-BB"
    assert pairing["flavor"] == "secure"
    assert pairing["serial"] == "0102030405"


def test_pair_no_camera_found(monkeypatch):
    mod = _fake_fujitrigger()
    mod.discover = lambda timeout=8.0: []
    monkeypatch.setitem(sys.modules, "fujitrigger", mod)
    ok, message, pairing = fuji_ble.pair()
    assert ok is False and pairing == {} and "No Fujifilm camera" in message


def test_vendor_error_is_translated_with_cause(monkeypatch):
    boom = ValueError("ble went sideways")
    mod = _fake_fujitrigger(discover_exc=boom)
    monkeypatch.setitem(sys.modules, "fujitrigger", mod)
    with pytest.raises(RuntimeError) as exc_info:
        fuji_ble.discover()
    assert exc_info.value.__cause__ is boom
