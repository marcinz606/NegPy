"""MSYS2 adapter tests, with native ABI probes when its runtime is installed."""

import ctypes as ct
import gc
import sys
from types import SimpleNamespace

import pytest

from negpy.infrastructure.capture import gphoto, gphoto_windows as win


def _runtime(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "libgphoto2-6.dll").touch()
    for name in ("libgphoto2", "libgphoto2_port"):
        drivers = tmp_path / "lib" / name / "1.0"
        drivers.mkdir(parents=True)
        (drivers / "driver.dll").touch()
    return tmp_path


def test_explicit_prefix_requires_both_plugin_trees(monkeypatch, tmp_path):
    monkeypatch.setenv("NEGPY_GPHOTO2_PREFIX", str(_runtime(tmp_path)))
    assert win.runtime_prefix() == tmp_path.resolve()
    (tmp_path / "lib" / "libgphoto2_port" / "1.0" / "driver.dll").unlink()
    with pytest.raises(FileNotFoundError, match="MSYS2"):
        win.runtime_prefix()


def test_windows_loader_reports_broken_runtime(monkeypatch):
    monkeypatch.setattr(gphoto.sys, "platform", "win32")
    monkeypatch.setattr(gphoto.simulated, "enabled", lambda: False)
    monkeypatch.setattr(win, "ensure_available", lambda: (_ for _ in ()).throw(OSError("missing dependency")))
    with pytest.raises(gphoto.CameraUnavailable, match="missing dependency"):
        gphoto._gp()
    assert not gphoto.camera_backend_available()


def test_event_path_is_copied_before_ucrt_free(monkeypatch):
    payload = win._FilePath(name=b"capture.ARW", folder=b"/store")
    freed = []

    def call(name, *args):
        if name == "gp_camera_new":
            ct.cast(args[0], ct.POINTER(ct.c_void_p))[0] = ct.c_void_p(123)
        elif name == "gp_camera_wait_for_event":
            ct.cast(args[2], ct.POINTER(ct.c_int))[0] = win.GP_EVENT_FILE_ADDED
            ct.cast(args[3], ct.POINTER(ct.c_void_p))[0] = ct.addressof(payload)
        return 0

    lib = SimpleNamespace(
        call=call, dll=SimpleNamespace(gp_camera_unref=lambda p: None), crt=SimpleNamespace(free=lambda p: freed.append(p.value))
    )
    monkeypatch.setattr(win, "_library", lambda: lib)
    camera = win.Camera()
    kind, path = camera.wait_for_event(50)
    assert kind == win.GP_EVENT_FILE_ADDED
    assert (path.name, path.folder) == ("capture.ARW", "/store")
    assert freed == [ct.addressof(payload)]
    payload.name = b"overwritten"
    assert path.name == "capture.ARW"


def test_failed_download_releases_the_native_file(monkeypatch):
    released = []

    def call(name, *args):
        if name in ("gp_camera_new", "gp_file_new"):
            ct.cast(args[0], ct.POINTER(ct.c_void_p))[0] = ct.c_void_p(123)
        elif name == "gp_camera_file_get":
            assert args[1:4] == (b"/store", b"capture.ARW", win.GP_FILE_TYPE_NORMAL)
            raise win.GPhoto2Error(-7, "I/O problem")
        return 0

    lib = SimpleNamespace(
        call=call,
        dll=SimpleNamespace(gp_camera_unref=lambda p: None, gp_file_unref=lambda p: released.append(p.value)),
    )
    monkeypatch.setattr(win, "_library", lambda: lib)
    camera = win.Camera()
    with pytest.raises(win.GPhoto2Error) as error:
        camera.file_get("/store", "capture.ARW", win.GP_FILE_TYPE_NORMAL)
    assert error.value.code == -7
    del error
    gc.collect()
    assert released == [123]


@pytest.fixture
def native():
    if sys.platform != "win32":
        pytest.skip("MSYS2 UCRT64 ABI probe")
    try:
        win.runtime_prefix()
    except FileNotFoundError:
        pytest.skip("MSYS2 libgphoto2 is not installed")
    lib = win._library()
    lib.dll.gp_widget_new.argtypes = [ct.c_int, ct.c_char_p, ct.POINTER(ct.c_void_p)]
    lib.dll.gp_widget_new.restype = ct.c_int
    lib.dll.gp_widget_add_choice.argtypes = [ct.c_void_p, ct.c_char_p]
    lib.dll.gp_widget_add_choice.restype = ct.c_int
    lib.dll.gp_file_set_data_and_size.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_ulong]
    lib.dll.gp_file_set_data_and_size.restype = ct.c_int
    lib.crt.malloc.argtypes = [ct.c_size_t]
    lib.crt.malloc.restype = ct.c_void_p
    return lib


@pytest.mark.parametrize(
    "kind,value",
    [(win.GP_WIDGET_TEXT, "PC Remote"), (win.GP_WIDGET_RADIO, "1/125"), (win.GP_WIDGET_TOGGLE, 1), (win.GP_WIDGET_RANGE, 2.5)],
)
def test_native_widget_value_abi(native, kind, value):
    pointer = ct.c_void_p()
    native.call("gp_widget_new", kind, b"test", ct.byref(pointer))
    widget = win._Widget(native, pointer, "gp_widget_unref")
    if kind == win.GP_WIDGET_RADIO:
        assert gphoto._safe_value(win, widget) is None
        native.call("gp_widget_add_choice", pointer, b"1/125")
        assert widget.get_value() is None
        assert widget.get_choice(0) == "1/125"
    widget.set_value(value)
    assert widget.get_value() == value


def test_native_file_copies_binary_data(native):
    data = b"\x00RAW\xff\x00DATA"
    allocated = native.crt.malloc(len(data))
    assert allocated
    ct.memmove(allocated, data, len(data))
    file = win.Camera()._file()
    native.call("gp_file_set_data_and_size", file._pointer, allocated, len(data))
    assert file.get_data_and_size() == data
    file._release()
    assert data == b"\x00RAW\xff\x00DATA"
