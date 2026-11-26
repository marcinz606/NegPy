"""The libgphoto2 surface used by capture, backed by MSYS2 UCRT64 DLLs."""

from __future__ import annotations

import ctypes as ct
import os
import weakref
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from negpy.kernel.system.paths import get_resource_path

GP_WIDGET_TEXT, GP_WIDGET_RANGE, GP_WIDGET_TOGGLE = 2, 3, 4
GP_WIDGET_RADIO, GP_WIDGET_MENU, GP_WIDGET_DATE = 5, 6, 8
GP_CAPTURE_IMAGE, GP_FILE_TYPE_NORMAL = 0, 1
GP_EVENT_TIMEOUT, GP_EVENT_FILE_ADDED = 1, 2
GP_OPERATION_CAPTURE_PREVIEW, GP_OPERATION_CONFIG = 1 << 3, 1 << 4


class GPhoto2Error(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def runtime_prefix() -> Path:
    """Find a complete bundled runtime or an MSYS2 UCRT64 prefix."""
    configured = os.environ.get("NEGPY_GPHOTO2_PREFIX")
    candidates = [Path(configured)] if configured else [Path(get_resource_path("gphoto2_runtime")), Path("C:\\msys64\\ucrt64")]
    for prefix in candidates:
        if (prefix / "bin" / "libgphoto2-6.dll").is_file() and all(
            len(list((prefix / "lib" / name).glob("*/*.dll"))) > 0 for name in ("libgphoto2", "libgphoto2_port")
        ):
            return prefix.resolve()
    raise FileNotFoundError(
        "Install MSYS2 UCRT64 libgphoto2 with `pacman -S mingw-w64-ucrt-x86_64-libgphoto2`. "
        "Set NEGPY_GPHOTO2_PREFIX to its ucrt64 directory. See docs/CAMERA_SCANNING.md."
    )


def plugin_directory(prefix: Path, name: str) -> Path:
    directories = sorted({p.parent for p in (prefix / "lib" / name).glob("*/*.dll")})
    if len(directories) != 1:
        raise OSError(f"Expected one {name} driver directory in {prefix}, found {len(directories)}")
    return directories[0]


class _FilePath(ct.Structure):
    _fields_ = [("name", ct.c_char * 128), ("folder", ct.c_char * 1024)]


class _Abilities(ct.Structure):
    _fields_ = [
        ("model", ct.c_char * 128),
        ("status", ct.c_int),
        ("port", ct.c_int),
        ("speed", ct.c_int * 64),
        ("operations", ct.c_int),
        ("file_operations", ct.c_int),
        ("folder_operations", ct.c_int),
        ("usb", ct.c_int * 5),
        ("library", ct.c_char * 1024),
        ("id", ct.c_char * 1024),
        ("device_type", ct.c_int),
        ("reserved", ct.c_int * 7),
    ]


class _Library:
    def __init__(self, prefix: Path) -> None:
        binary_dir = str(prefix / "bin")
        self._dll_directory = os.add_dll_directory(binary_dir)
        # libltdl uses LoadLibrary for drivers, whose dependencies also need the legacy search path.
        os.environ["PATH"] = binary_dir + os.pathsep + os.environ.get("PATH", "")
        os.environ["CAMLIBS"] = str(plugin_directory(prefix, "libgphoto2"))
        os.environ["IOLIBS"] = str(plugin_directory(prefix, "libgphoto2_port"))
        self.dll = ct.CDLL(str(prefix / "bin" / "libgphoto2-6.dll"))
        self.crt = ct.CDLL("ucrtbase.dll")
        self.crt.free.argtypes = [ct.c_void_p]
        self.crt.free.restype = None
        pointer, string, integer = ct.c_void_p, ct.c_char_p, ct.c_int
        out = ct.POINTER(pointer)
        signatures = {
            "gp_camera_new": [out],
            "gp_camera_unref": [pointer],
            "gp_camera_autodetect": [pointer, pointer],
            "gp_camera_init": [pointer, pointer],
            "gp_camera_exit": [pointer, pointer],
            "gp_camera_get_summary": [pointer, pointer, pointer],
            "gp_camera_get_abilities": [pointer, ct.POINTER(_Abilities)],
            "gp_camera_get_single_config": [pointer, string, out, pointer],
            "gp_camera_set_single_config": [pointer, string, pointer, pointer],
            "gp_camera_capture": [pointer, integer, ct.POINTER(_FilePath), pointer],
            "gp_camera_capture_preview": [pointer, pointer, pointer],
            "gp_camera_file_get": [pointer, string, string, integer, pointer, pointer],
            "gp_camera_file_delete": [pointer, string, string, pointer],
            "gp_camera_wait_for_event": [pointer, integer, ct.POINTER(integer), out, pointer],
            "gp_list_new": [out],
            "gp_list_unref": [pointer],
            "gp_list_count": [pointer],
            "gp_list_get_name": [pointer, integer, ct.POINTER(string)],
            "gp_list_get_value": [pointer, integer, ct.POINTER(string)],
            "gp_file_new": [out],
            "gp_file_unref": [pointer],
            "gp_file_get_data_and_size": [pointer, out, ct.POINTER(ct.c_ulong)],
            "gp_widget_unref": [pointer],
            "gp_widget_get_type": [pointer, ct.POINTER(integer)],
            "gp_widget_get_readonly": [pointer, ct.POINTER(integer)],
            "gp_widget_get_name": [pointer, ct.POINTER(string)],
            "gp_widget_count_choices": [pointer],
            "gp_widget_get_choice": [pointer, integer, ct.POINTER(string)],
            "gp_widget_get_value": [pointer, pointer],
            "gp_widget_set_value": [pointer, pointer],
        }
        for name, args in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes, function.restype = args, integer
        self.dll.gp_result_as_string.argtypes = [integer]
        self.dll.gp_result_as_string.restype = string

    def call(self, name: str, *args: Any) -> int:
        result = getattr(self.dll, name)(*args)
        if result < 0:
            message = self.dll.gp_result_as_string(result)
            raise GPhoto2Error(result, message.decode("utf-8", errors="replace") if message else name)
        return result


@lru_cache(maxsize=1)
def _library() -> _Library:
    return _Library(runtime_prefix())


def _text(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if value else ""


def _path(value: _FilePath) -> SimpleNamespace:
    return SimpleNamespace(name=_text(value.name), folder=_text(value.folder))


class _Handle:
    def __init__(self, lib: _Library, pointer: ct.c_void_p, release: str) -> None:
        self._lib, self._pointer = lib, pointer
        self._release = weakref.finalize(self, getattr(lib.dll, release), pointer)


class _List(_Handle):
    def count(self) -> int:
        return self._lib.call("gp_list_count", self._pointer)

    def _get(self, name: str, index: int) -> str:
        value = ct.c_char_p()
        self._lib.call(name, self._pointer, index, ct.byref(value))
        return _text(value.value)

    def get_name(self, index: int) -> str:
        return self._get("gp_list_get_name", index)

    def get_value(self, index: int) -> str:
        return self._get("gp_list_get_value", index)


class _File(_Handle):
    def get_data_and_size(self) -> bytes:
        data, size = ct.c_void_p(), ct.c_ulong()
        self._lib.call("gp_file_get_data_and_size", self._pointer, ct.byref(data), ct.byref(size))
        return ct.string_at(data, size.value) if size.value else b""


class _Widget(_Handle):
    def _integer(self, name: str) -> int:
        value = ct.c_int()
        self._lib.call(name, self._pointer, ct.byref(value))
        return value.value

    def get_type(self) -> int:
        return self._integer("gp_widget_get_type")

    def get_readonly(self) -> int:
        return self._integer("gp_widget_get_readonly")

    def get_name(self) -> str:
        value = ct.c_char_p()
        self._lib.call("gp_widget_get_name", self._pointer, ct.byref(value))
        return _text(value.value)

    def count_choices(self) -> int:
        return self._lib.call("gp_widget_count_choices", self._pointer)

    def get_choice(self, index: int) -> str:
        value = ct.c_char_p()
        self._lib.call("gp_widget_get_choice", self._pointer, index, ct.byref(value))
        return _text(value.value)

    def get_value(self) -> str | float | int | None:
        kind = self.get_type()
        if kind in (GP_WIDGET_TEXT, GP_WIDGET_RADIO, GP_WIDGET_MENU):
            value = ct.c_char_p()
            self._lib.call("gp_widget_get_value", self._pointer, ct.byref(value))
            return _text(value.value) if value.value is not None else None
        if kind in (GP_WIDGET_RANGE, GP_WIDGET_TOGGLE, GP_WIDGET_DATE):
            number = ct.c_float() if kind == GP_WIDGET_RANGE else ct.c_int()
            self._lib.call("gp_widget_get_value", self._pointer, ct.byref(number))
            return number.value
        raise GPhoto2Error(-6, f"Unsupported widget type {kind}")

    def set_value(self, value: str | float | int) -> None:
        kind = self.get_type()
        if kind in (GP_WIDGET_TEXT, GP_WIDGET_RADIO, GP_WIDGET_MENU):
            self._lib.call("gp_widget_set_value", self._pointer, str(value).encode("utf-8"))
        elif kind in (GP_WIDGET_RANGE, GP_WIDGET_TOGGLE, GP_WIDGET_DATE):
            number = ct.c_float(float(value)) if kind == GP_WIDGET_RANGE else ct.c_int(int(value))
            self._lib.call("gp_widget_set_value", self._pointer, ct.byref(number))
        else:
            raise GPhoto2Error(-6, f"Unsupported widget type {kind}")


class Camera(_Handle):
    def __init__(self) -> None:
        lib, pointer = _library(), ct.c_void_p()
        lib.call("gp_camera_new", ct.byref(pointer))
        super().__init__(lib, pointer, "gp_camera_unref")

    @staticmethod
    def autodetect() -> _List:
        lib, pointer = _library(), ct.c_void_p()
        lib.call("gp_list_new", ct.byref(pointer))
        found = _List(lib, pointer, "gp_list_unref")
        lib.call("gp_camera_autodetect", pointer, None)
        return found

    def init(self) -> None:
        self._lib.call("gp_camera_init", self._pointer, None)

    def exit(self) -> None:
        self._lib.call("gp_camera_exit", self._pointer, None)

    def get_summary(self) -> str:
        text = ct.create_string_buffer(32 * 1024)
        self._lib.call("gp_camera_get_summary", self._pointer, text, None)
        return _text(text.value)

    def get_abilities(self) -> SimpleNamespace:
        abilities = _Abilities()
        self._lib.call("gp_camera_get_abilities", self._pointer, ct.byref(abilities))
        return SimpleNamespace(model=_text(abilities.model), operations=abilities.operations)

    def get_single_config(self, name: str) -> _Widget:
        pointer = ct.c_void_p()
        self._lib.call("gp_camera_get_single_config", self._pointer, name.encode("utf-8"), ct.byref(pointer), None)
        return _Widget(self._lib, pointer, "gp_widget_unref")

    def set_single_config(self, name: str, widget: _Widget) -> None:
        self._lib.call("gp_camera_set_single_config", self._pointer, name.encode("utf-8"), widget._pointer, None)

    def capture(self, kind: int) -> SimpleNamespace:
        path = _FilePath()
        self._lib.call("gp_camera_capture", self._pointer, kind, ct.byref(path), None)
        return _path(path)

    def _file(self) -> _File:
        pointer = ct.c_void_p()
        self._lib.call("gp_file_new", ct.byref(pointer))
        return _File(self._lib, pointer, "gp_file_unref")

    def capture_preview(self) -> _File:
        file = self._file()
        self._lib.call("gp_camera_capture_preview", self._pointer, file._pointer, None)
        return file

    def file_get(self, folder: str, name: str, kind: int) -> _File:
        file = self._file()
        self._lib.call("gp_camera_file_get", self._pointer, folder.encode("utf-8"), name.encode("utf-8"), kind, file._pointer, None)
        return file

    def file_delete(self, folder: str, name: str) -> None:
        self._lib.call("gp_camera_file_delete", self._pointer, folder.encode("utf-8"), name.encode("utf-8"), None)

    def wait_for_event(self, timeout: int) -> tuple[int, SimpleNamespace | None]:
        kind, data = ct.c_int(), ct.c_void_p()
        try:
            self._lib.call("gp_camera_wait_for_event", self._pointer, timeout, ct.byref(kind), ct.byref(data), None)
            path = _path(ct.cast(data, ct.POINTER(_FilePath)).contents) if data and kind.value in (2, 3, 5) else None
            return kind.value, path
        finally:
            # Event payloads belong to the caller and use MSYS2's UCRT allocator.
            if data:
                self._lib.crt.free(data)


def ensure_available() -> None:
    _library()
