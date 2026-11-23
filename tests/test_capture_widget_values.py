"""Widget value checks without a camera or an unsafe native string conversion."""

import ctypes
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from negpy.infrastructure.capture import gphoto


GP = SimpleNamespace(GP_WIDGET_TEXT=2, GP_WIDGET_RADIO=5, GP_WIDGET_MENU=6)


class _Widget:
    def __init__(self, kind: int, value: Any, choices: int = 1) -> None:
        self.kind, self.value, self.choices = kind, value, choices
        self.this = 1
        self.reads = 0

    def get_type(self) -> int:
        return self.kind

    def count_choices(self) -> int:
        return self.choices

    def get_name(self) -> str:
        return "f-number"

    def get_value(self) -> Any:
        self.reads += 1
        assert self.value is not None, "the binding must not convert a NULL string"
        return self.value


@pytest.fixture(autouse=True)
def _clear_library_cache() -> Any:
    lookup = gphoto._widget_value_library
    lookup.cache_clear()
    yield
    lookup.cache_clear()


@pytest.mark.parametrize("kind", [GP.GP_WIDGET_RADIO, GP.GP_WIDGET_MENU, GP.GP_WIDGET_TEXT])
def test_null_string_never_reaches_binding(kind: int, monkeypatch: pytest.MonkeyPatch) -> None:
    reader = Mock(return_value=0)  # Leave the output pointer NULL despite nonempty choices.
    monkeypatch.setattr(gphoto, "_widget_value_library", lambda: SimpleNamespace(gp_widget_get_value=reader))
    widget = _Widget(kind, None)

    assert gphoto._safe_value(GP, widget) is None
    reader.assert_called_once()
    assert widget.reads == 0


@pytest.mark.parametrize("value", ["100", ""])
def test_nonnull_string_preserves_binding_value(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = ctypes.create_string_buffer(value.encode())

    def read(pointer: int, output: Any) -> int:
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.addressof(storage)
        return 0

    monkeypatch.setattr(gphoto, "_widget_value_library", lambda: SimpleNamespace(gp_widget_get_value=read))
    widget = _Widget(GP.GP_WIDGET_RADIO, value)

    assert gphoto._safe_value(GP, widget) == value
    assert widget.reads == 1


@pytest.mark.parametrize("kind", [3, 4, 8])  # Range, toggle, date.
@pytest.mark.parametrize("value", [0, 0.0])
def test_numeric_zero_does_not_use_string_reader(kind: int, value: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = Mock(side_effect=AssertionError("numeric values are not string pointers"))
    monkeypatch.setattr(gphoto, "_widget_value_library", lookup)
    widget = _Widget(kind, value)

    assert gphoto._safe_value(GP, widget) == value
    lookup.assert_not_called()
    assert widget.reads == 1


def test_injected_widget_does_not_load_native_library(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = Mock(side_effect=AssertionError("fake widgets have no native library"))
    monkeypatch.setattr(gphoto, "_widget_value_library", lookup)
    widget = _Widget(GP.GP_WIDGET_RADIO, "100")
    del widget.this

    assert gphoto._safe_value(GP, widget) == "100"
    lookup.assert_not_called()


@pytest.mark.parametrize("kind", [GP.GP_WIDGET_RADIO, GP.GP_WIDGET_MENU])
def test_choiceless_widget_does_not_load_native_library(kind: int, monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = Mock(side_effect=AssertionError("choiceless widgets must not be read"))
    monkeypatch.setattr(gphoto, "_widget_value_library", lookup)
    widget = _Widget(kind, None, choices=0)

    assert gphoto._safe_value(GP, widget) is None
    lookup.assert_not_called()
    assert widget.reads == 0


def test_missing_library_never_falls_through_to_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gphoto, "_widget_value_library", lambda: None)
    widget = _Widget(GP.GP_WIDGET_RADIO, None)

    assert gphoto._safe_value(GP, widget) is None
    assert widget.reads == 0


def test_native_error_never_falls_through_to_binding(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    reader = Mock(return_value=-2)
    monkeypatch.setattr(gphoto, "_widget_value_library", lambda: SimpleNamespace(gp_widget_get_value=reader))
    widget = _Widget(GP.GP_WIDGET_RADIO, None)

    assert gphoto._safe_value(GP, widget) is None
    assert widget.reads == 0
    assert "f-number" in caplog.text
    assert "error -2" in caplog.text


def test_library_lookup_uses_extension_and_caches_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(__file__="/test/gphoto2/_widget.so")
    importer = Mock(return_value=module)
    library = Mock()
    load = Mock(return_value=library)
    monkeypatch.setattr(gphoto, "import_module", importer)
    monkeypatch.setattr(gphoto.ctypes, "CDLL", load)

    assert gphoto._widget_value_library() is library
    assert gphoto._widget_value_library() is library
    importer.assert_called_once_with("gphoto2._widget")
    load.assert_called_once_with(module.__file__)
    assert library.gp_widget_get_value.argtypes == [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    assert library.gp_widget_get_value.restype is ctypes.c_int


@pytest.mark.parametrize("failure", ["import", "load", "symbol"])
def test_library_lookup_failure_is_logged_and_cached(
    failure: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    importer = Mock(return_value=SimpleNamespace(__file__="/test/gphoto2/_widget.so"))
    load = Mock(return_value=SimpleNamespace())
    if failure == "import":
        importer.side_effect = ImportError("not installed")
    elif failure == "load":
        load.side_effect = OSError("cannot load")
    monkeypatch.setattr(gphoto, "import_module", importer)
    monkeypatch.setattr(gphoto.ctypes, "CDLL", load)

    assert gphoto._widget_value_library() is None
    assert gphoto._widget_value_library() is None
    importer.assert_called_once()
    assert caplog.text.count("cannot check widget string values") == 1


@pytest.mark.parametrize("kind", [GP.GP_WIDGET_RADIO, GP.GP_WIDGET_MENU, GP.GP_WIDGET_TEXT])
@pytest.mark.parametrize("value", [None, "", "100"])
def test_native_string_widget(kind: int, value: Any) -> None:
    gp = pytest.importorskip("gphoto2")
    module = gphoto.import_module("gphoto2._widget")
    library = ctypes.CDLL(module.__file__)
    library.gp_widget_new.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    library.gp_widget_new.restype = ctypes.c_int
    library.gp_widget_add_choice.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    library.gp_widget_add_choice.restype = ctypes.c_int
    library.gp_widget_set_value.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    library.gp_widget_set_value.restype = ctypes.c_int
    library.gp_widget_unref.argtypes = [ctypes.c_void_p]
    library.gp_widget_unref.restype = ctypes.c_int
    pointer = ctypes.c_void_p()
    assert library.gp_widget_new(kind, b"probe", ctypes.byref(pointer)) == 0
    try:
        if kind in (gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU):
            assert library.gp_widget_add_choice(pointer, b"100") == 0
        if value is not None:
            assert library.gp_widget_set_value(pointer, value.encode()) == 0
        widget = _Widget(kind, value)
        widget.this = pointer.value

        assert gphoto._safe_value(gp, widget) == value
        assert widget.reads == (0 if value is None else 1)
    finally:
        library.gp_widget_unref(pointer)
