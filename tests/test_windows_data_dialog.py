import ctypes as ct
from pathlib import Path
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Native Windows dialog")


@pytest.fixture
def dialog(monkeypatch):
    from negpy.desktop import windows_data_dialog as native
    from negpy.desktop import startup

    messages = []
    monkeypatch.setattr(startup, "_message", lambda text, flags: messages.append(text))
    return native, messages


def _mock_show(monkeypatch, native, action, button=100, result=0):
    def show(config_pointer, button_pointer, *args):
        config = ct.cast(config_pointer, ct.POINTER(native._TaskConfig)).contents
        assert config.size == ct.sizeof(native._TaskConfig)
        assert config.buttons[0].text == "Use This Folder"
        action(config)
        ct.cast(button_pointer, ct.POINTER(ct.c_int)).contents.value = button
        return result

    monkeypatch.setattr(native.ct.windll.comctl32, "TaskDialogIndirect", show)


def _link(config, text):
    value = ct.create_unicode_buffer(text)
    config.callback(None, 3, 0, ct.addressof(value), 0)


@pytest.mark.parametrize("button, expected", [(100, Path("C:/suggested")), (2, None)])
def test_accept_or_cancel(monkeypatch, dialog, button, expected):
    native, _ = dialog
    _mock_show(monkeypatch, native, lambda config: None, button)
    assert native.choose_directory(Path("C:/source"), Path("C:/suggested")) == expected


def test_click_path_updates_selection_and_display(monkeypatch, dialog):
    native, messages = dialog
    target = Path("C:/custom Å & data")
    monkeypatch.setattr(native, "_browse_directory", lambda *args: target)
    updates = []
    monkeypatch.setattr(native, "_send", lambda *args: updates.append(args[-1]))
    _mock_show(monkeypatch, native, lambda config: _link(config, "folder"))
    assert native.choose_directory(Path("C:/source"), Path("C:/suggested")) == target
    assert updates == [native._content(Path("C:/source"), target)]
    assert messages == []


def test_cancel_picker_keeps_suggestion(monkeypatch, dialog):
    native, _ = dialog
    monkeypatch.setattr(native, "_browse_directory", lambda *args: None)
    _mock_show(monkeypatch, native, lambda config: _link(config, "folder"))
    assert native.choose_directory(Path("C:/source"), Path("C:/suggested")) == Path("C:/suggested")


def test_help_does_not_select_or_open_external_content(monkeypatch, dialog):
    native, messages = dialog
    _mock_show(monkeypatch, native, lambda config: _link(config, "help"), button=2)
    assert native.choose_directory(Path("C:/source"), Path("C:/suggested")) is None
    assert len(messages) == 1
    assert "Keep folder protection enabled" in messages[0]


def test_unknown_link_is_ignored(monkeypatch, dialog):
    native, messages = dialog
    _mock_show(monkeypatch, native, lambda config: _link(config, "https://untrusted.example"), button=2)
    assert native.choose_directory(Path("C:/source"), Path("C:/suggested")) is None
    assert messages == []


def test_picker_error_is_reported_without_losing_suggestion(monkeypatch, dialog):
    native, messages = dialog

    def fail(*args):
        raise OSError("picker failed")

    monkeypatch.setattr(native, "_browse_directory", fail)
    _mock_show(monkeypatch, native, lambda config: _link(config, "folder"))
    assert native.choose_directory(Path("C:/source"), Path("C:/suggested")) == Path("C:/suggested")
    assert "picker failed" in messages[0]


def test_native_failure_raises_visible_startup_error(monkeypatch, dialog):
    native, _ = dialog
    _mock_show(monkeypatch, native, lambda config: None, result=-1)
    with pytest.raises(OSError, match="Cannot show"):
        native.choose_directory(Path("C:/source"), Path("C:/suggested"))


def test_paths_cannot_inject_links(dialog):
    native, _ = dialog
    content = native._content(Path('C:/<a href="help">'), Path("C:/Å & data"))
    assert content.count('<a href="') == 1
    assert "‹a" in content
    assert "Å && data" in content
    assert "No files are copied or moved" in content
