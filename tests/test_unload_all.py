from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from negpy.desktop.view import keyboard_shortcuts
from negpy.desktop.view.confirm import confirm_unload_all
from negpy.desktop.view.keyboard_shortcuts import _unload_all
from negpy.desktop.view.shortcut_registry import REGISTRY


def _controller(files=("a.tif",), roll_id="r1"):
    state = SimpleNamespace(uploaded_files=list(files), active_roll_id=roll_id)
    return MagicMock(session=MagicMock(state=state))


@pytest.fixture
def asked(monkeypatch):
    names: list = []
    answer = {"yes": True}

    def fake(_parent, name):
        names.append(name)
        return answer["yes"]

    monkeypatch.setattr(keyboard_shortcuts, "confirm_unload_all", fake)
    monkeypatch.setattr(keyboard_shortcuts.rolls, "roll_for_id", lambda _repo, rid: {"name": "Portra Lisbon"} if rid == "r1" else None)
    return names, answer


def test_unloading_names_the_open_roll_and_clears_the_strip(asked):
    names, _ = asked
    ctrl = _controller()
    _unload_all(None, ctrl)
    assert names == ["Portra Lisbon"]
    ctrl.session.clear_files.assert_called_once()


def test_with_no_roll_open_it_asks_to_unload_all(asked):
    names, _ = asked
    ctrl = _controller(roll_id=None)
    _unload_all(None, ctrl)
    assert names == [None]
    ctrl.session.clear_files.assert_called_once()


def test_cancel_keeps_the_strip(asked):
    _, answer = asked
    answer["yes"] = False
    ctrl = _controller()
    _unload_all(None, ctrl)
    ctrl.session.clear_files.assert_not_called()


def test_an_empty_strip_does_not_ask(asked):
    names, _ = asked
    ctrl = _controller(files=())
    _unload_all(None, ctrl)
    assert names == []
    ctrl.session.clear_files.assert_not_called()


@pytest.mark.parametrize(
    ("roll_name", "text", "informative"),
    [
        ("Portra Lisbon", "Unload all frames of “Portra Lisbon” from the Film Strip?", "The roll stays in the Library."),
        (None, "Unload all frames from the Film Strip?", "Your saved edits stay in the database."),
    ],
)
def test_one_dialog_names_the_open_roll_and_keeps_it_in_the_library(monkeypatch, roll_name, text, informative):
    from negpy.desktop.view import confirm

    seen = []
    monkeypatch.setattr(confirm, "confirm_action", lambda *a, **k: seen.append(a) or True)
    assert confirm_unload_all(None, roll_name)
    _parent, title, shown, verb, note = seen[0]
    assert (title, shown, verb) == ("Unload All", text, "Unload")
    assert note.endswith(informative)
    assert (roll_name is None) == ("Library" not in note)


def test_unload_all_is_a_registered_action_without_a_default_key():
    assert REGISTRY["close_roll"].default_key == ""
