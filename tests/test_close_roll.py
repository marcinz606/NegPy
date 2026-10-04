from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from negpy.desktop.view import keyboard_shortcuts
from negpy.desktop.view.keyboard_shortcuts import _close_roll, close_roll_label
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

    monkeypatch.setattr(keyboard_shortcuts, "confirm_close_roll", fake)
    monkeypatch.setattr(keyboard_shortcuts.rolls, "roll_for_id", lambda _repo, rid: {"name": "Portra Lisbon"} if rid == "r1" else None)
    return names, answer


def test_closing_names_the_open_roll_and_clears_the_strip(asked):
    names, _ = asked
    ctrl = _controller()
    _close_roll(None, ctrl)
    assert names == ["Portra Lisbon"]
    ctrl.session.clear_files.assert_called_once()


def test_with_no_roll_open_it_asks_to_unload_all(asked):
    names, _ = asked
    ctrl = _controller(roll_id=None)
    _close_roll(None, ctrl)
    assert names == [None]
    ctrl.session.clear_files.assert_called_once()


def test_cancel_keeps_the_strip(asked):
    _, answer = asked
    answer["yes"] = False
    ctrl = _controller()
    _close_roll(None, ctrl)
    ctrl.session.clear_files.assert_not_called()


def test_an_empty_strip_does_not_ask(asked):
    names, _ = asked
    ctrl = _controller(files=())
    _close_roll(None, ctrl)
    assert names == []
    ctrl.session.clear_files.assert_not_called()


def test_label_follows_whether_a_roll_is_open():
    assert close_roll_label(SimpleNamespace(active_roll_id="r1")) == "Close Roll…"
    assert close_roll_label(SimpleNamespace(active_roll_id=None)) == "Unload All…"


def test_close_roll_is_a_registered_action_without_a_default_key():
    assert REGISTRY["close_roll"].default_key == ""
