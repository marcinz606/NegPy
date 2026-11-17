from __future__ import annotations

import sys
import types

import pytest

from negpy.infrastructure.scanners import nkscan_log


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """A stand-in nkscan that records each init_logging call on stderr."""
    made: list[tuple] = []
    fake = types.ModuleType("nkscan")
    fake.init_logging = lambda level=None: made.append((level,))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nkscan", fake)
    monkeypatch.setattr(nkscan_log, "_level", "off")
    return made


def test_a_level_sets_nkscans_stderr_level(calls: list[tuple]) -> None:
    assert nkscan_log.set_level("debug")

    assert calls == [("debug",)]


def test_off_stops_the_log(calls: list[tuple]) -> None:
    nkscan_log.set_level("trace")
    calls.clear()

    assert nkscan_log.set_level("off")

    assert calls == [("off",)]


def test_the_same_level_again_changes_nothing(calls: list[tuple]) -> None:
    nkscan_log.set_level("debug")
    calls.clear()

    assert nkscan_log.set_level("debug")

    assert calls == []


def test_an_unknown_level_is_off(calls: list[tuple]) -> None:
    nkscan_log.set_level("debug")
    calls.clear()

    nkscan_log.set_level("verbose")

    assert calls == [("off",)]


def test_a_missing_nkscan_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "nkscan", None)
    assert nkscan_log.set_level("trace") is False
    assert nkscan_log.set_level("off") is True
