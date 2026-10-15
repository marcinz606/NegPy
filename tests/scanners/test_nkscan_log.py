from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from negpy.infrastructure.scanners import nkscan_log


@pytest.fixture
def calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """A stand-in nkscan that records each init_logging call, and a log in tmp_path."""
    made: list[tuple] = []
    fake = types.ModuleType("nkscan")
    fake.init_logging = lambda level=None, path=None: made.append((level, path))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nkscan", fake)
    monkeypatch.setattr(nkscan_log, "log_path", lambda: str(tmp_path / nkscan_log.LOG_NAME))
    monkeypatch.setattr(nkscan_log, "_level", "off")
    return made


def test_a_level_sends_nkscans_log_to_the_file(calls: list[tuple]) -> None:
    assert nkscan_log.set_level("debug")

    path = nkscan_log.log_path()
    assert calls == [("off", None), ("debug", path)]
    assert "nkscan" in Path(path).read_text().splitlines()[1]  # the session header


def test_off_stops_the_log(calls: list[tuple]) -> None:
    nkscan_log.set_level("trace")
    calls.clear()

    assert nkscan_log.set_level("off")

    assert calls == [("off", None)]


def test_the_same_level_again_changes_nothing(calls: list[tuple]) -> None:
    nkscan_log.set_level("debug")
    calls.clear()

    assert nkscan_log.set_level("debug")

    assert calls == []


def test_an_unknown_level_is_off(calls: list[tuple]) -> None:
    nkscan_log.set_level("debug")
    calls.clear()

    nkscan_log.set_level("verbose")

    assert calls == [("off", None)]


def test_a_missing_nkscan_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "nkscan", None)
    assert nkscan_log.set_level("trace") is False
    assert nkscan_log.set_level("off") is True


def test_a_full_log_rolls_over(calls: list[tuple], monkeypatch: pytest.MonkeyPatch) -> None:
    path = Path(nkscan_log.log_path())
    path.write_bytes(b"0123456789ab\n")
    monkeypatch.setattr(nkscan_log, "_MAX_BYTES", 10)

    nkscan_log.set_level("debug")

    assert Path(f"{path}.1").read_bytes() == b"0123456789ab\n"
    assert path.read_text().startswith("\n--- ")
