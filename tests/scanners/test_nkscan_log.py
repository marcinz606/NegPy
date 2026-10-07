from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from negpy.infrastructure.scanners import nkscan_log

# Stands in for the Rust extension, which writes to stdout (0.12) or to stderr (as documented).
_FAKE_NKSCAN = """
import os

def init_logging(level=None):
    pass

def emit(text, level="DEBUG", fd=1):
    os.write(fd, f"2026-09-27T01:02:03.456789Z {level:>5} {text}\\n".encode())
"""

_DRIVER = """
import os
import time
import nkscan
from negpy.infrastructure.scanners import nkscan_log

assert nkscan_log.set_level("debug")
nkscan.emit("debug at debug")
nkscan.emit("trace at debug", "TRACE")
nkscan.emit("stderr at debug", fd=2)
os.write(1, b"other stdout\\n")
os.write(2, b"other stderr\\n")
time.sleep(0.3)
nkscan_log.set_level("trace")
nkscan.emit("trace at trace", "TRACE")
time.sleep(0.3)
nkscan_log.set_level("off")
nkscan.emit("debug at off")
time.sleep(0.3)
"""


_DRIVER_ERROR_AT_EXIT = """
from negpy.infrastructure.scanners import nkscan_log

assert nkscan_log.set_level("trace")
print("last stdout line")

def inner():
    raise RuntimeError("boom at exit")

def outer():
    inner()

outer()
"""


def _run(tmp_path: Path, driver: str = _DRIVER) -> subprocess.CompletedProcess[str]:
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "nkscan.py").write_text(_FAKE_NKSCAN)
    env = {k: v for k, v in os.environ.items() if k != "RUST_LOG"}
    # The asserts read the child's traceback as plain text; PYTHON_COLORS outranks an inherited FORCE_COLOR.
    env.update(PYTHONPATH=os.pathsep.join([str(fake), str(Path.cwd())]), NEGPY_USER_DIR=str(tmp_path), PYTHON_COLORS="0")
    return subprocess.run([sys.executable, "-c", textwrap.dedent(driver)], env=env, capture_output=True, text=True, timeout=30)


def test_nkscan_lines_reach_the_log_at_the_chosen_level(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr

    log = (tmp_path / nkscan_log.LOG_NAME).read_text()
    assert "nkscan" in log.splitlines()[1]  # the session header
    assert "debug at debug" in log and "stderr at debug" in log and "trace at trace" in log
    assert "trace at debug" not in log and "debug at off" not in log
    assert "other" not in log


def test_only_other_lines_reach_the_console(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert "other stdout" in result.stdout and "other stderr" in result.stderr
    assert "at debug" not in result.stdout + result.stderr


def test_output_at_exit_is_not_cut_off(tmp_path: Path) -> None:
    result = _run(tmp_path, _DRIVER_ERROR_AT_EXIT)

    assert result.returncode == 1
    assert "last stdout line" in result.stdout
    assert "in outer" in result.stderr and "in inner" in result.stderr
    assert result.stderr.rstrip().endswith("RuntimeError: boom at exit")


def test_a_missing_nkscan_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "nkscan", None)
    assert nkscan_log.set_level("trace") is False


def test_a_full_log_rolls_over(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / nkscan_log.LOG_NAME
    monkeypatch.setattr(nkscan_log, "log_path", lambda: str(path))
    monkeypatch.setattr(nkscan_log, "_MAX_BYTES", 10)
    monkeypatch.setattr(nkscan_log, "_file", None)

    nkscan_log._write(b"0123456789ab\n")
    nkscan_log._write(b"next\n")

    assert (tmp_path / f"{nkscan_log.LOG_NAME}.1").read_bytes() == b"0123456789ab\n"
    assert path.read_bytes() == b"next\n"
    nkscan_log._file.close()
