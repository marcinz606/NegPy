"""Every parallel kernel must go through the gate, and the platform default must not move.

Numba's workqueue threading layer terminates the process outright when two threads enter it
at once — no exception, no dialog, nothing in the log. `parallel_njit` serialises every
parallel invocation behind one lock, which holds only while *every* parallel kernel is
dispatched through it. A kernel compiled `parallel=True` directly would bypass the lock and
reintroduce the abort, and it would show up as a rare crash nobody could attribute.
"""

import ast
import pathlib

import pytest

from negpy.kernel.system.parallel import default_cpu_parallel, resolve_cpu_parallel

ROOT = pathlib.Path(__file__).resolve().parent.parent / "negpy"
GATE = ROOT / "kernel" / "system" / "parallel.py"


def _python_files():
    return [p for p in ROOT.rglob("*.py") if p != GATE]


class TestNothingBypassesTheGate:
    def test_no_kernel_is_compiled_parallel_directly(self):
        """`@njit(parallel=True)` outside parallel_njit is the bypass."""
        offenders = []
        for path in _python_files():
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for kw in node.keywords:
                    if kw.arg == "parallel" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        offenders.append(f"{path.relative_to(ROOT.parent)}:{node.lineno}")
        assert not offenders, "compile these through parallel_njit, or they skip the workqueue lock: " + ", ".join(offenders)

    def test_prange_only_appears_in_gated_kernels(self):
        """A file using prange but not parallel_njit is either a bypass or a kernel that
        silently degraded to a serial loop — both worth failing on."""
        offenders = [
            str(p.relative_to(ROOT.parent)) for p in _python_files() if "prange" in (src := p.read_text()) and "parallel_njit" not in src
        ]
        assert not offenders, "prange outside a parallel_njit kernel: " + ", ".join(offenders)

    def test_the_parallel_variant_is_never_called_directly(self):
        offenders = [str(p.relative_to(ROOT.parent)) for p in _python_files() if ".parallel(" in p.read_text()]
        assert not offenders, "call the dispatcher, not the parallel variant: " + ", ".join(offenders)


class TestPlatformDefaultIsUnmoved:
    """Surfacing the setting must not silently re-decide it for anyone who never touches it."""

    @pytest.mark.parametrize("platform,expected", [("win32", True), ("linux", True), ("darwin", False)])
    def test_untouched_installs_keep_their_platform_behaviour(self, platform, expected):
        assert default_cpu_parallel(platform) is expected
        assert resolve_cpu_parallel(None, None) is None, "None must fall through to the platform default"

    def test_the_saved_setting_beats_the_platform_default(self):
        assert resolve_cpu_parallel(None, True) is True
        assert resolve_cpu_parallel(None, False) is False

    def test_the_override_file_beats_the_saved_setting(self):
        """override.toml is the escape hatch for a machine that cannot boot far enough to
        reach the menu."""
        assert resolve_cpu_parallel(False, True) is False
        assert resolve_cpu_parallel(True, False) is True
