import sys

import pytest
from numba.core import caching, config

from negpy.kernel.system import jit


def test_frozen_njit_disables_requested_disk_cache(monkeypatch):
    call = {}

    def fake_njit(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return "dispatcher"

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(jit, "_njit", fake_njit)

    assert jit.njit(cache=True, fastmath=True) == "dispatcher"
    assert call == {"args": (), "kwargs": {"cache": False, "fastmath": True}}


def test_frozen_njit_runs_without_a_usable_cache(monkeypatch, tmp_path):
    namespace = {"__name__": "frozen_test"}
    source_path = str(tmp_path / "frozen_logic.py")
    exec(compile("def kernel(value):\n    return value + 1\n", source_path, "exec"), namespace)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(config, "CACHE_LOCATOR_CLASSES", "")

    def denied_cache(_locator):
        raise PermissionError("Cache directory is not writable")

    # Frozen functions can use UserWideCacheLocator even without a source file.
    monkeypatch.setattr(caching._CacheLocator, "ensure_cache_path", denied_cache)
    with pytest.raises(RuntimeError, match="no locator available"):
        jit._njit(cache=True)(namespace["kernel"])

    dispatcher = jit.njit(cache=True)(namespace["kernel"])

    assert dispatcher(2) == 3
    assert dispatcher.nopython_signatures


def test_unfrozen_njit_keeps_requested_disk_cache(monkeypatch):
    call = {}

    def fake_njit(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return "dispatcher"

    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(jit, "_njit", fake_njit)

    assert jit.njit(cache=True) == "dispatcher"
    assert call == {"args": (), "kwargs": {"cache": True}}
