import sys

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


def test_frozen_njit_accepts_a_source_less_function(monkeypatch):
    namespace = {"__name__": "frozen_test"}
    exec(compile("def kernel(value):\n    return value + 1\n", r"negpy\kernel\image\frozen_logic.py", "exec"), namespace)
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    dispatcher = jit.njit(cache=True)(namespace["kernel"])

    assert dispatcher(2) == 3


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
