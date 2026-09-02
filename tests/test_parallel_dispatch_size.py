import numpy as np

from negpy.kernel.system import parallel as par


def test_small_arrays_take_serial_variant(monkeypatch):
    @par.parallel_njit(cache=False)
    def double(x):
        out = np.empty_like(x)
        for i in par.prange(x.shape[0]) if hasattr(par, "prange") else range(x.shape[0]):
            out[i] = x[i] * 2
        return out

    calls = []
    monkeypatch.setattr(double, "serial", lambda *a, **k: calls.append("serial") or a[0] * 2)
    monkeypatch.setattr(double, "parallel", lambda *a, **k: calls.append("parallel") or a[0] * 2)
    monkeypatch.setattr(par, "_parallel_enabled", True)
    double(np.ones(8, dtype=np.float32))
    double(np.ones(par.SERIAL_MAX_ELEMENTS, dtype=np.float32))
    assert calls == ["serial", "parallel"]
