"""The multi-core failsafe: parallel_njit dual dispatch, platform policy, override."""

import threading

import numpy as np
import pytest
from numba import prange

from negpy.kernel.system import parallel as par
from negpy.kernel.system.override import OverrideConfig, _parse
from negpy.kernel.system.parallel import default_cpu_parallel, parallel_njit


@pytest.fixture(autouse=True)
def _restore_flag():
    before = par.parallel_enabled()
    yield
    par.set_parallel_enabled(before)


@parallel_njit(fastmath=True)
def _double(arr):
    out = np.empty_like(arr)
    for i in prange(arr.shape[0]):
        out[i] = arr[i] * 2.0
    return out


def test_serial_variant_never_uses_disk_cache():
    # Numba's disk cache is keyed by source location, so the serial and parallel
    # variants of one kernel share a cache slot: whichever compiles first, the
    # other silently loads its binary — a "serial" call would execute the cached
    # PARALLEL object, defeating the failsafe. Guard: serial must be cache-less.
    from negpy.features.exposure.logic import _apply_print_curve_kernel
    from negpy.infrastructure.display.icc_lut import _apply_lut_f32_jit

    for kernel in (_double, _apply_print_curve_kernel, _apply_lut_f32_jit):
        assert type(kernel.serial._cache).__name__ == "NullCache"


def test_platform_policy():
    assert default_cpu_parallel("darwin") is False
    assert default_cpu_parallel("win32") is True
    assert default_cpu_parallel("linux") is True


def test_dispatch_uses_selected_variant():
    arr = np.arange(64, dtype=np.float32)

    par.set_parallel_enabled(False)
    np.testing.assert_array_equal(_double(arr), arr * 2.0)
    assert _double.serial.signatures  # serial variant compiled and used
    assert not _double.parallel.signatures  # parallel never touched while disabled

    par.set_parallel_enabled(True)
    np.testing.assert_array_equal(_double(arr), arr * 2.0)
    assert _double.parallel.signatures


def test_print_curve_parity_serial_vs_parallel():
    from negpy.features.exposure.logic import apply_characteristic_curve

    rng = np.random.default_rng(7)
    img = rng.random((64, 96, 3), dtype=np.float32)

    par.set_parallel_enabled(True)
    res_par = apply_characteristic_curve(img, (0.5, 2.0), (0.5, 2.0), (0.5, 2.0))
    par.set_parallel_enabled(False)
    res_ser = apply_characteristic_curve(img, (0.5, 2.0), (0.5, 2.0), (0.5, 2.0))

    np.testing.assert_allclose(res_par, res_ser, rtol=0, atol=1e-6)


def test_lab_roundtrip_parity_serial_vs_parallel():
    from negpy.kernel.image.logic import rgb_to_lab_working

    rng = np.random.default_rng(11)
    img = rng.random((32, 48, 3), dtype=np.float32)

    par.set_parallel_enabled(True)
    lab_par = rgb_to_lab_working(img)
    par.set_parallel_enabled(False)
    lab_ser = rgb_to_lab_working(img)

    np.testing.assert_allclose(lab_par, lab_ser, rtol=0, atol=1e-4)


def test_concurrent_parallel_invocation_is_serialized():
    # Regression for the workqueue concurrent-access abort: many threads invoking
    # a parallel kernel simultaneously must be safe (the gate serializes entry).
    par.set_parallel_enabled(True)
    arr = np.arange(200_000, dtype=np.float32)
    errors: list[Exception] = []

    def hammer():
        try:
            for _ in range(20):
                _double(arr)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=hammer) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_override_parsing_and_apply():
    assert _parse({"performance": {"cpu_parallel": False}}).cpu_parallel is False
    assert _parse({"performance": {"cpu_parallel": True}}).cpu_parallel is True
    assert _parse({"performance": {}}).cpu_parallel is None
    assert _parse({"performance": {"cpu_parallel": "yes"}}).cpu_parallel is None  # non-bool ignored

    from negpy.kernel.system.override import apply as apply_override
    from negpy.domain.types import AppConfig

    cfg = AppConfig(
        thumbnail_size=1,
        max_workers=1,
        preview_render_size=1,
        max_history_steps=1,
        edits_db_path="",
        settings_db_path="",
        presets_dir="",
        cache_dir="",
        user_icc_dir="",
        crosstalk_dir="",
        gear_dir="",
        contact_sheet_templates_dir="",
        default_export_dir="",
        adobe_rgb_profile="",
    )
    assert cfg.cpu_parallel is None
    apply_override(OverrideConfig(cpu_parallel=False), cfg)
    assert cfg.cpu_parallel is False


def test_configure_cpu_parallel_override_wins():
    par.configure_cpu_parallel(False)
    assert par.parallel_enabled() is False
    par.configure_cpu_parallel(True)
    assert par.parallel_enabled() is True
    par.configure_cpu_parallel(None)
    assert par.parallel_enabled() is default_cpu_parallel()
