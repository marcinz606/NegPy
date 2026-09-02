import pytest

from negpy.infrastructure.gpu.cffi_warm import warm_wgpu_cffi_types


def test_warm_matches_parser_types():
    from wgpu.backends.wgpu_native._ffi import ffi

    warm_wgpu_cffi_types()
    for cdecl in ("WGPUBufferDescriptor *", "WGPUBindGroupEntry[]", "WGPUTextureFormat[]", "char []", "uint32_t []"):
        cached = ffi._parsed_types[cdecl][0]
        parsed = ffi._parser.parse_type(cdecl)
        with ffi._lock:
            assert ffi._get_cached_btype(parsed) is cached


def test_first_render_parses_few_types():
    from negpy.infrastructure.gpu.device import GPUDevice

    if not GPUDevice.get().is_available:
        pytest.skip("GPU unavailable")
    import numpy as np

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.gpu_engine import GPUEngine
    from wgpu.backends.wgpu_native._ffi import ffi

    misses = []
    orig = type(ffi)._typeof_locked

    def counting(self, cdecl):
        if cdecl not in self._parsed_types:
            misses.append(cdecl)
        return orig(self, cdecl)

    type(ffi)._typeof_locked = counting
    try:
        engine = GPUEngine()
        img = np.full((64, 96, 3), 0.4, dtype=np.float32)
        engine.process_to_texture(img, WorkspaceConfig(), scale_factor=1.0, render_size_ref=96.0)
        engine.destroy_all()
    finally:
        type(ffi)._typeof_locked = orig
    # Only ffi.callback signatures are left to parse; every struct/array/pointer type is pre-cached.
    assert all("(" in m for m in misses), misses
