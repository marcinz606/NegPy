import importlib

import pytest

_KERNELS = [
    ("negpy.kernel.image.logic", name)
    for name in (
        "_get_luminance_jit",
        "_to_uint16_jit",
        "_to_uint8_jit",
        "uint8_to_float32",
        "rgb_to_rgba_into",
        "rgba_to_rgb_into",
        "uint16_to_float32",
        "_float_to_uint8_luma_jit",
        "_float_to_uint16_luma_jit",
    )
] + [
    ("negpy.features.exposure.normalization", "_normalize_log_image_jit"),
    ("negpy.features.exposure.logic", "_apply_print_curve_kernel"),
    ("negpy.infrastructure.display.icc_lut", "_apply_lut_u16_jit"),
    ("negpy.infrastructure.display.icc_lut", "_apply_lut_f32_jit"),
    ("negpy.infrastructure.display.icc_lut", "_matrix_trc_u8_jit"),
]


@pytest.mark.parametrize("module, name", _KERNELS)
def test_full_image_kernels_release_the_gil(module, name):
    kernel = getattr(importlib.import_module(module), name)
    dispatchers = (kernel.serial, kernel.parallel) if hasattr(kernel, "serial") else (kernel,)
    for d in dispatchers:
        assert d.targetoptions.get("nogil") is True
