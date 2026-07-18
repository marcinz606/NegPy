"""Characterization guard: pin the default + exposure-only full-engine output so the
look doesn't drift. Goldens are the scene-linear pipeline encoded with the Adobe RGB
working TRC."""

from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import ExposureConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.process.models import ProcessConfig
from negpy.services.rendering.engine import DarkroomEngine

_POINTS = [(0, 0), (8, 8), (20, 30), (32, 32), (50, 50), (60, 10)]

# Golden pixel values (Adobe-RGB-TRC-encoded engine output, scene-linear pipeline,
# default config — paper_dmin off, true_black on, cast_removal_strength 0.5, sharpen 0.25,
# chroma_damping 0.5).
_GOLDEN = {
    "default": [
        (0.322883, 0.939246, 0.917734),
        (0.322776, 0.930625, 0.928195),
        (0.282558, 0.276802, 0.274650),
        (0.210360, 0.213957, 0.212846),
        (0.211338, 0.146376, 0.834764),
        (0.939059, 0.942757, 0.274310),
    ],
    "expo_dark": [
        (0.471856, 0.977380, 0.972680),
        (0.473081, 0.977112, 0.973068),
        (0.814318, 0.807314, 0.804601),
        (0.697333, 0.704595, 0.702376),
        (0.393388, 0.364368, 0.898300),
        (0.997007, 0.988844, 0.458169),
    ],
    # WB CMY sliders are absolute CC density (divided by the stretch range).
    "expo_cmy": [
        (0.315156, 0.949618, 0.867968),
        (0.316424, 0.943101, 0.887874),
        (0.242077, 0.310282, 0.211843),
        (0.179748, 0.240598, 0.162881),
        (0.195144, 0.147744, 0.783009),
        (0.910561, 0.953286, 0.266756),
    ],
}


def _synthetic_image(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((64, 64, 3), dtype=np.float32)
    for y in range(64):
        for x in range(64):
            img[y, x] = 0.1 + 0.8 * ((x + y) / 126.0)
    img[0:16, 0:16] = [0.9, 0.1, 0.1]
    img[0:16, 48:64] = [0.1, 0.9, 0.1]
    img[48:64, 0:16] = [0.1, 0.1, 0.9]
    img[48:64, 48:64] = [0.9, 0.9, 0.1]
    img += rng.normal(0, 0.005, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _base_settings() -> WorkspaceConfig:
    geo = GeometryConfig(
        rotation=0,
        fine_rotation=0.0,
        flip_horizontal=False,
        flip_vertical=False,
        manual_crop_rect=(0.0, 0.0, 1.0, 1.0),
        autocrop_offset=0,
    )
    return replace(
        WorkspaceConfig(),
        geometry=geo,
        process=replace(ProcessConfig(), white_point_offset=0.0, black_point_offset=0.0),
    )


def test_full_engine_output_preserved_after_relocation():
    base = _base_settings()
    configs = {
        "default": base,
        "expo_dark": replace(base, exposure=ExposureConfig(density=-1.0, grade=2.0)),
        "expo_cmy": replace(base, exposure=ExposureConfig(wb_cyan=0.3, wb_magenta=-0.2, wb_yellow=0.5)),
    }
    img = _synthetic_image()
    eng = DarkroomEngine()
    for name, cfg in configs.items():
        out = eng.process(img, cfg, f"relocation_{name}")
        got = np.array([out[y, x] for (y, x) in _POINTS], dtype=np.float32)
        want = np.array(_GOLDEN[name], dtype=np.float32)
        np.testing.assert_allclose(got, want, atol=1e-3, err_msg=f"config={name}")
