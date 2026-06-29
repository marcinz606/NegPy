"""Characterization guard: pin the default + exposure-only full-engine output so the
look doesn't drift. Goldens are the scene-linear pipeline encoded with the ProPhoto RGB
working TRC (ROMM)."""

from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import ExposureConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.process.models import ProcessConfig
from negpy.services.rendering.engine import DarkroomEngine

_POINTS = [(0, 0), (8, 8), (20, 30), (32, 32), (50, 50), (60, 10)]

# Golden pixel values (ProPhoto-RGB-TRC-encoded engine output, scene-linear pipeline,
# default config).
_GOLDEN = {
    "default": [
        (0.075390, 0.886933, 0.868244),
        (0.075686, 0.879093, 0.880298),
        (0.220732, 0.214988, 0.212375),
        (0.158489, 0.161844, 0.160701),
        (0.075630, 0.074537, 0.868469),
        (0.865319, 0.881498, 0.075432),
    ],
    "expo_dark": [
        (0.285538, 0.920233, 0.919681),
        (0.287373, 0.920007, 0.920043),
        (0.755331, 0.747448, 0.743701),
        (0.636751, 0.645447, 0.642520),
        (0.287026, 0.280221, 0.919688),
        (0.919591, 0.920077, 0.285803),
    ],
    "expo_cmy": [
        (0.066436, 0.896528, 0.810552),
        (0.066634, 0.890726, 0.833632),
        (0.177395, 0.249646, 0.148358),
        (0.128725, 0.186996, 0.114138),
        (0.066597, 0.082522, 0.810975),
        (0.833651, 0.892511, 0.062316),
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
