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
# default config — paper_dmin off, paper_black off, cast_removal_strength 0.5, sharpen 0.25,
# chroma_damping 0.5).
_GOLDEN = {
    "default": [
        (0.330028, 0.939552, 0.918772),
        (0.329881, 0.931057, 0.928883),
        (0.276760, 0.270905, 0.269126),
        (0.204649, 0.208192, 0.207280),
        (0.214037, 0.146760, 0.831148),
        (0.941396, 0.943747, 0.280440),
    ],
    "expo_dark": [
        (0.472653, 0.976357, 0.971520),
        (0.473850, 0.976112, 0.971873),
        (0.815229, 0.808012, 0.805741),
        (0.696288, 0.703601, 0.701740),
        (0.389825, 0.359756, 0.893031),
        (0.997129, 0.988443, 0.458347),
    ],
    # WB CMY sliders are absolute CC density (divided by the stretch range).
    "expo_cmy": [
        (0.322609, 0.949710, 0.869808),
        (0.323910, 0.943332, 0.889480),
        (0.236722, 0.304218, 0.206959),
        (0.174551, 0.234601, 0.158040),
        (0.198129, 0.147219, 0.780075),
        (0.913318, 0.954011, 0.273669),
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
