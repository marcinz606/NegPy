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
# dye_separation 1.0).
_GOLDEN = {
    "default": [
        (0.095545, 0.959373, 0.940689),
        (0.084938, 0.947674, 0.950774),
        (0.277507, 0.270613, 0.268520),
        (0.204215, 0.208390, 0.207317),
        (0.084829, 0.083255, 0.938062),
        (0.932800, 0.948873, 0.081130),
    ],
    "expo_dark": [
        (0.345343, 0.994696, 0.994101),
        (0.347270, 0.994423, 0.994522),
        (0.816296, 0.807605, 0.804870),
        (0.695260, 0.704071, 0.701831),
        (0.346328, 0.341166, 0.993554),
        (0.993984, 0.994508, 0.346161),
    ],
    # WB CMY sliders are absolute CC density (divided by the stretch range).
    "expo_cmy": [
        (0.065897, 0.966940, 0.879480),
        (0.066365, 0.960373, 0.903259),
        (0.226797, 0.306151, 0.192477),
        (0.167046, 0.237717, 0.148179),
        (0.066275, 0.097582, 0.879914),
        (0.898741, 0.959553, 0.044090),
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
