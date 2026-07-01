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
# default config — cast_removal_strength default 0.5).
_GOLDEN = {
    "default": [
        (0.092235, 0.861299, 0.837324),
        (0.092613, 0.850244, 0.853355),
        (0.242108, 0.236516, 0.234428),
        (0.181938, 0.185318, 0.184274),
        (0.092542, 0.091389, 0.837615),
        (0.833332, 0.853584, 0.092328),
    ],
    "expo_dark": [
        (0.299548, 0.920606, 0.919175),
        (0.301166, 0.919962, 0.920146),
        (0.709704, 0.701544, 0.698399),
        (0.598064, 0.605943, 0.603535),
        (0.300861, 0.295918, 0.919193),
        (0.918923, 0.920160, 0.299948),
    ],
    "expo_cmy": [
        (0.080090, 0.875624, 0.769273),
        (0.080379, 0.866844, 0.795187),
        (0.200609, 0.268509, 0.171894),
        (0.151706, 0.209890, 0.136479),
        (0.080324, 0.101300, 0.769735),
        (0.794916, 0.869504, 0.073777),
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
