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
# default config — paper_dmin off, true_black on, cast_removal_strength 0.5, sharpen 0.5).
_GOLDEN = {
    "default": [
        (0.056208, 0.939594, 0.914423),
        (0.056636, 0.928128, 0.931380),
        (0.218408, 0.212310, 0.210037),
        (0.153415, 0.157037, 0.155919),
        (0.056555, 0.055249, 0.914736),
        (0.910131, 0.931619, 0.056314),
    ],
    "expo_dark": [
        (0.281944, 0.993794, 0.992784),
        (0.283758, 0.993346, 0.993475),
        (0.768494, 0.758779, 0.755028),
        (0.634111, 0.643652, 0.640735),
        (0.283415, 0.277878, 0.992797),
        (0.992602, 0.993485, 0.282392),
    ],
    # WB CMY sliders are absolute CC density (divided by the stretch range).
    "expo_cmy": [
        (0.042539, 0.953732, 0.841580),
        (0.042881, 0.944891, 0.870689),
        (0.174562, 0.246521, 0.144534),
        (0.121855, 0.182790, 0.106151),
        (0.042817, 0.066099, 0.842105),
        (0.869013, 0.947591, 0.035223),
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
