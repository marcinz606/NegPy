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
# default config — paper_dmin off, true_black on, cast_removal_strength 0.5, sharpen 0.25,
# chroma_damping 0.5).
_GOLDEN = {
    "default": [
        (0.204851, 0.927244, 0.896463),
        (0.203398, 0.915851, 0.909803),
        (0.217849, 0.212541, 0.210557),
        (0.153749, 0.156905, 0.155930),
        (0.072774, 0.047537, 0.761169),
        (0.914429, 0.929906, 0.120987),
    ],
    "expo_dark": [
        (0.373914, 0.978381, 0.969779),
        (0.375290, 0.977941, 0.970339),
        (0.767470, 0.759197, 0.756000),
        (0.635119, 0.643248, 0.640762),
        (0.287566, 0.276198, 0.865637),
        (0.994327, 0.992783, 0.365315),
    ],
    # WB CMY sliders are absolute CC density (divided by the stretch range).
    "expo_cmy": [
        (0.201257, 0.941433, 0.834892),
        (0.200378, 0.932615, 0.859163),
        (0.181375, 0.244389, 0.154415),
        (0.127646, 0.181057, 0.113393),
        (0.063280, 0.058972, 0.703570),
        (0.878620, 0.943941, 0.105783),
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
