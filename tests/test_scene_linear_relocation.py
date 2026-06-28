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
# default saturation = DEFAULT_SATURATION).
_GOLDEN = {
    "default": [
        (0.218817, 0.873366, 0.845754),
        (0.217784, 0.865650, 0.855250),
        (0.222888, 0.217278, 0.212564),
        (0.160508, 0.163282, 0.160711),
        (0.089764, 0.069016, 0.706038),
        (0.871984, 0.880190, 0.149239),
    ],
    "expo_dark": [
        (0.368100, 0.905842, 0.898391),
        (0.369569, 0.905627, 0.898749),
        (0.758184, 0.751020, 0.733920),
        (0.642077, 0.649583, 0.629815),
        (0.292174, 0.283094, 0.806559),
        (0.921165, 0.919462, 0.353217),
    ],
    "expo_cmy": [
        (0.215503, 0.882976, 0.797457),
        (0.214987, 0.877234, 0.816583),
        (0.187310, 0.249517, 0.160285),
        (0.136313, 0.186884, 0.122779),
        (0.082885, 0.077226, 0.661457),
        (0.845878, 0.889456, 0.135215),
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
