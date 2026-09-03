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
        (0.080695, 0.963675, 0.945836),
        (0.067880, 0.952575, 0.955525),
        (0.276857, 0.269768, 0.267612),
        (0.200415, 0.204841, 0.203704),
        (0.067753, 0.065932, 0.943329),
        (0.938839, 0.954274, 0.064929),
    ],
    "expo_dark": [
        (0.345724, 0.993943, 0.993527),
        (0.347660, 0.993754, 0.993823),
        (0.820011, 0.811040, 0.808217),
        (0.695486, 0.704488, 0.702199),
        (0.346713, 0.341523, 0.992975),
        (0.993445, 0.993813, 0.346546),
    ],
    # WB CMY sliders are absolute CC density (divided by the stretch range).
    "expo_cmy": [
        (0.046363, 0.970597, 0.884897),
        (0.046871, 0.964545, 0.908936),
        (0.224309, 0.306133, 0.188023),
        (0.160541, 0.235667, 0.139895),
        (0.046772, 0.082567, 0.885339),
        (0.904451, 0.963619, 0.000000),
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
        crop_rect=(0.0, 0.0, 1.0, 1.0),
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
