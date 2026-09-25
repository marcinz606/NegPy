"""Characterization of the two slide renders — as captured and Positive —
with every live control moved at once. The CPU goldens pin the render so a refactor of
the slide path cannot move it silently; the GPU cases pin the two engines together on
the same configs, since a parity case with a control left neutral agrees trivially."""

from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.interfaces import PipelineContext
from negpy.features.process.models import ProcessMode, cast_removal_for_mode
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
from negpy.services.rendering.engine import DarkroomEngine

CAM_XYZ = [
    [0.6988, -0.1384, -0.0714],
    [-0.5631, 1.3410, 0.2447],
    [-0.1485, 0.2204, 0.7318],
]

E6_MATRIX = (1.0, -0.05, -0.002, -0.29, 1.0, -0.05, -0.09, -0.19, 1.0)

# Per sub-path: Positive, then Auto Density and Auto Grade.
SUB_PATHS = {
    "as_captured": (False, False),
    "positive": (True, True),
}

# Per sub-path: per-channel mean, then the pixels at SAMPLES, all of the CPU render.
SAMPLES = ((4, 4), (20, 40), (40, 70), (60, 90))
GOLDEN = {
    "as_captured": [
        [0.3753, 0.3461, 0.2733],
        [0.0863, 0.0371, 0.0453],
        [0.0869, 0.1725, 0.1452],
        [0.4617, 0.4149, 0.2979],
        [0.8909, 0.8747, 0.8077],
    ],
    "positive": [
        [0.3654, 0.3486, 0.2943],
        [0.1631, 0.0996, 0.1194],
        [0.1753, 0.2366, 0.219],
        [0.4045, 0.3845, 0.3176],
        [0.8026, 0.77, 0.6762],
    ],
}


def _config(sub_path: str):
    cfg = DEFAULT_WORKSPACE_CONFIG
    positive_source, autos = SUB_PATHS[sub_path]
    process = replace(
        cfg.process,
        process_mode=ProcessMode.E6,
        white_point_offset=0.04,
        black_point_offset=-0.03,
        white_point_trim_red=0.02,
        black_point_trim_blue=0.02,
        crosstalk_strength=1.0,
        crosstalk_process=ProcessMode.E6,
        crosstalk_matrix=E6_MATRIX,
        positive_source=positive_source,
    )
    exposure = replace(
        cfg.exposure,
        auto_exposure=autos,
        auto_normalize_contrast=autos,
        cast_removal_strength=0.6,
        density=1.2,
        grade=85.0,
        toe=0.4,
        shoulder=-0.3,
        toe_width=3.5,
        shoulder_width=1.8,
        toe_trim_red=0.2,
        shoulder_trim_blue=-0.2,
        toe_width_trim_green=0.5,
        shoulder_width_trim_red=-0.3,
        wb_cyan=0.2,
        wb_magenta=-0.1,
        wb_yellow=0.15,
        shadow_cyan=0.1,
        highlight_yellow=-0.1,
        shadow_density=-0.3,
        highlight_density=0.2,
        dye_separation=1.25,
        dye_separation_trim_red=0.15,
        dye_separation_trim_blue=-0.1,
        separation_damping=0.5,
    )
    geometry = replace(cfg.geometry, autocrop_offset=0)
    return replace(cfg, process=process, exposure=exposure, geometry=geometry)


def _image() -> np.ndarray:
    rng = np.random.default_rng(1167)
    h, w = 64, 96
    y = np.geomspace(2e-3, 0.8, h, dtype=np.float32)[:, None]
    x = np.linspace(0.6, 1.0, w, dtype=np.float32)[None, :]
    base = y * x
    img = np.stack([base, base * np.float32(0.8), base * np.float32(0.6)], axis=-1)
    img[16:32, 24:48] *= np.array([0.5, 1.1, 1.4], dtype=np.float32)
    return np.ascontiguousarray(img + rng.uniform(0, 1e-3, img.shape).astype(np.float32))


def _cpu(sub_path: str) -> np.ndarray:
    img = _image()
    cfg = _config(sub_path)
    ctx = PipelineContext(
        original_size=img.shape[:2],
        scale_factor=1.0,
        process_mode=cfg.process.process_mode,
        cam_xyz=CAM_XYZ,
        wants_uv_grid=False,
        cache_stages=False,
    )
    return np.asarray(DarkroomEngine().process(img, cfg, f"slide-char-{sub_path}", ctx), dtype=np.float64)


def _summary(out: np.ndarray) -> list:
    rows = [out[:, :, :3].mean(axis=(0, 1)).tolist()]
    rows += [out[y, x, :3].tolist() for y, x in SAMPLES]
    return [[round(v, 4) for v in row] for row in rows]


@pytest.mark.parametrize("sub_path", list(SUB_PATHS))
def test_cpu_render_is_pinned(sub_path):
    np.testing.assert_allclose(_summary(_cpu(sub_path)), GOLDEN[sub_path], atol=2e-4)


@pytest.mark.parametrize("sub_path", list(SUB_PATHS))
def test_every_moved_control_is_live(sub_path):
    """Guards the goldens: a config whose controls do nothing would pin the wrong thing."""
    shipped = DEFAULT_WORKSPACE_CONFIG.exposure
    autos = SUB_PATHS[sub_path][1]
    neutral = replace(
        _config(sub_path),
        exposure=replace(
            shipped,
            cast_removal_strength=cast_removal_for_mode(ProcessMode.E6, shipped.cast_removal_strength),
            auto_exposure=autos,
            auto_normalize_contrast=autos,
        ),
    )
    img = _image()
    ctx = PipelineContext(
        original_size=img.shape[:2], scale_factor=1.0, process_mode=ProcessMode.E6, cam_xyz=CAM_XYZ, wants_uv_grid=False, cache_stages=False
    )
    base = np.asarray(DarkroomEngine().process(img, neutral, "slide-char-neutral", ctx), dtype=np.float64)
    assert float(np.abs(_cpu(sub_path) - base).max()) > 0.02


@pytest.mark.parametrize("sub_path", list(SUB_PATHS))
def test_gpu_matches_cpu(sub_path):
    from negpy.services.rendering.image_processor import ImageProcessor

    processor = ImageProcessor()
    if processor.engine_gpu is None:
        pytest.skip("GPU engine not initialised")
    img = _image()
    cfg = _config(sub_path)

    def render(prefer_gpu):
        result, _ = processor.run_pipeline(
            img.copy(),
            cfg,
            f"slide-char-{sub_path}-{prefer_gpu}",
            render_size_ref=float(max(img.shape[:2])),
            prefer_gpu=prefer_gpu,
            readback_metrics=False,
            cam_xyz=CAM_XYZ,
        )
        arr = np.asarray(result.readback()) if hasattr(result, "readback") else np.asarray(result)
        return arr[:, :, :3].astype(np.float64)

    cpu, gpu = render(False), render(True)
    assert cpu.shape == gpu.shape
    assert float(np.mean(np.abs(cpu - gpu))) < 0.01
    assert float(np.max(np.abs(cpu - gpu))) < 0.04
