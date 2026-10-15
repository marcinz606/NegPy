"""CPU stage-cache keys follow every input of the stage they guard."""

from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.services.rendering.engine import DarkroomEngine


def _img(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.1, 0.8, (96, 128, 3)).astype(np.float32)


def _ctx(img, cfg, crop_preview_full=False) -> PipelineContext:
    return PipelineContext(
        scale_factor=1.0, original_size=img.shape[:2], process_mode=cfg.process.process_mode, crop_preview_full=crop_preview_full
    )


def test_tilt_moves_the_full_frame_crop_preview():
    img, cfg = _img(0), WorkspaceConfig()
    engine = DarkroomEngine()
    engine.process(img, cfg, "a", _ctx(img, cfg, True))
    tilted = replace(cfg, geometry=replace(cfg.geometry, converge_v=10.0))
    out = engine.process(img, tilted, "a", _ctx(img, tilted, True))
    fresh = DarkroomEngine().process(img, tilted, "a", _ctx(img, tilted, True))
    np.testing.assert_allclose(out, fresh, atol=1e-5)


def test_the_contrast_mask_plane_follows_the_source():
    cfg = WorkspaceConfig()
    cfg = replace(cfg, exposure=replace(cfg.exposure, contrast_mask=0.6))
    a, b = _img(1), _img(2)
    engine = DarkroomEngine()
    engine.process(a, cfg, "a", _ctx(a, cfg))
    out = engine.process(b, cfg, "b", _ctx(b, cfg))
    fresh = DarkroomEngine().process(b, cfg, "b", _ctx(b, cfg))
    np.testing.assert_allclose(out, fresh, atol=1e-5)
