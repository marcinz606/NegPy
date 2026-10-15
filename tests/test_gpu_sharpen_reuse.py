"""A lab-only change reuses the sharpen blur; anything upstream recomputes it."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.lab.models import SharpenMethod
from negpy.infrastructure.gpu.device import GPUDevice

pytestmark = pytest.mark.skipif(not GPUDevice.get().is_available, reason="GPU not available")


def _cfg(method: SharpenMethod, saturation: float = 1.0, density: float = 0.0) -> WorkspaceConfig:
    cfg = WorkspaceConfig()
    return replace(
        cfg,
        lab=replace(cfg.lab, sharpen=0.8, sharpen_method=method, sharpen_radius=1.5, saturation=saturation),
        exposure=replace(cfg.exposure, density=cfg.exposure.density + density),
    )


def _render(engine, img, cfg):
    tex, _ = engine.process_to_texture(img, cfg, source_hash="s", readback_metrics=False)
    return engine._readback_downsampled(tex)


@pytest.mark.parametrize("method,blur_pass", [(SharpenMethod.RL, "rl_blur_h"), (SharpenMethod.USM, "lab_sharpen_h")])
def test_a_lab_only_change_skips_the_blur_and_matches_a_fresh_render(method, blur_pass):
    from negpy.services.rendering.gpu_engine import GPUEngine

    img = np.random.default_rng(1).uniform(0.1, 0.7, (120, 160, 3)).astype(np.float32)
    engine, fresh = GPUEngine(), GPUEngine()
    try:
        _render(engine, img, _cfg(method))
        with patch.object(GPUEngine, "_dispatch_pass", autospec=True, side_effect=GPUEngine._dispatch_pass) as dispatch:
            reused = _render(engine, img, _cfg(method, saturation=1.3))
        assert blur_pass not in [c.args[2] for c in dispatch.call_args_list]
        np.testing.assert_array_equal(reused, _render(fresh, img, _cfg(method, saturation=1.3)))

        # Upstream moved, then a lab-only step: the blur follows the new input.
        _render(engine, img, _cfg(method, saturation=1.3, density=0.2))
        moved = _render(engine, img, _cfg(method, saturation=1.1, density=0.2))
        np.testing.assert_array_equal(moved, _render(fresh, img, _cfg(method, saturation=1.1, density=0.2)))
    finally:
        engine.destroy_all()
        fresh.destroy_all()
