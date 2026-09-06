from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import ProcessMode
from negpy.services.rendering.engine import DarkroomEngine


@pytest.mark.parametrize("mode", list(ProcessMode))
def test_uncached_render_matches_cached_and_leaves_no_stage_arrays(mode):
    source = np.random.default_rng(91).uniform(0.02, 0.85, (48, 64, 3)).astype(np.float32)
    original = source.copy()
    config = WorkspaceConfig()
    config = replace(config, process=replace(config.process, process_mode=mode))
    engine = DarkroomEngine()
    cached = engine.process(source, config, "same-source")
    assert engine.cache.base is not None
    context = PipelineContext(
        original_size=(48, 64), scale_factor=64 / engine.config.preview_render_size, process_mode=mode, cache_stages=False
    )
    uncached = engine.process(source, config, "same-source", context)
    np.testing.assert_array_equal(cached, uncached)
    np.testing.assert_array_equal(source, original)
    for name in ("base", "exposure", "clahe", "lab"):
        assert getattr(engine.cache, name) is None
    preview = engine.process(source, config, "same-source")
    np.testing.assert_array_equal(preview, cached)
    assert engine.cache.base is not None
