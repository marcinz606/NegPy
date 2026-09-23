"""Roll/scene Cast Removal: pooling per-frame neutral axes and rendering with the pool."""

from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.normalization import (
    analyze_log_exposure_bounds,
    measure_neutral_axis,
    pool_neutral_axis,
)
from negpy.features.process.models import ProcessConfig, ProcessMode, pooled_neutral_axis
from negpy.services.assets import rolls
from negpy.services.rendering.engine import DarkroomEngine


def _axis(v: float, conf: float = 1.0, highlight: bool = True) -> tuple:
    return ((v, v + 0.1, v + 0.2), (v - 1.0, v - 0.9, v - 0.8), (v + 1.0, v + 1.1, v + 1.2) if highlight else None, conf)


def test_pool_is_the_confidence_weighted_median():
    pooled = pool_neutral_axis([_axis(0.0, 0.2), _axis(1.0, 0.9), _axis(2.0, 0.2)], np.zeros(3, bool))

    assert pooled[0] == (1.0, 1.1, 1.2)
    assert pooled[3] == 0.2


def test_heavy_weight_wins_over_the_count():
    pooled = pool_neutral_axis([_axis(0.0, 0.1), _axis(0.0, 0.1), _axis(5.0, 0.9)], np.zeros(3, bool))

    assert pooled[0][0] == 5.0


def test_outliers_and_missing_axes_are_skipped():
    pooled = pool_neutral_axis([_axis(9.0), None, _axis(1.0), _axis(0.0, 0.0)], np.array([True, False, False, False]))

    assert pooled[0][0] == 1.0


def test_highlight_pools_only_when_half_the_frames_have_one():
    frames = [_axis(0.0), _axis(1.0, highlight=False), _axis(2.0, highlight=False)]

    assert pool_neutral_axis(frames, np.zeros(3, bool))[2] is None
    assert pool_neutral_axis(frames[:2], np.zeros(2, bool))[2] == (1.0, 1.1, 1.2)


def test_nothing_to_pool_is_none():
    assert pool_neutral_axis([None, _axis(0.0, 0.0)], np.zeros(2, bool)) is None


def test_pooled_axis_reads_only_when_on_and_color_negative():
    axis = _axis(0.0)
    on = ProcessConfig(use_cast_average=True, locked_neutral_axis=axis)

    assert pooled_neutral_axis(on) == axis
    assert pooled_neutral_axis(replace(on, use_cast_average=False)) is None
    assert pooled_neutral_axis(replace(on, process_mode=ProcessMode.E6)) is None
    assert pooled_neutral_axis(ProcessConfig(use_cast_average=True)) is None


class _Repo:
    def __init__(self):
        self.settings: dict = {}

    def get_global_setting(self, key, default=None):
        return self.settings.get(key, default)

    def save_global_setting(self, key, value):
        self.settings[key] = value


def test_a_frame_loaded_later_takes_the_rolls_axis():
    repo = _Repo()
    roll_id = rolls.create_virtual_roll(repo, "500T", [])
    axis = _axis(-1.0, 0.8, highlight=False)
    rolls.set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9), axis=axis)
    config = replace(WorkspaceConfig(), process=ProcessConfig(use_color_average=True, use_cast_average=True))

    resolved = rolls.resolve_roll_baseline(repo, roll_id, "h", config)

    assert resolved.process.locked_neutral_axis == axis


def _negative() -> np.ndarray:
    e = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    log = np.stack([-0.2 + m - g * e - c * e * e for m, g, c in ((0.0, 0.66, 0.0), (-0.12, 0.71, 0.3), (-0.22, 0.68, 0.12))], -1)
    return np.ascontiguousarray(np.broadcast_to((10.0**log)[:, None, :], (256, 256, 3))).astype(np.float32)


def _render(img: np.ndarray, settings: WorkspaceConfig) -> tuple[np.ndarray, dict]:
    ctx = PipelineContext(scale_factor=1.0, original_size=(img.shape[1], img.shape[0]), process_mode=settings.process.process_mode)
    return DarkroomEngine().process(img, settings, "h", ctx), ctx.metrics


def test_the_frames_own_axis_as_the_pool_renders_like_its_own_meter():
    img = _negative()
    base = WorkspaceConfig()
    own, metrics = _render(img, base)
    pooled = replace(base, process=replace(base.process, use_cast_average=True, locked_neutral_axis=metrics["neutral_axis_refs"]))

    np.testing.assert_allclose(_render(img, pooled)[0], own, atol=1e-6)


def test_a_different_pooled_axis_changes_the_render():
    img = _negative()
    mid, shadow, highlight, conf = measure_neutral_axis(img, analyze_log_exposure_bounds(img))
    axis = ((mid[0] + 0.05, mid[1], mid[2] - 0.05), shadow, highlight, conf)
    base = WorkspaceConfig()
    pooled = replace(base, process=replace(base.process, use_cast_average=True, locked_neutral_axis=axis))

    assert np.abs(_render(img, pooled)[0] - _render(img, base)[0]).max() > 1e-2
