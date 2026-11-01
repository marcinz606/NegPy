import numpy as np
import pytest

from negpy.features.exposure.normalization import LogNegativeBounds, resolve_bounds_detailed
from negpy.features.process.models import ProcessConfig

_AUTO = LogNegativeBounds((-2.0, -2.1, -2.2), (-0.3, -0.4, -0.5))
_DMIN = (-0.1, -0.2, -0.6)
_DMAX = (-1.8, -1.9, -2.4)


def _t(b: LogNegativeBounds) -> tuple:
    return tuple(b.floors), tuple(b.ceils)


def _resolve(**kw) -> tuple:
    final, base = resolve_bounds_detailed(ProcessConfig(**kw), lambda: _AUTO)
    return _t(final), _t(base)


@pytest.mark.parametrize(
    ("picks", "expected"),
    [
        ({}, (_AUTO.floors, _AUTO.ceils)),
        ({"manual_dmin": _DMIN}, (_AUTO.floors, _DMIN)),
        ({"manual_dmax": list(_DMAX)}, (_DMAX, _AUTO.ceils)),
        ({"manual_dmin": _DMIN, "manual_dmax": _DMAX}, (_DMAX, _DMIN)),
    ],
)
def test_a_picked_end_replaces_that_end_of_the_analysis(picks, expected):
    final, base = _resolve(**picks)
    assert base == expected
    assert final == expected


def test_roll_average_on_both_axes_wins_over_picks():
    locked = LogNegativeBounds((-1.5, -1.5, -1.5), (-0.2, -0.2, -0.2))
    final, _ = _resolve(
        manual_dmin=_DMIN,
        manual_dmax=_DMAX,
        use_luma_average=True,
        use_color_average=True,
        locked_floors=locked.floors,
        locked_ceils=locked.ceils,
    )
    assert final == _t(locked)


def _controller():
    from dataclasses import replace
    from unittest.mock import MagicMock

    from negpy.desktop.controller import AppController
    from negpy.desktop.session import AppState

    ctrl = AppController.__new__(AppController)
    ctrl.state = AppState()
    ctrl.state.config = replace(ctrl.state.config, process=replace(ctrl.state.config.process, local_floors=(-1.0,) * 3, lock_bounds=True))
    ctrl.state.last_metrics = {"final_bounds": LogNegativeBounds((-1.0, -1.0, -1.0), (0.0, 0.0, 0.0))}
    ctrl.set_status = MagicMock()
    ctrl.set_active_tool = MagicMock()
    ctrl.tool_sync_requested = MagicMock()
    ctrl.apply_config = lambda config, **_k: setattr(ctrl.state, "config", config)
    return ctrl


def _click(ctrl, field, value):
    from unittest.mock import MagicMock

    ctrl._sample_normalized_patch = MagicMock(return_value=[value] * 8 + [(1.0, 1.0, 1.0)])
    ctrl._handle_bounds_pick(0.5, 0.5, field)


def test_each_tool_sets_its_own_end_and_disarms():
    from negpy.desktop.session import ToolMode

    ctrl = _controller()
    _click(ctrl, "manual_dmax", (0.1, 0.2, 0.3))
    proc = ctrl.state.config.process
    assert proc.manual_dmax == pytest.approx((-0.9, -0.8, -0.7)) and proc.manual_dmin is None
    assert proc.local_floors == (0.0, 0.0, 0.0)  # measured again even under Lock Bounds
    ctrl.set_active_tool.assert_called_once_with(ToolMode.NONE)

    _click(ctrl, "manual_dmin", (0.9, 0.8, 0.7))
    assert ctrl.state.config.process.manual_dmin == pytest.approx((-0.1, -0.2, -0.3))
    assert ctrl.state.config.process.manual_dmin_at == (0.5, 0.5)


def test_clear_forgets_only_its_own_end():
    ctrl = _controller()
    _click(ctrl, "manual_dmin", (0.9, 0.8, 0.7))
    _click(ctrl, "manual_dmax", (0.1, 0.2, 0.3))
    ctrl.clear_bounds_pick("manual_dmin")
    proc = ctrl.state.config.process
    assert proc.manual_dmin is None and proc.manual_dmin_at is None
    assert proc.manual_dmax is not None and proc.manual_dmax_at is not None
    assert ctrl.has_bounds_picks()


def test_the_tools_are_registered():
    from negpy.desktop.session import UNCROPPED_PREVIEW_TOOLS, ToolMode
    from negpy.desktop.view.canvas.toolbar import _CANVAS_TOOLS
    from negpy.desktop.view.shortcut_registry import REGISTRY

    for action, mode in (("dmin_pick", ToolMode.DMIN_PICK), ("dmax_pick", ToolMode.DMAX_PICK)):
        assert action in REGISTRY
        assert mode in UNCROPPED_PREVIEW_TOOLS
        assert any(not isinstance(spec, str) and spec[0] is mode for spec in _CANVAS_TOOLS)


def test_both_engines_render_with_the_picked_bounds():
    from dataclasses import replace

    from negpy.domain.interfaces import PipelineContext
    from negpy.domain.models import WorkspaceConfig
    from negpy.features.process.models import ProcessMode
    from negpy.infrastructure.gpu.device import GPUDevice
    from negpy.infrastructure.simulated.images import film

    if not GPUDevice.get().is_available:
        pytest.skip("GPU not available")
    from negpy.services.rendering.engine import DarkroomEngine
    from negpy.services.rendering.gpu_engine import GPUEngine

    img = np.ascontiguousarray(film(600, 900, 3, 24)[0], dtype=np.float32)
    cfg = WorkspaceConfig()
    cfg = replace(cfg, process=replace(cfg.process, manual_dmin=(-0.2, -0.3, -0.5), manual_dmax=(-1.6, -1.7, -2.0)))
    engine = GPUEngine()
    try:
        _tex, gpu = engine.process_to_texture(
            img, cfg, scale_factor=1.0, apply_layout=False, readback_metrics=True, source_hash="pick", analysis_source_hash="pick"
        )
    finally:
        engine.destroy_all()
    context = PipelineContext(scale_factor=1.0, original_size=(900, 600), process_mode=ProcessMode.C41)
    DarkroomEngine().process(img, cfg, "pick", context=context)

    for engine_bounds in (gpu["log_bounds_base"], context.metrics["log_bounds_base"]):
        floors, ceils = _t(engine_bounds)
        assert floors == pytest.approx((-1.6, -1.7, -2.0)) and ceils == pytest.approx((-0.2, -0.3, -0.5))


def test_the_readout_colors_each_channel():
    from negpy.desktop.view.sidebar.process import _picked_densities
    from negpy.desktop.view.styles.theme import THEME

    text = _picked_densities((-0.21, -0.48, -0.83))
    for color, part in ((THEME.channel_red_text, "R 0.21"), (THEME.channel_green_text, "G 0.48"), (THEME.channel_blue_text, "B 0.83")):
        assert f'<span style="color:{color}">{part}</span>' in text
    assert _picked_densities(None) == "Auto"


def test_a_pick_location_never_reaches_the_thumbnail():
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.assets.thumbnail_fingerprint import thumbnail_fingerprint

    cfg = WorkspaceConfig()
    moved = replace(cfg, process=replace(cfg.process, manual_dmin_at=(0.1, 0.2)))

    def fp(c):
        return thumbnail_fingerprint(c, workspace_color_space="Adobe RGB", input_icc_path=None)

    assert fp(cfg) == fp(moved)
