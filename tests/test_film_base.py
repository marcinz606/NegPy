"""The Film Base tool: a base the user sets on the roll wins over the rebate reader."""

from __future__ import annotations

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.normalization import LogNegativeBounds, log_from_val
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.simulated.images import film
from negpy.services.assets import rolls


def _negative() -> np.ndarray:
    return np.ascontiguousarray(film(600, 900, 3, 24)[0], dtype=np.float32)


def test_log_from_val_undoes_the_window():
    bounds = LogNegativeBounds(floors=(-1.5, -1.4, -1.3), ceils=(-0.3, -0.2, -0.1))
    val = tuple((v - f) / (c - f) for v, f, c in zip((-0.9, -0.8, -0.7), bounds.floors, bounds.ceils))

    r, g, b = log_from_val(val, bounds)

    assert abs(r + 0.9) < 1e-6 and abs(g + 0.8) < 1e-6 and abs(b + 0.7) < 1e-6


class _Repo:
    def __init__(self) -> None:
        self.store: dict = {}

    def get_global_setting(self, key, default=None):
        return self.store.get(key, default)

    read_global_setting = get_global_setting

    def save_global_setting(self, key, value):
        self.store[key] = value


def _repo_with_roll():
    repo = _Repo()
    roll_id = rolls.recognize_folder(repo, "/tmp/roll-a") or rolls.folder_roll_id_for_path(repo, "/tmp/roll-a/x.ARW")
    return repo, roll_id


def test_the_roll_keeps_the_base_it_was_given():
    repo, roll_id = _repo_with_roll()
    assert roll_id is not None

    assert rolls.roll_film_base(repo, roll_id) is None
    assert rolls.set_roll_film_base(repo, roll_id, (-0.33, -0.32, -0.31)) is True
    assert rolls.roll_film_base(repo, roll_id) == (-0.33, -0.32, -0.31)
    assert rolls.set_roll_film_base(repo, roll_id, None) is True
    assert rolls.roll_film_base(repo, roll_id) is None
    assert rolls.roll_film_base(repo, None) is None
    assert rolls.set_roll_film_base(repo, "no-such-roll", (0.0, 0.0, 0.0)) is False


def test_a_picked_base_round_trips_through_the_crosstalk_unmix():
    """The tool stores M⁻¹·picked; a render unmixes it back to what was clicked."""
    from negpy.features.exposure.normalization import unmix_log_image

    m = np.array([[1.0, -0.05, 0.0], [-0.04, 1.0, -0.03], [0.0, -0.06, 1.0]])
    clicked = np.array([-0.9, -0.8, -0.7])
    stored = np.linalg.solve(m, clicked)

    back = unmix_log_image(stored.reshape(1, 1, 3).astype(np.float32), m).reshape(3)

    assert np.allclose(back, clicked, atol=1e-5)


def test_a_pick_after_a_cached_render_reads_above_it(monkeypatch):
    """The base stage's cache key carries the base, so a pick with no other change re-reads."""
    from negpy.services.rendering.engine import DarkroomEngine

    img, cfg, engine = _negative(), WorkspaceConfig(), DarkroomEngine()
    first = PipelineContext(scale_factor=1.0, original_size=(900, 600), process_mode=ProcessMode.C41)
    engine.process(img, cfg, "same", context=first)
    second = PipelineContext(scale_factor=1.0, original_size=(900, 600), process_mode=ProcessMode.C41)
    second.metrics["film_base"] = (-1.5, -1.5, -1.5)
    engine.process(img, cfg, "same", context=second)

    assert first.metrics["base_source"] == "read"
    assert second.metrics["base_source"] == "picked"
    assert second.metrics["density_above_base"] != first.metrics["density_above_base"]


def test_the_cpu_engine_prefers_the_picked_base(monkeypatch):
    from negpy.services.rendering import base_read
    from negpy.services.rendering.engine import DarkroomEngine

    monkeypatch.setattr(base_read, "measure_base_log", lambda image: (_ for _ in ()).throw(AssertionError("the reader must not run")))
    picked = (-0.33, -0.32, -0.31)
    context = PipelineContext(scale_factor=1.0, original_size=(900, 600), process_mode=ProcessMode.C41)
    context.metrics["film_base"] = picked

    DarkroomEngine().process(_negative(), WorkspaceConfig(), "picked", context=context)

    assert context.metrics["base_source"] == "picked"
    assert context.metrics["base_log"] == picked
    assert context.metrics["density_above_base"] is not None


def test_both_engines_agree_on_a_picked_base():
    from negpy.infrastructure.gpu.device import GPUDevice

    if not GPUDevice.get().is_available:
        import pytest

        pytest.skip("GPU not available")
    from negpy.services.rendering.engine import DarkroomEngine
    from negpy.services.rendering.gpu_engine import GPUEngine

    img, cfg, picked = _negative(), WorkspaceConfig(), (-0.33, -0.32, -0.31)
    engine = GPUEngine()
    try:
        _tex, gpu = engine.process_to_texture(
            img,
            cfg,
            scale_factor=1.0,
            apply_layout=False,
            readback_metrics=True,
            source_hash="pick",
            analysis_source_hash="pick",
            film_base=picked,
        )
    finally:
        engine.destroy_all()
    context = PipelineContext(scale_factor=1.0, original_size=(900, 600), process_mode=ProcessMode.C41)
    context.metrics["film_base"] = picked
    DarkroomEngine().process(img, cfg, "pick", context=context)

    assert gpu["base_source"] == "picked"
    assert all(abs(c - g) < 0.02 for c, g in zip(context.metrics["density_above_base"], gpu["density_above_base"]))


def test_the_tool_is_registered():
    from negpy.desktop.session import ToolMode
    from negpy.desktop.view.canvas.toolbar import _CANVAS_TOOLS
    from negpy.desktop.view.shortcut_registry import REGISTRY

    assert "film_base" in REGISTRY
    assert any(not isinstance(spec, str) and spec[0] is ToolMode.FILM_BASE for spec in _CANVAS_TOOLS)


def _controller(repo, *, roll_id=None, own=None):
    """A controller over *repo* whose frame /tmp/roll-a/1.tif carries *own* as its picked base."""
    from dataclasses import replace
    from unittest.mock import MagicMock

    from negpy.desktop.controller import AppController
    from negpy.desktop.session import AppState

    ctrl = AppController.__new__(AppController)
    ctrl.session = MagicMock()
    ctrl.session.repo = repo
    ctrl.state = AppState()
    ctrl.state.active_roll_id = roll_id
    ctrl.state.uploaded_files = [{"path": "/tmp/roll-a/1.tif", "hash": "h1"}]
    ctrl.state.selected_file_idx = 0
    ctrl.state.config = replace(ctrl.state.config, process=replace(ctrl.state.config.process, film_base=own))
    ctrl.set_status = MagicMock()
    ctrl.set_active_tool = MagicMock()
    ctrl.tool_sync_requested = MagicMock()
    ctrl.request_render = MagicMock()
    ctrl.apply_config = lambda config, **_k: setattr(ctrl.state, "config", config)
    return ctrl


ROLL_BASE = (-0.33, -0.32, -0.31)
OWN_BASE = (-0.21, -0.2, -0.19)


def test_an_open_rolls_base_wins_and_a_loose_frames_own_base_wins():
    repo, roll_id = _repo_with_roll()
    rolls.set_roll_film_base(repo, roll_id, ROLL_BASE)
    assert _controller(repo, roll_id=roll_id, own=OWN_BASE).film_base() == ROLL_BASE
    # No roll open: the frame's own base, else its library roll's (a search result).
    assert _controller(repo, own=OWN_BASE).film_base() == OWN_BASE
    assert _controller(repo).film_base() == ROLL_BASE
    rolls.set_roll_film_base(repo, roll_id, None)
    assert _controller(repo, roll_id=roll_id, own=OWN_BASE).film_base() == OWN_BASE


def _pick(ctrl):
    from unittest.mock import MagicMock

    bounds = LogNegativeBounds(floors=(-1.0, -1.0, -1.0), ceils=(0.0, 0.0, 0.0))
    ctrl.state.last_metrics = {"final_bounds": bounds}
    ctrl._sample_normalized_patch = MagicMock(return_value=[(0.7, 0.7, 0.7)] * 9)
    ctrl._handle_film_base_pick(0.5, 0.5)


def test_a_pick_with_no_roll_open_sets_the_frames_own_base():
    repo, roll_id = _repo_with_roll()
    ctrl = _controller(repo)
    _pick(ctrl)
    assert ctrl.state.config.process.film_base is not None
    assert rolls.roll_film_base(repo, roll_id) is None
    ctrl.set_status.assert_called_once_with("Film base set for this frame", 3000)


def test_a_pick_in_an_open_roll_sets_the_rolls_base():
    repo, roll_id = _repo_with_roll()
    ctrl = _controller(repo, roll_id=roll_id)
    _pick(ctrl)
    assert rolls.roll_film_base(repo, roll_id) is not None
    assert ctrl.state.config.process.film_base is None


def test_clear_forgets_the_base_in_use():
    repo, roll_id = _repo_with_roll()
    rolls.set_roll_film_base(repo, roll_id, ROLL_BASE)
    ctrl = _controller(repo, own=OWN_BASE)
    ctrl.clear_film_base()
    assert ctrl.state.config.process.film_base is None and rolls.roll_film_base(repo, roll_id) == ROLL_BASE

    ctrl = _controller(repo, roll_id=roll_id, own=OWN_BASE)
    ctrl.clear_film_base()
    assert rolls.roll_film_base(repo, roll_id) is None and ctrl.state.config.process.film_base == OWN_BASE


def test_a_frames_own_base_round_trips_and_leaves_thumbnails_alone():
    from dataclasses import replace

    from negpy.services.assets.thumbnail_fingerprint import thumbnail_fingerprint

    config = WorkspaceConfig()
    picked = replace(config, process=replace(config.process, film_base=OWN_BASE))
    assert WorkspaceConfig.from_flat_dict(picked.to_dict()).process.film_base == OWN_BASE

    def fingerprint(c):
        return thumbnail_fingerprint(c, workspace_color_space="Adobe RGB", input_icc_path=None)

    assert fingerprint(picked) == fingerprint(config)


def test_with_no_roll_open_clear_leaves_the_library_rolls_base_alone():
    repo, roll_id = _repo_with_roll()
    rolls.set_roll_film_base(repo, roll_id, ROLL_BASE)
    ctrl = _controller(repo)
    assert ctrl.film_base() == ROLL_BASE and not ctrl.clearable_film_base()
    ctrl.clear_film_base()
    assert rolls.roll_film_base(repo, roll_id) == ROLL_BASE
    assert _controller(repo, roll_id=roll_id).clearable_film_base()
