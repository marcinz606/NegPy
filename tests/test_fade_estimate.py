"""`fade_estimate.run_estimate`, the Fade card's Estimate, without Qt."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from fade_crossover_scene import CROSSOVER, build_scene, estimate_grid, scan_image
from negpy.features.process import fade_estimate
from negpy.features.process.fade_estimate import GrayMemo, run_estimate
from negpy.features.process.models import ProcessConfig, ProcessMode


@pytest.fixture(scope="module")
def grid():
    return estimate_grid(scan_image(CROSSOVER, build_scene(96, 72)))


def _conf(**fields) -> ProcessConfig:
    return replace(ProcessConfig(), process_mode=ProcessMode.E6, **fields)


def test_a_press_sets_survivals_gray_mix_and_a_table(grid):
    out = run_estimate(_conf(), grid, None, None, GrayMemo(), white_anchor=False)

    assert out.survivals is not None
    assert out.changes["fade_ratio_g"] == out.survivals[1] and out.changes["fade_ratio_b"] == out.survivals[2]
    assert "fade_neutrals_mix" in out.changes and out.mix == out.changes["fade_neutrals_mix"]
    assert "fade_crossover_table" in out.changes
    assert out.red_hint is not None


def test_a_live_mix_drag_writes_no_table_and_reads_the_grays_once(grid):
    memo = GrayMemo()
    with patch.object(fade_estimate, "gray_readings", wraps=fade_estimate.gray_readings) as readings:
        for mix in (0.2, 0.4):
            out = run_estimate(_conf(), grid, None, None, memo, white_anchor=False, mix=mix, live=True)
    assert readings.call_count == 1
    assert "fade_crossover_table" not in out.changes
    assert out.changes["fade_neutrals_mix"] == 0.4 and out.mix is None
    assert out.red_hint is None


def test_a_failed_reading_keeps_the_survivals():
    out = run_estimate(_conf(), np.zeros((32, 32, 3), np.float32), None, None, GrayMemo(), white_anchor=False)

    assert out.survivals is None
    assert not any(k.startswith("fade_ratio") for k in out.changes)
    assert out.hint


def test_restore_range_on_takes_its_fit_and_measures_the_table_after_it(grid):
    fit = ({"fade_ratio_r": 0.6, "fade_density_shift": -0.2}, "")
    with patch.object(fade_estimate, "auto_red_changes", return_value=fit):
        out = run_estimate(_conf(fade_auto_red=True), grid, None, None, GrayMemo(), white_anchor=False)

    assert (out.changes["fade_ratio_r"], out.changes["fade_density_shift"]) == (0.6, -0.2)
    assert out.survivals is not None and out.survivals[0] == 0.6
    assert "fade_crossover_table" in out.changes
    assert not out.suggest_red


def test_a_short_range_suggests_restore_range_while_it_is_off(grid):
    with patch.object(fade_estimate, "short_range", return_value=True):
        out = run_estimate(_conf(), grid, None, None, GrayMemo(), white_anchor=False)
    assert out.suggest_red and "Restore Range" in (out.red_hint or "")


def test_estimate_keeps_a_hand_trim_and_fits_the_shift_for_the_red_it_sets(grid):
    plain = run_estimate(_conf(fade_auto_red=True), grid, None, None, GrayMemo(), white_anchor=False)
    trimmed = run_estimate(_conf(fade_auto_red=True, fade_red_trim=0.8), grid, None, None, GrayMemo(), white_anchor=False)

    assert plain.changes["fade_red_trim"] == 1.0 and trimmed.changes["fade_red_trim"] == 0.8
    assert trimmed.changes["fade_ratio_r"] == pytest.approx(round(plain.changes["fade_ratio_r"] * 0.8, 3), abs=1e-9)
    assert trimmed.changes["fade_density_shift"] != plain.changes["fade_density_shift"]


def test_the_points_at_the_conf_s_own_red_are_read_once_per_grid_and_conf(grid):
    from negpy.features.process import fade_auto_red

    conf = _conf(fade_ratio_r=0.9)
    other = grid.copy()
    with patch.object(fade_estimate, "survival_points", wraps=fade_auto_red.survival_points) as read:
        fade_estimate.auto_red_changes(conf, None, None, grid, 0.7)
        first = read.call_count
        fade_estimate.auto_red_changes(replace(conf, fade_red_trim=0.9), None, None, grid, 0.6)
        assert read.call_count == first + 1
        fade_estimate.auto_red_changes(conf, None, None, other, 0.7)
        assert read.call_count == first * 2 + 1
        fade_estimate.auto_red_changes(replace(conf, fade_ratio_g=0.8), None, None, other, 0.7)
        assert read.call_count == first * 3 + 1


def test_the_estimate_grid_ignores_restore_range_and_the_trim():
    from negpy.desktop.workers import chart_fit
    from negpy.domain.models import WorkspaceConfig

    img = np.full((16, 16, 3), 0.4, np.float32)
    cfg = WorkspaceConfig()
    rendered = np.zeros((4, 4, 3), np.float32)
    with patch.object(chart_fit, "_render_estimate_grid", return_value=rendered) as render:
        for process in (cfg.process, replace(cfg.process, fade_auto_red=True, fade_red_trim=0.7)):
            chart_fit.fade_estimate_grid(chart_fit.ChartSource(img, replace(cfg, process=process), "h", None, None, (), 1))
    assert render.call_count == 1
