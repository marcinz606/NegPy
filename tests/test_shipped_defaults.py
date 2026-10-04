from dataclasses import fields, is_dataclass
from unittest.mock import MagicMock

import pytest

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def test_a_fresh_session_starts_on_the_shipped_config():
    assert AppState().config == DEFAULT_WORKSPACE_CONFIG


def test_the_dataclass_defaults_carry_the_shipped_values():
    bare, shipped = WorkspaceConfig(), DEFAULT_WORKSPACE_CONFIG
    for section in fields(bare):
        bare_section, shipped_section = getattr(bare, section.name), getattr(shipped, section.name)
        if not is_dataclass(bare_section):
            assert bare_section == shipped_section, section.name
            continue
        for f in fields(bare_section):
            assert getattr(bare_section, f.name) == getattr(shipped_section, f.name), f"{section.name}.{f.name}"


@pytest.fixture
def panel(qapp):
    controller = MagicMock()
    controller.state = AppState()
    return ControlsPanel(controller)


def test_sliders_double_click_to_the_value_their_card_resets_to(panel):
    assert panel.tone_sidebar.grade_slider._default == DEFAULT_WORKSPACE_CONFIG.exposure.grade
    assert panel.sensor_sidebar.crosstalk_strength_slider._default == DEFAULT_WORKSPACE_CONFIG.process.crosstalk_strength


def test_the_grade_slider_travels_exactly_the_curves_own_clamp(panel):
    slider = panel.tone_sidebar.grade_slider
    assert (slider._min, slider._max) == (EXPOSURE_CONSTANTS["iso_r_min"], EXPOSURE_CONSTANTS["iso_r_max"])
