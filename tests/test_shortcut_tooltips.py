"""apply_shortcut_tooltips(), the keyboard action map and the tool-mode sync all reach
into the sidebars by attribute name, and none of them is exercised by building a panel, so
a widget that moves to a different sidebar breaks them only at launch."""

import re
from pathlib import Path

import pytest

from conftest import FakeController as _Controller, FakeRepo as _Repo
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel

_VIEW = Path("negpy/desktop/view")
_ATTR_CHAIN = re.compile(r"\bcontrols(?:_panel)?\.([a-z_]+_sidebar)\.([a-z_0-9]+)")


@pytest.fixture(scope="module")
def controls(qapp):
    return ControlsPanel(_Controller(_Repo()))


def test_apply_shortcut_tooltips_reaches_every_widget_it_names(controls):
    controls.apply_shortcut_tooltips()


def test_every_sidebar_widget_the_view_layer_names_exists(controls):
    named = {pair for path in _VIEW.rglob("*.py") for pair in _ATTR_CHAIN.findall(path.read_text())}
    missing = [f"{sidebar}.{widget}" for sidebar, widget in named if not hasattr(getattr(controls, sidebar, None), widget)]
    assert missing == []
