"""A panel slider resets to NegPy's default, never to the config the panel was built from,
which holds the sticky values a fresh frame carries."""

import dataclasses

from conftest import FakeController, FakeRepo

from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG

# Each sets its reset value on sync: the meter-driven Tone sliders and Masking from the meter,
# Brush Size from the user's saved size.
_OWN_RESET = {
    "density_slider",
    "grade_slider",
    "shadow_density_slider",
    "highlight_density_slider",
    "sharpen_masking_slider",
    "manual_size_slider",
}


def _moved(config):
    """*config* with every plain float field moved off its value."""

    def section(sec):
        floats = {f.name: getattr(sec, f.name) + 0.03 for f in dataclasses.fields(sec) if type(getattr(sec, f.name)) is float}
        return dataclasses.replace(sec, **floats)

    sections = {
        f.name: section(getattr(config, f.name)) for f in dataclasses.fields(config) if dataclasses.is_dataclass(getattr(config, f.name))
    }
    return dataclasses.replace(config, **sections)


def _reset_values(qapp, config) -> dict:
    from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
    from negpy.desktop.view.widgets.sliders import CompactSlider

    controller = FakeController(FakeRepo())
    controller.state.config = config
    panel = ControlsPanel(controller)
    found = {}
    for sidebar in vars(panel).values():
        for name, widget in vars(sidebar).items() if hasattr(sidebar, "__dict__") else ():
            if isinstance(widget, CompactSlider) and name not in _OWN_RESET:
                found[f"{type(sidebar).__name__}.{name}"] = widget.default_value()
    return found


def test_no_slider_takes_its_reset_value_from_the_build_config(qapp):
    shipped = _reset_values(qapp, DEFAULT_WORKSPACE_CONFIG)
    carried = _reset_values(qapp, _moved(DEFAULT_WORKSPACE_CONFIG))

    assert len(shipped) > 50
    moved = sorted(name for name in shipped if carried.get(name) != shipped[name])
    assert not moved, "Pass NegPy's default as the reset value and the frame's value as value=:\n" + "\n".join(moved)
