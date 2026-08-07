"""A CompactSlider's title label must show the same tooltip as its groove: a child
with its own tooltip shadows the parent's, so the label used to repeat only the
control name while the explanatory text sat on the container."""

from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.widgets.sliders import CompactSlider


def test_label_tooltip_defaults_to_name_plus_reset_hint():
    s = CompactSlider("Density", 0.0, 4.0, 1.0)

    assert s.toolTip().startswith("<qt>")
    assert "Density" in s.toolTip()
    assert "Double-click to reset" in s.toolTip()
    assert s.label.toolTip() == s.toolTip()


def test_explanatory_tooltip_reaches_both_halves():
    s = CompactSlider("Density", 0.0, 4.0, 1.0)
    s.setToolTip("Overall print density — lower = brighter")

    assert "Overall print density" in s.toolTip()
    assert s.toolTip().count("Double-click to reset") == 1
    assert s.label.toolTip() == s.toolTip()


def test_shortcut_chips_survive_unescaped():
    s = CompactSlider("Density", 0.0, 4.0, 1.0)
    s.setToolTip(tooltip_with_shortcut("Print density", ["density_up", "density_down"]))

    assert "<table" in s.toolTip()
    assert "&lt;table" not in s.toolTip()
    # The hint follows the chips, inside the <qt> document.
    assert s.toolTip().index("<table") < s.toolTip().index("Double-click to reset")
    assert s.toolTip().endswith("</qt>")
    assert s.label.toolTip() == s.toolTip()


def test_plain_text_still_escaped():
    s = CompactSlider("Ratio", 0.0, 4.0, 1.0)
    s.setToolTip("a < b")

    assert "a &lt; b" in s.toolTip()


def test_export_form_sliders_tooltip_the_container():
    """These four used to tooltip only .label, leaving the groove bare."""
    from negpy.desktop.view.widgets.export_settings_form import ExportSettingsForm

    form = ExportSettingsForm()

    assert "libjxl" in form.jxl_distance_spin.toolTip()
    assert form.jxl_distance_spin.label.toolTip() == form.jxl_distance_spin.toolTip()
    assert "Encoder effort" in form.webp_method_spin.toolTip()
