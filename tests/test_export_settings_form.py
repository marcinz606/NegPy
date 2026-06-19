"""Round-trip tests for the shared ExportSettingsForm widget."""

from negpy.desktop.view.widgets.export_settings_form import ExportSettingsForm
from negpy.domain.models import (
    AspectRatio,
    ColorSpace,
    ExportFormat,
    ExportPresetOutputMode,
    ExportResolutionMode,
)


def _values(**overrides) -> dict:
    base = {
        "export_fmt": ExportFormat.JPEG,
        "jpeg_quality": 88,
        "export_resolution_mode": ExportResolutionMode.PRINT.value,
        "paper_aspect_ratio": AspectRatio.ORIGINAL,
        "export_print_size": 24.0,
        "export_dpi": 360,
        "export_target_long_edge_px": 3000,
        "output_mode": ExportPresetOutputMode.SUBFOLDER_OF_SOURCE,
        "output_subfolder": "web",
        "output_path": "/tmp/out",
        "filename_pattern": "{{ original_name }}_{{ size }}",
        "overwrite": False,
        "export_color_space": ColorSpace.SRGB.value,
        "icc_input_path": None,
        "icc_output_path": None,
    }
    base.update(overrides)
    return base


def test_load_then_values_round_trip(qapp):
    form = ExportSettingsForm()
    v = _values()
    form.load(v)
    out = form.values()
    for key, expected in v.items():
        assert out[key] == expected, key


def test_jpeg_quality_hidden_for_non_jpeg(qapp):
    form = ExportSettingsForm()
    form.load(_values(export_fmt=ExportFormat.TIFF))
    assert not form._quality_container.isVisible()
    form.load(_values(export_fmt=ExportFormat.JPEG))
    # Visibility flag flips even though the widget isn't shown on screen.
    assert not form._quality_container.isHidden()


def test_destination_subfields_track_output_mode(qapp):
    form = ExportSettingsForm()
    form.load(_values(output_mode=ExportPresetOutputMode.ABSOLUTE))
    assert not form._abspath_container.isHidden()
    assert form._subfolder_container.isHidden()
    form.load(_values(output_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE))
    assert not form._subfolder_container.isHidden()
    assert form._abspath_container.isHidden()


def test_load_does_not_emit_changed(qapp):
    form = ExportSettingsForm()
    fired = []
    form.changed.connect(lambda: fired.append(True))
    form.load(_values())
    assert not fired
