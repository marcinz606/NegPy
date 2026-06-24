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
        "jxl_lossless": False,
        "jxl_distance": 2.0,
        "jxl_effort": 5,
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


def test_jxl_controls_visible_only_for_jxl(qapp):
    form = ExportSettingsForm()
    form.load(_values(export_fmt=ExportFormat.JPEG))
    assert form._jxl_container.isHidden()
    form.load(_values(export_fmt=ExportFormat.JXL, export_color_space=ColorSpace.SRGB.value))
    assert not form._jxl_container.isHidden()


def test_jxl_supported_space_not_blocked(qapp):
    form = ExportSettingsForm()
    form.load(_values(export_fmt=ExportFormat.JXL, export_color_space=ColorSpace.REC2020.value))
    assert not form.is_export_blocked()

    # Non-JXL formats are never blocked by colour space.
    form.load(_values(export_fmt=ExportFormat.TIFF, export_color_space=ColorSpace.ADOBE_RGB.value))
    assert not form.is_export_blocked()


def test_jxl_greys_unsupported_color_spaces_and_disables_output_icc(qapp):
    form = ExportSettingsForm()
    form.load(_values(export_fmt=ExportFormat.JXL, export_color_space=ColorSpace.SRGB.value))

    model = form.color_space_combo.model()
    for i in range(form.color_space_combo.count()):
        space = form.color_space_combo.itemText(i)
        supported = space in {
            ColorSpace.SRGB.value,
            ColorSpace.P3_D65.value,
            ColorSpace.REC2020.value,
            ColorSpace.GREYSCALE.value,
            ColorSpace.SAME_AS_SOURCE.value,
        }
        assert model.item(i).isEnabled() == supported, space

    # Custom output ICC override would mistag — forced off and disabled for JXL.
    assert not form.icc_output_combo.isEnabled()
    assert form.icc_output_combo.currentIndex() == 0


def test_jxl_switches_unsupported_current_space_to_srgb(qapp):
    form = ExportSettingsForm()
    form.load(_values(export_fmt=ExportFormat.JPEG, export_color_space=ColorSpace.ADOBE_RGB.value))
    # Switching to JXL while on an unsupported space snaps to sRGB.
    form.fmt_combo.setCurrentText(ExportFormat.JXL)
    assert form.color_space_combo.currentText() == ColorSpace.SRGB.value
    assert not form.is_export_blocked()


def test_leaving_jxl_re_enables_color_spaces_and_output_icc(qapp):
    form = ExportSettingsForm()
    form.load(_values(export_fmt=ExportFormat.JXL, export_color_space=ColorSpace.SRGB.value))
    form.fmt_combo.setCurrentText(ExportFormat.TIFF)
    model = form.color_space_combo.model()
    assert all(model.item(i).isEnabled() for i in range(form.color_space_combo.count()))
    assert form.icc_output_combo.isEnabled()


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
