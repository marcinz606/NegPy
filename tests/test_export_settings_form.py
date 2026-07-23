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


def test_destination_mode_restored_from_persisted_plain_string(qapp):
    """Saved settings come back from JSON as plain strings, not StrEnum members —
    the combo must still land on the saved destination mode."""
    form = ExportSettingsForm()
    for mode in ExportPresetOutputMode:
        form.load(_values(output_mode=str(mode.value)))
        assert form.output_mode_combo.currentData() == mode, mode
        assert form.values()["output_mode"] == mode, mode


def test_destination_mode_falls_back_to_absolute_when_unknown(qapp):
    form = ExportSettingsForm()
    form.load(_values(output_mode="not_a_mode"))
    assert form.values()["output_mode"] == ExportPresetOutputMode.ABSOLUTE


def test_load_does_not_emit_changed(qapp):
    form = ExportSettingsForm()
    fired = []
    form.changed.connect(lambda: fired.append(True))
    form.load(_values())
    assert not fired


def test_flat_mode_hides_format_section(qapp):
    form = ExportSettingsForm()
    form.load(_values())
    assert not form._format_section.isHidden()
    form.set_flat_mode(True)
    assert form._format_section.isHidden()
    assert form.flat_mode()
    form.set_flat_mode(False)
    assert not form._format_section.isHidden()


def test_flat_mode_hides_paper_ratio_for_original(qapp):
    form = ExportSettingsForm()
    form.load(_values(export_resolution_mode=ExportResolutionMode.ORIGINAL.value))
    form.set_flat_mode(True)
    assert form._ratio_row_widget.isHidden()
    form.mode_target_px_btn.setChecked(True)
    assert not form._ratio_row_widget.isHidden()


def test_flat_mode_skips_jxl_export_block(qapp):
    form = ExportSettingsForm()
    form.load(_values(export_fmt=ExportFormat.JXL, export_color_space=ColorSpace.SRGB.value))
    form._flat_mode = True
    assert not form.is_export_blocked()
    form.set_flat_mode(False)
    form.set_flat_mode(True)
    assert not form.is_export_blocked()


def test_flat_preset_editor_limits_format_choices(qapp):
    form = ExportSettingsForm()
    form.load(_values(export_fmt=ExportFormat.JPEG))
    form.set_flat_mode(True, preset_editor=True)
    assert [form.fmt_combo.itemText(i) for i in range(form.fmt_combo.count())] == [
        ExportFormat.TIFF.value,
        ExportFormat.DNG.value,
    ]
    form.set_flat_mode(False, preset_editor=True)
    assert ExportFormat.JPEG.value in [form.fmt_combo.itemText(i) for i in range(form.fmt_combo.count())]
