"""Round-trip tests for the shared ExportSettingsForm widget."""

from negpy.desktop.view.widgets.export_settings_form import ExportSettingsForm
from negpy.domain.models import (
    JXL_TAGGABLE_SPACES,
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
        "webp_quality": 90,
        "webp_lossless": False,
        "webp_method": 4,
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

    # Non-JXL formats are never blocked by color space.
    form.load(_values(export_fmt=ExportFormat.TIFF, export_color_space=ColorSpace.ADOBE_RGB.value))
    assert not form.is_export_blocked()


def test_jxl_switches_same_as_source_to_srgb(qapp):
    """'Same as Source' resolves per-file at export time — usually to the Adobe
    RGB working space for scans/raws — which JXL can't tag. The form must not let
    a user park on that combination; it self-heals to sRGB like any other
    untaggable space (test_jxl_switches_unsupported_current_space_to_srgb)."""
    form = ExportSettingsForm()
    form.load(_values(export_fmt=ExportFormat.TIFF, export_color_space=ColorSpace.SAME_AS_SOURCE.value))
    form.fmt_combo.setCurrentText(ExportFormat.JXL)
    assert form.color_space_combo.currentText() == ColorSpace.SRGB.value
    assert not form.is_export_blocked()


def test_jxl_greys_unsupported_color_spaces_and_disables_output_icc(qapp):
    form = ExportSettingsForm()
    form.load(_values(export_fmt=ExportFormat.JXL, export_color_space=ColorSpace.SRGB.value))

    model = form.color_space_combo.model()
    for i in range(form.color_space_combo.count()):
        space = form.color_space_combo.itemText(i)
        supported = space in JXL_TAGGABLE_SPACES
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


def test_flat_mode_limits_format_choices(qapp):
    form = ExportSettingsForm()
    form.load(_values(export_fmt=ExportFormat.JPEG))
    assert not form._format_section.isHidden()
    form.set_flat_mode(True)
    assert not form._format_section.isHidden()
    assert form.flat_mode()
    assert [form.fmt_combo.itemData(i) for i in range(form.fmt_combo.count())] == [ExportFormat.TIFF.value, ExportFormat.JXL.value]
    assert form.fmt_combo.itemText(1) == "JXL (lossless)"
    form.set_flat_mode(False)
    assert not form._format_section.isHidden()
    assert ExportFormat.JPEG.value in [form.fmt_combo.itemText(i) for i in range(form.fmt_combo.count())]


def test_linear_mode_keeps_destination_and_drops_the_rest(qapp):
    """Linear Output has no use for format, size or color, but needs the destination
    rules as much as print does — hiding the whole form was what left it with none (#859)."""
    form = ExportSettingsForm()
    form.load(_values(output_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE))

    form.set_linear_mode(True)
    assert form.linear_mode()
    assert form._format_section.isHidden()
    assert form._size_section.isHidden()
    assert form._color_section.isHidden()
    assert not form._subfolder_container.isHidden()
    assert not form.filename_edit.isHidden()

    form.set_linear_mode(False)
    assert not form._format_section.isHidden()
    assert not form._size_section.isHidden()
    assert not form._color_section.isHidden()


def test_flat_mode_does_not_reveal_format_under_linear(qapp):
    """Switching intents runs set_flat_mode first, which re-shows FORMAT unconditionally."""
    form = ExportSettingsForm()
    form.load(_values())
    form.set_linear_mode(True)
    form.set_flat_mode(True)
    assert form._format_section.isHidden()


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


def test_flat_mode_forces_jxl_lossless_and_hides_the_toggle(qapp):
    """flat_export_config() always overrides jxl_lossless=True for a flat master —
    the lossy toggle/distance row would be silently ignored, so flat mode hides it
    and pins the checkbox rather than showing a control with no effect."""
    form = ExportSettingsForm()
    form.load(_values(export_fmt=ExportFormat.JXL, jxl_lossless=False, export_color_space=ColorSpace.SRGB.value))
    assert not form.jxl_lossless_check.isChecked()
    assert not form.jxl_lossless_check.isHidden()

    form.set_flat_mode(True)
    assert form.fmt_combo.currentData() == ExportFormat.JXL.value
    assert form.jxl_lossless_check.isChecked()
    assert form.jxl_lossless_check.isHidden()
    assert form.jxl_distance_spin.isHidden()

    form.set_flat_mode(False)
    assert form.fmt_combo.currentData() == ExportFormat.JXL.value
    assert not form.jxl_lossless_check.isHidden()
    assert not form.jxl_distance_spin.isHidden()
