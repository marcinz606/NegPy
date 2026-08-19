from datetime import datetime
from negpy.domain.models import ExportConfig, ExportFormat, ExportResolutionMode
from negpy.features.metadata.models import MetadataConfig
from negpy.services.export.templating import parse_capture_stem, render_export_filename


# ── Existing tests (unchanged behavior) ──────────────────────────────────────


def test_basic_templating():
    conf = ExportConfig(filename_pattern="test_{{ original_name }}_{{ colorspace }}", export_color_space="Adobe RGB")
    result = render_export_filename("/path/to/image.orf", conf)
    assert result == "test_image_Adobe_RGB"


def test_date_templating():
    conf = ExportConfig(filename_pattern="{{ date }}_{{ original_name }}")
    today = datetime.now().strftime("%Y%m%d")
    result = render_export_filename("my_scan.tiff", conf)
    assert result == f"{today}_my_scan"


def test_size_and_dpi_normal():
    conf = ExportConfig(
        export_resolution_mode=ExportResolutionMode.PRINT.value,
        export_print_size=30.0,
        export_dpi=300,
        filename_pattern="{{ original_name }}_{{ size }}_{{ dpi }}",
    )
    result = render_export_filename("shot.jpg", conf)
    assert result == "shot_30cm_300dpi"


def test_size_and_dpi_original_res():
    conf = ExportConfig(
        export_resolution_mode=ExportResolutionMode.ORIGINAL.value,
        export_print_size=30.0,
        export_dpi=300,
        filename_pattern="{{ original_name }}_{{ size }}_{{ dpi }}_end",
    )
    result = render_export_filename("shot.jpg", conf)
    assert result == "shot_end"


def test_target_px_filename_var():
    conf = ExportConfig(
        export_resolution_mode=ExportResolutionMode.TARGET_PX.value,
        export_target_long_edge_px=2048,
        filename_pattern="{{ original_name }}_{{ target_px }}",
    )
    result = render_export_filename("shot.jpg", conf)
    assert result == "shot_2048px"


def test_target_px_var_empty_in_print_mode():
    conf = ExportConfig(
        export_resolution_mode=ExportResolutionMode.PRINT.value,
        filename_pattern="{{ original_name }}_{{ target_px }}_end",
    )
    result = render_export_filename("shot.jpg", conf)
    assert result == "shot_end"


def test_border_logic():
    conf_border = ExportConfig(filename_pattern="{{ original_name }}_{{ border }}")
    assert render_export_filename("img.jpg", conf_border, border_size=1.5) == "img_border"

    conf_no_border = ExportConfig(filename_pattern="{{ original_name }}_{{ border }}")
    assert render_export_filename("img.jpg", conf_no_border, border_size=0.0) == "img"


def test_cleanup_logic():
    conf = ExportConfig(filename_pattern="{{ original_name }} - {{ colorspace }} --- final", export_color_space="Adobe RGB")
    # Structural template separators cleaned; original_name content preserved verbatim.
    result = render_export_filename("my scan.jpg", conf)
    assert result == "my scan_Adobe_RGB_final"


def test_format_and_ratio():
    conf = ExportConfig(
        export_fmt=ExportFormat.TIFF,
        paper_aspect_ratio="3:2",
        filename_pattern="{{ original_name }}_{{ format }}_{{ paper_ratio }}",
    )
    result = render_export_filename("img.jpg", conf)
    assert result == "img_TIFF_3:2"


def test_empty_template_fallback():
    conf = ExportConfig(filename_pattern="")
    result = render_export_filename("img.jpg", conf)
    assert result == "img"


def test_invalid_template_fallback():
    conf = ExportConfig(filename_pattern="{{ invalid_var }}")
    result = render_export_filename("img.jpg", conf)
    assert result == "img"


# ── original_name preservation ────────────────────────────────────────────────


def test_original_name_dash_preserved():
    """Dashes in the filename must not be converted to underscores."""
    conf = ExportConfig(filename_pattern="print_{{ original_name }}")
    assert render_export_filename("/shots/IMG-0001.orf", conf) == "print_IMG-0001"


def test_original_name_dash_and_underscore_preserved():
    """Files with both dashes and underscores keep both."""
    conf = ExportConfig(filename_pattern="print_{{ original_name }}")
    assert render_export_filename("scan_001-A.orf", conf) == "print_scan_001-A"


def test_original_name_multiple_underscores_preserved():
    """Double (or more) underscores inside the filename are kept as-is."""
    conf = ExportConfig(filename_pattern="print_{{ original_name }}")
    assert render_export_filename("IMG__0001.orf", conf) == "print_IMG__0001"


def test_original_name_leading_underscore_preserved():
    """A leading underscore in the filename is not stripped."""
    conf = ExportConfig(filename_pattern="print_{{ original_name }}")
    assert render_export_filename("_scan.orf", conf) == "print__scan"


def test_original_name_trailing_underscore_preserved():
    """A trailing underscore in the filename is not stripped."""
    conf = ExportConfig(filename_pattern="{{ original_name }}_end")
    assert render_export_filename("scan_.orf", conf) == "scan__end"


def test_original_name_leading_and_trailing_underscores_preserved():
    """Both leading and trailing underscores survive."""
    conf = ExportConfig(filename_pattern="print_{{ original_name }}")
    assert render_export_filename("_scan_001_.orf", conf) == "print__scan_001_"


def test_original_name_space_preserved():
    """Spaces inside the original filename are kept verbatim."""
    conf = ExportConfig(filename_pattern="print_{{ original_name }}")
    assert render_export_filename("my scan.jpg", conf) == "print_my scan"


def test_original_name_only_pattern():
    """Pattern with only original_name — no structural parts to clean."""
    conf = ExportConfig(filename_pattern="{{ original_name }}")
    assert render_export_filename("IMG-0001.orf", conf) == "IMG-0001"


def test_original_name_repeated_in_pattern():
    """original_name appearing twice is substituted correctly both times."""
    conf = ExportConfig(filename_pattern="{{ original_name }}_copy_{{ original_name }}")
    assert render_export_filename("IMG-0001.orf", conf) == "IMG-0001_copy_IMG-0001"


def test_structural_dashes_cleaned_but_original_name_untouched():
    """Dashes as template separators → underscores; dashes inside original_name → preserved."""
    conf = ExportConfig(filename_pattern="{{ original_name }}-{{ colorspace }}-final", export_color_space="Adobe RGB")
    result = render_export_filename("IMG-0001.orf", conf)
    assert result == "IMG-0001_Adobe_RGB_final"


def test_empty_pattern_fallback_preserves_dashes():
    """Fallback path (empty pattern) gives verbatim original_name."""
    conf = ExportConfig(filename_pattern="")
    assert render_export_filename("IMG-0001.orf", conf) == "IMG-0001"


def test_invalid_pattern_fallback_preserves_original_name():
    """Fallback path (bad template) gives verbatim original_name."""
    conf = ExportConfig(filename_pattern="{{ invalid_var }}")
    assert render_export_filename("IMG-0001.orf", conf) == "IMG-0001"


# ── Metadata / capture-roll variables ─────────────────────────────────────────


def test_gear_vars_render():
    meta = MetadataConfig(
        camera_make="Mamiya",
        camera_model="7",
        lens_model="80mm f/4",
        film="Portra 400",
        film_iso=400,
        film_manufacturer="Kodak",
        format="35mm",
        developer="D-76 1+1",
        push_pull=1,
        scanning="DSLR copy-stand",
        exposure_override="1/125s f/2.8",
        focal_length_mm=80.0,
    )
    conf = ExportConfig(
        filename_pattern=(
            "{{ film }}_{{ film_iso }}_{{ camera }}_{{ lens }}_{{ focal_length }}_"
            "{{ film_format }}_{{ developer }}_{{ push_pull }}_{{ scanning }}_{{ exposure }}_"
            "{{ original_name }}"
        )
    )
    result = render_export_filename("DSC0123.orf", conf, metadata=meta)
    assert result == ("Portra_400_400_Mamiya_7_80mm_f4_80_35mm_D_76_1+1_1_DSLR_copy_stand_1125s_f2.8_DSC0123")


def test_empty_gear_collapses_separators():
    conf = ExportConfig(filename_pattern="{{ film }}_{{ camera }}_{{ original_name }}_end")
    result = render_export_filename("shot.jpg", conf, metadata=MetadataConfig())
    assert result == "shot_end"


def test_format_vs_film_format_no_collision():
    meta = MetadataConfig(format="120")
    conf = ExportConfig(
        export_fmt=ExportFormat.JPEG,
        filename_pattern="{{ original_name }}_{{ format }}_{{ film_format }}",
    )
    result = render_export_filename("img.jpg", conf, metadata=meta)
    assert result == "img_JPEG_120"


def test_film_format_other():
    meta = MetadataConfig(format="Other", format_other="6×7")
    conf = ExportConfig(filename_pattern="{{ film_format }}_{{ original_name }}")
    assert render_export_filename("img.jpg", conf, metadata=meta) == "6×7_img"


def test_roll_frame_from_metadata():
    meta = MetadataConfig(capture_roll="Summer24", capture_frame=7)
    conf = ExportConfig(filename_pattern='{{ roll }}_Frame{{ "%03d" % frame }}_{{ original_name }}')
    result = render_export_filename("ignored_name.tif", conf, metadata=meta)
    assert result == "Summer24_Frame007_ignored_name"


def test_roll_frame_parse_fallback_from_stem():
    conf = ExportConfig(filename_pattern='{{ roll }}_Frame{{ "%03d" % frame }}_{{ film }}')
    result = render_export_filename(
        "/rolls/Roll001_Frame012.tif",
        conf,
        metadata=MetadataConfig(film="Portra"),
    )
    assert result == "Roll001_Frame012_Portra"


def test_metadata_roll_wins_over_stem_parse():
    meta = MetadataConfig(capture_roll="ManualRoll", capture_frame=3)
    conf = ExportConfig(filename_pattern="{{ roll }}_{{ frame }}")
    result = render_export_filename("Roll001_Frame012.tif", conf, metadata=meta)
    assert result == "ManualRoll_3"


def test_path_unsafe_chars_sanitized():
    meta = MetadataConfig(camera_model='Foo/Bar:Baz*?"<>|', film="A\\B")
    conf = ExportConfig(filename_pattern="{{ camera_model }}_{{ film }}_{{ original_name }}")
    result = render_export_filename("shot.jpg", conf, metadata=meta)
    assert result == "FooBarBaz_AB_shot"
    assert "/" not in result
    assert ":" not in result
    assert "*" not in result


def test_half_frame_with_metadata():
    meta = MetadataConfig(film="HP5", capture_roll="R1", capture_frame=1)
    conf = ExportConfig(filename_pattern="{{ roll }}_{{ film }}_{{ original_name }}")
    result = render_export_filename("/x/IMG420.tif", conf, half=2, metadata=meta)
    assert result == "R1_HP5_IMG420_2"


def test_parse_capture_stem():
    assert parse_capture_stem("Roll001_Frame012") == ("Roll001", 12)
    assert parse_capture_stem("roll001_frame7") == ("roll001", 7)
    assert parse_capture_stem("DSC0123") == ("", None)


def test_pad_filter_with_missing_frame_keeps_other_vars():
    """Missing frame must not abort the whole pattern when using |pad."""
    meta = MetadataConfig(capture_roll="Summer24", film="Portra", film_iso=400)
    conf = ExportConfig(filename_pattern="{{ roll }}_Frame{{ frame|pad(3) }}_{{ film }}_{{ film_iso }}_{{ original_name }}")
    result = render_export_filename("HighDef2 (3).tif", conf, metadata=meta)
    assert result == "Summer24_Frame_Portra_400_HighDef2 (3)"


def test_frame_padded_var():
    meta = MetadataConfig(capture_frame=12)
    conf = ExportConfig(filename_pattern="{{ frame_padded }}_{{ original_name }}")
    assert render_export_filename("shot.tif", conf, metadata=meta) == "012_shot"


def test_percent_format_missing_frame_falls_back_to_original_name():
    """Legacy '%03d' % frame with unset frame still falls back (documented limitation)."""
    conf = ExportConfig(filename_pattern='{{ roll }}_Frame{{ "%03d" % frame }}_{{ original_name }}')
    result = render_export_filename("HighDef2 (3).tif", conf, metadata=MetadataConfig(capture_roll="R1"))
    assert result == "HighDef2 (3)"


def test_percent_format_with_frame_set():
    meta = MetadataConfig(capture_roll="R1", capture_frame=12)
    conf = ExportConfig(filename_pattern='{{ roll }}_Frame{{ "%03d" % frame }}')
    assert render_export_filename("shot.tif", conf, metadata=meta) == "R1_Frame012"


def test_capture_date_vars():
    meta = MetadataConfig(capture_date="1998-07")
    conf = ExportConfig(filename_pattern="{{ capture_year }}_{{ capture_date }}_{{ original_name }}")
    assert render_export_filename("shot.tif", conf, metadata=meta) == "1998_19980701_shot"


def test_capture_date_vars_empty_when_unset():
    conf = ExportConfig(filename_pattern="{{ capture_year }}_{{ original_name }}")
    assert render_export_filename("shot.tif", conf, metadata=MetadataConfig()) == "shot"
