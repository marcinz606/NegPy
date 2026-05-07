from datetime import datetime
from negpy.domain.models import ExportConfig, ExportFormat, ExportResolutionMode
from negpy.services.export.templating import render_export_filename


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
