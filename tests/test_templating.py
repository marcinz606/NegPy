from datetime import datetime
from src.domain.models import ExportConfig, ExportFormat
from src.services.export.templating import render_export_filename


def test_basic_templating():
    conf = ExportConfig(filename_pattern="test_{{ original_name }}_{{ colorspace }}")
    result = render_export_filename("/path/to/image.orf", conf)
    assert result == "test_image_Adobe_RGB"


def test_date_templating():
    conf = ExportConfig(filename_pattern="{{ date }}_{{ original_name }}")
    today = datetime.now().strftime("%Y%m%d")
    result = render_export_filename("my_scan.tiff", conf)
    assert result == f"{today}_my_scan"


def test_size_and_dpi_normal():
    conf = ExportConfig(
        use_original_res=False,
        export_print_size=30.0,
        export_dpi=300,
        filename_pattern="{{ original_name }}_{{ size }}_{{ dpi }}",
    )
    result = render_export_filename("shot.jpg", conf)
    assert result == "shot_30cm_300dpi"


def test_size_and_dpi_original_res():
    conf = ExportConfig(
        use_original_res=True,
        export_print_size=30.0,
        export_dpi=300,
        filename_pattern="{{ original_name }}_{{ size }}_{{ dpi }}_end",
    )
    # size and dpi should be empty strings, cleanup should collapse underscores
    result = render_export_filename("shot.jpg", conf)
    assert result == "shot_end"


def test_border_logic():
    # With border
    conf_border = ExportConfig(
        export_border_size=1.5, filename_pattern="{{ original_name }}_{{ border }}"
    )
    assert render_export_filename("img.jpg", conf_border) == "img_border"

    # Without border
    conf_no_border = ExportConfig(
        export_border_size=0.0, filename_pattern="{{ original_name }}_{{ border }}"
    )
    assert render_export_filename("img.jpg", conf_no_border) == "img"


def test_cleanup_logic():
    conf = ExportConfig(
        filename_pattern="{{ original_name }} - {{ colorspace }} --- final"
    )
    # Spaces, dashes and multiple underscores should be collapsed to single underscore
    result = render_export_filename("my scan.jpg", conf)
    assert result == "my_scan_Adobe_RGB_final"


def test_format_and_ratio():
    conf = ExportConfig(
        export_fmt=ExportFormat.TIFF,
        paper_aspect_ratio="3:2",
        filename_pattern="{{ original_name }}_{{ format }}_{{ paper_ratio }}",
    )
    result = render_export_filename("img.jpg", conf)
    # Note: 3:2 has a colon, cleanup might replace it if we were strict,
    # but current regex [ _-]+ only targets spaces, underscores and dashes.
    # Let's see what happens.
    assert result == "img_TIFF_3:2"


def test_empty_template_fallback():
    conf = ExportConfig(filename_pattern="")
    result = render_export_filename("img.jpg", conf)
    assert result == "positive_img"


def test_invalid_template_fallback():
    conf = ExportConfig(filename_pattern="{{ invalid_var }}")
    # Jinja2 by default renders undefined as empty string if not configured otherwise
    # Cleanup will handle the empty result
    result = render_export_filename("img.jpg", conf)
    assert result == "positive_img"
