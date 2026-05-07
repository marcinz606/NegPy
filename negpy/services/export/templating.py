import os
import re
from datetime import datetime
from jinja2 import Template
from negpy.domain.models import ExportConfig, ExportResolutionMode


def render_export_filename(
    original_path: str,
    export_settings: ExportConfig,
    border_size: float = 0.0,
) -> str:
    """
    Renders the export filename using Jinja2 templates.
    Supported variables:
    - original_name: Original filename without extension
    - colorspace: Target color space
    - format: JPEG/TIFF
    - paper_ratio: e.g. 3:2
    - size: Export size in cm (PRINT mode only, else empty)
    - dpi: Export DPI (PRINT mode only, else empty)
    - target_px: Target long edge in pixels (TARGET_PX mode only, else empty)
    - border: "border" if border size > 0, else empty
    - date: Current date in YYYYMMDD format
    """
    original_name = os.path.splitext(os.path.basename(original_path))[0]

    # Null-byte placeholder protects original_name from the cleanup regex.
    # Null bytes cannot appear in filesystem paths, so collision is impossible.
    _PLACEHOLDER = "\x00ORIG\x00"

    mode = export_settings.export_resolution_mode
    is_print = mode == ExportResolutionMode.PRINT
    is_target_px = mode == ExportResolutionMode.TARGET_PX

    context = {
        "original_name": _PLACEHOLDER,
        "colorspace": export_settings.export_color_space,
        "format": export_settings.export_fmt,
        "paper_ratio": export_settings.paper_aspect_ratio,
        "size": f"{export_settings.export_print_size:.0f}cm" if is_print else "",
        "dpi": f"{export_settings.export_dpi}dpi" if is_print else "",
        "target_px": f"{export_settings.export_target_long_edge_px}px" if is_target_px else "",
        "border": "border" if border_size > 0 else "",
        "date": datetime.now().strftime("%Y%m%d"),
    }

    try:
        template = Template(export_settings.filename_pattern)
        rendered = template.render(**context)

        # Clean up structural separators (spaces/dashes/underscores in the template
        # skeleton). original_name is still a placeholder here, so it's untouched.
        rendered = re.sub(r"[ _-]+", "_", rendered).strip("_")

        # Restore original_name verbatim — dashes, spaces, and underscores intact.
        rendered = rendered.replace(_PLACEHOLDER, original_name)

        if not rendered:
            return original_name

        return rendered
    except Exception:
        return original_name
