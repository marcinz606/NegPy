import os
import re
from datetime import datetime
from typing import Optional, Union

from jinja2.sandbox import SandboxedEnvironment

from negpy.domain.models import ExportConfig, ExportPreset, ExportResolutionMode
from negpy.features.metadata.models import MetadataConfig
from negpy.kernel.system.logging import get_logger

logger = get_logger("export.templating")

_CAPTURE_STEM_RE = re.compile(r"^(?P<roll>.+)_Frame(?P<frame>\d+)", re.IGNORECASE)
_PATH_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def parse_capture_stem(stem: str) -> tuple[str, Optional[int]]:
    """Parse Scanlight `{roll}_Frame{NNN}` from a basename stem. Returns ("", None) if no match."""
    m = _CAPTURE_STEM_RE.match(stem)
    if not m:
        return "", None
    return m.group("roll"), int(m.group("frame"))


def _path_safe(value: object) -> str:
    """Make a template value safe for use in a filename (empty if unset)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value == int(value):
            text = str(int(value))
        else:
            text = str(value)
    else:
        text = str(value).strip()
    if not text:
        return ""
    return _PATH_UNSAFE_RE.sub("", text).strip(" .")


def _film_format(metadata: MetadataConfig) -> str:
    if metadata.format == "Other":
        return metadata.format_other.strip()
    return metadata.format.strip()


def pad_filter(value: object, width: int = 3) -> str:
    """Zero-pad an int for filenames; empty when unset (safe for missing frame)."""
    if value is None or value == "":
        return ""
    try:
        return f"{int(value):0{int(width)}d}"
    except (TypeError, ValueError):
        return ""


def _frame_padded(frame: Optional[int]) -> str:
    return pad_filter(frame, 3)


def _metadata_context(original_stem: str, metadata: Optional[MetadataConfig]) -> dict:
    meta = metadata or MetadataConfig()
    parsed_roll, parsed_frame = parse_capture_stem(original_stem)

    roll = meta.capture_roll.strip() or parsed_roll or ""
    frame: Optional[int] = None
    if meta.capture_frame is not None:
        frame = meta.capture_frame
    elif parsed_frame is not None:
        frame = parsed_frame

    camera = f"{meta.camera_make} {meta.camera_model}".strip()
    lens = meta.lens_model.strip() or meta.lens_make.strip()

    return {
        "roll": _path_safe(roll),
        "frame": frame,
        "frame_padded": _frame_padded(frame),
        "camera": _path_safe(camera),
        "camera_make": _path_safe(meta.camera_make),
        "camera_model": _path_safe(meta.camera_model),
        "lens": _path_safe(lens),
        "lens_make": _path_safe(meta.lens_make),
        "lens_model": _path_safe(meta.lens_model),
        "focal_length": _path_safe(meta.focal_length_mm),
        "film": _path_safe(meta.film),
        "film_iso": _path_safe(meta.film_iso),
        "film_manufacturer": _path_safe(meta.film_manufacturer),
        "film_color_type": _path_safe(meta.film_color_type),
        "film_format": _path_safe(_film_format(meta)),
        "developer": _path_safe(meta.developer),
        "push_pull": meta.push_pull,
        "scanning": _path_safe(meta.scanning),
        "exposure": _path_safe(meta.exposure_override),
    }


def _jinja_env() -> SandboxedEnvironment:
    env = SandboxedEnvironment()
    env.filters["pad"] = pad_filter
    return env


def render_export_filename(
    original_path: str,
    export_settings: Union[ExportConfig, ExportPreset],
    border_size: float = 0.0,
    half: int = 0,
    metadata: Optional[MetadataConfig] = None,
    composite: str = "",
) -> str:
    """
    Renders the export filename using Jinja2 templates.
    Supported variables:
    - original_name: Original filename without extension
    - colorspace: Target color space
    - format: JPEG/TIFF (export file format)
    - paper_ratio: e.g. 3:2
    - size: Export size in cm (PRINT mode only, else empty)
    - dpi: Export DPI (PRINT mode only, else empty)
    - target_px: Target long edge in pixels (TARGET_PX mode only, else empty)
    - border: "border" if border size > 0, else empty
    - date: Current date in YYYYMMDD format
    - roll / frame / frame_padded: Scanlight capture roll and frame (metadata or stem parse)
    - camera / camera_make / camera_model, lens / lens_make / lens_model / focal_length
    - film / film_iso / film_manufacturer / film_color_type / film_format
    - developer / push_pull / scanning / exposure

    ``composite`` suffixes the original name (e.g. "HDR"), so a merged frame does not
    write over the export of the source file it is named after.
    """
    original_name = os.path.splitext(os.path.basename(original_path))[0]
    if half:
        # halves share the source file — suffix so outputs don't collide
        original_name = f"{original_name}_{half}"
    if composite:
        # A composite is built from source files that can also be exported on their own,
        # so it needs its own name for the same reason a half does.
        original_name = f"{original_name}-{composite}"

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
    # Stem for parse fallback is the unsuffixed basename (half suffix is export-only).
    stem_for_parse = os.path.splitext(os.path.basename(original_path))[0]
    context.update(_metadata_context(stem_for_parse, metadata))

    env = _jinja_env()

    try:
        template = env.from_string(export_settings.filename_pattern)
        rendered = template.render(**context)

        # Clean up structural separators (spaces/dashes/underscores in the template
        # skeleton). original_name is still a placeholder here, so it's untouched.
        rendered = re.sub(r"[ _-]+", "_", rendered).strip("_")

        # Restore original_name verbatim — dashes, spaces, and underscores intact.
        rendered = rendered.replace(_PLACEHOLDER, original_name)

        if not rendered:
            return original_name

        return rendered
    except Exception as exc:
        logger.warning(
            "Export filename pattern failed (%r); falling back to original_name: %s",
            export_settings.filename_pattern,
            exc,
        )
        return original_name
