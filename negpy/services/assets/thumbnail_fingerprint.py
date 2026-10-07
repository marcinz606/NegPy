"""Thumbnail fingerprints, stored in the JPEG comment so the record lives and dies with the image.

Anything that cannot be fingerprinted reads as stale: a false stale costs one render, a false current leaves a wrong thumbnail up."""

import functools
import hashlib
import json
import os
from dataclasses import asdict, fields
from typing import Any, Optional

from negpy.domain.models import WorkspaceConfig

# Bump in any change that alters rendered pixels for unchanged settings, a changed default included.
THUMBNAIL_RENDER_VERSION = 2

# Marks a thumbnail made from the source preview, which runs none of the frame's settings.
QUICK = "quick"

_COMMENT_PREFIX = "negpy-thumb:"

# Sections that never reach the pixels; every other section counts as shaping the thumbnail.
_NON_PIXEL_SECTIONS = frozenset({"metadata", "export"})

# Fields left out of a hashed section: export-only, labels, and detail too fine for a thumbnail.
_UNHASHED_FIELDS: dict[str, frozenset[str]] = {
    "process": frozenset({"demosaic_export", "roll_name", "baseline_source", "demosaic_preview", "crosstalk_profile"}),
    "lab": frozenset({"sharpen", "sharpen_method", "sharpen_radius", "sharpen_masking", "chroma_denoise"}),
}

_BELOW_THUMBNAIL_SECTIONS = frozenset({"retouch"})


def _file_identity(path: Optional[str]) -> Optional[str]:
    """Path plus size and mtime: a file replaced under the same name changes the render."""
    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return f"{path}|missing"
    return f"{path}|{stat.st_size}|{stat.st_mtime_ns}"


def _serialize(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


@functools.cache
def _section_defaults(section_type: type) -> dict[str, str]:
    return {name: _serialize(value) for name, value in asdict(section_type()).items()}


def _changed_fields(section: Any, skip: frozenset[str]) -> dict[str, Any]:
    """The section's fields that differ from their defaults. A field at its default is left
    out, so a new defaulted field leaves every fingerprint as it was."""
    defaults = _section_defaults(type(section))
    return {k: v for k, v in asdict(section).items() if k not in skip and _serialize(v) != defaults.get(k)}


def _companion_paths(config: WorkspaceConfig) -> list[str]:
    """Files a composite reads besides the frame's own, which its hash already covers."""
    triplets = [path for pair in config.stitch.stitch_triplets for path in pair]
    paths = [*config.hdr.hdr_paths, config.rgbscan.green_path, config.rgbscan.blue_path, *config.stitch.stitch_paths, *triplets]
    return [p for p in paths if p]


def thumbnail_fingerprint(
    config: WorkspaceConfig,
    *,
    workspace_color_space: str,
    input_icc_path: Optional[str],
) -> str:
    """``config`` must be the resolved config the render ran on.

    The display transform is left out, so a monitor-profile or soft-proof change re-renders nothing.
    """
    sections = {}
    for f in fields(config):
        if f.name in _NON_PIXEL_SECTIONS or f.name in _BELOW_THUMBNAIL_SECTIONS:
            continue
        sections[f.name] = _changed_fields(getattr(config, f.name), _UNHASHED_FIELDS.get(f.name, frozenset()))
    payload = {
        "v": THUMBNAIL_RENDER_VERSION,
        "config": sections,
        "workspace": workspace_color_space,
        "input_icc": _file_identity(input_icc_path),
        "companions": [_file_identity(p) for p in _companion_paths(config)],
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def encode_comment(fingerprint: Optional[str]) -> Optional[bytes]:
    if not fingerprint:
        return None
    return f"{_COMMENT_PREFIX}{fingerprint}".encode("ascii")


def decode_comment(raw: Any) -> Optional[str]:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("ascii")
        except UnicodeDecodeError:
            return None
    if not isinstance(raw, str) or not raw.startswith(_COMMENT_PREFIX):
        return None
    return raw[len(_COMMENT_PREFIX) :] or None


def is_current(stored: Optional[str], current: str) -> bool:
    return stored is not None and stored != QUICK and stored == current
