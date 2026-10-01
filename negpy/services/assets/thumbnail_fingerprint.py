"""Which rendered settings a cached thumbnail shows, so staleness survives a restart.

A thumbnail JPEG carries its fingerprint in the JPEG comment field. The record then
lives and dies with the image: ``clear_thumbnails`` removes both, and a crash cannot
leave one without the other. A thumbnail is current only when its stored fingerprint
equals the fingerprint of the frame's current resolved settings.

Every mismatch errs toward "stale": a false stale costs one background render, a false
"current" leaves a wrong thumbnail up indefinitely. So anything that cannot be
fingerprinted (legacy files, the quick source-preview thumbnail, a render whose config
is unknown) reads as stale.
"""

import hashlib
import json
import os
from dataclasses import asdict, fields
from typing import Any, Optional

from negpy.domain.models import WorkspaceConfig

# Bump when a pipeline change alters rendered pixels for unchanged settings, so every
# thumbnail rendered by older code reads as stale once.
THUMBNAIL_RENDER_VERSION = 1

# Marker for a thumbnail made from the source preview (``get_thumbnail_worker``), which
# does not run the frame's settings at all.
QUICK = "quick"

_COMMENT_PREFIX = "negpy-thumb:"

# Sections that never reach the pixels: metadata is text, export settings apply only to
# the exported file. Everything else is assumed to shape the thumbnail.
_NON_PIXEL_SECTIONS = frozenset({"metadata", "export"})

# Fields left out of a hashed section. Export-only and label fields never reach a
# thumbnail. The rest change only detail a thumbnail is too small to show, and a roll push
# or paste of them would otherwise cost a full source read per frame.
_UNHASHED_FIELDS: dict[str, frozenset[str]] = {
    "process": frozenset({"demosaic_export", "roll_name", "baseline_source", "demosaic_preview"}),
    "lab": frozenset({"sharpen", "sharpen_method", "sharpen_radius", "sharpen_masking", "chroma_denoise"}),
}

# Sections whose every field is below thumbnail size: dust, scratch and heal repairs.
_BELOW_THUMBNAIL_SECTIONS = frozenset({"retouch"})


def _file_identity(path: Optional[str]) -> Optional[str]:
    """A profile path plus size and mtime: replacing the file under the same name changes
    the render, so it has to change the fingerprint too."""
    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return f"{path}|missing"
    return f"{path}|{stat.st_size}|{stat.st_mtime_ns}"


def _companion_paths(config: WorkspaceConfig) -> list[str]:
    """Files besides the frame's own that a composite reads. The frame's own content is in
    its hash, so the thumbnail key covers it; these are named only by path."""
    triplets = [path for pair in config.stitch.stitch_triplets for path in pair]
    paths = [*config.hdr.hdr_paths, config.rgbscan.green_path, config.rgbscan.blue_path, *config.stitch.stitch_paths, *triplets]
    return [p for p in paths if p]


def thumbnail_fingerprint(
    config: WorkspaceConfig,
    *,
    workspace_color_space: str,
    input_icc_path: Optional[str],
) -> str:
    """Identity of the render a frame's thumbnail shows.

    ``config`` must be the frame's *resolved* config (roll defaults applied), the one the
    render ran on. The display transform is left out: the JPEG has it baked in, but a
    monitor-profile or soft-proof change would otherwise make every thumbnail stale and
    force a full source read per frame for a color shift.
    """
    sections = {}
    for f in fields(config):
        if f.name in _NON_PIXEL_SECTIONS or f.name in _BELOW_THUMBNAIL_SECTIONS:
            continue
        skip = _UNHASHED_FIELDS.get(f.name, frozenset())
        sections[f.name] = {k: v for k, v in asdict(getattr(config, f.name)).items() if k not in skip}
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
    """JPEG comment payload for a fingerprint, or None to write no comment."""
    if not fingerprint:
        return None
    return f"{_COMMENT_PREFIX}{fingerprint}".encode("ascii")


def decode_comment(raw: Any) -> Optional[str]:
    """The fingerprint stored in a JPEG comment, or None when absent or not ours."""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("ascii")
        except UnicodeDecodeError:
            return None
    if not isinstance(raw, str) or not raw.startswith(_COMMENT_PREFIX):
        return None
    return raw[len(_COMMENT_PREFIX) :] or None


def is_current(stored: Optional[str], current: str) -> bool:
    """True only for a real fingerprint that matches. Unknown and quick read as stale."""
    return stored is not None and stored != QUICK and stored == current
