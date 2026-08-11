"""Identity of an assembled source buffer.

Two configs with the same `source_token` decode to the same pixels, before any pipeline
stage runs. That makes it the one thing every source cache is keyed on, and the one test
for whether a config change needs a re-decode or only a re-render.

Source assembly happens in more than one place (the full-res decode, the export path, the
preview service), and each used to build its own cache key by hand. A key that forgot a
field served the previous buffer for a setting the user had just changed.
"""

from negpy.domain.models import WorkspaceConfig
from negpy.features.flatfield.logic import flatfield_token
from negpy.features.hdr.models import hdr_token
from negpy.features.rgbscan.logic import rgbscan_token
from negpy.features.stitch.models import stitch_token


def source_token(config: WorkspaceConfig) -> str:
    """What the decode reads, folded into one string."""
    parts = [
        # Decides use_camera_wb, so it changes the decoded numbers themselves.
        f"|lr{int(config.process.linear_raw)}",
        rgbscan_token(config.rgbscan),
        stitch_token(config.stitch),
        hdr_token(config.hdr),
    ]
    if config.stitch.stitch_enabled:
        # Stitch is the only assembly that flat-fields during the decode (per part, before
        # the warp — a canvas-wide gain map would stretch across the seam). Everywhere else
        # flat-field is a render stage, and including it here would force a needless
        # re-decode every time the profile is switched.
        parts.append(flatfield_token(config.flatfield))
    return "".join(parts)
