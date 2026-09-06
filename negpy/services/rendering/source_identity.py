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
from negpy.features.process.logic import demosaic_token, effective_linear_raw
from negpy.features.rgbscan.logic import rgbscan_token
from negpy.features.stitch.models import stitch_token
from negpy.services.rendering.lens import lens_decode_token, metadata_lens_enabled


def source_token(config: WorkspaceConfig) -> str:
    """What the decode reads, folded into one string."""
    parts = [
        # Decides use_camera_wb, so it changes the decoded numbers themselves.
        f"|lr{int(effective_linear_raw(config.process, config.exposure.render_intent))}",
        # Preview only: folding the export choice in would re-decode the open frame whenever
        # an export setting moved.
        demosaic_token(config.process.demosaic_preview),
        rgbscan_token(config.rgbscan),
        stitch_token(config.stitch),
        hdr_token(config.hdr),
        lens_decode_token(metadata_lens_enabled(config), config.flatfield),
    ]
    if config.stitch.stitch_enabled:
        # Stitch flat-fields each part before assembly; embedded lens mode carries this
        # dependency in lens_decode_token. Other frames flat-field at render time.
        parts.append(flatfield_token(config.flatfield))
    return "".join(parts)
