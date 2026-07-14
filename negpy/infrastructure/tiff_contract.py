"""Versioned TIFF contracts shared by scanner writers and image loaders."""

from typing import Any, Mapping

# TIFF reserves 65000-65535 for reusable application-private tags.  A private
# tag keeps this machine-readable source contract separate from tag 270
# (ImageDescription), which NegPy legitimately uses for human captions and film
# metadata.  The exact ASCII payload is deliberately versioned: loaders must
# only bypass a transfer-function decode for a contract they fully understand.
LINEAR_SCANNER_RGB_TAG = 65000
LINEAR_SCANNER_RGB_MARKER = "negpy.scanner-rgb.linear.uint16.v1"
LINEAR_SCANNER_RGB_EXTRATAG = (
    LINEAR_SCANNER_RGB_TAG,
    "s",
    0,
    LINEAR_SCANNER_RGB_MARKER,
    False,
)


def has_linear_scanner_rgb_marker(tags: Mapping[Any, Any]) -> bool:
    """Return whether ``tags`` contains NegPy's exact linear scanner contract."""

    tag = tags.get(LINEAR_SCANNER_RGB_TAG)
    return tag is not None and tag.value == LINEAR_SCANNER_RGB_MARKER
