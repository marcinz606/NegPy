"""Camera + light-source capture layer — no Qt, no NegPy file model.

Drives a trichromatic narrowband-RGB film "scanner" built from a Scanlight
light source (USB-CDC serial) and a tethered camera (via libgphoto2).
One frame is captured as three exposures — red, then green, then blue — which
NegPy's existing RGB-Scan mode (`negpy.features.rgbscan`) merges and inverts.
"""

from negpy.infrastructure.capture.base import (
    CAPTURE_ORDER,
    Camera,
    CaptureResult,
    CaptureSettings,
    Channel,
    LightSource,
)

__all__ = [
    "CAPTURE_ORDER",
    "Camera",
    "CaptureResult",
    "CaptureSettings",
    "Channel",
    "LightSource",
]
