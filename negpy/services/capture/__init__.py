"""Capture orchestration — drives a Scanlight + camera to shoot an R/G/B triplet.

No Qt dependencies (mirrors `negpy.services.scanning`).
"""

from negpy.services.capture.service import CaptureError, CaptureService

__all__ = ["CaptureError", "CaptureService"]
