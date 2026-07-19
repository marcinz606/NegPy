"""Roll-scanning orchestration -- drives a coolscanpy Device + Roll extension
through preview, spacing correction, approval and batch fine-scanning.

No Qt dependencies (mirrors `negpy.services.capture` and
`negpy.services.scanning`).
"""

from negpy.services.roll.service import RollFrameOutput, RollScanningError, RollScanningService, available

__all__ = [
    "RollFrameOutput",
    "RollScanningError",
    "RollScanningService",
    "available",
]
