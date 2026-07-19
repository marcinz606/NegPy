"""Scanner device layer — no Qt, no NegPy file model."""

from negpy.infrastructure.scanners.base import (
    ScannerBackend,
    ScannerCapabilities,
    ScannerDevice,
    ScannerSession,
    ScannerUnavailable,
    TransientScanError,
)
from negpy.infrastructure.scanners.params import ScanMode, ScanParams
from negpy.infrastructure.scanners.result import ScanResult

__all__ = [
    "ScanMode",
    "ScannerBackend",
    "ScannerCapabilities",
    "ScannerDevice",
    "ScannerSession",
    "ScannerUnavailable",
    "ScanParams",
    "ScanResult",
    "TransientScanError",
]
