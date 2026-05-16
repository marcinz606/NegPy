from dataclasses import dataclass
from enum import StrEnum


class ScanMode(StrEnum):
    NEGATIVE = "Negative"
    POSITIVE = "Positive"
    TRANSPARENCY = "Transparency"


@dataclass(frozen=True)
class ScanParams:
    dpi: int
    depth: int
    capture_ir: bool
    area: tuple[float, float, float, float] | None = None
