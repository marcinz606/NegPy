from dataclasses import dataclass
from enum import StrEnum


class ScanMode(StrEnum):
    NEGATIVE = "Negative"
    POSITIVE = "Positive"
    TRANSPARENCY = "Transparency"


@dataclass(frozen=True)
class ScanParams:
    dpi: int
    depth: int  # 8 or 16
    capture_ir: bool  # ignored if device cap.ir_channel is False
    area: tuple[float, float, float, float] | None = None  # (tl_x, tl_y, br_x, br_y) in mm; None = device default frame
