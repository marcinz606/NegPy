from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScanResult:
    rgb: np.ndarray  # (H, W, 3) uint8 or uint16
    ir: np.ndarray | None  # (H, W) uint8/uint16 if capture_ir, else None
    dpi: int
    device_model: str
