from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScanResult:
    rgb: np.ndarray
    ir: np.ndarray | None
    dpi: int
    device_model: str
