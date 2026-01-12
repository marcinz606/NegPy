import os
import numpy as np
from typing import Any, List, Dict, ContextManager
from src.domain.interfaces import IImageLoader
from src.infrastructure.loaders.tiff_loader import NonStandardFileWrapper
from src.kernel.image.logic import uint16_to_float32


class PakonLoader(IImageLoader):
    """
    Loader for Pakon planar RAWs.
    """

    PAKON_SPECS: List[Dict[str, Any]] = [
        {"size": 36000000, "res": (2000, 3000), "desc": "F135 Plus High Res"},
        {"size": 9000000, "res": (1000, 1500), "desc": "F135 Plus Low Res"},
        {"size": 24000000, "res": (2000, 2000), "desc": "Pakon 2k Square"},
        {"size": 48000000, "res": (2000, 4000), "desc": "Pakon Panoram"},
        {"size": 72000000, "res": (4000, 3000), "desc": "F335 High Res"},
    ]

    @classmethod
    def can_handle(cls, file_path: str) -> bool:
        file_size = os.path.getsize(file_path)
        return any(abs(file_size - s["size"]) < 1024 for s in cls.PAKON_SPECS)

    def load(self, file_path: str) -> ContextManager[Any]:
        file_size = os.path.getsize(file_path)
        spec = next(s for s in self.PAKON_SPECS if abs(file_size - s["size"]) < 1024)
        h, w = spec["res"]
        expected_pixels = h * w * 3

        with open(file_path, "rb") as f:
            data = np.fromfile(f, dtype="<u2", count=expected_pixels)

        data = data.reshape((3, h, w)).transpose((1, 2, 0))
        return NonStandardFileWrapper(uint16_to_float32(np.ascontiguousarray(data)))
