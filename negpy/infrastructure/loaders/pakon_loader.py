import os
import numpy as np
from typing import Any, List, Dict, ContextManager, Tuple
from negpy.domain.interfaces import IImageLoader
from negpy.infrastructure.loaders.tiff_loader import NonStandardFileWrapper
from negpy.kernel.image.logic import uint16_to_float32_seq


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
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ["", ".raw", ".dat", ".bin"]:
            return False

        try:
            file_size = os.path.getsize(file_path)
            return any(abs(file_size - s["size"]) < 1024 for s in cls.PAKON_SPECS)
        except OSError:
            return False

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        try:
            file_size = os.path.getsize(file_path)
            spec = next(s for s in self.PAKON_SPECS if abs(file_size - s["size"]) < 1024)
            h, w = spec["res"]
            expected_pixels = h * w * 3

            with open(file_path, "rb") as f:
                data = np.fromfile(f, dtype="<u2", count=expected_pixels)

            if len(data) < expected_pixels:
                raise ValueError(f"File too small: expected {expected_pixels} pixels, got {len(data)}")

            # Heuristic: Detect Planar vs Interleaved layout
            sample_size = min(len(data), 6000)
            sample = data[:sample_size].astype(np.float32)

            adj_diff = np.mean(np.abs(sample[1:] - sample[:-1]))
            step3_diff = np.mean(np.abs(sample[3:] - sample[:-3]))

            if adj_diff > step3_diff * 1.5:
                data = data.reshape((h, w, 3))[..., ::-1]
            else:
                data = data.reshape((3, h, w)).transpose((1, 2, 0))

            metadata = {"orientation": 0}
            return NonStandardFileWrapper(uint16_to_float32_seq(np.ascontiguousarray(data))), metadata
        except Exception as e:
            # Fallback to Rawpy or re-raise to be caught by worker
            raise RuntimeError(f"Pakon Load Failure: {e}") from e
