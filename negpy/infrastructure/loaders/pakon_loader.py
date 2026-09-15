import os
import numpy as np
from collections.abc import Callable
from typing import Any, List, Dict, ContextManager, Optional, Tuple
from PIL import Image
from negpy.domain.interfaces import IImageLoader
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, linear_uint16_to_display_uint8
from negpy.kernel.image.logic import uint16_to_float32


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

            # Skip optional 16-byte header by anchoring pixels to file end.
            byte_offset = max(0, file_size - expected_pixels * 2)
            with open(file_path, "rb") as f:
                f.seek(byte_offset)
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

            metadata = {"orientation": 0, "ir": None}
            return NonStandardFileWrapper(uint16_to_float32(np.ascontiguousarray(data))), metadata
        except Exception as e:
            # Fallback to Rawpy or re-raise to be caught by worker
            raise RuntimeError(f"Pakon Load Failure: {e}") from e

    def load_bounded_preview(
        self,
        file_path: str,
        max_edge: int,
        *,
        fast_only: bool = False,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Optional[Image.Image]:
        if should_cancel is not None and should_cancel():
            raise InterruptedError("preview cancelled")
        file_size = os.path.getsize(file_path)
        spec = next((item for item in self.PAKON_SPECS if abs(file_size - item["size"]) < 1024), None)
        if spec is None:
            return None
        height, width = spec["res"]
        expected_samples = height * width * 3
        byte_offset = max(0, file_size - expected_samples * 2)
        mapped = np.memmap(file_path, dtype="<u2", mode="r", offset=byte_offset, shape=(expected_samples,))
        sample = np.asarray(mapped[: min(expected_samples, 6000)], dtype=np.float32)
        adjacent = np.mean(np.abs(sample[1:] - sample[:-1]))
        step_three = np.mean(np.abs(sample[3:] - sample[:-3]))
        stride = max(1, int(np.ceil(max(height, width) / max(1, max_edge))))
        if adjacent > step_three * 1.5:
            preview = mapped.reshape(height, width, 3)[::stride, ::stride, ::-1]
        else:
            preview = mapped.reshape(3, height, width)[:, ::stride, ::stride].transpose(1, 2, 0)
        array = np.ascontiguousarray(linear_uint16_to_display_uint8(preview))
        del preview
        del mapped
        image = Image.fromarray(array)
        edge = max(1, max_edge)
        image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        return image
