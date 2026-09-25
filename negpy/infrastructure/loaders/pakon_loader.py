import os
import struct
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

    HEADER_SIZE = 16

    @classmethod
    def read_header(cls, file_path: str) -> Optional[Tuple[int, int]]:
        """(height, width) from a TLX header of uint32 [16, width, height, 48], if it matches the file size."""
        try:
            file_size = os.path.getsize(file_path)
            with open(file_path, "rb") as f:
                raw = f.read(cls.HEADER_SIZE)
        except OSError:
            return None
        if len(raw) < cls.HEADER_SIZE:
            return None
        header_size, width, height, bits = struct.unpack("<4I", raw)
        if header_size != cls.HEADER_SIZE or bits != 48 or width == 0 or height == 0:
            return None
        if cls.HEADER_SIZE + width * height * 6 != file_size:
            return None
        return height, width

    @classmethod
    def resolve_layout(cls, file_path: str) -> Optional[Tuple[int, int, Optional[bool]]]:
        """(height, width, planar) or None. planar is None when only the size table matched and layout must be guessed."""
        header = cls.read_header(file_path)
        if header is not None:
            return header[0], header[1], True
        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            return None
        spec = next((s for s in cls.PAKON_SPECS if abs(file_size - s["size"]) < 1024), None)
        if spec is None:
            return None
        height, width = spec["res"]
        return height, width, None

    @classmethod
    def can_handle(cls, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ["", ".raw", ".dat", ".bin"]:
            return False
        return cls.resolve_layout(file_path) is not None

    @staticmethod
    def _looks_interleaved(samples: np.ndarray) -> bool:
        sample = np.asarray(samples[: min(len(samples), 6000)], dtype=np.float32)
        adj_diff = np.mean(np.abs(sample[1:] - sample[:-1]))
        step3_diff = np.mean(np.abs(sample[3:] - sample[:-3]))
        return bool(adj_diff > step3_diff * 1.5)

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        try:
            layout = self.resolve_layout(file_path)
            if layout is None:
                raise ValueError("Unknown Pakon dimensions")
            h, w, planar = layout
            expected_pixels = h * w * 3

            # Anchoring pixels to the file end skips the header of a table-matched file too.
            byte_offset = max(0, os.path.getsize(file_path) - expected_pixels * 2)
            with open(file_path, "rb") as f:
                f.seek(byte_offset)
                data = np.fromfile(f, dtype="<u2", count=expected_pixels)

            if len(data) < expected_pixels:
                raise ValueError(f"File too small: expected {expected_pixels} pixels, got {len(data)}")

            if planar is None:
                planar = not self._looks_interleaved(data)
            if planar:
                data = data.reshape((3, h, w)).transpose((1, 2, 0))
            else:
                data = data.reshape((h, w, 3))[..., ::-1]

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
        layout = self.resolve_layout(file_path)
        if layout is None:
            return None
        height, width, planar = layout
        expected_samples = height * width * 3
        byte_offset = max(0, os.path.getsize(file_path) - expected_samples * 2)
        mapped = np.memmap(file_path, dtype="<u2", mode="r", offset=byte_offset, shape=(expected_samples,))
        if planar is None:
            planar = not self._looks_interleaved(mapped)
        stride = max(1, int(np.ceil(max(height, width) / max(1, max_edge))))
        if planar:
            preview = mapped.reshape(3, height, width)[:, ::stride, ::stride].transpose(1, 2, 0)
        else:
            preview = mapped.reshape(height, width, 3)[::stride, ::stride, ::-1]
        array = np.ascontiguousarray(linear_uint16_to_display_uint8(preview))
        del preview
        del mapped
        image = Image.fromarray(array)
        edge = max(1, max_edge)
        image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        return image
