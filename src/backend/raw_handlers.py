import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from src.logging_config import get_logger

logger = get_logger(__name__)


class PlanarRawWrapper:
    """
    A compatibility wrapper that mimics the rawpy.RawPy interface
    for headerless or custom planar RAW files.
    """

    def __init__(self, data: np.ndarray):
        """
        Expects data as (H, W, 3) float32 in [0, 1] range.
        """
        self.data = data

    def __enter__(self) -> "PlanarRawWrapper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def postprocess(self, **kwargs: Any) -> np.ndarray:
        """
        Mimics rawpy's postprocess. Returns uint16 array to match
        the pipeline's expectation for high-bit depth linear data.
        """
        res = self.data * 65535.0
        return res.astype(np.uint16)


# --- FORMAT HANDLERS ---


def load_pakon_raw(file_path: str) -> Optional[PlanarRawWrapper]:
    """


    Handler for Pakon F135/F235/F335 planar RAW files.


    Identifies via file size with fuzzy matching.


    """

    pakon_specs: List[Dict[str, Any]] = [
        {"size": 36000000, "res": (2000, 3000), "desc": "F135 Plus High Res"},
        {"size": 9000000, "res": (1000, 1500), "desc": "F135 Plus Low Res"},
        {"size": 24000000, "res": (2000, 2000), "desc": "Pakon 2k Square"},
        {"size": 48000000, "res": (2000, 4000), "desc": "Pakon Panoram"},
        {"size": 72000000, "res": (4000, 3000), "desc": "F335 High Res"},
    ]

    file_size = os.path.getsize(file_path)

    spec: Optional[Dict[str, Any]] = None

    for s in pakon_specs:
        size_val: int = s["size"]

        if abs(file_size - size_val) < 1024:
            spec = s

            break

    if not spec:
        return None

    h, w = cast(Tuple[int, int], spec["res"])

    logger.info(f"Specialty Loader: Detected {spec['desc']} for {file_path}")

    try:
        expected_pixels = h * w * 3
        with open(file_path, "rb") as f:
            data = np.fromfile(f, dtype="<u2", count=expected_pixels)

        if data.size != expected_pixels:
            return None

        # Planar RGB -> Interleaved RGB
        data = data.reshape((3, h, w)).transpose((1, 2, 0))
        data_f32 = data.astype(np.float32) / 65535.0

        return PlanarRawWrapper(data_f32)
    except Exception as e:
        logger.error(f"Pakon loader failed: {e}")
        return None


# --- REGISTRY ---

# List of specialty loading functions.
# To support a new "weird" format, simply add its handler function here.
SPECIAL_LOADERS = [
    load_pakon_raw,
]


def load_special_raw(file_path: str) -> Optional[Any]:
    """
    Entry point for non-standard RAW loading.
    Attempts all registered loaders until one succeeds.
    """
    for loader in SPECIAL_LOADERS:
        try:
            result = loader(file_path)
            if result:
                return result
        except Exception:
            continue
    return None
