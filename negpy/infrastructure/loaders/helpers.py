from typing import Any

import numpy as np
import rawpy

from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS


class NonStandardFileWrapper:
    """
    numpy -> rawpy-like interface.
    """

    def __init__(self, data: np.ndarray):
        self.data = data

    def __enter__(self) -> "NonStandardFileWrapper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def postprocess(self, **kwargs: Any) -> np.ndarray:
        bps = kwargs.get("output_bps", 8)
        half_size = kwargs.get("half_size", False)
        data = self.data
        if half_size:
            data = data[::2, ::2]

        if bps == 16:
            return (data * 65535.0).astype(np.uint16)
        return (data * 255.0).astype(np.uint8)


def get_best_demosaic_algorithm(raw: Any) -> Any:
    """
    Selects optimal demosaicing algorithm based on sensor type.
    """
    try:
        if raw.raw_type == rawpy.RawType.XTrans:
            return rawpy.DemosaicAlgorithm.XT_3PASS

        elif raw.raw_type == rawpy.RawType.Bayer:
            if hasattr(rawpy.DemosaicAlgorithm, "AMAZE") and rawpy.DemosaicAlgorithm.AMAZE.isSupported:
                return rawpy.DemosaicAlgorithm.AMAZE
            return rawpy.DemosaicAlgorithm.DHT

        else:
            return rawpy.DemosaicAlgorithm.LINEAR

    except Exception:
        return rawpy.DemosaicAlgorithm.LINEAR


def get_supported_raw_wildcards() -> str:
    """
    Returns raw formats as string for file dialogs.
    """
    wildcards = []
    for ext in sorted(SUPPORTED_RAW_EXTENSIONS):
        base = ext.lstrip(".")
        wildcards.append(f"*.{base}")
        wildcards.append(f"*.{base.upper()}")

    return " ".join(wildcards)
