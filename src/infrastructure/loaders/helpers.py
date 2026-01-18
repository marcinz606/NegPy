import rawpy
from typing import Any
from src.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS


def get_best_demosaic_algorithm(raw: Any) -> Any:
    """
    Selects optimal demosaicing algorithm based on sensor type.
    """
    try:
        if raw.raw_type == rawpy.RawType.XTrans:
            return rawpy.DemosaicAlgorithm.XT_3PASS
        return rawpy.DemosaicAlgorithm.AHD
    except AttributeError:
        return None


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
