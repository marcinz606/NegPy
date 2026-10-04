from pathlib import Path

import numpy as np
import tifffile

from negpy.infrastructure.simulated.images import negative


def write_demo(folder: Path) -> Path:
    """The tour's demo negative, an untagged 16-bit TIFF, which the loader reads as linear."""
    path = folder / "demo_negative.tif"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        rgb, _ir = negative(1600, 2400, seed=7)
        tifffile.imwrite(path, np.ascontiguousarray(rgb), photometric="rgb")
    return path
