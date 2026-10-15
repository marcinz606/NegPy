import tracemalloc

import numpy as np
import tifffile

from negpy.infrastructure.loaders.helpers import _read_exif_uncached


def test_tiff_exif_is_read_without_loading_the_pixels(tmp_path) -> None:
    path = str(tmp_path / "big.tif")
    pixels = np.zeros((2000, 5000, 3), dtype=np.uint16)  # 60 MB
    tifffile.imwrite(path, pixels, extratags=[(306, "s", 0, "2024:01:02 03:04:05", True)])

    tracemalloc.start()
    exif = _read_exif_uncached(path)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert exif["0th"][306] == b"2024:01:02 03:04:05"
    assert peak < 8 * 1024 * 1024
