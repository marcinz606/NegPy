from typing import TypeAlias, Tuple
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass


# Float32 (0.0-1.0) [Height, Width, Channels]
ImageBuffer: TypeAlias = npt.NDArray[np.float32]

# Geometry
# (y1, y2, x1, x2)
ROI: TypeAlias = Tuple[int, int, int, int]
# (Height, Width)
Dimensions: TypeAlias = Tuple[int, int]

# Rec. 709 Luma
LUMA_COEFFS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
LUMA_R = 0.2126
LUMA_G = 0.7152
LUMA_B = 0.0722

# Histogram types
HistogramData: TypeAlias = Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


@dataclass
class AppConfig:
    thumbnail_size: int
    max_workers: int
    preview_render_size: int
    edits_db_path: str
    settings_db_path: str
    presets_dir: str
    cache_dir: str
    user_icc_dir: str
    default_export_dir: str
    adobe_rgb_profile: str
    use_gpu: bool = True
