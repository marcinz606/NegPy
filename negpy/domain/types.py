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
    max_history_steps: int
    edits_db_path: str
    settings_db_path: str
    presets_dir: str
    cache_dir: str
    user_icc_dir: str
    default_export_dir: str
    adobe_rgb_profile: str
    use_gpu: bool = True
    override_toml_path: str = ""
    max_texture_size: int | None = None
    force_hq_preview: bool | None = None
    # Preview buffer LRU (decoded float preview before render pipeline)
    preview_cache_max_entries: int = 8
    preview_cache_max_bytes: int = 1_200_000_000
    # Canvas zoom (1.0 = 100%)
    canvas_zoom_min: float = 0.25
    canvas_zoom_max: float = 8.0
