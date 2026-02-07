import numpy as np
import cv2
from typing import Tuple, Optional


class CoordinateMapping:
    """
    Raw <-> Viewport coordinate transforms.
    """

    @staticmethod
    def create_uv_grid(
        rh_orig: int,
        rw_orig: int,
        rotation: int,
        fine_rot: float,
        flip_h: bool = False,
        flip_v: bool = False,
        autocrop: bool = False,
        autocrop_params: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Generates UV map for geometric state.
        """
        u_raw, v_raw = np.meshgrid(np.linspace(0, 1, rw_orig), np.linspace(0, 1, rh_orig))
        uv_grid = np.stack([u_raw, v_raw], axis=-1).astype(np.float32)

        if rotation != 0:
            uv_grid = np.rot90(uv_grid, k=-rotation).astype(np.float32)

        if flip_h:
            uv_grid = np.fliplr(uv_grid).astype(np.float32)

        if flip_v:
            uv_grid = np.flipud(uv_grid).astype(np.float32)

        if fine_rot != 0.0:
            h_r, w_r = uv_grid.shape[:2]
            m_mat = cv2.getRotationMatrix2D((w_r / 2.0, h_r / 2.0), fine_rot, 1.0)
            uv_grid = cv2.warpAffine(uv_grid, m_mat, (w_r, h_r), flags=cv2.INTER_LINEAR).astype(np.float32)

        if autocrop and autocrop_params:
            y1, y2, x1, x2 = autocrop_params["roi"]
            uv_grid = uv_grid[y1:y2, x1:x2].astype(np.float32)

        return uv_grid

    @staticmethod
    def map_click_to_raw(nx: float, ny: float, uv_grid: np.ndarray) -> Tuple[float, float]:
        """
        Viewport (0-1) -> Raw (0-1).
        """
        h_uv, w_uv = uv_grid.shape[:2]
        px = int(np.clip(nx * (w_uv - 1), 0, w_uv - 1))
        py = int(np.clip(ny * (h_uv - 1), 0, h_uv - 1))
        raw_uv = uv_grid[py, px]
        return float(raw_uv[0]), float(raw_uv[1])
