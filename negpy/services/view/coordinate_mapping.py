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
        distortion_k1: float = 0.0,
        converge_v: float = 0.0,
        converge_h: float = 0.0,
    ) -> np.ndarray:
        """
        Generates UV map for geometric state (output pixel -> raw uv it samples), so it
        carries the same forward transforms as the image, distortion included.
        """
        u_raw, v_raw = np.meshgrid(
            np.linspace(0, 1, rw_orig, dtype=np.float32),
            np.linspace(0, 1, rh_orig, dtype=np.float32),
        )
        uv_grid = np.stack([u_raw, v_raw], axis=-1)

        if rotation != 0:
            # Must match GPUEngine rotation direction (CCW)
            uv_grid = np.rot90(uv_grid, k=rotation)

        if flip_h:
            uv_grid = np.fliplr(uv_grid)

        if flip_v:
            uv_grid = np.flipud(uv_grid)

        # rot90/flips return views; consumers need one contiguous copy
        uv_grid = np.ascontiguousarray(uv_grid)

        if fine_rot != 0.0:
            h_r, w_r = uv_grid.shape[:2]
            m_mat = cv2.getRotationMatrix2D((w_r / 2.0, h_r / 2.0), fine_rot, 1.0)
            uv_grid = cv2.warpAffine(uv_grid, m_mat, (w_r, h_r), flags=cv2.INTER_LINEAR)

        if distortion_k1 != 0.0:
            from negpy.features.geometry.logic import apply_radial_distortion

            uv_grid = np.ascontiguousarray(apply_radial_distortion(uv_grid, distortion_k1))

        if converge_v != 0.0 or converge_h != 0.0:
            from negpy.features.geometry.logic import apply_keystone

            uv_grid = np.ascontiguousarray(apply_keystone(uv_grid, converge_v, converge_h))

        if autocrop and autocrop_params:
            y1, y2, x1, x2 = autocrop_params["roi"]
            # copy so the ROI slice doesn't pin the full-size parent
            uv_grid = np.ascontiguousarray(uv_grid[y1:y2, x1:x2])

        return uv_grid

    @staticmethod
    def _grid_homography(uv_grid: np.ndarray) -> np.ndarray:
        """The grid as a projective map, viewport (0-1) -> raw (0-1).

        Four interior samples, exact for every projective op (rotation, flips, fine
        rotation, crop, keystone); only distortion stays approximate, and this is used
        only off the frame. Samples come from the middle, since a fine rotation fills
        the border with zeros and those are not coordinates.
        """
        h_uv, w_uv = uv_grid.shape[:2]
        x0, x1 = w_uv // 4, w_uv - 1 - w_uv // 4
        y0, y1 = h_uv // 4, h_uv - 1 - h_uv // 4
        src = np.float32(
            [
                [x0 / (w_uv - 1), y0 / (h_uv - 1)],
                [x1 / (w_uv - 1), y0 / (h_uv - 1)],
                [x1 / (w_uv - 1), y1 / (h_uv - 1)],
                [x0 / (w_uv - 1), y1 / (h_uv - 1)],
            ]
        )
        dst = np.float32([uv_grid[y0, x0], uv_grid[y0, x1], uv_grid[y1, x1], uv_grid[y1, x0]])
        return cv2.getPerspectiveTransform(src, dst).astype(np.float64)

    @staticmethod
    def _apply_homography(m: np.ndarray, nx: float, ny: float) -> Tuple[float, float]:
        den = m[2, 0] * nx + m[2, 1] * ny + m[2, 2]
        if abs(den) < 1e-12:
            return nx, ny
        return (
            float((m[0, 0] * nx + m[0, 1] * ny + m[0, 2]) / den),
            float((m[1, 0] * nx + m[1, 1] * ny + m[1, 2]) / den),
        )

    @staticmethod
    def map_click_to_raw(nx: float, ny: float, uv_grid: np.ndarray) -> Tuple[float, float]:
        """
        Viewport (0-1) -> Raw (0-1).

        A point off the frame has no grid sample, so the projective model of the grid
        gives it. Dodge/burn masks use this, because a card edge must start outside the
        picture to cover a corner when you tilt it.
        """
        h_uv, w_uv = uv_grid.shape[:2]
        if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
            raw_uv = uv_grid[int(ny * (h_uv - 1)), int(nx * (w_uv - 1))]
            return float(raw_uv[0]), float(raw_uv[1])
        return CoordinateMapping._apply_homography(CoordinateMapping._grid_homography(uv_grid), nx, ny)

    @staticmethod
    def map_raw_to_viewport(rx: float, ry: float, uv_grid: np.ndarray, buckets: int = 100) -> Tuple[float, float]:
        """
        Raw (0-1) -> Viewport (0-1): the inverse of map_click_to_raw.

        Two-stage nearest-neighbour: a coarse pass over a `buckets`-decimated grid
        locates the neighbourhood cheaply, then a full-resolution pass over that
        bucket's window pins the exact pixel. The coarse pass alone snapped results
        to bucket centres (± step/2 grid pixels ≈ 3-20px depending on preview size,
        magnified by zoom) — enough to draw a heal outline entirely off the healed
        spot even though the heal itself landed exactly where clicked.
        """
        h_uv, w_uv = uv_grid.shape[:2]
        step = max(1, h_uv // buckets)
        small = uv_grid[::step, ::step]
        dist = (small[..., 0] - rx) ** 2 + (small[..., 1] - ry) ** 2
        idx = int(np.argmin(dist))
        vy, vx = divmod(idx, small.shape[1])

        # Refine: exact search across the coarse cell and its neighbours.
        py, px = vy * step, vx * step
        y0, y1 = max(0, py - step), min(h_uv, py + step + 1)
        x0, x1 = max(0, px - step), min(w_uv, px + step + 1)
        window = uv_grid[y0:y1, x0:x1]
        wdist = (window[..., 0] - rx) ** 2 + (window[..., 1] - ry) ** 2
        widx = int(np.argmin(wdist))
        wy, wx = divmod(widx, window.shape[1])

        nx, ny = min((x0 + wx + 0.5) / w_uv, 1.0), min((y0 + wy + 0.5) / h_uv, 1.0)

        # The nearest sample is more than one grid step away only if the raw point is off the
        # frame. The projective model then gives the answer, the inverse of what
        # map_click_to_raw does there.
        if float(wdist.flat[widx]) > (2.0 / max(h_uv, w_uv)) ** 2:
            inv = np.linalg.inv(CoordinateMapping._grid_homography(uv_grid))
            return CoordinateMapping._apply_homography(inv, rx, ry)
        return nx, ny
