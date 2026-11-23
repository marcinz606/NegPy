import negpy_core
import numpy as np
from typing import Tuple, Optional

from negpy.kernel.image.logic import apply_geometry_orientation


def _invert_affine(m: np.ndarray) -> np.ndarray:
    """cv2.invertAffineTransform in its operation order."""
    d = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    d = 1.0 / d if d != 0 else 0.0
    a11, a22, a12, a21 = m[1, 1] * d, m[0, 0] * d, -m[0, 1] * d, -m[1, 0] * d
    return np.array([[a11, a12, -a11 * m[0, 2] - a12 * m[1, 2]], [a21, a22, -a21 * m[0, 2] - a22 * m[1, 2]]])


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

        uv_grid = apply_geometry_orientation(uv_grid, rotation, flip_h, flip_v)

        if fine_rot != 0.0:
            # The grid is affine in pixel position, so the rotated grid is solved exactly. A
            # warpAffine fills the corner wedges with 0, which sends a click there to (0, 0).
            h_r, w_r = uv_grid.shape[:2]
            m_inv = _invert_affine(negpy_core.rotation_matrix((w_r / 2.0, h_r / 2.0), fine_rot, 1.0))
            o = uv_grid[0, 0].astype(np.float64)
            du = (uv_grid[0, -1] - o) / max(w_r - 1, 1)
            dv = (uv_grid[-1, 0] - o) / max(h_r - 1, 1)
            uv_grid = negpy_core.affine_uv_grid(h_r, w_r, m_inv, tuple(o), tuple(du), tuple(dv))

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
        the border with zeros and those are not coordinates. Each grid entry (y, x) is
        the sample at pixel center ((x + 0.5) / w_uv, (y + 0.5) / h_uv).
        """
        h_uv, w_uv = uv_grid.shape[:2]
        x0, x1 = w_uv // 4, w_uv - 1 - w_uv // 4
        y0, y1 = h_uv // 4, h_uv - 1 - h_uv // 4
        src = np.float32(
            [
                [(x0 + 0.5) / w_uv, (y0 + 0.5) / h_uv],
                [(x1 + 0.5) / w_uv, (y0 + 0.5) / h_uv],
                [(x1 + 0.5) / w_uv, (y1 + 0.5) / h_uv],
                [(x0 + 0.5) / w_uv, (y1 + 0.5) / h_uv],
            ]
        )
        dst = np.float32([uv_grid[y0, x0], uv_grid[y0, x1], uv_grid[y1, x1], uv_grid[y1, x0]])
        return negpy_core.perspective_transform(src, dst).astype(np.float64)

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
    def _sample_bilinear(uv_grid: np.ndarray, gx: float, gy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate piecewise-bilinear `uv_grid` and its `(df/dgx, df/dgy)` Jacobian at `(gx, gy)`.

        Allows `gx` in `[-0.5, w_uv - 0.5]` and `gy` in `[-0.5, h_uv - 0.5]` by linear
        extrapolation on the outer half-pixel rim so viewport `[0, 1]` is continuous right to
        the frame edge.
        """
        h_uv, w_uv = uv_grid.shape[:2]
        cx = min(max(int(np.floor(gx)), 0), w_uv - 2)
        cy = min(max(int(np.floor(gy)), 0), h_uv - 2)
        tx = gx - cx
        ty = gy - cy
        c00 = uv_grid[cy, cx].astype(np.float64)
        c10 = uv_grid[cy, cx + 1].astype(np.float64)
        c01 = uv_grid[cy + 1, cx].astype(np.float64)
        c11 = uv_grid[cy + 1, cx + 1].astype(np.float64)
        val = (1.0 - ty) * ((1.0 - tx) * c00 + tx * c10) + ty * ((1.0 - tx) * c01 + tx * c11)
        df_dx = (1.0 - ty) * (c10 - c00) + ty * (c11 - c01)
        df_dy = (1.0 - tx) * (c01 - c00) + tx * (c11 - c10)
        return val, df_dx, df_dy

    @staticmethod
    def map_click_to_raw(nx: float, ny: float, uv_grid: np.ndarray) -> Tuple[float, float]:
        """
        Viewport (0-1) -> Raw (0-1), sub-pixel interpolated around pixel centers.

        Each grid cell `(y, x)` sits at viewport center `((x + 0.5) / w_uv, (y + 0.5) / h_uv)`.
        Bilinear interpolation eliminates the 0.5-1.5 cell top-left floor snap that integer
        grid indexing caused when zoomed in closely. Points off the frame use the projective
        model of the grid so dodge/burn handles outside the picture continue smoothly.
        """
        h_uv, w_uv = uv_grid.shape[:2]
        if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
            if h_uv < 2 or w_uv < 2:
                raw_uv = uv_grid[0, 0]
                return float(raw_uv[0]), float(raw_uv[1])
            gx = nx * w_uv - 0.5
            gy = ny * h_uv - 0.5
            val, _, _ = CoordinateMapping._sample_bilinear(uv_grid, gx, gy)
            return float(val[0]), float(val[1])
        return CoordinateMapping._apply_homography(CoordinateMapping._grid_homography(uv_grid), nx, ny)

    @staticmethod
    def map_raw_to_viewport(rx: float, ry: float, uv_grid: np.ndarray, buckets: int = 100) -> Tuple[float, float]:
        """
        Raw (0-1) -> Viewport (0-1): continuous sub-pixel inverse of `map_click_to_raw`.

        Locates the nearest grid sample via a two-stage coarse-to-fine search, then refines
        continuous `(gx, gy)` with Newton steps on the bilinear cell so zoomed-in retouch
        overlays stay locked onto the exact sub-pixel cursor position.
        """
        h_uv, w_uv = uv_grid.shape[:2]
        if h_uv < 2 or w_uv < 2:
            return 0.5, 0.5
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

        # If the nearest grid sample is more than ~2 grid steps away, the raw point is off-frame.
        if float(wdist.flat[widx]) > (2.0 / max(h_uv, w_uv)) ** 2:
            inv = np.linalg.inv(CoordinateMapping._grid_homography(uv_grid))
            return CoordinateMapping._apply_homography(inv, rx, ry)

        gx = float(x0 + wx)
        gy = float(y0 + wy)
        for _ in range(3):
            val, df_dx, df_dy = CoordinateMapping._sample_bilinear(uv_grid, gx, gy)
            ru = rx - val[0]
            rv = ry - val[1]
            det = df_dx[0] * df_dy[1] - df_dx[1] * df_dy[0]
            if abs(det) < 1e-18:
                break
            dgx = (ru * df_dy[1] - rv * df_dy[0]) / det
            dgy = (df_dx[0] * rv - df_dx[1] * ru) / det
            gx = float(np.clip(gx + dgx, -0.5, w_uv - 0.5))
            gy = float(np.clip(gy + dgy, -0.5, h_uv - 0.5))

        nx = float(np.clip((gx + 0.5) / w_uv, 0.0, 1.0))
        ny = float(np.clip((gy + 0.5) / h_uv, 0.0, 1.0))
        return nx, ny
