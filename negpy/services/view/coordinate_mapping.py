import math
from typing import Optional, Tuple

import negpy_core
import numpy as np

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
    def _sample_bilinear(uv_grid: np.ndarray, gx: float, gy: float) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Evaluate piecewise-bilinear `uv_grid` and its `(df/dgx, df/dgy)` Jacobian at `(gx, gy)`.

        Allows `gx` in `[-0.5, w_uv - 0.5]` and `gy` in `[-0.5, h_uv - 0.5]` by linear
        extrapolation on the outer half-pixel rim so viewport `[0, 1]` stays continuous to
        the frame edge.
        """
        h_uv, w_uv = uv_grid.shape[:2]
        cx = min(max(math.floor(gx), 0), w_uv - 2)
        cy = min(max(math.floor(gy), 0), h_uv - 2)
        tx = gx - cx
        ty = gy - cy
        ((u00, v00), (u10, v10)), ((u01, v01), (u11, v11)) = uv_grid[cy : cy + 2, cx : cx + 2].tolist()
        omx, omy = 1.0 - tx, 1.0 - ty
        u = omy * (omx * u00 + tx * u10) + ty * (omx * u01 + tx * u11)
        v = omy * (omx * v00 + tx * v10) + ty * (omx * v01 + tx * v11)
        du_dx = omy * (u10 - u00) + ty * (u11 - u01)
        dv_dx = omy * (v10 - v00) + ty * (v11 - v01)
        du_dy = omx * (u01 - u00) + tx * (u11 - u10)
        dv_dy = omx * (v01 - v00) + tx * (v11 - v10)
        return (u, v), (du_dx, dv_dx), (du_dy, dv_dy)

    @staticmethod
    def map_click_to_raw(nx: float, ny: float, uv_grid: np.ndarray) -> Tuple[float, float]:
        """
        Viewport (0-1) -> Raw (0-1), sub-pixel interpolated around pixel centers.

        Each grid cell `(y, x)` sits at viewport center `((x + 0.5) / w_uv, (y + 0.5) / h_uv)`.
        Bilinear interpolation removes the top-left floor snap that integer grid indexing
        caused at high zoom. Points off the frame use the projective model of the grid so
        dodge/burn handles outside the picture continue smoothly.
        """
        h_uv, w_uv = uv_grid.shape[:2]
        if h_uv < 2 or w_uv < 2:
            raw_uv = uv_grid[0, 0]
            return float(raw_uv[0]), float(raw_uv[1])
        if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
            gx = nx * w_uv - 0.5
            gy = ny * h_uv - 0.5
            (u, v), _, _ = CoordinateMapping._sample_bilinear(uv_grid, gx, gy)
            return u, v
        try:
            return CoordinateMapping._apply_homography(CoordinateMapping._grid_homography(uv_grid), nx, ny)
        except ValueError:
            gx = min(max(nx * w_uv - 0.5, -0.5), w_uv - 0.5)
            gy = min(max(ny * h_uv - 0.5, -0.5), h_uv - 0.5)
            (u, v), _, _ = CoordinateMapping._sample_bilinear(uv_grid, gx, gy)
            return u, v

    @staticmethod
    def map_raw_to_viewport(rx: float, ry: float, uv_grid: np.ndarray, buckets: int = 100) -> Tuple[float, float]:
        """
        Raw (0-1) -> Viewport (0-1): continuous sub-pixel inverse of `map_click_to_raw`.

        Locates the nearest grid sample via a two-stage coarse-to-fine search, then refines
        continuous `(gx, gy)` with unclamped Newton steps on the bilinear cell. Points whose
        solution lies outside the viewport grid domain project through the inverse homography.
        """
        h_uv, w_uv = uv_grid.shape[:2]
        if h_uv < 2 or w_uv < 2:
            return 0.5, 0.5
        step_y = max(1, h_uv // buckets)
        step_x = max(1, w_uv // buckets)
        small = uv_grid[::step_y, ::step_x]
        dist = (small[..., 0] - rx) ** 2 + (small[..., 1] - ry) ** 2
        idx = int(np.argmin(dist))
        vy, vx = divmod(idx, small.shape[1])

        # Refine: exact search across the coarse cell and its neighbours.
        py, px = vy * step_y, vx * step_x
        y0, y1 = max(0, py - step_y), min(h_uv, py + step_y + 1)
        x0, x1 = max(0, px - step_x), min(w_uv, px + step_x + 1)
        window = uv_grid[y0:y1, x0:x1]
        wdist = (window[..., 0] - rx) ** 2 + (window[..., 1] - ry) ** 2
        widx = int(np.argmin(wdist))
        wy, wx = divmod(widx, window.shape[1])

        gx = float(x0 + wx)
        gy = float(y0 + wy)
        degenerate = False
        for _ in range(4):
            (u, v), (du_dx, dv_dx), (du_dy, dv_dy) = CoordinateMapping._sample_bilinear(uv_grid, gx, gy)
            ru = rx - u
            rv = ry - v
            det = du_dx * dv_dy - dv_dx * du_dy
            if abs(det) < 1e-18:
                degenerate = ru * ru + rv * rv > 1e-12
                break
            gx += (ru * dv_dy - rv * du_dy) / det
            gy += (du_dx * rv - dv_dx * ru) / det
            if gx < -2.0 or gx > w_uv + 1.0 or gy < -2.0 or gy > h_uv + 1.0:
                break

        eps = 1e-3
        if degenerate or not (-0.5 - eps <= gx <= w_uv - 0.5 + eps and -0.5 - eps <= gy <= h_uv - 0.5 + eps):
            try:
                inv = np.linalg.inv(CoordinateMapping._grid_homography(uv_grid))
            except (ValueError, np.linalg.LinAlgError):
                return min(max((gx + 0.5) / w_uv, 0.0), 1.0), min(max((gy + 0.5) / h_uv, 0.0), 1.0)
            return CoordinateMapping._apply_homography(inv, rx, ry)

        return min(max((gx + 0.5) / w_uv, 0.0), 1.0), min(max((gy + 0.5) / h_uv, 0.0), 1.0)
