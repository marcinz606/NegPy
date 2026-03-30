from typing import Any, Tuple

import numpy as np
from numba import njit  # type: ignore

from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image


def _expit(x: Any) -> Any:
    """Numpy implementation of the logistic sigmoid function (scipy.special.expit fallback)."""
    return 1.0 / (1.0 + np.exp(-x))


@njit(inline="always")
def _fast_sigmoid(x: float) -> float:
    """
    Fast implementation of the logistic sigmoid function.
    expit(x) = 1 / (1 + exp(-x))
    """
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    else:
        z = np.exp(x)
        return float(z / (1.0 + z))


@njit(cache=True, fastmath=True)
def _apply_photometric_fused_kernel(
    img: np.ndarray,
    pivots: np.ndarray,
    slopes: np.ndarray,
    toe: float,
    toe_width: float,
    shoulder: float,
    shoulder_width: float,
    cmy_offsets: np.ndarray,
    shadow_cmy: np.ndarray,
    highlight_cmy: np.ndarray,
    d_max: float = 4.0,
    gamma: float = 2.2,
    mode: int = 0,
) -> np.ndarray:
    """
    Fused JIT kernel for H&D curve application with hybrid toe/shoulder control.
    """
    h, w, c = img.shape
    res = np.empty_like(img)
    inv_gamma = 1.0 / gamma

    for y in range(h):
        for x in range(w):
            for ch in range(3):
                val = img[y, x, ch] + cmy_offsets[ch]
                diff = val - pivots[ch]
                epsilon = 1e-6

                # Toe Mask (Shadows): Active at high diff (Positive/Dense in negative space)
                t_val = toe_width * (diff / max(1.0 - float(pivots[ch]), epsilon) - 0.5)
                toe_mask = _fast_sigmoid(t_val)

                # Shoulder Mask (Highlights): Active at low diff (Negative/Thin in negative space)
                s_val = -shoulder_width * (diff / max(float(pivots[ch]), epsilon) + 0.5)
                shoulder_mask = _fast_sigmoid(s_val)

                toe_density_offset = toe * toe_mask * 0.1
                shoulder_density_offset = shoulder * shoulder_mask * 0.1

                shadow_color_offset = shadow_cmy[ch] * toe_mask
                highlight_color_offset = highlight_cmy[ch] * shoulder_mask

                diff_adj = diff + shadow_color_offset + highlight_color_offset - toe_density_offset + shoulder_density_offset

                damp_toe = toe * toe_mask * 0.5
                damp_shoulder = shoulder * shoulder_mask * 0.5

                k_mod = 1.0 - damp_toe - damp_shoulder
                if k_mod < 0.1:
                    k_mod = 0.1
                elif k_mod > 2.0:
                    k_mod = 2.0

                slope = slopes[ch]
                density = d_max * _fast_sigmoid(float(slope) * diff_adj * k_mod)

                transmittance = 10.0 ** (-density)
                final_val = transmittance**inv_gamma

                if final_val < 0.0:
                    final_val = 0.0
                elif final_val > 1.0:
                    final_val = 1.0

                res[y, x, ch] = final_val
    return res


class LogisticSigmoid:
    """
    Sigmoid approximation of the H&D curve with hybrid toe/shoulder.
    """

    def __init__(
        self,
        contrast: float,
        pivot: float,
        d_max: float = 4.0,
        toe: float = 0.0,
        toe_width: float = 3.0,
        shoulder: float = 0.0,
        shoulder_width: float = 3.0,
        shadow_cmy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        highlight_cmy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.k = contrast
        self.x0 = pivot
        self.L = d_max
        self.toe = toe
        self.toe_width = toe_width
        self.shoulder = shoulder
        self.shoulder_width = shoulder_width
        self.shadow_cmy = shadow_cmy
        self.highlight_cmy = highlight_cmy

    def __call__(self, x: ImageBuffer) -> ImageBuffer:
        diff = x - self.x0
        epsilon = 1e-6

        t_val = self.toe_width * (diff / max(1.0 - self.x0, epsilon) - 0.5)
        toe_mask = _expit(t_val)

        s_val = -self.shoulder_width * (diff / max(self.x0, epsilon) + 0.5)
        shoulder_mask = _expit(s_val)

        toe_density_offset = self.toe * toe_mask * 0.3
        shoulder_density_offset = self.shoulder * shoulder_mask * 0.3

        diff_adj = diff - toe_density_offset + shoulder_density_offset

        k_mod = 1.0 - (self.toe * toe_mask) - (self.shoulder * shoulder_mask)
        k_mod = np.clip(k_mod, 0.1, 2.0)

        val = self.k * diff_adj * k_mod
        res = self.L * _expit(val)
        return ensure_image(res)


def apply_characteristic_curve(
    img: ImageBuffer,
    params_r: Tuple[float, float],
    params_g: Tuple[float, float],
    params_b: Tuple[float, float],
    toe: float = 0.0,
    toe_width: float = 3.0,
    shoulder: float = 0.0,
    shoulder_width: float = 3.0,
    shadow_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    highlight_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cmy_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    mode: int = 0,
) -> ImageBuffer:
    """
    Applies a film/paper characteristic curve (Sigmoid) per channel in Log-Density space.
    """
    pivots = np.ascontiguousarray(np.array([params_r[0], params_g[0], params_b[0]], dtype=np.float32))
    slopes = np.ascontiguousarray(np.array([params_r[1], params_g[1], params_b[1]], dtype=np.float32))
    offsets = np.ascontiguousarray(np.array(cmy_offsets, dtype=np.float32))
    s_cmy = np.ascontiguousarray(np.array(shadow_cmy, dtype=np.float32))
    h_cmy = np.ascontiguousarray(np.array(highlight_cmy, dtype=np.float32))

    res = _apply_photometric_fused_kernel(
        np.ascontiguousarray(img.astype(np.float32)),
        pivots,
        slopes,
        float(toe),
        float(toe_width),
        float(shoulder),
        float(shoulder_width),
        offsets,
        s_cmy,
        h_cmy,
        mode=mode,
    )

    return ensure_image(res)


def cmy_to_density(val: float, log_range: float = 1.0) -> float:
    """
    Converts a CMY slider value (-1.0..1.0) to a physical density shift (D).
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    absolute_density = val * EXPOSURE_CONSTANTS["cmy_max_density"]
    return float(absolute_density / max(log_range, 1e-6))


def density_to_cmy(density: float, log_range: float = 1.0) -> float:
    """
    Converts a physical density shift (D) back to a normalized CMY slider value.
    """
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    absolute_density = density * log_range
    return float(absolute_density / EXPOSURE_CONSTANTS["cmy_max_density"])


def calculate_wb_shifts(sampled_rgb: np.ndarray) -> Tuple[float, float]:
    """
    Calculates Magenta and Yellow shifts to neutralize sampled color in positive space.
    """
    r, g, b = np.clip(sampled_rgb, 1e-6, 1.0)
    d_m = np.log10(g) - np.log10(r)
    d_y = np.log10(b) - np.log10(r)

    shift_m = density_to_cmy(d_m)
    shift_y = density_to_cmy(d_y)

    return float(shift_m), float(shift_y)


def calculate_wb_shifts_from_log(sampled_log_rgb: np.ndarray) -> Tuple[float, float]:
    """
    Calculates Magenta and Yellow shifts from data in Negative Log-Density space.
    """
    r, g, b = sampled_log_rgb[:3]
    d_m = r - g
    d_y = r - b

    shift_m = density_to_cmy(d_m)
    shift_y = density_to_cmy(d_y)

    return float(shift_m), float(shift_y)
