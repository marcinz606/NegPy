import numpy as np
from scipy.special import expit
from typing import Tuple
from src.core.types import ImageBuffer
from src.core.validation import ensure_image


class LogisticSigmoid:
    """
    Models a photometric H&D Characteristic Curve using a logistic sigmoid.
    D = L / (1 + exp(-k * (x - x0)))
    """

    def __init__(
        self,
        contrast: float,
        pivot: float,
        d_max: float = 3.5,
        toe: float = 0.0,
        toe_width: float = 3.0,
        toe_hardness: float = 1.0,
        shoulder: float = 0.0,
        shoulder_width: float = 3.0,
        shoulder_hardness: float = 1.0,
    ):
        self.k = contrast
        self.x0 = pivot
        self.L = d_max
        self.toe = toe
        self.toe_width = toe_width
        self.toe_hardness = toe_hardness
        self.shoulder = shoulder
        self.shoulder_width = shoulder_width
        self.shoulder_hardness = shoulder_hardness

    def __call__(self, x: ImageBuffer) -> ImageBuffer:
        # Avoid log(0)
        diff = x - self.x0
        epsilon = 1e-6

        # --- SHOULDER (Shadows) ---
        w_s = expit(self.shoulder_width * (diff / max(self.x0, epsilon)))
        prot_s = (4.0 * ((w_s - 0.5) ** 2)) ** self.shoulder_hardness
        damp_shoulder = self.shoulder * (1.0 - w_s) * prot_s

        # --- TOE (Highlights) ---
        w_t = expit(self.toe_width * (diff / max(1.0 - self.x0, epsilon)))
        prot_t = (4.0 * ((w_t - 0.5) ** 2)) ** self.toe_hardness
        damp_toe = self.toe * w_t * prot_t

        k_mod = 1.0 - damp_toe - damp_shoulder
        k_mod = np.clip(k_mod, 0.1, 2.0)

        val = self.k * diff
        res = self.L * expit(val * k_mod)
        return ensure_image(res)


def apply_characteristic_curve(
    img: ImageBuffer,
    params_r: Tuple[float, float],
    params_g: Tuple[float, float],
    params_b: Tuple[float, float],
    toe: float = 0.0,
    toe_width: float = 3.0,
    toe_hardness: float = 1.0,
    shoulder: float = 0.0,
    shoulder_width: float = 3.0,
    shoulder_hardness: float = 1.0,
    cmy_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ImageBuffer:
    """
    Applies a film/paper characteristic curve (Sigmoid) per channel in Log-Density space.
    Input 'img' is expected to be Normalized Log Negative (or similar).
    """
    d_max = 3.5

    # Unpack parameters (Pivot, Slope)
    pivot_r, slope_r = params_r
    pivot_g, slope_g = params_g
    pivot_b, slope_b = params_b

    # Initialize Curves
    curve_r = LogisticSigmoid(
        slope_r,
        pivot_r,
        d_max,
        toe,
        toe_width,
        toe_hardness,
        shoulder,
        shoulder_width,
        shoulder_hardness,
    )
    curve_g = LogisticSigmoid(
        slope_g,
        pivot_g,
        d_max,
        toe,
        toe_width,
        toe_hardness,
        shoulder,
        shoulder_width,
        shoulder_hardness,
    )
    curve_b = LogisticSigmoid(
        slope_b,
        pivot_b,
        d_max,
        toe,
        toe_width,
        toe_hardness,
        shoulder,
        shoulder_width,
        shoulder_hardness,
    )

    # Use the input directly (assumed to be pre-logged by Normalization step)
    log_exp = img

    # Apply per-channel Sigmoid -> Density
    d_r = curve_r(log_exp[:, :, 0] + cmy_offsets[0])
    d_g = curve_g(log_exp[:, :, 1] + cmy_offsets[1])
    d_b = curve_b(log_exp[:, :, 2] + cmy_offsets[2])

    # Calculate Transmittance (Positive Reflection)
    t_r = np.power(10.0, -d_r)
    t_g = np.power(10.0, -d_g)
    t_b = np.power(10.0, -d_b)

    res = np.stack([t_r, t_g, t_b], axis=-1)

    # Gamma 2.2 for display/print linearity
    res_gamma = np.power(res, 1.0 / 2.2)

    return ensure_image(np.clip(res_gamma, 0.0, 1.0))


def cmy_to_density(val: float, log_range: float = 1.0) -> float:
    """
    Converts a CMY slider value (-1.0..1.0) to a density shift.
    """
    from src.core.constants import PIPELINE_CONSTANTS

    absolute_density = val * PIPELINE_CONSTANTS["cmy_max_density"]
    return float(absolute_density / max(log_range, 1e-6))


def density_to_cmy(density: float, log_range: float = 1.0) -> float:
    """
    Converts a density shift back to a CMY slider value (-1.0..1.0).
    """
    from src.core.constants import PIPELINE_CONSTANTS

    absolute_density = density * log_range
    return float(absolute_density / PIPELINE_CONSTANTS["cmy_max_density"])
