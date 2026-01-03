import numpy as np
from typing import Tuple, Dict, List
from src.helpers import get_luminance, density_to_cmy
from src.domain_objects import LogNegativeBounds
from src.config import PIPELINE_CONSTANTS


class LogisticSigmoid:
    """
    Models a photometric H&D Characteristic Curve using a logistic sigmoid.
    D = L / (1 + exp(-k * (x - x0)))

    Attributes:
        k (float): Contrast/Slope of the curve.
        x0 (float): Pivot/Speed point (Log Exposure shift).
        L (float): Maximum Density (D_max).
        toe (float): Highlights roll-off softness (0-1).
        shoulder (float): Shadows roll-off softness (0-1).
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

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Avoid log(0)
        diff = x - self.x0
        epsilon = 1e-6

        # --- SHOULDER (Shadows, Low x / Negative diff) ---
        # In our engine, low negative density = Shadows = High Paper Exposure.
        # This corresponds to the Shoulder of the paper response.
        # Normalize diff by distance to boundary [0, x0]
        w_s = 1.0 / (
            1.0 + np.exp(-self.shoulder_width * (diff / max(self.x0, epsilon)))
        )
        prot_s = (4.0 * ((w_s - 0.5) ** 2)) ** self.shoulder_hardness
        damp_shoulder = self.shoulder * (1.0 - w_s) * prot_s

        # --- TOE (Highlights, High x / Positive diff) ---
        # In our engine, high negative density = Highlights = Low Paper Exposure.
        # This corresponds to the Toe of the paper response.
        # Normalize diff by distance to boundary [x0, 1]
        w_t = 1.0 / (
            1.0 + np.exp(-self.toe_width * (diff / max(1.0 - self.x0, epsilon)))
        )
        prot_t = (4.0 * ((w_t - 0.5) ** 2)) ** self.toe_hardness
        damp_toe = self.toe * w_t * prot_t

        # Apply dynamic slope damping
        # k_mod can be > 1.0 for hardening (negative toe/shoulder) or < 1.0 for softening
        k_mod = 1.0 - damp_toe - damp_shoulder
        k_mod = np.clip(k_mod, 0.1, 2.0)

        # Apply base Sigmoid with damped slope
        val = -self.k * diff
        return self.L / (1.0 + np.exp(val * k_mod))


def apply_film_characteristic_curve(
    img: np.ndarray,
    params_r: Tuple[float, float],
    params_g: Tuple[float, float],
    params_b: Tuple[float, float],
    toe: float = 0.0,
    toe_width: float = 3.0,
    toe_hardness: float = 1.0,
    shoulder: float = 0.0,
    shoulder_width: float = 3.0,
    shoulder_hardness: float = 1.0,
    pre_logged: bool = False,
) -> np.ndarray:
    """
    Applies a film/paper characteristic curve (Sigmoid) per channel in Log-Density space.
    """
    epsilon = 1e-6
    d_max = 3.5

    # Unpack parameters
    pivot_r, slope_r = params_r
    pivot_g, slope_g = params_g
    pivot_b, slope_b = params_b

    # Initialize Curves
    curve_r = LogisticSigmoid(
        contrast=slope_r,
        pivot=pivot_r,
        d_max=d_max,
        toe=toe,
        toe_width=toe_width,
        toe_hardness=toe_hardness,
        shoulder=shoulder,
        shoulder_width=shoulder_width,
        shoulder_hardness=shoulder_hardness,
    )
    curve_g = LogisticSigmoid(
        contrast=slope_g,
        pivot=pivot_g,
        d_max=d_max,
        toe=toe,
        toe_width=toe_width,
        toe_hardness=toe_hardness,
        shoulder=shoulder,
        shoulder_width=shoulder_width,
        shoulder_hardness=shoulder_hardness,
    )
    curve_b = LogisticSigmoid(
        contrast=slope_b,
        pivot=pivot_b,
        d_max=d_max,
        toe=toe,
        toe_width=toe_width,
        toe_hardness=toe_hardness,
        shoulder=shoulder,
        shoulder_width=shoulder_width,
        shoulder_hardness=shoulder_hardness,
    )

    # Calculate Log Exposure
    if pre_logged:
        log_exp = img
    else:
        log_exp = np.log10(np.clip(img, epsilon, None))

    # Apply per-channel Sigmoid -> Density
    d_r = curve_r(log_exp[:, :, 0])
    d_g = curve_g(log_exp[:, :, 1])
    d_b = curve_b(log_exp[:, :, 2])

    # Calculate Transmittance (Positive Reflection)
    t_r = np.power(10.0, -d_r)
    t_g = np.power(10.0, -d_g)
    t_b = np.power(10.0, -d_b)

    # Stack channels to form the Positive Image
    res = np.stack([t_r, t_g, t_b], axis=-1)

    # Brightness Fix: Apply Gamma 2.2
    res_gamma: np.ndarray = np.power(res, 1.0 / 2.2)

    return np.clip(res_gamma, 0.0, 1.0)


def measure_log_negative_bounds(img: np.ndarray) -> LogNegativeBounds:
    """
    Finds the robust floor and ceiling of each channel in Log10 space.
    Returns LogNegativeBounds containing floors and ceils.
    """
    epsilon = 1e-6
    img_log = np.log10(np.clip(img, epsilon, 1.0))
    floors: List[float] = []
    ceils: List[float] = []
    for ch in range(3):
        # 1st and 99.5th percentiles capture the usable density range
        f, c = np.percentile(img_log[:, :, ch], [1.0, 99.5])
        floors.append(float(f))
        ceils.append(float(c))

    return LogNegativeBounds(
        floors=(floors[0], floors[1], floors[2]),
        ceils=(ceils[0], ceils[1], ceils[2]),
    )


def solve_photometric_exposure(img: np.ndarray) -> Dict[str, float]:
    """
    Analyzes a raw negative to determine the optimal Photometric Exposure settings.
    """
    epsilon = 1e-6

    # 1. Measure Log Bounds & Normalize Subject Area
    h, w = img.shape[:2]
    mh, mw = int(h * 0.20), int(w * 0.20)
    subject_linear = img[mh : h - mh, mw : w - mw]

    # Get global log bounds
    bounds = measure_log_negative_bounds(img)
    floors, ceils = bounds.floors, bounds.ceils

    # Normalize subject in log domain
    subject_log = np.log10(np.clip(subject_linear, epsilon, 1.0))
    norm_subject_log = np.zeros_like(subject_log)
    for ch in range(3):
        f, c = floors[ch], ceils[ch]
        norm_subject_log[:, :, ch] = (subject_log[:, :, ch] - f) / (max(c - f, epsilon))

    # 2. isolate Red Channel for tonality
    red_subject_log = norm_subject_log[:, :, 0]

    # 3. Contrast Analysis
    p10, p90 = np.percentile(red_subject_log, [10, 90])
    input_range = float(max(p90 - p10, 0.1))
    target_slope = PIPELINE_CONSTANTS["auto_grade_target"] / input_range

    # Map Slope to UI Grade slider (0-5)
    auto_grade = (target_slope - 1.0) / PIPELINE_CONSTANTS["grade_multiplier"]

    # 4. Density Analysis
    p75_subject = np.percentile(red_subject_log, 75.0)
    auto_density = (
        p75_subject - PIPELINE_CONSTANTS["auto_density_target"]
    ) / PIPELINE_CONSTANTS["density_multiplier"]

    # 5. Calculate White Balance
    med_r = np.median(norm_subject_log[:, :, 0])
    med_g = np.median(norm_subject_log[:, :, 1])
    med_b = np.median(norm_subject_log[:, :, 2])

    wb_magenta_norm = med_r - med_g
    wb_yellow_norm = med_r - med_b

    # Convert to CMY units
    wb_cyan = 0.0
    wb_magenta = density_to_cmy(wb_magenta_norm, ceils[1] - floors[1])
    wb_yellow = density_to_cmy(wb_yellow_norm, ceils[2] - floors[2])

    # Rounding
    auto_density = float(np.round(auto_density * 20.0) / 20.0)
    auto_grade = float(np.round(auto_grade * 20.0) / 20.0)
    wb_magenta = float(np.round(wb_magenta * 20.0) / 20.0)
    wb_yellow = float(np.round(wb_yellow * 20.0) / 20.0)

    return {
        "density": auto_density,
        "grade": auto_grade,
        "wb_cyan": wb_cyan,
        "wb_magenta": wb_magenta,
        "wb_yellow": wb_yellow,
    }


def apply_chromaticity_preserving_black_point(
    img: np.ndarray, percentile: float
) -> np.ndarray:
    """
    Neutralizes the overall black level of the print.
    """
    lum = get_luminance(img)
    bp = np.percentile(lum, percentile)
    res: np.ndarray = (img - bp) / (1.0 - bp + 1e-6)
    return np.clip(res, 0.0, 1.0)
