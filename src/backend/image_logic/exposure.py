import numpy as np
from scipy.special import expit
from typing import Tuple, List
from src.helpers import get_luminance, ensure_array
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
        w_s = expit(self.shoulder_width * (diff / max(self.x0, epsilon)))
        prot_s = (4.0 * ((w_s - 0.5) ** 2)) ** self.shoulder_hardness
        damp_shoulder = self.shoulder * (1.0 - w_s) * prot_s

        # --- TOE (Highlights, High x / Positive diff) ---
        # In our engine, high negative density = Highlights = Low Paper Exposure.
        # This corresponds to the Toe of the paper response.
        # Normalize diff by distance to boundary [x0, 1]
        w_t = expit(self.toe_width * (diff / max(1.0 - self.x0, epsilon)))
        prot_t = (4.0 * ((w_t - 0.5) ** 2)) ** self.toe_hardness
        damp_toe = self.toe * w_t * prot_t

        # Apply dynamic slope damping
        # k_mod can be > 1.0 for hardening (negative toe/shoulder) or < 1.0 for softening
        k_mod = 1.0 - damp_toe - damp_shoulder
        k_mod = np.clip(k_mod, 0.1, 2.0)

        # Apply base Sigmoid with damped slope
        val = self.k * diff
        return ensure_array(self.L * expit(val * k_mod))


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
    cmy_offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
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
    # Subtractive filtration: Channel_Density_Final = Channel_Density_Input + User_CMY_Offset
    d_r = curve_r(log_exp[:, :, 0] + cmy_offsets[0])
    d_g = curve_g(log_exp[:, :, 1] + cmy_offsets[1])
    d_b = curve_b(log_exp[:, :, 2] + cmy_offsets[2])

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


def prepare_exposure_analysis(img: np.ndarray) -> Tuple[np.ndarray, LogNegativeBounds]:
    """
    Helper to normalize raw negative image for sensitometric analysis.
    Extracts bounds and normalizes the subject area (center crop) to density-like space.
    """
    epsilon = 1e-6

    # 1. Measure Log Bounds
    bounds = measure_log_negative_bounds(img)
    floors, ceils = bounds.floors, bounds.ceils

    # 2. Extract Subject Area (Center 60%)
    h, w = img.shape[:2]
    mh, mw = int(h * 0.20), int(w * 0.20)
    subject_linear = img[mh : h - mh, mw : w - mw]

    # 3. Normalize subject in log domain
    # This transforms the input into a normalized log-space where:
    # 0.0 approx corresponds to the floor (Clearer/Base in some conventions, or Denser in others depending on 'img')
    # But based on usage: (val - floor) / range.
    subject_log = np.log10(np.clip(subject_linear, epsilon, 1.0))
    norm_subject_log = np.zeros_like(subject_log)
    for ch in range(3):
        f, c = floors[ch], ceils[ch]
        norm_subject_log[:, :, ch] = (subject_log[:, :, ch] - f) / (max(c - f, epsilon))

    return norm_subject_log, bounds


def analyze_sensitometry(norm_subject_log: np.ndarray) -> Tuple[float, float]:
    """
    Calculates Sensitometric Metrics for the Red Channel (Structure).

    Returns:
        Tuple[float, float]: (measured_dr, midpoint_r)
        - measured_dr: Dynamic Range (P99 - P1)
        - midpoint_r: Midpoint of the range (P99 + P1) / 2
    """
    p1_r = float(np.percentile(norm_subject_log[:, :, 0], 1.0))
    p99_r = float(np.percentile(norm_subject_log[:, :, 0], 99.0))
    measured_dr = max(p99_r - p1_r, 0.1)  # Avoid division by zero
    midpoint_r = (p99_r + p1_r) / 2.0
    return measured_dr, midpoint_r


def solve_photometric_exposure(
    norm_subject_log: np.ndarray, bounds: LogNegativeBounds
) -> Tuple[float, float, float, float, float]:
    """
    Analyzes a raw negative to determine the optimal Photometric Exposure settings using a Sensitometric Solver.

    Logic:
        1. Auto-Filtration (CMY): Align Film Base (D-min) of G/B to R.
        2. Auto-Grade (Contrast): Match Negative Dynamic Range to Paper Dynamic Range.
        3. Auto-Density (Exposure): Range-Based Anchoring (Midpoint to Sigmoid Center).

    Args:
        norm_subject_log: Normalized log-density map.
                          Low values = Clearer Negative (Film Base).
                          High values = Denser Negative (Shadows).
        bounds: Measured log bounds (floors, ceils) of the original image.

    Returns:
        Tuple[float, float, float, float, float]: (cyan, magenta, yellow, density, grade)
        Returned values are in UI Slider units.
    """
    # --- 1. Auto-Filtration (CMY): "Base Neutralization" ---
    # Align the Minimum Density (Film Base) of Green and Blue to Red.
    # We use the Red channel as the reference because it's usually the most stable on color negatives.

    # Calculate Base Density (D-min) for each channel using the 0.1th percentile (robust minimum)
    base_r = float(np.percentile(norm_subject_log[:, :, 0], 0.1))
    base_g = float(np.percentile(norm_subject_log[:, :, 1], 0.1))
    base_b = float(np.percentile(norm_subject_log[:, :, 2], 0.1))

    # Calculate offsets to align Green and Blue D-min to Red D-min
    # If Green Base is denser (higher value) than Red, we need a negative offset to bring it down.
    cyan_offset = 0.0
    magenta_offset = base_r - base_g
    yellow_offset = base_r - base_b

    # --- 2. Sensitometry Analysis ---
    measured_dr, midpoint_r = analyze_sensitometry(norm_subject_log)

    # --- 3. Auto-Grade (Contrast) ---
    # Calculate required physical slope to match the Paper Range
    physical_slope = PIPELINE_CONSTANTS["target_paper_range"] / measured_dr
    physical_slope = float(np.clip(physical_slope, 1.0, 4.0))

    # Convert to UI Grade Units
    # physical_slope = 1.0 + (grade_ui * grade_multiplier)
    auto_grade_ui = (physical_slope - 1.0) / PIPELINE_CONSTANTS["grade_multiplier"]

    # --- 4. Auto-Density (Exposure) ---
    # We want to align the Sigmoid's Pivot (Center) to this Midpoint.
    # In PhotometricProcessor: Pivot = 1.0 - Exposure_Shift
    # So: 1.0 - Exposure_Shift = Midpoint
    # Exposure_Shift = 1.0 - Midpoint

    exposure_shift = 1.0 - midpoint_r

    # Convert to UI Density Units
    # Exposure_Shift = 0.1 + (density_ui * density_multiplier)
    # density_ui = (Exposure_Shift - 0.1) / density_multiplier
    auto_density_ui = (exposure_shift - 0.1) / PIPELINE_CONSTANTS["density_multiplier"]

    return (
        cyan_offset,
        magenta_offset,
        yellow_offset,
        auto_density_ui,
        auto_grade_ui,
    )


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
