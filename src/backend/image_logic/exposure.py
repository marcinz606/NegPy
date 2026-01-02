import numpy as np
from typing import Tuple, cast, Dict
from src.helpers import get_luminance
from src.backend.image_logic.color import measure_film_base, calculate_log_medians


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
        shoulder: float = 0.0
    ):
        self.k = contrast
        self.x0 = pivot
        self.L = d_max
        self.toe = toe
        self.shoulder = shoulder

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Avoid log(0)
        diff = x - self.x0
        
        # Calculate mixing factor w
        # w is 0.0 in Highlights (Low x), 1.0 in Shadows (High x)
        # Using 5.0 for a tighter midtone protection zone.
        w = 1.0 / (1.0 + np.exp(-5.0 * diff))
        
        # Protection factor to preserve midtone contrast (k)
        # At midpoint (w=0.5), effect is 0.0. At extremes (w=0 or 1), effect is 1.0.
        protection = 4.0 * ((w - 0.5) ** 2)
        
        # Calculate Dynamic Slope damping with midtone protection
        # We want Toe/Shoulder to fade to 0 effect at the midtones.
        damp_toe = self.toe * (1.0 - w) * protection
        damp_shoulder = self.shoulder * w * protection
        
        k_mod = 1.0 - damp_toe - damp_shoulder
        k_mod = np.clip(k_mod, 0.1, 1.0)
        
        # Apply base Sigmoid with damped slope
        # This ensures midtones stay punchy while extremes can roll off gently.
        val = -self.k * diff
        return self.L / (1.0 + np.exp(val * k_mod))


def apply_contrast(img: np.ndarray, contrast: float) -> np.ndarray:
    """
    Applies simple contrast adjustment around 0.5 midpoint.
    Retained for legacy pipeline compatibility.
    """
    if contrast == 1.0:
        return img
    res = (img - 0.5) * contrast + 0.5
    return np.clip(res, 0.0, 1.0)


def apply_film_characteristic_curve(
    img: np.ndarray,
    params_r: Tuple[float, float],
    params_g: Tuple[float, float],
    params_b: Tuple[float, float],
    toe: float = 0.0,
    shoulder: float = 0.0,
    pre_logged: bool = False,
) -> np.ndarray:
    """
    Applies a film/paper characteristic curve (Sigmoid) per channel in Log-Density space.
    
    Pipeline:
    1. Input `img` is Linear Negative Scan (or Pre-Logged data).
    2. We treat this as Transmittance of Negative -> Exposure on Paper.
    3. Calculate Reflection Density D = Sigmoid(log10(Exposure)).
    4. Calculate Positive Reflection T = 10^-D (Bright=White Paper, Dark=Black Paper).
    
    Args:
        img: RGB image (Negative).
        params_r: (pivot, slope) for Red channel.
        params_g: (pivot, slope) for Green channel.
        params_b: (pivot, slope) for Blue channel.
        toe: Highlights roll-off softness (0-1).
        shoulder: Shadows roll-off softness (0-1).
        pre_logged: If True, skip Log10 conversion.
                            
    Returns:
        np.ndarray: Positive image (0=Black, 1=White).
    """
    # Avoid log(0)
    epsilon = 1e-6
    d_max = 3.5 

    # Unpack parameters
    pivot_r, slope_r = params_r
    pivot_g, slope_g = params_g
    pivot_b, slope_b = params_b

    # Initialize Curves
    curve_r = LogisticSigmoid(contrast=slope_r, pivot=pivot_r, d_max=d_max, toe=toe, shoulder=shoulder)
    curve_g = LogisticSigmoid(contrast=slope_g, pivot=pivot_g, d_max=d_max, toe=toe, shoulder=shoulder)
    curve_b = LogisticSigmoid(contrast=slope_b, pivot=pivot_b, d_max=d_max, toe=toe, shoulder=shoulder)

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
    # T = 10^-D. This is the photometric inversion.
    # D=0 (White) -> T=1.0
    # D=3 (Black) -> T=0.001
    t_r = np.power(10.0, -d_r)
    t_g = np.power(10.0, -d_g)
    t_b = np.power(10.0, -d_b)
    
    # Stack channels to form the Positive Image
    res = np.stack([t_r, t_g, t_b], axis=-1)
    
    # Brightness Fix: Apply Gamma 2.2 to linear transmittance
    # This matches the user's requirement for correct brightness.
    res = cast(np.ndarray, np.power(res, 1.0/2.2))
    
    return np.clip(res, 0.0, 1.0)


def calculate_auto_exposure_params(
    img_raw: np.ndarray, wb_r: float, wb_g: float, wb_b: float
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Solves for optimal Sigmoid Shift (Pivot) and Slope (Contrast) for each channel independently.
    
    Args:
        img_raw: Input RGB image (Linear Negative).
        wb_r, wb_g, wb_b: White Balance gains to neutralize the image before analysis.
        
    Returns:
        ((pivot_r, slope_r), (pivot_g, slope_g), (pivot_b, slope_b))
    """
    # 1. Neutralize
    img = img_raw.copy()
    img[:, :, 0] *= wb_r
    img[:, :, 1] *= wb_g
    img[:, :, 2] *= wb_b
    
    # 2. Log-Exposure
    epsilon = 1e-6
    log_exp = np.log10(np.clip(img, epsilon, None))
    
    results = []
    
    # 3. Analyze per channel
    for ch in range(3):
        vals = log_exp[:, :, ch]
        
        # Find Signal Range
        p5, p95 = np.percentile(vals, [5, 95])
        
        # Target Densities
        # Input Shadow (High Value on Neg) -> p95 (High Log) -> Should map to D_max (Black on Print)
        # Input Highlight (Low Value on Neg) -> p5 (Low Log) -> Should map to D_min (White on Print)
        
        d_target_highlight = 0.2  # White paper (Low Density)
        d_target_shadow = 2.2     # Black ink (High Density)
        d_max = 3.5 
        
        # Inverse Sigmoid logic:
        # y = L / (1 + exp(-k(x-x0)))
        # -k(x-x0) = ln(L/y - 1)
        # k(x-x0) = -ln(L/y - 1)
        
        # At p95 (Shadow input), we want d_target_shadow (High Density)
        term_shadow = -np.log(d_max / d_target_shadow - 1.0)
        
        # At p5 (Highlight input), we want d_target_highlight (Low Density)
        term_highlight = -np.log(d_max / d_target_highlight - 1.0)
        
        # k * p95 - k * x0 = term_shadow
        # k * p5  - k * x0 = term_highlight
        # Subtract: k * (p95 - p5) = term_shadow - term_highlight
        
        if (p95 - p5) < 0.1:
            results.append((-2.0, 1.0))
            continue

        k = (term_shadow - term_highlight) / (p95 - p5)
        
        # Solve for x0 using Shadow point:
        # k * x0 = k * p95 - term_shadow
        x0 = p95 - (term_shadow / k)
        
        results.append((float(x0), float(k)))
        
    return (
        results[0],
        results[1],
        results[2]
    )


def measure_log_negative_bounds(img: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Finds the robust floor and ceiling of each channel in Log10 space.
    Returns (log_floors, log_ceils).
    """
    epsilon = 1e-6
    img_log = np.log10(np.clip(img, epsilon, 1.0))
    floors = []
    ceils = []
    for ch in range(3):
        # 1st and 99.5th percentiles capture the usable density range
        f, c = np.percentile(img_log[:, :, ch], [1.0, 99.5])
        floors.append(float(f))
        ceils.append(float(c))
    return (floors[0], floors[1], floors[2]), (ceils[0], ceils[1], ceils[2])


def solve_photometric_exposure(img: np.ndarray) -> Dict[str, float]:
    """
    Analyzes a raw negative to determine the optimal Photometric Exposure settings.
    
    Log-Expansion Strategy:
    1. Log-Normalize: Stretching performed in Log space to preserve stop ratios.
    2. High-Latitude Slope: Target a wide output range (2.7) for punchy prints.
    3. Truncated Stats: Focus on core subject (10-90th percentile).
    4. Calibration: Calibrated for log-normalized [0,1] input.
    """
    epsilon = 1e-6
    
    # 1. Measure Log Bounds & Normalize Subject Area
    h, w = img.shape[:2]
    mh, mw = int(h * 0.20), int(w * 0.20) 
    subject_linear = img[mh:h-mh, mw:w-mw]
    
    # Get global log bounds
    floors, ceils = measure_log_negative_bounds(img)
    
    # Normalize subject in log domain: [D-max, Base] -> [0.0, 1.0]
    subject_log = np.log10(np.clip(subject_linear, epsilon, 1.0))
    norm_subject_log = np.zeros_like(subject_log)
    for ch in range(3):
        f, c = floors[ch], ceils[ch]
        norm_subject_log[:, :, ch] = (subject_log[:, :, ch] - f) / (max(c - f, epsilon))
    
    # 2. isolate Red Channel for tonality
    red_subject_log = norm_subject_log[:, :, 0]
    
    # 3. Contrast Analysis (Core Subject Range: 10th to 90th percentile)
    p10, p90 = np.percentile(red_subject_log, [10, 90])
    input_range = float(max(p90 - p10, 0.1))
    target_slope = 2.7 / input_range
    
    # Map Slope to UI Grade slider (0-5)
    auto_grade = (target_slope - 1.0) / 1.2
    
    # 4. Density Analysis (Subject Highlights: 75th percentile)
    p75_subject = np.percentile(red_subject_log, 75.0)
    auto_density = (p75_subject - 0.45) / 0.20
    
    # 5. Calculate White Balance (Residual offsets after normalization)
    meds = np.median(norm_subject_log, axis=(0, 1))
    med_r, med_g, med_b = meds[0], meds[1], meds[2]
    
    scale = 100.0
    wb_cyan = 0.0
    wb_magenta = (med_r - med_g) * scale
    wb_yellow = (med_r - med_b) * scale
    
    # Rounding Fix: Lab-style rounding to nearest 0.05 for Density/Grade, 0.5 for WB.
    auto_density = float(np.round(auto_density * 20.0) / 20.0)
    auto_grade = float(np.round(auto_grade * 20.0) / 20.0)
    wb_magenta = float(np.round(wb_magenta * 2.0) / 2.0)
    wb_yellow = float(np.round(wb_yellow * 2.0) / 2.0)

    # Safety clips
    auto_density = float(np.clip(auto_density, -1.0, 3.0))
    auto_grade = float(np.clip(auto_grade, 0.0, 5.0))
    wb_magenta = float(np.clip(wb_magenta, -50.0, 170.0))
    wb_yellow = float(np.clip(wb_yellow, -50.0, 170.0))
    
    return {
        "density": auto_density,
        "grade": auto_grade,
        "wb_cyan": wb_cyan,
        "wb_magenta": wb_magenta,
        "wb_yellow": wb_yellow
    }


def apply_chromaticity_preserving_black_point(
    img: np.ndarray, percentile: float
) -> np.ndarray:
    """
    Neutralizes the overall black level of the print.
    """
    lum = get_luminance(img)
    bp = np.percentile(lum, percentile)
    res = (img - bp) / (1.0 - bp + 1e-6)
    return cast(np.ndarray, np.clip(res, 0.0, 1.0))