import numpy as np
from typing import Any, cast


def apply_gamma_to_img(img: np.ndarray, gamma_val: Any) -> np.ndarray:
    """
    Applies gamma correction (per-channel power law) to an image.
    """
    if isinstance(gamma_val, (list, tuple, np.ndarray)):
        gammas = np.array(gamma_val)
    else:
        gammas = np.array([gamma_val, gamma_val, gamma_val])

    if np.all(gammas == 1.0):
        return img

    g_inv = 1.0 / np.maximum(0.01, gammas)
    res = np.power(img, g_inv)
    return cast(np.ndarray, np.clip(res, 0.0, 1.0))


def calculate_balancing_gammas(img: np.ndarray, target_percentile: float) -> np.ndarray:
    """
    Calculates per-channel gamma required to map the specified percentile of
    each channel to the Green channel's value.
    Formula: P^g = T => g = log(T) / log(P)
    """
    points = np.percentile(img, target_percentile, axis=(0, 1))
    target = np.clip(points[1], 0.01, 0.99)  # Anchor to Green
    points = np.clip(points, 0.01, 0.99)
    return cast(np.ndarray, np.log(target) / (np.log(points) + 1e-6))
