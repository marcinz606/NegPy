import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from typing import Tuple
from src.domain.types import ImageBuffer
from src.features.exposure.logic import LogisticSigmoid
from src.features.exposure.models import ExposureConfig, EXPOSURE_CONSTANTS
from src.kernel.image.validation import ensure_image
from src.kernel.image.logic import get_luminance


def plot_histogram(
    img_arr: ImageBuffer, figsize: Tuple[float, float] = (3, 1.4), dpi: int = 150
) -> plt.Figure:
    """
    RGB + Luma histogram plot.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#000000")
    fig.patch.set_facecolor("#000000")

    lum = get_luminance(img_arr)
    colors = ("#ff4b4b", "#28df99", "#3182ce")

    for i, color in enumerate(colors):
        hist, bins = np.histogram(img_arr[..., i], bins=256, range=(0, 256))
        ax.plot(bins[:-1], hist, color=color, lw=1.2, alpha=0.8)
        ax.fill_between(bins[:-1], hist, color=color, alpha=0.1)

    l_hist, bins = np.histogram(lum, bins=256, range=(0, 256))
    l_hist = ndimage.gaussian_filter1d(l_hist, sigma=1)
    ax.plot(bins[:-1], l_hist, color="#e0e0e0", lw=1.5, alpha=0.9, label="Luma")
    ax.fill_between(bins[:-1], l_hist, color="#e0e0e0", alpha=0.05)

    ax.axvline(x=128, color="#7d7d7d", alpha=0.3, lw=1, ls="--")
    ax.axvline(x=64, color="#7d7d7d", alpha=0.2, lw=0.8, ls=":")
    ax.axvline(x=192, color="#7d7d7d", alpha=0.2, lw=0.8, ls=":")

    ax.set_xlim(0, 256)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    return fig


def plot_photometric_curve(
    params: ExposureConfig, figsize: Tuple[float, float] = (3, 1.4), dpi: int = 150
) -> plt.Figure:
    """
    Sigmoid curve visualization.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#000000")
    fig.patch.set_facecolor("#000000")

    master_ref = 1.0
    exposure_shift = 0.1 + (params.density * EXPOSURE_CONSTANTS["density_multiplier"])
    pivot = master_ref - exposure_shift
    slope = 1.0 + (params.grade * EXPOSURE_CONSTANTS["grade_multiplier"])

    curve = LogisticSigmoid(
        contrast=slope,
        pivot=pivot,
        d_max=3.5,
        toe=params.toe,
        toe_width=params.toe_width,
        toe_hardness=params.toe_hardness,
        shoulder=params.shoulder,
        shoulder_width=params.shoulder_width,
        shoulder_hardness=params.shoulder_hardness,
    )

    plt_x = np.linspace(-0.1, 1.1, 100)
    x_log_exp = 1.0 - plt_x

    d = curve(ensure_image(x_log_exp))
    t = np.power(10.0, -d)
    y = np.power(t, 1.0 / 2.2)

    ax.plot(plt_x, y, color="#e0e0e0", lw=2, alpha=0.9)
    ax.fill_between(plt_x, y, color="#e0e0e0", alpha=0.1)

    ax.axvline(x=1.0 - pivot, color="#ff4b4b", alpha=0.4, lw=1, ls="--")
    ax.axvspan(0, 1, color="#28df99", alpha=0.05)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    return fig
