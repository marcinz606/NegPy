import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from typing import Tuple
from src.helpers import get_luminance
from src.domain_objects import ImageSettings
from src.backend.image_logic.exposure import LogisticSigmoid
from src.config import PIPELINE_CONSTANTS


def plot_histogram(
    img_arr: np.ndarray, figsize: Tuple[float, float] = (3, 1.4), dpi: int = 150
) -> plt.Figure:
    """
    Generates a professional RGB + Luminance histogram plot.
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
    params: ImageSettings, figsize: Tuple[float, float] = (3, 1.4), dpi: int = 150
) -> plt.Figure:
    """
    Plots the Photometric H&D Curve (Sigmoid) representing the Positive response.
    X-axis: Subject Brightness (0=Black, 1=White)
    Y-axis: Print Brightness (0=Black, 1=White)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#000000")
    fig.patch.set_facecolor("#000000")

    master_ref = 1.0
    exposure_shift = 0.1 + (params.density * PIPELINE_CONSTANTS["density_multiplier"])
    pivot = master_ref - exposure_shift
    slope = 1.0 + (params.grade * PIPELINE_CONSTANTS["grade_multiplier"])

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

    # Subject Brightness range for plotting (0=Black, 1=White)
    plt_x = np.linspace(-0.1, 1.1, 100)

    # Map Subject Brightness to Log Exposure on paper (1=Shadows, 0=Highlights)
    # This effectively 'reverses' the curve to represent the positive.
    x_log_exp = 1.0 - plt_x

    d = curve(x_log_exp)
    t = np.power(10.0, -d)
    y = np.power(t, 1.0 / 2.2)

    # Plotting Subject Brightness vs Output Brightness
    ax.plot(plt_x, y, color="#e0e0e0", lw=2, alpha=0.9)
    ax.fill_between(plt_x, y, color="#e0e0e0", alpha=0.1)

    # Pivot marker (transformed to plt_x space)
    ax.axvline(x=1.0 - pivot, color="#ff4b4b", alpha=0.4, lw=1, ls="--")

    # Highlight the usable range [0, 1]
    ax.axvspan(0, 1, color="#28df99", alpha=0.05)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    return fig
