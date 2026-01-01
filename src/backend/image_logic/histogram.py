import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from typing import Tuple

def plot_histogram(img_arr: np.ndarray, figsize: Tuple[float, float] = (6, 1), dpi: int = 150) -> plt.Figure:
    """
    Generates a professional RGB + Luminance histogram plot.
    
    Args:
        img_arr (np.ndarray): Image data as uint8 array.
        figsize (Tuple[float, float]): Figure size in inches.
        dpi (int): Plot resolution.
        
    Returns:
        plt.Figure: The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor('#000000') 
    fig.patch.set_facecolor('#000000')
    
    lum = 0.2126 * img_arr[..., 0] + 0.7152 * img_arr[..., 1] + 0.0722 * img_arr[..., 2]
    colors = ('#ff4b4b', '#28df99', '#3182ce')
    
    for i, color in enumerate(colors):
        hist, bins = np.histogram(img_arr[..., i], bins=256, range=(0, 256))
        ax.plot(bins[:-1], hist, color=color, lw=1.2, alpha=0.8)
        ax.fill_between(bins[:-1], hist, color=color, alpha=0.1)

    l_hist, bins = np.histogram(lum, bins=256, range=(0, 256))
    l_hist = ndimage.gaussian_filter1d(l_hist, sigma=1)
    ax.plot(bins[:-1], l_hist, color='#e0e0e0', lw=1.5, alpha=0.9, label='Luma')
    ax.fill_between(bins[:-1], l_hist, color='#e0e0e0', alpha=0.05)
    
    ax.axvline(x=128, color='#7d7d7d', alpha=0.3, lw=1, ls='--')
    ax.axvline(x=64, color='#7d7d7d', alpha=0.2, lw=0.8, ls=':')
    ax.axvline(x=192, color='#7d7d7d', alpha=0.2, lw=0.8, ls=':')

    ax.set_xlim(0, 256)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    return fig
