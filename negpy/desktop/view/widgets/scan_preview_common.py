"""Shared display helpers for scan-preview dialogs (StripPreviewDialog, QuickScanPreviewDialog)."""

import numpy as np


def preview_positive(rgb: np.ndarray) -> np.ndarray:
    """Cheap negative->positive for a scan preview: per-channel invert + auto-level.

    Not the real develop pipeline — just enough to read the scene through the
    orange mask. Each channel is inverted and stretched between its 1st/99th
    percentiles, which both flips the negative and neutralizes the base cast.
    """
    a = rgb.astype(np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        ch = a[..., c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        out[..., c] = 0.0 if hi <= lo else np.clip((hi - ch) / (hi - lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)
