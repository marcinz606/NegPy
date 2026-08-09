"""Shared display helpers for scan-preview dialogs (StripPreviewDialog, QuickScanPreviewDialog)."""

import numpy as np


class RollPreviewSignalsMixin:
    """Wires a dialog's four scan_roll_preview_ready/scan_roll_preview_finished/
    scan_error/scan_cancelled handlers onto its controller, and tears the connections
    down on close.

    Both preview dialogs (whole-strip and single-shot) drive the same
    RollPreviewRequest/roll-preview signal pair and differ only in what their four
    handlers do with a result — StripPreviewDialog updates one of N tiles and tracks
    a batch selection, QuickScanPreviewDialog has just one frame. The wiring itself
    doesn't vary, so it lives here once rather than being copy-pasted per dialog.

    A subclass must set ``self._controller`` before calling ``_connect_preview_signals()``
    (typically the first thing __init__ does) and implement the four handlers.
    """

    def _preview_signal_pairs(self):
        c = self._controller
        return (
            (c.scan_roll_preview_ready, self._on_preview_ready),
            (c.scan_roll_preview_finished, self._on_preview_finished),
            (c.scan_error, self._on_error),
            (c.scan_cancelled, self._on_cancelled),
        )

    def _connect_preview_signals(self) -> None:
        for signal, slot in self._preview_signal_pairs():
            signal.connect(slot)

    def closeEvent(self, ev) -> None:
        for signal, slot in self._preview_signal_pairs():
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(ev)


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
