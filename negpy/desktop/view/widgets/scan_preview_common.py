"""Shared display helpers for scan-preview dialogs (StripPreviewDialog, QuickScanPreviewDialog)."""

import numpy as np

from negpy.infrastructure.scanners.params import FilmType, film_reads_positive


class RollPreviewSignalsMixin:
    """Wires a dialog's preview handlers onto its controller, drives its progress bar,
    and stops the scanner when the dialog goes away.

    Both preview dialogs (whole-strip and single-shot) drive the same
    RollPreviewRequest/roll-preview signal pair and differ only in what their
    handlers do with a result — StripPreviewDialog updates one of N tiles and tracks
    a batch selection, QuickScanPreviewDialog has just one frame. The wiring itself
    doesn't vary, so it lives here once rather than being copy-pasted per dialog.

    A subclass must set ``self._controller``, ``self._previewing`` and
    ``self.status_strip`` before calling ``_connect_preview_signals()``, and
    implement the four result handlers.

    Leaving a dialog mid-preview cancels the request: the transport holds the unit for
    the whole pass, so an abandoned preview leaves the next scan refused as busy.
    """

    def _preview_signal_pairs(self):
        c = self._controller
        return (
            (c.scan_roll_preview_ready, self._on_preview_ready),
            (c.scan_roll_preview_finished, self._on_preview_finished),
            (c.scan_progress, self._on_preview_progress),
            (c.scan_error, self._on_error),
            (c.scan_cancelled, self._on_cancelled),
        )

    def _connect_preview_signals(self) -> None:
        for signal, slot in self._preview_signal_pairs():
            signal.connect(slot)

    def _on_preview_progress(self, fraction: float, phase: str = "Scanning") -> None:
        if not self._previewing:
            return
        self.status_strip.set_progress(f"{phase}… %p%", float(fraction))

    def stop_preview(self) -> None:
        """Ask the transport to abandon the pass in flight. The worker answers with
        `cancelled`, which the dialog's own handler turns into idle state."""
        if self._previewing:
            self._controller.cancel_scan()

    def _teardown_preview(self) -> None:
        self.stop_preview()
        self._previewing = False
        for signal, slot in self._preview_signal_pairs():
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

    def done(self, result: int) -> None:
        # accept() and reject() both route through done() and never raise a close event,
        # so this is the only hook that catches Use / Scan / Cancel / Esc.
        self._teardown_preview()
        super().done(result)

    def closeEvent(self, ev) -> None:
        self._teardown_preview()
        super().closeEvent(ev)


def preview_positive(rgb: np.ndarray, film_type: str = FilmType.NEGATIVE.value) -> np.ndarray:
    """Cheap scan preview: per-channel auto-level, inverted for negative stock.

    Not the real develop pipeline — just enough to read the scene through the
    orange mask. Each channel is stretched between its 1st/99th percentiles, which
    neutralizes the base cast; reversal stock is already a positive and only gets
    the stretch. Mirrors services.assets.thumbnails.preview_positive, which decides
    the same thing off the frame's stored process mode.
    """
    a = rgb.astype(np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    positive = film_reads_positive(film_type)
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        ch = a[..., c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        if hi <= lo:
            out[..., c] = 0.0
            continue
        scaled = (ch - lo) if positive else (hi - ch)
        out[..., c] = np.clip(scaled / (hi - lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)
