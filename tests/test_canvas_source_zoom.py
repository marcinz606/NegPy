"""Zoom is quoted against source pixels, whatever resolution the frame rendered at."""

from __future__ import annotations

from types import SimpleNamespace

from negpy.desktop.view.canvas.widget import ImageCanvas


class _Canvas:
    """Enough of ImageCanvas for the zoom math: the fit scale is supplied directly."""

    _source_scale = ImageCanvas._source_scale
    current_zoom_percent = ImageCanvas.current_zoom_percent
    zoom_to_percent = ImageCanvas.zoom_to_percent
    zoom_to_original = ImageCanvas.zoom_to_original
    zoom_note = ImageCanvas.zoom_note

    def __init__(self, fit_scale: float, render_long_edge: int, original_res: tuple[int, int]) -> None:
        self.zoom_level = 1.0
        self.pan_offset = None
        self._fs = fit_scale
        self.state = SimpleNamespace(
            last_metrics={"render_long_edge": render_long_edge},
            original_res=original_res,
            hq_preview=False,
        )

    def _fit_scale(self):
        return self._fs

    def set_zoom(self, z):
        self.zoom_level = z

    def _sync_transform(self):
        pass


def test_full_res_buffer_is_one_to_one():
    c = _Canvas(fit_scale=0.25, render_long_edge=6000, original_res=(6000, 4000))
    c.zoom_to_original()
    assert round(c.zoom_level, 6) == 4.0
    assert c.current_zoom_percent() == 100


def test_downscaled_preview_still_reaches_source_pixels():
    """A preview-size buffer needs 3.75x more zoom for one scan pixel per screen pixel."""
    c = _Canvas(fit_scale=0.25, render_long_edge=1600, original_res=(6000, 4000))
    c.zoom_to_original()
    assert round(c.zoom_level, 6) == 15.0
    assert c.current_zoom_percent() == 100


def test_readout_is_invariant_to_the_buffer_swap():
    """Swapping a proxy frame for a full-res one changes neither the view nor the readout."""
    proxy = _Canvas(fit_scale=1.0, render_long_edge=1600, original_res=(6400, 4000))
    proxy.zoom_level = 2.0
    full = _Canvas(fit_scale=0.25, render_long_edge=6400, original_res=(6400, 4000))
    full.zoom_level = 2.0
    assert proxy.current_zoom_percent() == full.current_zoom_percent() == 50


def test_small_source_needs_no_correction():
    c = _Canvas(fit_scale=0.5, render_long_edge=1200, original_res=(1200, 800))
    c.zoom_to_original()
    assert round(c.zoom_level, 6) == 2.0


def test_missing_metrics_fall_back_to_buffer_pixels():
    c = _Canvas(fit_scale=0.5, render_long_edge=0, original_res=(0, 0))
    assert c._source_scale() == 1.0
    c.zoom_to_original()
    assert round(c.zoom_level, 6) == 2.0


def _note(fit_scale, render_long_edge, original_res, zoom_level, hq):
    c = _Canvas(fit_scale=fit_scale, render_long_edge=render_long_edge, original_res=original_res)
    c.zoom_level = zoom_level
    c.state.hq_preview = hq
    return c.zoom_note()


def test_note_marks_an_upscaled_preview_at_one_to_one():
    assert _note(0.25, 1600, (6000, 4000), 15.0, hq=False) == "preview res · HQ off"


def test_note_is_silent_below_one_to_one():
    """Fit view on a preview buffer says nothing, however large the display."""
    assert _note(2.0, 1600, (6000, 4000), 1.0, hq=False) == ""


def test_note_is_silent_on_hq():
    """An interactive proxy frame renders small while HQ is on; the settled frame does not."""
    assert _note(0.25, 1600, (6000, 4000), 15.0, hq=True) == ""


def test_note_is_silent_when_the_source_fits_the_preview_budget():
    assert _note(0.5, 1200, (1200, 800), 2.0, hq=False) == ""
