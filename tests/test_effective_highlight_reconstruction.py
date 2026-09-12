"""Reconstruction only makes sense against a slide's own blown highlights, so the level
stored on `ProcessConfig` must never reach a negative decode — not even a stale value left
over from a mode switch. `effective_highlight_reconstruction` is the single gate every
decode call asks; a nonzero stored value off `ProcessMode.E6` must resolve to Clip (0).
"""

from dataclasses import replace

import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.logic import effective_highlight_reconstruction, highlight_reconstruction_token
from negpy.features.process.models import ProcessConfig, ProcessMode


def cfg(mode=ProcessMode.C41, level=0, narrowband_scan=False) -> ProcessConfig:
    return replace(ProcessConfig(), process_mode=mode, highlight_reconstruction=level, narrowband_scan=narrowband_scan)


class TestEffectiveHighlightReconstruction:
    def test_default_is_off(self):
        assert ProcessConfig().highlight_reconstruction == 0
        assert effective_highlight_reconstruction(ProcessConfig()) == 0

    @pytest.mark.parametrize("level", [2, 3, 4, 5, 6, 7, 8, 9])
    def test_a_valid_level_passes_through_on_a_slide(self, level):
        assert effective_highlight_reconstruction(cfg(ProcessMode.E6, level)) == level

    @pytest.mark.parametrize("mode", [ProcessMode.C41, ProcessMode.BW])
    @pytest.mark.parametrize("level", [0, 2, 5, 9])
    def test_never_reaches_a_negative_regardless_of_the_stored_value(self, mode, level):
        """The critical case: a stale or copied config must not silently start
        reconstructing a negative's genuinely-clipped base."""
        assert effective_highlight_reconstruction(cfg(mode, level)) == 0

    @pytest.mark.parametrize("level", [1, 10, -1, 100])
    def test_an_out_of_range_stored_value_resolves_to_off(self, level):
        """1 (Ignore) is reserved for a separate decode-correctness fix and must never be
        passed here; anything else out of libraw's 0/2-9 range is a bogus/corrupt value."""
        assert effective_highlight_reconstruction(cfg(ProcessMode.E6, level)) == 0

    def test_narrowband_forces_off_even_on_a_slide(self):
        """A triplet's single raw channel per exposure has no 'highlight color' to
        reconstruct — same reasoning should_fold_camera_wb applies to the WB fold."""
        assert effective_highlight_reconstruction(cfg(ProcessMode.E6, 5, narrowband_scan=True)) == 0

    def test_the_token_follows_the_effective_value_not_the_stored_one(self):
        """A stale value off E-6 must not be able to change the cache key even though it
        never reaches the decode."""
        assert highlight_reconstruction_token(cfg(ProcessMode.C41, 5)) == highlight_reconstruction_token(cfg(ProcessMode.C41, 0))
        assert highlight_reconstruction_token(cfg(ProcessMode.E6, 5)) != highlight_reconstruction_token(cfg(ProcessMode.E6, 0))


class TestSourceIdentity:
    def test_source_token_differs_on_a_slide(self):
        from negpy.services.rendering.source_identity import source_token

        base = WorkspaceConfig()
        off = replace(base, process=replace(base.process, process_mode=ProcessMode.E6, highlight_reconstruction=0))
        on = replace(base, process=replace(base.process, process_mode=ProcessMode.E6, highlight_reconstruction=5))
        assert source_token(off) != source_token(on)

    def test_source_token_is_unaffected_on_a_negative(self):
        """Forced to 0 either way, so the stored value must not perturb the decode's cache
        identity off the E-6 path."""
        from negpy.services.rendering.source_identity import source_token

        base = WorkspaceConfig()
        off = replace(base, process=replace(base.process, process_mode=ProcessMode.C41, highlight_reconstruction=0))
        on = replace(base, process=replace(base.process, process_mode=ProcessMode.C41, highlight_reconstruction=5))
        assert source_token(off) == source_token(on)


class TestDecodeAsksTheGate:
    """Every site that sets `highlight_mode` on a decode must go through
    `effective_highlight_reconstruction`, the same discipline `effective_linear_raw` is held
    to (see `test_effective_linear_raw.py::TestDecodeAndMatrixAgree`) — a hardcoded or
    unguarded value here is how reconstruction reaches a negative.
    """

    def test_the_render_decode_asks_the_shared_helper(self):
        import inspect

        from negpy.services.rendering import image_processor as ip

        assert "effective_highlight_reconstruction" in inspect.getsource(ip)

    def test_the_preview_decode_asks_the_shared_helper(self):
        import inspect

        from negpy.desktop import controller
        from negpy.desktop.workers import render

        # preview_manager.py itself only forwards a precomputed highlight_mode (same
        # pattern as use_camera_wb); the gate is asked where the value originates.
        assert "effective_highlight_reconstruction" in inspect.getsource(controller)
        assert "effective_highlight_reconstruction" in inspect.getsource(render)
