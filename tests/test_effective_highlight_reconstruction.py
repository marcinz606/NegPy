"""Reconstruction only makes sense against a slide's own blown highlights, so the level
stored on `ProcessConfig` must never reach a negative decode — not even a stale value left
over from a mode switch. `effective_highlight_reconstruction` is the single gate every
decode call asks; a nonzero stored value off `ProcessMode.E6` must resolve to Clip (0).
"""

from dataclasses import replace

import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.logic import (
    effective_highlight_reconstruction,
    highlight_reconstruction_bakes_wb,
    highlight_reconstruction_bakes_wb_token,
    highlight_reconstruction_token,
    linear_raw_token,
)
from negpy.features.process.models import ProcessConfig, ProcessMode


def cfg(
    mode=ProcessMode.C41,
    level=0,
    narrowband_scan=False,
    e6_normalize=False,
    positive_source=False,
    linear_raw=False,
) -> ProcessConfig:
    return replace(
        ProcessConfig(),
        process_mode=mode,
        highlight_reconstruction=level,
        narrowband_scan=narrowband_scan,
        e6_normalize=e6_normalize,
        positive_source=positive_source,
        linear_raw=linear_raw,
    )


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


class TestHighlightReconstructionBakesWb:
    """The transfer path ('as captured', Normalize off) decodes neutral and folds white
    balance back in downstream. Libraw's own reconstruction reads the decode's per-channel
    multipliers to decide what is clipped, which are all 1.0 on that neutral decode, so its
    threshold sits at the raw ADC ceiling and the clipping this feature targets — which only
    exists after the downstream fold — is invisible to it. An active reconstruction on that
    path must bake the real white balance in instead, and skip the fold to match.
    """

    def test_off_by_default(self):
        assert not highlight_reconstruction_bakes_wb(cfg())

    def test_off_when_reconstruction_itself_is_off(self):
        assert not highlight_reconstruction_bakes_wb(cfg(ProcessMode.E6, level=0))

    def test_on_for_the_default_transfer_path_with_reconstruction_active(self):
        assert highlight_reconstruction_bakes_wb(cfg(ProcessMode.E6, level=5))

    def test_the_explicit_linear_raw_toggle_is_never_overridden(self):
        """The critical case: an explicit request for neutral data must stay neutral,
        even with reconstruction on and everything else identical to the True case above."""
        assert not highlight_reconstruction_bakes_wb(cfg(ProcessMode.E6, level=5, linear_raw=True))

    def test_off_for_a_positive_source(self):
        """No camera matrix to fold in the first place, so nothing to bake either."""
        assert not highlight_reconstruction_bakes_wb(cfg(ProcessMode.E6, level=5, positive_source=True))

    def test_off_with_normalize_on(self):
        """That path already decodes with real white balance and needs no override."""
        assert not highlight_reconstruction_bakes_wb(cfg(ProcessMode.E6, level=5, e6_normalize=True))

    @pytest.mark.parametrize("mode", [ProcessMode.C41, ProcessMode.BW])
    def test_off_on_a_negative(self, mode):
        assert not highlight_reconstruction_bakes_wb(cfg(mode, level=5))

    def test_off_under_narrowband(self):
        assert not highlight_reconstruction_bakes_wb(cfg(ProcessMode.E6, level=5, narrowband_scan=True))

    def test_token_catches_what_linear_raw_token_alone_would_miss(self):
        """The explicit Linear RAW toggle and the transfer path's own default neutral
        decode both read as effective_linear_raw()==True, so linear_raw_token cannot tell
        a config that bakes white balance in apart from one that stays neutral because the
        user asked for it — yet the two decode differently once reconstruction is active.
        """
        default_transfer = cfg(ProcessMode.E6, level=5)
        explicit_linear_raw = cfg(ProcessMode.E6, level=5, linear_raw=True)
        assert linear_raw_token(default_transfer) == linear_raw_token(explicit_linear_raw)
        assert highlight_reconstruction_bakes_wb_token(default_transfer) != highlight_reconstruction_bakes_wb_token(explicit_linear_raw)


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

    def test_every_decode_path_asks_whether_to_bake_white_balance(self):
        """Unlike the level itself, preview_manager.py DOES ask this one directly — it
        decides use_camera_wb locally rather than only forwarding a precomputed flag."""
        import inspect

        from negpy.desktop import controller
        from negpy.desktop.workers import render
        from negpy.services.rendering import image_processor as ip
        from negpy.services.rendering import preview_manager

        for module in (ip, preview_manager, controller, render):
            assert "highlight_reconstruction_bakes_wb" in inspect.getsource(module)
