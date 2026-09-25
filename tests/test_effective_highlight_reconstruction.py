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
    positive_source=False,
    linear_raw=False,
) -> ProcessConfig:
    return replace(
        ProcessConfig(),
        process_mode=mode,
        highlight_reconstruction=level,
        narrowband_scan=narrowband_scan,
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
    """The transfer path ('as captured') decodes neutral and folds white
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

    def test_the_export_decode_asks_whether_to_bake_white_balance(self):
        import inspect

        from negpy.services.rendering import image_processor as ip

        assert "highlight_reconstruction_bakes_wb" in inspect.getsource(ip)

    def test_the_preview_bake_flag_is_asked_where_it_originates(self):
        """Like highlight_mode itself, preview_manager.py only forwards a precomputed
        bake_camera_wb flag; the gate is asked where the value originates."""
        import inspect

        from negpy.desktop import controller
        from negpy.desktop.workers import render

        assert "highlight_reconstruction_bakes_wb" in inspect.getsource(controller)
        assert "highlight_reconstruction_bakes_wb" in inspect.getsource(render)


class _SpyRaw:
    """Records the kwargs its own postprocess() call received, and answers
    camera_whitebalance the way rawpy would."""

    raw_type = None
    raw_pattern = None
    sizes = None
    camera_whitebalance = (1.9, 1.0, 1.55, 1.0)
    rgb_xyz_matrix = None
    white_level = 16383
    camera_white_level_per_channel = None
    black_level_per_channel = [0, 0, 0, 0]

    def __init__(self) -> None:
        self.seen: dict = {}

    def __enter__(self) -> "_SpyRaw":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def postprocess(self, **kwargs: object):
        import numpy as np

        self.seen.update(kwargs)
        return np.zeros((4, 4, 3), dtype=np.uint16)


class TestBrightGainReachesTheDecode:
    """A baked decode must not just carry the real white balance -- it must also offset
    the exposure libraw's own highlight-mode scaling takes back out (see
    highlight_reconstruction_bright_gain), or turning Reconstruct on visibly darkens the
    render.
    """

    def test_a_baked_export_decode_passes_the_compensating_bright(self):
        from unittest.mock import patch

        from negpy.services.rendering.image_processor import ImageProcessor

        raw = _SpyRaw()
        with patch("negpy.services.rendering.image_processor.loader_factory") as lf:
            lf.get_loader.return_value = (raw, {})
            ImageProcessor()._decode_sensor_rgb("/x.nef", linear_raw=True, highlight_mode=3, bake_camera_wb=True)

        assert raw.seen["use_camera_wb"] is True
        assert raw.seen["highlight_mode"] == 3
        assert raw.seen["bright"] == pytest.approx(1.9 / 1.0)

    def test_a_clip_mode_export_decode_passes_no_compensation(self):
        from unittest.mock import patch

        from negpy.services.rendering.image_processor import ImageProcessor

        raw = _SpyRaw()
        with patch("negpy.services.rendering.image_processor.loader_factory") as lf:
            lf.get_loader.return_value = (raw, {})
            ImageProcessor()._decode_sensor_rgb("/x.nef", linear_raw=True, highlight_mode=0, bake_camera_wb=False)

        assert raw.seen["bright"] == 1.0

    def test_a_baked_preview_decode_passes_the_compensating_bright(self):
        from unittest.mock import patch

        from negpy.services.rendering.preview_manager import PreviewManager

        raw = _SpyRaw()
        with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
            lf.get_loader.return_value = (raw, {"color_space": "Adobe RGB"})
            PreviewManager().load_linear_preview("/x.nef", highlight_mode=3, bake_camera_wb=True)

        assert raw.seen["bright"] == pytest.approx(1.9 / 1.0)

    def test_a_bracket_siblings_wb_override_pads_to_the_length_rawpy_requires(self):
        """camera_wb_multipliers only ever supplies [R, G, B]; rawpy's user_wb needs
        [R, G, B, G2], or Params.__init__ raises AssertionError."""
        from unittest.mock import patch

        from negpy.services.rendering.image_processor import ImageProcessor

        raw = _SpyRaw()
        with patch("negpy.services.rendering.image_processor.loader_factory") as lf:
            lf.get_loader.return_value = (raw, {})
            ImageProcessor()._decode_sensor_rgb("/x.nef", linear_raw=True, highlight_mode=3, wb_override=[1.9, 1.0, 1.55])

        assert raw.seen["user_wb"] == [1.9, 1.0, 1.55, 1.0]
        assert raw.seen["bright"] == pytest.approx(1.9 / 1.0)

    def test_a_preview_bracket_siblings_wb_override_pads_to_the_length_rawpy_requires(self):
        """Preview counterpart of the export decode's own padding: a bracket preview
        pins its siblings to the reference's white balance the same way the export merge
        and the HDR solve do (see test_hdr_preview_white_balance.py)."""
        from unittest.mock import patch

        from negpy.services.rendering.preview_manager import PreviewManager

        raw = _SpyRaw()
        with patch("negpy.services.rendering.preview_manager.loader_factory") as lf:
            lf.get_loader.return_value = (raw, {"color_space": "Adobe RGB"})
            PreviewManager().load_linear_preview("/x.nef", highlight_mode=3, wb_override=[1.9, 1.0, 1.55])

        assert raw.seen["user_wb"] == [1.9, 1.0, 1.55, 1.0]
        assert raw.seen["use_camera_wb"] is False
        assert raw.seen["bright"] == pytest.approx(1.9 / 1.0)


class TestTheSolveGatesHighlightModeBeforeTheMergeExists:
    """WorkspaceConfig.__post_init__ zeroes highlight_reconstruction once `hdr` names the
    bracket (see test_highlight_reconstruction_on_merges.py), but request_hdr_merge_selected
    calls the solve on each frame's own pre-merge config -- the merge does not exist yet, so
    `hdr_active` reads false there and the invariant has nothing to catch. The solve must
    zero it explicitly instead, or its clip detection sees a per-frame reconstruction guess
    the same way the merge's own would.
    """

    def test_the_solve_decodes_at_clip_despite_each_frame_predating_the_merge(self):
        import numpy as np
        from unittest.mock import MagicMock

        from negpy.desktop.workers.hdr import HdrTask, HdrWorker

        params = replace(
            WorkspaceConfig(),
            process=replace(WorkspaceConfig().process, process_mode=ProcessMode.E6, highlight_reconstruction=5),
        )
        assert params.process.highlight_reconstruction == 5, "the premise: not yet part of a merge"
        seen_levels = []

        class _Processor:
            def _decode_oriented_f32(self, path, p, fast_decode=False, wb_override=None):
                seen_levels.append(p.process.highlight_reconstruction)
                return np.full((4, 4, 3), 0.5, dtype=np.float32), None, "sRGB"

            def camera_wb_for(self, path):
                return [1.9, 1.0, 1.55]

            def cleanup(self, **kw):
                pass

        worker = HdrWorker.__new__(HdrWorker)
        worker._processor = _Processor()
        worker._cancel = MagicMock()
        worker._cancel.is_set.return_value = False
        for signal in ("progress", "solved", "cancelled", "error"):
            setattr(worker, signal, MagicMock())
        files = tuple({"path": f"/x/{i}.nef", "name": f"{i}.nef"} for i in range(3))
        worker.run(HdrTask(files=files, params_by_path={f["path"]: params for f in files}))

        worker.error.emit.assert_not_called()
        assert seen_levels == [0, 0, 0]
