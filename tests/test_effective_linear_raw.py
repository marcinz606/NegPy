"""The transfer path decodes without camera white balance, whatever the stored flag says.

Linear RAW is documented as inert on the E-6 as-captured path — the render applies the
camera matrix, which folds the as-shot multipliers back in itself. But the decode read the
stored flag regardless, so a hidden, stale toggle decided whether white balance was
applied, and two frames of one bracket could decode on different scales.
"""

from dataclasses import replace

import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import RenderIntent
from negpy.features.process.logic import effective_linear_raw, linear_raw_token
from negpy.features.process.models import ProcessConfig, ProcessMode


def cfg(mode=ProcessMode.C41, normalize=False, linear_raw=False) -> ProcessConfig:
    return replace(ProcessConfig(), process_mode=mode, e6_normalize=normalize, linear_raw=linear_raw)


class TestEffectiveLinearRaw:
    def test_transfer_path_decodes_neutral_even_with_the_flag_off(self):
        """The reported fault: the flag is hidden on this path but was still obeyed."""
        assert effective_linear_raw(cfg(ProcessMode.E6, normalize=False, linear_raw=False))

    def test_the_user_flag_still_wins_everywhere_else(self):
        assert effective_linear_raw(cfg(ProcessMode.C41, linear_raw=True))
        assert effective_linear_raw(cfg(ProcessMode.BW, linear_raw=True))
        assert effective_linear_raw(cfg(ProcessMode.E6, normalize=True, linear_raw=True))

    @pytest.mark.parametrize(
        "mode,normalize",
        [(ProcessMode.C41, False), (ProcessMode.C41, True), (ProcessMode.BW, False), (ProcessMode.E6, True)],
    )
    def test_nothing_else_is_forced(self, mode, normalize):
        """Only the transparency transfer is affected — a metered E-6 render and every
        negative path keep the decode they had."""
        assert not effective_linear_raw(cfg(mode, normalize=normalize, linear_raw=False))

    def test_the_flat_intent_is_not_the_transfer_path(self):
        assert not effective_linear_raw(cfg(ProcessMode.E6, normalize=False), RenderIntent.FLAT)

    def test_the_token_follows_the_effective_value(self):
        """Caches key on this. Keying on the stored flag would serve a buffer decoded the
        other way round, which is invisible until the colors are wrong."""
        transfer = cfg(ProcessMode.E6, normalize=False, linear_raw=False)
        assert linear_raw_token(transfer) == linear_raw_token(cfg(ProcessMode.C41, linear_raw=True))
        assert linear_raw_token(transfer) != linear_raw_token(cfg(ProcessMode.C41, linear_raw=False))


class TestDecodeAndMatrixAgree:
    """The decode and the camera matrix must make the same choice.

    `camera_to_working_matrix` folds the as-shot multipliers in only for a buffer decoded
    without them. Apply white balance at one and not the other and the render is tinted by
    the raw green-to-red ratio, roughly 2:1.
    """

    def test_both_sides_read_one_helper(self):
        import inspect

        from negpy.features.exposure import processor as cpu
        from negpy.services.rendering import image_processor as ip

        for mod, what in ((cpu, "the CPU matrix"), (ip, "the decode")):
            src = inspect.getsource(mod)
            assert "effective_linear_raw" in src, f"{what} no longer asks the shared helper"

    def test_batch_analysis_decodes_on_the_render_path_s_white_balance(self):
        """Its own comment is the requirement: the roll-average bounds are applied to the
        render-decoded image, so analysing in a different white balance shifts per-channel
        floors and ceils and prints a color cast. Missed on the first pass — the site read
        the stored flag while the render path had moved to the helper."""
        import inspect

        from negpy.desktop.workers import render

        src = inspect.getsource(render)
        assert "params.process.linear_raw if params else" not in src, "Batch Analysis is back on the stored flag"
        assert "effective_linear_raw" in src

    def test_neighbour_prefetch_keys_on_the_same_decode(self):
        """The warm buffer has to land under the key load_file will look for, or navigation
        re-decodes and the prefetch is wasted."""
        import inspect

        from negpy.desktop import controller

        src = inspect.getsource(controller)
        assert "saved.process.linear_raw if saved else False" not in src, "prefetch key will not match the decode"

    def test_the_gpu_matrix_agrees_with_the_cpu_one(self):
        import inspect

        from negpy.services.rendering import gpu_engine

        assert "effective_linear_raw" in inspect.getsource(gpu_engine), "GPU matrix would drift from the CPU one"


class TestSourceIdentity:
    def test_a_transfer_render_is_not_cached_as_a_wb_decode(self):
        from negpy.services.rendering.source_identity import source_token

        base = WorkspaceConfig()
        slide = replace(base, process=replace(base.process, process_mode=ProcessMode.E6, e6_normalize=False))
        negative = replace(base, process=replace(base.process, process_mode=ProcessMode.C41))
        assert source_token(slide) != source_token(negative)
