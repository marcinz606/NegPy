"""with_process_mode / with_positive_source: the one route every mode change takes."""

from dataclasses import replace

from negpy.domain.models import WorkspaceConfig
from negpy.desktop.settings_catalog import all_rows, apply_selected_fields
from negpy.features.process.models import ProcessMode, with_positive_source, with_process_mode, with_slide_bw


def _slide(**exposure) -> WorkspaceConfig:
    cfg = with_process_mode(WorkspaceConfig(), ProcessMode.E6)
    return replace(cfg, exposure=replace(cfg.exposure, **exposure))


def test_entering_slide_starts_cast_removal_at_zero_and_leaving_restores_it():
    neg = WorkspaceConfig()
    slide = with_process_mode(neg, ProcessMode.E6)
    assert slide.exposure.cast_removal_strength == 0.0
    assert with_process_mode(slide, ProcessMode.C41).exposure.cast_removal_strength == neg.exposure.cast_removal_strength


def test_the_same_mode_keeps_a_deliberate_slide_strength():
    slide = _slide(cast_removal_strength=WorkspaceConfig().exposure.cast_removal_strength)
    assert with_process_mode(slide, ProcessMode.E6) is slide


def test_leaving_slide_drops_positive_and_restores_the_autos():
    positive = with_positive_source(_slide(), True)
    assert positive.exposure.auto_exposure is False
    neg = with_process_mode(positive, ProcessMode.BW)
    assert neg.process.positive_source is False
    assert neg.exposure.auto_exposure is True


def test_positive_is_refused_off_slide_and_a_repeat_is_a_no_op():
    neg = WorkspaceConfig()
    assert with_positive_source(neg, True) is neg
    positive = with_positive_source(_slide(), True)
    assert with_positive_source(positive, True) is positive


def _chroma(cfg: WorkspaceConfig, saturation: float) -> WorkspaceConfig:
    return replace(cfg, lab=replace(cfg.lab, saturation=saturation))


def test_leaving_bw_restores_a_hidden_chroma_of_zero():
    bw = _chroma(with_process_mode(WorkspaceConfig(), ProcessMode.BW), 0.0)
    assert with_process_mode(bw, ProcessMode.C41).lab.saturation == 1.0
    assert with_process_mode(bw, ProcessMode.E6).lab.saturation == 1.0


def test_leaving_bw_keeps_any_other_chroma():
    bw = _chroma(with_process_mode(WorkspaceConfig(), ProcessMode.BW), 0.4)
    assert with_process_mode(bw, ProcessMode.C41).lab.saturation == 0.4


def test_entering_bw_keeps_a_chroma_of_zero():
    assert with_process_mode(_chroma(WorkspaceConfig(), 0.0), ProcessMode.BW).lab.saturation == 0.0


def test_turning_slide_bw_off_restores_a_chroma_of_zero():
    slide = with_slide_bw(_chroma(_slide(), 0.0), True)
    assert slide.lab.saturation == 0.0
    assert with_slide_bw(slide, False).lab.saturation == 1.0


def test_a_pasted_chroma_of_zero_wins_over_the_switch():
    source = _chroma(WorkspaceConfig(), 0.0)
    target = with_process_mode(WorkspaceConfig(), ProcessMode.BW)
    rows = [r for r in all_rows() if r.id in ("process.process_mode", "lab.saturation")]
    out = apply_selected_fields(source, target, rows)
    assert (out.process.process_mode, out.lab.saturation) == (ProcessMode.C41, 0.0)
