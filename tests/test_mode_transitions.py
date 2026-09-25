"""with_process_mode / with_positive_source: the one route every mode change takes."""

from dataclasses import replace

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import ProcessMode, with_positive_source, with_process_mode


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
