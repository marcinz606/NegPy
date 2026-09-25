"""Slide Normalize retired: a saved slide keeps its metering intent on the transfer curve."""

import pytest

from negpy.domain.models import WorkspaceConfig


def _load(**fields) -> WorkspaceConfig:
    return WorkspaceConfig.from_flat_dict(
        {"process_mode": "Transparency", "auto_exposure": True, "auto_normalize_contrast": True, **fields}
    )


def test_a_normalized_slide_meters_on_the_transfer_curve():
    cfg = _load(e6_normalize=True, auto_exposure=False)
    assert cfg.exposure.auto_exposure and cfg.exposure.auto_normalize_contrast


def test_a_raw_slide_keeps_its_inert_autos_off():
    cfg = _load(e6_normalize=False)
    assert not cfg.exposure.auto_exposure and not cfg.exposure.auto_normalize_contrast


def test_a_positive_frame_keeps_its_autos():
    cfg = _load(e6_normalize=False, positive_source=True)
    assert cfg.exposure.auto_exposure and cfg.exposure.auto_normalize_contrast


@pytest.mark.parametrize("mode", ["Color Negative", "B&W Negative"])
def test_a_negative_is_untouched(mode):
    cfg = WorkspaceConfig.from_flat_dict({"process_mode": mode, "e6_normalize": False, "auto_exposure": True})
    assert cfg.exposure.auto_exposure


def test_a_row_saved_after_the_retirement_is_untouched():
    """The key is gone after one save, so a later choice is never rewritten."""
    assert _load().exposure.auto_exposure
    assert "e6_normalize" not in _load(e6_normalize=True).to_dict()
