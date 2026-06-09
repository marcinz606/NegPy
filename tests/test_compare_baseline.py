"""
Before/After baseline config: resets only the creative sections, preserves the automatic
and structural parts (process bounds + mode, geometry/crop) so the inversion stays valid.
"""

from dataclasses import replace

from negpy.desktop.controller import baseline_compare_config
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import ExposureConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.lab.models import LabConfig
from negpy.features.process.models import ProcessMode
from negpy.features.retouch.models import RetouchConfig
from negpy.features.toning.models import ToningConfig


def _edited_config() -> WorkspaceConfig:
    cfg = WorkspaceConfig()
    # A graded edit with non-default creative sections + meaningful process/geometry.
    process = replace(
        cfg.process,
        process_mode=ProcessMode.BW,  # non-default (default is C41) to prove preservation
        local_floors=(0.1, 0.2, 0.3),
        local_ceils=(0.8, 0.9, 0.95),
        lock_bounds=True,
    )
    geometry = replace(cfg.geometry, fine_rotation=2.5)
    return replace(
        cfg,
        process=process,
        geometry=geometry,
        exposure=replace(cfg.exposure, density=0.7),
        lab=replace(cfg.lab, saturation=1.6),
        toning=replace(cfg.toning, selenium_strength=0.5),
        finish=replace(cfg.finish, vignette_strength=0.4),
    )


def test_baseline_resets_creative_sections() -> None:
    base = baseline_compare_config(_edited_config())
    assert base.exposure == ExposureConfig()
    assert base.lab == LabConfig()
    assert base.toning == ToningConfig()
    assert base.finish == FinishConfig()
    assert base.retouch == RetouchConfig()


def test_baseline_preserves_process_and_geometry() -> None:
    edited = _edited_config()
    base = baseline_compare_config(edited)
    # Normalization bounds + mode must survive so the auto conversion is valid.
    assert base.process == edited.process
    assert base.process.local_floors == (0.1, 0.2, 0.3)
    assert base.process.local_ceils == (0.8, 0.9, 0.95)
    # Same framing.
    assert base.geometry == edited.geometry
