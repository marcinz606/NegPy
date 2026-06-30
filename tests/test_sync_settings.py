"""Pure-merge logic for the Files sidebar "Apply to selected" bulk action."""

from dataclasses import replace

from negpy.desktop.session import _source_effective_bounds, build_synced_config
from negpy.domain.models import WorkspaceConfig

_BOUNDS = ((0.11, 0.22, 0.33), (0.88, 0.77, 0.66))


def _source() -> WorkspaceConfig:
    c = WorkspaceConfig()
    return replace(
        c,
        exposure=replace(c.exposure, density=2.0, grade=130.0),
        lab=replace(c.lab, saturation=1.5),
        toning=replace(c.toning, sepia_strength=0.4),
        finish=replace(c.finish, vignette_strength=0.3),
        process=replace(c.process, process_mode="E-6", analysis_buffer=0.2),
        geometry=replace(c.geometry, rotation=90, flip_horizontal=True, manual_crop_rect=(0.1, 0.1, 0.9, 0.9)),
        retouch=replace(c.retouch, dust_threshold=0.5, manual_dust_spots=[(0.5, 0.5, 0.01)]),
    )


def _target() -> WorkspaceConfig:
    c = WorkspaceConfig()
    return replace(
        c,
        geometry=replace(c.geometry, rotation=270, manual_crop_rect=(0.2, 0.2, 0.8, 0.8)),
        retouch=replace(c.retouch, manual_dust_spots=[(0.1, 0.1, 0.02)]),
        process=replace(c.process, local_floors=(0.05, 0.05, 0.05), local_ceils=(0.95, 0.95, 0.95)),
    )


def test_crop_only_copies_crop_keeps_rotation_and_flips():
    src, tgt = _source(), _target()
    out = build_synced_config(src, tgt, frozenset({"crop"}), None)
    assert out.geometry.manual_crop_rect == src.geometry.manual_crop_rect
    assert out.geometry.rotation == tgt.geometry.rotation  # rotation preserved
    assert out.geometry.flip_horizontal == tgt.geometry.flip_horizontal  # flip preserved
    assert out.exposure == tgt.exposure


def test_rotation_only_copies_rotation_and_flips_keeps_crop():
    src, tgt = _source(), _target()
    out = build_synced_config(src, tgt, frozenset({"rotation"}), None)
    assert out.geometry.rotation == src.geometry.rotation
    assert out.geometry.flip_horizontal == src.geometry.flip_horizontal  # flips ride with rotation
    assert out.geometry.manual_crop_rect == tgt.geometry.manual_crop_rect  # crop preserved
    assert out.exposure == tgt.exposure


def test_crop_and_rotation_together_cover_full_geometry():
    src, tgt = _source(), _target()
    out = build_synced_config(src, tgt, frozenset({"crop", "rotation"}), None)
    assert out.geometry == src.geometry


def test_process_aspect_copies_setup_keeps_bounds():
    src, tgt = _source(), _target()
    out = build_synced_config(src, tgt, frozenset({"process"}), None)
    assert out.process.process_mode == src.process.process_mode
    assert out.process.analysis_buffer == src.process.analysis_buffer
    # bounds-related fields stay the target's own
    assert out.process.local_floors == tgt.process.local_floors
    assert out.process.local_ceils == tgt.process.local_ceils
    assert out.process.locked_floors == tgt.process.locked_floors
    assert out.process.locked_ceils == tgt.process.locked_ceils
    assert out.process.use_luma_average == tgt.process.use_luma_average
    assert out.process.use_colour_average == tgt.process.use_colour_average
    assert out.geometry == tgt.geometry


def test_exposure_aspect_copies_exposure_only():
    src, tgt = _source(), _target()
    out = build_synced_config(src, tgt, frozenset({"exposure"}), None)
    assert out.exposure == src.exposure
    assert out.lab == tgt.lab
    assert out.geometry == tgt.geometry


def test_color_aspect_copies_lab_and_toning():
    src, tgt = _source(), _target()
    out = build_synced_config(src, tgt, frozenset({"color"}), None)
    assert out.lab == src.lab
    assert out.toning == src.toning
    assert out.exposure == tgt.exposure


def test_finish_aspect_copies_retouch_and_finish_keeps_dust_spots():
    src, tgt = _source(), _target()
    out = build_synced_config(src, tgt, frozenset({"finish"}), None)
    assert out.finish == src.finish
    assert out.retouch.dust_threshold == src.retouch.dust_threshold
    assert out.retouch.manual_dust_spots == tgt.retouch.manual_dust_spots  # frame-specific, preserved
    assert out.geometry == tgt.geometry


def test_all_aspects_checked_keeps_frame_specifics_and_untouched_fields():
    src, tgt = _source(), _target()
    aspects = frozenset({"process", "crop", "rotation", "exposure", "color", "finish", "bounds_luma", "bounds_colour"})
    out = build_synced_config(src, tgt, aspects, _BOUNDS)
    assert out.geometry == src.geometry
    assert out.exposure == src.exposure
    assert out.lab == src.lab
    assert out.toning == src.toning
    assert out.finish == src.finish
    assert out.process.process_mode == src.process.process_mode
    assert out.retouch.manual_dust_spots == tgt.retouch.manual_dust_spots
    assert out.process.local_floors == tgt.process.local_floors  # per-frame meter preserved
    assert out.process.locked_floors == _BOUNDS[0]
    assert out.process.locked_ceils == _BOUNDS[1]
    assert out.process.use_luma_average and out.process.use_colour_average
    # not a sync category: stays the target's own
    assert out.flatfield == tgt.flatfield
    assert out.rgbscan == tgt.rgbscan
    assert out.metadata == tgt.metadata


def test_bounds_both_only_changes_baseline():
    src, tgt = _source(), _target()
    out = build_synced_config(src, tgt, frozenset({"bounds_luma", "bounds_colour"}), _BOUNDS)
    assert out.exposure == tgt.exposure
    assert out.lab == tgt.lab
    assert out.geometry == tgt.geometry
    assert out.process.local_floors == tgt.process.local_floors  # per-frame meter untouched
    assert out.process.locked_floors == _BOUNDS[0]
    assert out.process.locked_ceils == _BOUNDS[1]
    assert out.process.use_luma_average is True
    assert out.process.use_colour_average is True


def test_bounds_luma_forces_other_axis_off():
    out = build_synced_config(_source(), _target(), frozenset({"bounds_luma"}), _BOUNDS)
    assert out.process.locked_floors == _BOUNDS[0]
    assert out.process.use_luma_average is True
    assert out.process.use_colour_average is False


def test_bounds_colour_forces_other_axis_off():
    out = build_synced_config(_source(), _target(), frozenset({"bounds_colour"}), _BOUNDS)
    assert out.process.locked_ceils == _BOUNDS[1]
    assert out.process.use_colour_average is True
    assert out.process.use_luma_average is False


def test_source_effective_bounds_prefers_per_frame_meter():
    p = replace(WorkspaceConfig().process, local_floors=(0.1, 0.2, 0.3), local_ceils=(0.9, 0.8, 0.7))
    assert _source_effective_bounds(p) == ((0.1, 0.2, 0.3), (0.9, 0.8, 0.7))


def test_source_effective_bounds_uses_roll_baseline_when_active():
    p = replace(
        WorkspaceConfig().process,
        locked_floors=(0.4, 0.5, 0.6),
        locked_ceils=(0.7, 0.6, 0.5),
        use_luma_average=True,
    )
    assert _source_effective_bounds(p) == ((0.4, 0.5, 0.6), (0.7, 0.6, 0.5))


def test_source_effective_bounds_none_when_unanalysed():
    assert _source_effective_bounds(WorkspaceConfig().process) is None
