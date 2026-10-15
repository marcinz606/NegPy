"""Square Before Crop: Auto Crop squares a freshly armed frame from its own edges, then crops
the squared frame, once per arm. A frame with its own rotation, Tilt or Swing keeps them."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.logic import autocrop_detection_key
from negpy.features.geometry.models import holds_own_skew
from negpy.services.rendering.image_processor import ArmedCrop, _resolve_armed_autocrop

IMG = np.zeros((40, 60, 3), np.float32)


def _armed(**geometry) -> WorkspaceConfig:
    base = WorkspaceConfig()
    fields = {"crop_from_auto": True, "crop_rect": None, "autocrop_square": True, **geometry}
    return replace(base, geometry=replace(base.geometry, **fields))


def _square(fine=1.5, cv=0.7, ch=0.0):
    def fit(_img, geometry, _mode):
        return replace(geometry, fine_rotation=fine, converge_v=cv, converge_h=ch), None

    return fit


def test_a_fresh_arm_is_squared_then_cropped_on_the_squared_frame():
    seen = []

    def crop(_img, geometry, _size):
        seen.append(geometry)
        return (0.1, 0.1, 0.9, 0.9)

    with (
        patch("negpy.services.rendering.image_processor.squared_geometry", side_effect=_square()),
        patch("negpy.services.rendering.image_processor.resolve_autocrop_rect", side_effect=crop),
    ):
        settings, resolved = _resolve_armed_autocrop(IMG, _armed())

    assert (seen[0].fine_rotation, seen[0].converge_v) == (1.5, 0.7)  # the crop saw the squared frame
    assert resolved.skew == (1.5, 0.7, 0.0)
    assert resolved.key == autocrop_detection_key(settings.geometry)
    assert settings.geometry.crop_rect == (0.1, 0.1, 0.9, 0.9)


def test_square_off_crops_as_before():
    with (
        patch("negpy.services.rendering.image_processor.squared_geometry") as fit,
        patch("negpy.services.rendering.image_processor.resolve_autocrop_rect", return_value=(0.1, 0.1, 0.9, 0.9)),
    ):
        _, resolved = _resolve_armed_autocrop(IMG, _armed(autocrop_square=False))
    fit.assert_not_called()
    assert resolved.skew is None


def test_a_frame_with_its_own_angle_is_not_squared():
    with (
        patch("negpy.services.rendering.image_processor.squared_geometry") as fit,
        patch("negpy.services.rendering.image_processor.resolve_autocrop_rect", return_value=(0.1, 0.1, 0.9, 0.9)),
    ):
        _, resolved = _resolve_armed_autocrop(IMG, _armed(fine_rotation=0.3))
    fit.assert_not_called()
    assert resolved.skew is None


def test_a_re_detect_of_a_squared_arm_keeps_the_angle_it_has():
    """A rect under a stale key (a ratio change) re-crops at the frame's angle."""
    config = _armed(autocrop_squared=True)
    config = replace(config, geometry=replace(config.geometry, crop_rect=(0.2, 0.2, 0.8, 0.8), crop_detect_key="old"))
    with (
        patch("negpy.services.rendering.image_processor.squared_geometry") as fit,
        patch("negpy.services.rendering.image_processor.resolve_autocrop_rect", return_value=(0.1, 0.1, 0.9, 0.9)),
    ):
        _resolve_armed_autocrop(IMG, config)
    fit.assert_not_called()


def test_an_untrusted_fit_crops_the_frame_as_it_is():
    with (
        patch("negpy.services.rendering.image_processor.squared_geometry", return_value=None),
        patch("negpy.services.rendering.image_processor.resolve_autocrop_rect", return_value=(0.1, 0.1, 0.9, 0.9)),
    ):
        settings, resolved = _resolve_armed_autocrop(IMG, _armed())
    assert resolved.skew is None and resolved.squared and settings.geometry.fine_rotation == 0.0


def test_a_failed_squaring_with_no_crop_still_reports_the_attempt():
    """So the freeze can mark the arm and the costly fit does not run on every render."""
    with (
        patch("negpy.services.rendering.image_processor.squared_geometry", return_value=None),
        patch("negpy.services.rendering.image_processor.resolve_autocrop_rect", return_value=None),
    ):
        settings, resolved = _resolve_armed_autocrop(IMG, _armed())
    assert resolved == ArmedCrop(None, autocrop_detection_key(settings.geometry), None, True)


def test_an_arm_already_squared_is_not_squared_again():
    """The freeze marked the arm: a hand reset of the angle to 0 is not undone."""
    config = _armed(autocrop_squared=True)
    with (
        patch("negpy.services.rendering.image_processor.squared_geometry") as fit,
        patch("negpy.services.rendering.image_processor.resolve_autocrop_rect", return_value=None),
    ):
        _, resolved = _resolve_armed_autocrop(IMG, config)
    fit.assert_not_called()
    assert resolved is None


def test_a_squaring_is_kept_when_no_crop_is_found():
    with (
        patch("negpy.services.rendering.image_processor.squared_geometry", side_effect=_square(fine=-0.8, cv=0.0)),
        patch("negpy.services.rendering.image_processor.resolve_autocrop_rect", return_value=None),
    ):
        settings, resolved = _resolve_armed_autocrop(IMG, _armed())
    assert resolved == ArmedCrop(None, autocrop_detection_key(settings.geometry), (-0.8, 0.0, 0.0), True)
    assert settings.geometry.crop_rect is None


def test_holds_own_skew_reads_any_of_the_three():
    g = WorkspaceConfig().geometry
    assert not holds_own_skew(g)
    for field in ("fine_rotation", "converge_v", "converge_h"):
        assert holds_own_skew(replace(g, **{field: 0.2}))


def test_the_square_flag_is_keyed_only_when_on():
    """Off, every auto crop saved before Square existed keeps its key; on, the crop is stale."""
    g = WorkspaceConfig().geometry
    assert not autocrop_detection_key(g).endswith("square")
    assert autocrop_detection_key(replace(g, autocrop_square=True)) == autocrop_detection_key(g) + "|square"


def test_turning_square_on_squares_and_re_crops_a_cropped_frame():
    config = WorkspaceConfig()
    cropped = replace(config.geometry, crop_from_auto=True, crop_rect=(0.2, 0.2, 0.8, 0.8))
    cropped = replace(cropped, crop_detect_key=autocrop_detection_key(cropped))
    config = replace(config, geometry=replace(cropped, autocrop_square=True))
    with (
        patch("negpy.services.rendering.image_processor.squared_geometry", side_effect=_square()) as fit,
        patch("negpy.services.rendering.image_processor.resolve_autocrop_rect", return_value=(0.1, 0.1, 0.9, 0.9)),
    ):
        settings, resolved = _resolve_armed_autocrop(IMG, config)
    fit.assert_called_once()
    assert resolved.skew == (1.5, 0.7, 0.0) and resolved.rect == (0.1, 0.1, 0.9, 0.9)
    assert settings.geometry.crop_detect_key == autocrop_detection_key(settings.geometry)


def test_an_export_fits_once():
    """The export resolves twice; the first marks the settings squared, so the second does not fit."""
    with (
        patch("negpy.services.rendering.image_processor.squared_geometry", side_effect=_square()) as fit,
        patch("negpy.services.rendering.image_processor.resolve_autocrop_rect", return_value=None),
    ):
        settings, _ = _resolve_armed_autocrop(IMG, _armed())
        assert settings.geometry.autocrop_squared
        _resolve_armed_autocrop(IMG, settings)
    fit.assert_called_once()


def test_the_roll_run_takes_the_fits_tilt_and_swing_and_leaves_rotation_to_its_own_deskew():
    """The run's pooled deskew sets rotation; only the keystone comes from the squaring fit."""
    from negpy.desktop.workers.render import BatchAutoCropInput, BatchAutoCropTask, BatchAutoCropWorker
    from negpy.features.geometry.batch_autocrop import CropEvidence

    worker = BatchAutoCropWorker(preview_service=None)
    config = _armed()
    frame = BatchAutoCropInput(file_info={"hash": "h1", "name": "f1"}, config=config, fingerprint=())
    task = BatchAutoCropTask(frames=[frame], workspace_color_space="Adobe RGB", generation=1)
    geometries = []

    class Recorder:
        def __init__(self, geometry):
            geometries.append(geometry)

        def process(self, image, _context):
            return image

    def detect(key, image, **_kw):
        return CropEvidence(key=key, canvas_shape=image.shape[:2], roi=None, correction_angle=0.25, confidence=0.9)

    with (
        patch.object(worker, "_decode", return_value=(np.zeros((40, 60, 3), np.float32), {})),
        patch("negpy.desktop.workers.render.squared_geometry", side_effect=_square(fine=2.0, cv=0.5, ch=-0.3)),
        patch("negpy.desktop.workers.render.GeometryProcessor", Recorder),
        patch("negpy.desktop.workers.render.detect_crop_candidate", side_effect=detect),
    ):
        evidence, keystone = worker._frame_evidence(0, frame, task, 1)

    assert keystone == (0.5, -0.3)
    assert (geometries[0].fine_rotation, geometries[0].converge_v, geometries[0].converge_h) == (0.0, 0.5, -0.3)
    assert evidence.correction_angle == 0.25  # the run's own residual, unchanged

    hand = replace(config, geometry=replace(config.geometry, fine_rotation=0.4))
    with (
        patch.object(worker, "_decode", return_value=(np.zeros((40, 60, 3), np.float32), {})),
        patch("negpy.desktop.workers.render.squared_geometry") as fit,
        patch("negpy.desktop.workers.render.detect_crop_candidate", side_effect=detect),
    ):
        _, keystone = worker._frame_evidence(0, replace(frame, config=hand), task, 1)
    fit.assert_not_called()
    assert keystone is None


def test_a_new_arm_clears_the_squared_mark():
    from negpy.features.geometry.models import with_auto_crop

    config = _armed(autocrop_squared=True)
    assert not with_auto_crop(config, True).geometry.autocrop_squared
