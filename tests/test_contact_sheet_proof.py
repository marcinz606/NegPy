"""Straight Proof: one exposure and grade for the roll, with scan exposures evened out."""

import math
import os
import tempfile
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from negpy.domain.models import WorkspaceConfig
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.simulated.images import negative
from negpy.services.export import contact_sheet_roll as roll
from negpy.services.export.contact_sheet_roll import (
    PROOF_GRADE,
    FrameFacts,
    SheetFrame,
    read_scan_ev,
    scan_exposure_offsets,
    scene_breaks,
    scene_order,
    straight_proof,
    straight_proof_config,
    tile_params,
)
from negpy.services.rendering.image_processor import ImageProcessor

BASELINE = {"floors": (-2.0, -2.1, -2.2), "ceils": (-0.4, -0.5, -0.6)}


def _frame(name="a.nef", ev=None, mode=ProcessMode.C41, positive=False, **exposure):
    config = WorkspaceConfig()
    config = replace(
        config,
        process=replace(config.process, process_mode=mode, positive_source=positive),
        exposure=replace(config.exposure, **exposure) if exposure else config.exposure,
    )
    return SheetFrame({"name": name, "path": f"/r/{name}", "hash": name}, config, FrameFacts(scan_ev=ev))


def test_proof_config_prints_one_exposure_for_the_roll():
    edited = _frame(density=1.4, grade=90.0, auto_exposure=True, paper_black=True)
    burn = LocalMask(vertices=((0.1, 0.1), (0.5, 0.1), (0.5, 0.5)), stops=0.5)
    edited = replace(
        edited,
        config=replace(
            edited.config, local=LocalAdjustmentsConfig(masks=(burn,)), toning=replace(edited.config.toning, sepia_strength=0.5)
        ),
    )
    config = straight_proof_config(edited.config, BASELINE["floors"], BASELINE["ceils"], 0.3)
    exposure = config.exposure
    assert (exposure.auto_exposure, exposure.auto_normalize_contrast, exposure.cast_removal_strength) == (False, False, 0.0)
    assert exposure.grade == PROOF_GRADE and exposure.density == WorkspaceConfig().exposure.density
    assert exposure.paper_black is True
    process = config.process
    assert process.use_luma_average and process.use_color_average and not process.use_cast_average
    assert process.locked_floors == pytest.approx((-1.7, -1.8, -1.9))
    assert process.locked_ceils == pytest.approx((-0.1, -0.2, -0.3))
    assert config.local == WorkspaceConfig().local and config.toning == WorkspaceConfig().toning


def test_a_slide_keeps_its_own_window():
    config = straight_proof_config(_frame(mode=ProcessMode.E6).config, BASELINE["floors"], BASELINE["ceils"], 0.3)
    assert not config.process.use_luma_average
    assert config.exposure.auto_exposure is False


def test_scan_exposures_are_brought_to_the_roll_median():
    frames = [_frame(ev=10.0), _frame(ev=11.0), _frame(ev=12.0), _frame(ev=None)]
    shifts = scan_exposure_offsets(frames)
    assert shifts == pytest.approx([-math.log10(2), 0.0, math.log10(2), 0.0])
    assert scan_exposure_offsets([_frame(), _frame()]) == [0.0, 0.0]


def test_positives_cannot_be_proofed():
    proof = straight_proof([_frame(mode=ProcessMode.E6, positive=True), _frame()], BASELINE)
    assert not proof.available and "positives" in proof.reason


def test_negatives_need_the_roll_analysis():
    proof = straight_proof([_frame()], None)
    assert not proof.available and "Roll Analysis" in proof.reason
    slides = straight_proof([_frame(mode=ProcessMode.E6)], None)
    assert slides.available


def test_notes_say_what_the_proof_assumes_or_corrected():
    assert "one exposure" in straight_proof([_frame(), _frame()], BASELINE).note
    uneven = straight_proof([_frame(ev=10.0), _frame(ev=11.5), _frame(ev=None)], BASELINE)
    assert "differ by 1.5 stops" in uneven.note and "1 frames state no exposure" in uneven.note
    even = straight_proof([_frame(ev=10.0), _frame(ev=10.1)], BASELINE)
    assert "alike" in even.note
    shifts = [f.config.process.locked_floors[0] - BASELINE["floors"][0] for f in uneven.frames]
    assert shifts[0] < 0 < shifts[1] and shifts[2] == 0.0


def test_scan_ev_is_read_from_camera_raws_only(monkeypatch):
    class _Raw:
        other = SimpleNamespace(shutter_speed=1 / 125, iso_speed=100.0, aperture=8.0)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    import rawpy

    monkeypatch.setattr(rawpy, "imread", lambda path: _Raw())
    assert read_scan_ev("/r/a.NEF") == pytest.approx(math.log2((1 / 125) * 100 / 64))
    assert read_scan_ev("/r/a.tif") is None
    assert read_scan_ev("/r/a.jpg") is None
    monkeypatch.setattr(rawpy, "imread", lambda path: (_ for _ in ()).throw(OSError("unreadable")))
    assert read_scan_ev("/r/a.arw") is None
    assert roll.read_frame_facts({"path": "/r/missing.arw"}).scan_ev is None


def test_a_brighter_scan_prints_the_same_once_its_exposure_is_evened_out():
    """The proof's claim through the real pipeline: a negative scanned a stop brighter prints
    darker at the shared exposure, and prints alike with its shift."""
    rgb, _ir = negative(400, 600, 3)
    linear = rgb.astype(np.float64) / 65535.0
    darker = np.clip(linear * 0.45 / linear.max(), 0, 1)
    brighter = np.clip(darker * 2.0, 0, 1)
    log = np.log10(np.clip(darker, 1e-6, 1)).reshape(-1, 3)
    floors, ceils = tuple(np.percentile(log, 0.5, axis=0)), tuple(np.percentile(log, 99.5, axis=0))
    processor = ImageProcessor(use_gpu=False)
    with tempfile.TemporaryDirectory() as folder:
        paths = []
        for name, data in (("a.tif", darker), ("b.tif", brighter)):
            path = os.path.join(folder, name)
            tifffile.imwrite(path, (data * 65535).astype(np.uint16), photometric="rgb")
            paths.append(path)

        def render(path, shift):
            config = tile_params(straight_proof_config(WorkspaceConfig(), floors, ceils, shift))
            return processor.render_display_array(path, config, path, target_long_px=200, prefer_gpu=False).astype(float)

        reference = render(paths[0], 0.0)
        evened = render(paths[1], math.log10(2.0))
        as_scanned = render(paths[1], 0.0)
    assert np.abs(reference - evened).mean() < 1.0
    assert as_scanned.mean() < reference.mean() - 10


def _in_scene(frame, ordinal, scene_id):
    return replace(frame, asset={**frame.asset, "scene": (ordinal, scene_id, f"Scene {ordinal}")})


def test_scene_order_groups_scenes_and_puts_loose_frames_last():
    frames = [_frame("a"), _in_scene(_frame("b"), 2, "s2"), _in_scene(_frame("c"), 1, "s1"), _in_scene(_frame("d"), 2, "s2"), _frame("e")]
    order = scene_order(frames)
    assert [frames[i].name for i in order] == ["c", "b", "d", "a", "e"]
    assert scene_breaks([frames[i] for i in order]) == [1, 3]


def test_each_scene_proofs_at_its_own_metering():
    s1 = {"floors": (-3.0, -3.0, -3.0), "ceils": (-1.0, -1.0, -1.0)}
    frames = [_in_scene(_frame("a"), 1, "s1"), _in_scene(_frame("b"), 2, "s2"), _frame("c")]
    proof = straight_proof(frames, BASELINE, {"s1": s1, "s2": None})
    assert proof.available
    floors = [f.config.process.locked_floors for f in proof.frames]
    assert floors == [(-3.0, -3.0, -3.0), BASELINE["floors"], BASELINE["floors"]]
    assert "own metering" in proof.note and "1 scenes have none" in proof.note


def test_scene_metering_needs_some_analysis():
    proof = straight_proof([_in_scene(_frame("a"), 1, "s1")], None, {"s1": None})
    assert not proof.available and "Scene Analysis" in proof.reason


def test_scan_exposures_are_evened_out_within_each_scene():
    s1 = {"floors": (-3.0, -3.0, -3.0), "ceils": (-1.0, -1.0, -1.0)}
    s2 = {"floors": (-2.0, -2.0, -2.0), "ceils": (-0.5, -0.5, -0.5)}
    frames = [_in_scene(_frame("a", ev=10.0), 1, "s1"), _in_scene(_frame("b", ev=10.0), 1, "s1"), _in_scene(_frame("c", ev=12.0), 2, "s2")]
    proof = straight_proof(frames, BASELINE, {"s1": s1, "s2": s2})
    # Each scene's metering was taken at its own scan exposure, so a lone brighter scene is not moved.
    assert [f.config.process.locked_floors[0] for f in proof.frames] == [-3.0, -3.0, -2.0]
