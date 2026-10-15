"""NkscanRollSession: measure the strip once, preview what it found."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from negpy.infrastructure.scanners.base import ScannerDevice
from negpy.infrastructure.scanners.nkscan_roll import NkscanRollSession
from tests.scanners import fake_nkscan
from tests.scanners.fake_nkscan import DEVICE_ID, FRAMES, make_backend

_PITCH_MM = 36.0  # max_area_mm[1] on the nkscan caps: the fallback pitch


def _roll(**kwargs) -> tuple[NkscanRollSession, object, ScannerDevice]:
    backend, module = make_backend(**kwargs)
    device = backend.list_devices()[0]
    return backend.open_roll(device, dpi=500), module, device


def _previews(session, slots=(1, 2, 3), cancel=None):
    return list(session.preview(slots, cancel=cancel or threading.Event()))


# ── discovery ─────────────────────────────────────────────────────────────


def test_the_strip_is_measured_once_for_every_slot() -> None:
    session, module, _ = _roll()
    previews = _previews(session)

    assert [p.slot for p in previews] == [1, 2, 3]
    assert len(module.opened[-1].discoveries) == 1
    assert session.slot_count == 3


def test_the_detected_rects_are_cached_for_the_fine_scan() -> None:
    backend, _module = make_backend()
    device = backend.list_devices()[0]
    session = backend.open_roll(device, dpi=500)
    _previews(session)
    session.close()

    assert backend.frames(DEVICE_ID) == list(FRAMES)


def test_previewing_the_whole_strip_scans_nothing() -> None:
    """The measurement already read the film; a preview is a slice of that pass."""
    session, module, _ = _roll()
    previews = _previews(session)

    assert all(p.rgb is not None for p in previews)
    assert module.opened[-1].scans == []


def test_each_preview_is_cut_from_its_own_part_of_the_strip() -> None:
    session, _module, _ = _roll()
    previews = _previews(session)

    # The fake marks every frame's band with its slot number.
    assert [int(p.rgb[0, 0, 0]) for p in previews] == [1, 2, 3]
    assert previews[0].rgb.shape[2] == 3


def test_the_strip_pass_is_kept_for_the_operator() -> None:
    session, _module, _ = _roll()
    _previews(session, (1,))
    assert session.thumbnail is not None and session.thumbnail.shape[2] == 3


def test_a_mechanism_that_takes_no_thumbnail_scans_each_frame_instead() -> None:
    """A masked holder publishes its geometry, so nothing measures a strip pass to cut from."""
    session, module, _ = _roll(thumbnail=False)
    previews = _previews(session)

    assert session.thumbnail is None
    assert [p.rgb is not None for p in previews] == [True, True, True]
    assert [s["frame"] for s in module.opened[-1].scans] == list(FRAMES)
    asked = [s["exposures"] for s in module.opened[-1].scans]
    assert asked[0] is None and asked[1] == asked[2] == {"red": 1, "green": 2, "blue": 3}
    assert all(s["samples"] == 1 and not s["infrared"] and not s["clean"] for s in module.opened[-1].scans)


def test_the_film_format_reaches_the_measurement() -> None:
    backend, module = make_backend()
    device = backend.list_devices()[0]
    session = backend.open_roll(device, dpi=500, film_format="66")
    _previews(session, (1,))
    assert module.opened[-1].discoveries == ["66"]


def test_an_empty_strip_yields_nothing_rather_than_failing() -> None:
    session, _module, _ = _roll(frames=())
    assert _previews(session) == []


def test_slots_past_the_detected_count_are_skipped() -> None:
    session, _module, _ = _roll()
    assert [p.slot for p in _previews(session, (1, 2, 3, 4, 5))] == [1, 2, 3]


# ── boundaries ────────────────────────────────────────────────────────────


def test_every_boundary_needs_a_look_until_it_is_approved() -> None:
    session, _module, _ = _roll()
    assert _previews(session, (1,))[0].needs_approval is True

    session.approve(1)
    assert _previews(session, (1,))[0].needs_approval is False


def test_an_offset_slides_the_rect_by_the_same_film_distance_the_scan_uses() -> None:
    session, module, _ = _roll(thumbnail=False)  # the fallback shows the rect it asked for
    session.set_offset(1, 25.4 / _PITCH_MM)  # one inch of film
    _previews(session, (1,))

    assert module.opened[-1].scans[0]["frame"][0] == FRAMES[0][0] + 4000


def test_an_absolute_rect_can_be_re_addressed_backwards() -> None:
    session, _module, _ = _roll()
    assert session.offset_range == (-1.0, 1.0)
    _previews(session, (2,))  # measure first, so the offset only moves the slice

    session.set_offset(2, -1.0)
    moved = _previews(session, (2,))[0]
    # One pitch back off frame 2 lands on frame 1's band.
    assert int(moved.rgb[0, 0, 0]) == 1


def test_an_offset_beyond_the_range_is_clamped() -> None:
    session, _module, _ = _roll()
    session.set_offset(1, 4.0)
    assert _previews(session, (1,))[0].offset < 1.0


# ── failure and cancellation ──────────────────────────────────────────────


def test_one_failed_slot_does_not_cost_the_rest_of_the_strip() -> None:
    session, module, _ = _roll(thumbnail=False)
    previews = []
    for slot in (1, 2, 3):
        module.scan_error = fake_nkscan.MediaError("lost focus") if slot == 2 else None
        previews += _previews(session, (slot,))

    assert [p.slot for p in previews] == [1, 2, 3]
    assert previews[1].error == "lost focus" and previews[1].rgb is None
    assert previews[2].rgb is not None


def test_a_cancel_stops_the_strip_where_it_is() -> None:
    session, _module, _ = _roll()
    cancel = threading.Event()
    seen = []
    for preview in session.preview((1, 2, 3), cancel=cancel):
        seen.append(preview.slot)
        cancel.set()

    assert seen == [1]


def test_a_pre_set_cancel_never_touches_the_unit() -> None:
    session, module, _ = _roll()
    cancel = threading.Event()
    cancel.set()

    assert _previews(session, (1, 2, 3), cancel=cancel) == []
    assert module.opened[-1].scans == []


# ── lifetime ──────────────────────────────────────────────────────────────


def test_close_keeps_the_unit_open_and_is_idempotent() -> None:
    session, module, _ = _roll()
    session.close()
    session.close()
    assert not module.opened[-1].closed


def test_a_closed_strip_refuses_a_preview() -> None:
    session, _module, _ = _roll()
    session.close()
    with pytest.raises(RuntimeError, match="is closed"):
        _previews(session, (1,))


def test_a_second_strip_session_re_slices_rather_than_reading_the_film_again() -> None:
    """Nudging an offset and previewing again must not cost another pass over the strip."""
    backend, module = make_backend()
    device = backend.list_devices()[0]

    first = backend.open_roll(device, dpi=500)
    _previews(first)
    first.close()
    reads = sum(len(s.discoveries) for s in module.opened)
    scans = sum(len(s.scans) for s in module.opened)

    second = backend.open_roll(device, dpi=500)
    second.set_offset(2, 0.05)
    previews = _previews(second)
    second.close()

    assert [p.slot for p in previews] == [1, 2, 3]
    assert sum(len(s.discoveries) for s in module.opened) == reads
    assert sum(len(s.scans) for s in module.opened) == scans


def test_ejecting_forgets_the_strip_pass_because_that_film_is_gone() -> None:
    backend, module = make_backend(with_eject=True)
    device = backend.list_devices()[0]
    _previews(backend.open_roll(device, dpi=500))
    assert backend.strip_pass(device.id) is not None

    backend.eject(device.id)
    assert backend.strip_pass(device.id) is None


def test_a_tile_is_cut_at_the_address_the_scan_will_use() -> None:
    """The tile uses the column addresses the pass reports, not optical over thumbnail dpi."""
    backend, module = make_backend()
    device = backend.list_devices()[0]
    session = backend.open_roll(device, dpi=500)
    try:
        previews = _previews(session)
    finally:
        session.close()

    columns = backend.column_addresses(device.id)
    strip = backend.strip_pass(device.id)
    assert strip is not None and columns is not None
    for preview, (slot, rect) in zip(previews, enumerate(module.frames, 1)):
        expected = strip[:, np.searchsorted(columns, rect[0]) : np.searchsorted(columns, rect[2])]
        assert preview.rgb.shape[1] == expected.shape[1]
        # Every band carries its own slot, so a tile cut at the wrong column opens on the gap.
        assert set(np.unique(preview.rgb[:, 0])) == {slot}
        assert set(np.unique(preview.rgb)) <= {0, slot}


def test_a_tile_is_cut_from_unevenly_spaced_columns() -> None:
    """A unit that counts perforations resamples its pass, so its columns are not evenly spaced."""
    from negpy.infrastructure.scanners.nkscan_roll import slice_frame

    columns = np.array([0, 40, 82, 121, 168, 205, 250, 290])
    strip = np.arange(len(columns))[None, :, None].repeat(2, axis=0)

    tile = slice_frame(strip, (82, 0, 205, 10), columns)

    assert tile is not None
    assert list(tile[0, :, 0]) == [2, 3, 4]


def test_a_rect_past_the_strip_pass_stops_at_its_last_column() -> None:
    from negpy.infrastructure.scanners.nkscan_roll import slice_frame

    columns = np.array([0, 40, 82, 121])
    strip = np.arange(len(columns))[None, :, None].repeat(2, axis=0)

    tile = slice_frame(strip, (82, 0, 400, 10), columns)

    assert tile is not None and list(tile[0, :, 0]) == [2, 3]
    assert slice_frame(strip, (82, 0, 121, 10), None) is None


def test_a_frame_before_the_first_column_starts_at_the_first_column() -> None:
    from negpy.infrastructure.scanners.nkscan_roll import slice_frame

    columns = np.array([126, 168, 210, 252, 294])
    strip = np.arange(len(columns))[None, :, None].repeat(2, axis=0)

    tile = slice_frame(strip, (84, 0, 252, 10), columns)

    assert tile is not None
    assert list(tile[0, :, 0]) == [0, 1, 2]


def test_a_re_cut_of_a_measured_strip_opens_no_unit() -> None:
    """A nudge re-cuts from the strip pass, as Nikon Scan does; the unit is not touched."""
    backend, module = make_backend()
    device = backend.list_devices()[0]
    first = backend.open_roll(device, dpi=500)
    _previews(first)
    first.close()
    opened = len(module.opened)

    again = backend.open_roll(device, dpi=500)
    again.set_offset(2, 0.1)
    previews = _previews(again, slots=(2,))
    again.close()

    assert previews[0].rgb is not None
    assert len(module.opened) == opened


def test_a_nudge_past_the_strip_pass_is_held_at_its_end_and_not_scanned() -> None:
    backend, module = make_backend()
    device = backend.list_devices()[0]
    session = backend.open_roll(device, dpi=500)
    _previews(session)

    session.set_offset(3, 1.0)
    (preview,) = _previews(session, slots=(3,))
    session.close()

    assert preview.rgb is not None
    assert preview.offset < 1.0
    assert all(not s.scans for s in module.opened)


def test_the_strip_pass_reports_its_progress() -> None:
    """The dialog shows the progress of the strip pass, which is the long part of a preview."""
    session, _module, _ = _roll(progress_steps=0)
    seen = []
    list(session.preview((1,), cancel=threading.Event(), progress=lambda fraction, phase: seen.append(phase)))
    session.close()
    assert "Detecting frames" in seen


def test_a_cut_moves_the_tile_without_the_unit() -> None:
    """A nudge crops the strip pass again. The unit does nothing."""
    session, module, _ = _roll()
    first = _previews(session, slots=(2,))[0]
    unit = module.opened[-1]
    before = (len(unit.scans), len(unit.discoveries), unit.staged)

    assert first.cut is not None
    moved = first.cut(0.1)

    assert moved is not None and moved.slot == 2 and moved.offset > 0
    assert moved.rgb.shape == first.rgb.shape and not (moved.rgb == first.rgb).all()
    assert (len(unit.scans), len(unit.discoveries), unit.staged) == before
    assert moved.cut is not None


def test_without_a_strip_pass_a_preview_has_no_cut() -> None:
    session, _module, _ = _roll(thumbnail=False)
    assert all(p.cut is None for p in _previews(session))
