from datetime import date, datetime, time, timedelta, timezone

from negpy.features.metadata.capture_times import (
    AFTER_SCAN,
    ANCHORS_OUT_OF_ORDER,
    DST_CHANGE,
    NEXT_DAY,
    NO_DAY,
    NO_ZONE,
    Anchor,
    Segment,
    assign_times,
    day_of,
    segment_bounds,
)

FRAMES = tuple(f"F{n:02d}" for n in range(1, 8))
CAIRO = "Africa/Cairo"
EGYPT_DAY = date(2026, 10, 25)
PLUS_3 = timedelta(hours=3)


def _codes(issues) -> list[str]:
    return [issue.code for issue in issues]


def _clock(results) -> list[str]:
    return [r.local.strftime("%H:%M:%S") for r in results]


def _photo(hh: int, mm: int, ss: int = 0, offset: timedelta = PLUS_3, **kw) -> Anchor:
    return Anchor(local=datetime(2026, 10, 25, hh, mm, ss), offset=offset, ref=f"IMG_{hh}{mm}.HEIC", **kw)


class TestWithoutAnchors:
    def test_the_day_starts_at_noon_one_minute_apart(self) -> None:
        results, issues = assign_times(Segment(FRAMES[:4], day=EGYPT_DAY, zone=CAIRO))
        assert issues == []
        assert _clock(results) == ["12:00:00", "12:01:00", "12:02:00", "12:03:00"]
        assert {r.offset for r in results} == {PLUS_3}
        assert {r.source for r in results} == {"day"}
        assert results[0].capture_date_text() == "2026-10-25 12:00:00"

    def test_start_and_step_can_change(self) -> None:
        segment = Segment(FRAMES[:3], day=EGYPT_DAY, zone=CAIRO, start=time(9, 30), step=timedelta(minutes=5))
        assert _clock(assign_times(segment)[0]) == ["09:30:00", "09:35:00", "09:40:00"]

    def test_a_segment_without_a_day_is_an_error(self) -> None:
        results, issues = assign_times(Segment(FRAMES[:3], zone=CAIRO))
        assert NO_DAY in _codes(issues) and any(i.is_error for i in issues)
        assert all(r.local is None for r in results)

    def test_without_a_zone_there_is_no_offset(self) -> None:
        results, issues = assign_times(Segment(FRAMES[:2], day=EGYPT_DAY))
        assert _codes(issues) == [NO_ZONE]
        assert _clock(results) == ["12:00:00", "12:01:00"] and results[0].offset is None

    def test_a_late_start_moves_frames_to_the_next_day(self) -> None:
        results, issues = assign_times(Segment(FRAMES[:4], day=EGYPT_DAY, zone=CAIRO, start=time(23, 58)))
        next_day = [i for i in issues if i.code == NEXT_DAY]
        assert next_day and next_day[0].frames == ("F03", "F04")


class TestWithAnchors:
    def test_one_anchor_steps_two_minutes_each_way(self) -> None:
        segment = Segment(FRAMES[:5], day=EGYPT_DAY, zone=CAIRO, anchors={"F03": _photo(11, 9, 21)})
        results, _ = assign_times(segment)
        assert _clock(results) == ["11:05:21", "11:07:21", "11:09:21", "11:11:21", "11:13:21"]
        assert [r.source for r in results] == ["before", "before", "anchor", "after", "after"]
        assert results[2].anchor_ref == "IMG_119.HEIC"

    def test_frames_between_two_anchors_are_spread_evenly(self) -> None:
        segment = Segment(FRAMES[:5], day=EGYPT_DAY, zone=CAIRO, anchors={"F01": _photo(10, 0), "F05": _photo(10, 8)})
        results, issues = assign_times(segment)
        assert issues == []
        assert _clock(results) == ["10:00:00", "10:02:00", "10:04:00", "10:06:00", "10:08:00"]
        assert [r.source for r in results] == ["anchor", "between", "between", "between", "anchor"]

    def test_anchors_against_film_order_are_an_error(self) -> None:
        segment = Segment(FRAMES[:4], day=EGYPT_DAY, zone=CAIRO, anchors={"F01": _photo(11, 0), "F04": _photo(10, 0)})
        results, issues = assign_times(segment)
        assert [(i.code, i.frames) for i in issues] == [(ANCHORS_OUT_OF_ORDER, ("F04", "F01"))]
        assert all(r.local is None for r in results)

    def test_every_frame_gets_its_own_second(self) -> None:
        segment = Segment(FRAMES[:4], day=EGYPT_DAY, zone=CAIRO, anchors={"F01": _photo(10, 0, 0), "F04": _photo(10, 0, 2)})
        results, _ = assign_times(segment)
        instants = [r.local for r in results]
        assert all(b > a for a, b in zip(instants, instants[1:]))

    def test_a_phone_in_another_zone_keeps_the_same_moment(self) -> None:
        berlin_phone = _photo(9, 9, 21, offset=timedelta(hours=1))
        results, _ = assign_times(Segment(FRAMES[:1], day=EGYPT_DAY, zone=CAIRO, anchors={"F01": berlin_phone}))
        assert (_clock(results), results[0].offset) == (["11:09:21"], PLUS_3)

    def test_a_typed_anchor_is_local_time_in_the_zone(self) -> None:
        typed = Anchor(local=datetime(2026, 10, 25, 14, 0), source="typed")
        results, _ = assign_times(Segment(FRAMES[:2], day=EGYPT_DAY, zone=CAIRO, anchors={"F01": typed}))
        assert _clock(results) == ["14:00:00", "14:02:00"] and results[0].source == "typed"


class TestDaylightSaving:
    def test_the_spread_follows_real_time_across_the_change(self) -> None:
        # Berlin leaves summer time at 03:00 on 25 Oct 2026. 01:50 CEST to 02:10 CET is 80 minutes.
        before = Anchor(local=datetime(2026, 10, 25, 1, 50), offset=timedelta(hours=2))
        after = Anchor(local=datetime(2026, 10, 25, 2, 10), offset=timedelta(hours=1))
        segment = Segment(FRAMES[:5], day=EGYPT_DAY, zone="Europe/Berlin", anchors={"F01": before, "F05": after})
        results, _ = assign_times(segment)
        utc = [(r.local - r.offset).replace(tzinfo=timezone.utc) for r in results]
        gaps = {b - a for a, b in zip(utc, utc[1:])}
        assert gaps == {timedelta(minutes=20)}
        assert results[0].offset == timedelta(hours=2) and results[-1].offset == timedelta(hours=1)

    def test_a_typed_time_in_the_skipped_hour_moves_forward(self) -> None:
        typed = Anchor(local=datetime(2026, 3, 29, 2, 30), source="typed")
        results, issues = assign_times(Segment(FRAMES[:1], day=date(2026, 3, 29), zone="Europe/Berlin", anchors={"F01": typed}))
        assert DST_CHANGE in _codes(issues)
        assert (_clock(results), results[0].offset) == (["03:30:00"], timedelta(hours=2))


class TestPlace:
    def test_frames_take_the_nearest_gps_anchor(self) -> None:
        anchors = {
            "F02": _photo(10, 0, latitude=27.14, longitude=33.88),
            "F06": _photo(10, 8, latitude=27.09, longitude=33.85),
        }
        segment = Segment(FRAMES, day=EGYPT_DAY, zone=CAIRO, latitude=27.25, longitude=33.81, anchors=anchors)
        places = [(r.latitude, r.longitude) for r in assign_times(segment)[0]]
        near_first, near_second = (27.14, 33.88), (27.09, 33.85)
        assert places == [near_first, near_first, near_first, near_first, near_second, near_second, near_second]

    def test_without_gps_anchors_frames_take_the_segment_place(self) -> None:
        segment = Segment(FRAMES[:2], day=EGYPT_DAY, zone=CAIRO, latitude=27.25, longitude=33.81, anchors={"F01": _photo(10, 0)})
        assert {(r.latitude, r.longitude) for r in assign_times(segment)[0]} == {(27.25, 33.81)}


class TestChecks:
    def test_a_time_after_the_scan_is_flagged(self) -> None:
        scans = {"F02": datetime(2026, 10, 25, 12, 0, 30)}
        _, issues = assign_times(Segment(FRAMES[:2], day=EGYPT_DAY, zone=CAIRO), scan_times=scans)
        assert [(i.code, i.frames) for i in issues] == [(AFTER_SCAN, ("F02",))]


class TestReopen:
    def test_segments_are_runs_of_the_same_day_and_zone(self) -> None:
        d1, d2 = date(2026, 10, 25), date(2026, 10, 26)
        keys = [(d1, CAIRO), (d1, CAIRO), (d2, CAIRO), (d2, ""), (d2, "")]
        assert segment_bounds(keys) == [(0, 1), (2, 2), (3, 4)]

    def test_day_of_a_stored_date(self) -> None:
        assert day_of("2026-10-25 10:12:00") == date(2026, 10, 25)
        assert day_of("2026-10-25") == date(2026, 10, 25)
        assert day_of("2026-10") is None and day_of("") is None
