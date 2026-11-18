from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from negpy.features.metadata.capture_times import (
    AFTER_SCAN,
    DST_CHANGE,
    LONG_GAP_BETWEEN,
    NEXT_DAY,
    NO_DAY,
    NO_ZONE,
    OUT_OF_ORDER,
    Reference,
    Segment,
    assign_times,
    day_of,
)

FRAMES = tuple(f"F{n:02d}" for n in range(1, 8))
CAIRO = "Africa/Cairo"
EGYPT_DAY = date(2026, 10, 25)
PLUS_3 = timedelta(hours=3)
HURGHADA = dict(latitude=27.14, longitude=33.88)


def _codes(issues) -> list[str]:
    return [issue.code for issue in issues]


def _clock(results) -> list[str]:
    return [r.local.strftime("%H:%M:%S") for r in results]


def _photo(hh: int, mm: int, ss: int = 0, offset: timedelta = PLUS_3, day: date = EGYPT_DAY, **kw) -> Reference:
    return Reference(name=f"IMG_{hh:02d}{mm:02d}{ss:02d}.HEIC", local=datetime.combine(day, time(hh, mm, ss)), offset=offset, **kw)


def _segment(n: int, *references: tuple[int, Reference], **kw) -> Segment:
    return Segment(FRAMES[:n], day=EGYPT_DAY, zone=CAIRO, references=references, **kw)


class TestWithoutReferences:
    def test_the_day_starts_at_noon_one_minute_apart(self) -> None:
        results, issues = assign_times(_segment(4))
        assert issues == []
        assert _clock(results) == ["12:00:00", "12:01:00", "12:02:00", "12:03:00"]
        assert {r.offset for r in results} == {PLUS_3}
        assert {r.source for r in results} == {"start"}
        assert results[0].capture_date_text() == "2026-10-25 12:00:00"

    def test_start_and_step_can_change(self) -> None:
        segment = _segment(3, start=time(9, 30), step=timedelta(minutes=5))
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
        results, issues = assign_times(_segment(4, start=time(23, 58)))
        assert _codes(issues) == [NEXT_DAY] and issues[0].frames == ("F03", "F04")


class TestReferencesBetweenFrames:
    def test_one_reference_steps_two_minutes_each_way(self) -> None:
        photo = _photo(10, 12)
        results, issues = assign_times(_segment(4, (2, photo)))
        assert issues == []
        assert _clock(results) == ["10:08:00", "10:10:00", "10:14:00", "10:16:00"]
        assert [r.source for r in results] == ["before", "before", "after", "after"]
        assert results[1].basis == (photo.name,) and results[1].metadata_fields()["capture_anchor_ref"] == photo.name

    def test_frames_between_two_references_are_spread_by_position(self) -> None:
        first, second = _photo(10, 0), _photo(10, 40)
        results, _ = assign_times(_segment(5, (1, first), (4, second)))
        assert _clock(results) == ["09:58:00", "10:10:00", "10:20:00", "10:30:00", "10:42:00"]
        assert results[2].source == "between"
        assert results[2].metadata_fields()["capture_anchor_ref"] == f"{first.name} {second.name}"

    def test_references_at_both_ends(self) -> None:
        results, _ = assign_times(_segment(2, (0, _photo(10, 0)), (2, _photo(10, 30))))
        assert _clock(results) == ["10:10:00", "10:20:00"]

    def test_references_in_one_gap_keep_their_time_order(self) -> None:
        late, early = _photo(10, 5), _photo(10, 0)
        results, issues = assign_times(_segment(2, (1, late), (1, early)))
        assert issues == []
        assert _clock(results) == ["09:58:00", "10:07:00"]

    def test_references_against_film_order_are_an_error(self) -> None:
        results, issues = assign_times(_segment(4, (1, _photo(11, 0)), (3, _photo(10, 0))))
        assert _codes(issues) == [OUT_OF_ORDER] and issues[0].frames == ("IMG_100000.HEIC", "IMG_110000.HEIC")
        assert all(r.local is None for r in results)

    def test_every_frame_gets_its_own_second_after_the_reference_before_it(self) -> None:
        results, _ = assign_times(_segment(4, (1, _photo(10, 0, 0)), (4, _photo(10, 0, 2))))
        moments = [r.local for r in results]
        assert moments[1] > datetime(2026, 10, 25, 10, 0, 0)
        assert all(b > a for a, b in zip(moments, moments[1:]))

    def test_a_phone_in_another_zone_keeps_the_same_moment(self) -> None:
        berlin_phone = _photo(9, 0, offset=timedelta(hours=2))
        results, _ = assign_times(_segment(1, (0, berlin_phone)))
        assert (_clock(results), results[0].offset) == (["10:02:00"], PLUS_3)

    def test_a_reference_without_an_offset_is_local_at_its_place(self) -> None:
        photo = _photo(10, 0, offset=None, **HURGHADA)
        results, _ = assign_times(_segment(1, (0, photo)))
        assert (_clock(results), results[0].offset) == (["10:02:00"], PLUS_3)

    def test_a_typed_time_is_local_in_the_frame_zone(self) -> None:
        results, _ = assign_times(_segment(3, typed={"F02": datetime(2026, 10, 25, 14, 0)}))
        assert _clock(results) == ["13:58:00", "14:00:00", "14:02:00"]
        assert [r.source for r in results] == ["before", "typed", "after"]
        assert results[2].basis == ("F02",) and results[2].references == ()


class TestPlaceAndZone:
    def test_frames_take_the_place_and_zone_of_the_nearest_reference(self) -> None:
        day = date(2026, 10, 20)
        berlin = _photo(10, 0, day=day, offset=timedelta(hours=2), latitude=52.52, longitude=13.40)
        lisbon = _photo(13, 0, day=day, offset=timedelta(hours=1), latitude=38.72, longitude=-9.14)
        segment = Segment(FRAMES[:5], day=day, zone=CAIRO, references=((1, berlin), (4, lisbon)))
        results, issues = assign_times(segment)
        assert issues == []
        assert [r.zone for r in results] == ["Europe/Berlin"] * 3 + ["Europe/Lisbon"] * 2
        assert [r.offset for r in results] == [timedelta(hours=2)] * 3 + [timedelta(hours=1)] * 2
        assert (results[0].latitude, results[4].longitude) == (52.52, -9.14)
        assert {r.place_source for r in results} == {"photo"}

    def test_a_chosen_segment_place_wins_over_the_photos(self) -> None:
        segment = _segment(2, (1, _photo(10, 0, **HURGHADA)), latitude=27.25, longitude=33.81)
        results, _ = assign_times(segment)
        assert {(r.latitude, r.longitude, r.zone, r.place_source) for r in results} == {(27.25, 33.81, CAIRO, "segment")}

    def test_a_frame_keeps_its_own_place(self) -> None:
        berlin = _photo(10, 0, offset=timedelta(hours=2), latitude=52.52, longitude=13.40)
        results, _ = assign_times(_segment(2, (1, berlin), places={"F01": (27.25, 33.81)}))
        assert (results[0].latitude, results[0].zone, results[0].place_source) == (27.25, CAIRO, "frame")
        assert (results[1].latitude, results[1].zone, results[1].place_source) == (52.52, "Europe/Berlin", "photo")

    def test_a_frame_takes_the_place_of_the_photo_nearest_in_time(self) -> None:
        # The frame after the Berlin photo is bumped to one second after the Lisbon photo: nearer in time to Lisbon.
        berlin = Reference("B.HEIC", datetime(2026, 10, 20, 10, 0, 0), timedelta(hours=2), 52.52, 13.40)
        lisbon = Reference("L.HEIC", datetime(2026, 10, 20, 9, 0, 1), timedelta(hours=1), 38.72, -9.14)
        results, _ = assign_times(Segment(FRAMES[:2], zone=CAIRO, references=((0, berlin), (2, lisbon))))
        assert [r.zone for r in results] == ["Europe/Lisbon", "Europe/Lisbon"]

    def test_without_any_place_the_local_time_follows_the_photo(self) -> None:
        results, issues = assign_times(Segment(FRAMES[:1], references=((0, _photo(10, 0)),)))
        assert NO_ZONE in _codes(issues)
        assert (_clock(results), results[0].offset) == (["10:02:00"], None)


class TestDaylightSaving:
    def test_the_spread_follows_real_time_across_the_change(self) -> None:
        # Berlin leaves summer time at 03:00 on 25 Oct 2026. 01:50 CEST to 02:50 CET is two hours.
        before = Reference("A.HEIC", datetime(2026, 10, 25, 1, 50), timedelta(hours=2))
        after = Reference("B.HEIC", datetime(2026, 10, 25, 2, 50), timedelta(hours=1))
        segment = Segment(FRAMES[:3], day=EGYPT_DAY, zone="Europe/Berlin", references=((0, before), (3, after)))
        results, _ = assign_times(segment)
        utc = [(r.local - r.offset).replace(tzinfo=timezone.utc) for r in results]
        assert {b - a for a, b in zip(utc, utc[1:])} == {timedelta(minutes=30)}
        assert results[0].offset == timedelta(hours=2) and results[-1].offset == timedelta(hours=1)

    def test_a_typed_time_in_the_skipped_hour_moves_forward(self) -> None:
        segment = Segment(FRAMES[:1], day=date(2026, 3, 29), zone="Europe/Berlin", typed={"F01": datetime(2026, 3, 29, 2, 30)})
        results, issues = assign_times(segment)
        assert DST_CHANGE in _codes(issues)
        assert (_clock(results), results[0].offset) == (["03:30:00"], timedelta(hours=2))


class TestChecks:
    def test_frames_far_between_two_references_ask_for_a_split(self) -> None:
        first, second = _photo(13, 30), _photo(15, 20, day=date(2026, 10, 26))
        _, issues = assign_times(_segment(4, (1, first), (3, second)))
        assert [(i.code, i.frames) for i in issues] == [(LONG_GAP_BETWEEN, (first.name, second.name, "F02", "F03"))]

    def test_a_time_after_the_scan_is_flagged(self) -> None:
        scans = {"F01": datetime(2026, 10, 25, 9, 0, 30, tzinfo=timezone.utc), "F02": datetime(2026, 10, 25, 9, 0, 30, tzinfo=timezone.utc)}
        _, issues = assign_times(_segment(2), scan_times=scans)
        assert [(i.code, i.frames) for i in issues] == [(AFTER_SCAN, ("F02",))]

    def test_the_scan_check_compares_instants(self) -> None:
        # 12:00 in Cairo (UTC+3) comes before 11:30 in Berlin (UTC+1) on the same day.
        scans = {"F01": datetime(2026, 10, 25, 11, 30, tzinfo=ZoneInfo("Europe/Berlin"))}
        _, issues = assign_times(_segment(1), scan_times=scans)
        assert issues == []

    def test_day_of_a_stored_date(self) -> None:
        assert day_of("2026-10-25 12:00:00") == EGYPT_DAY
        assert day_of("2026-10") is None and day_of("") is None
