import json
from dataclasses import replace
from datetime import date, datetime, time, timedelta

from negpy.features.metadata.capture_plan import CapturePlan, Place, PlanFrame
from negpy.features.metadata.capture_times import NO_DAY, SEGMENTS_OUT_OF_ORDER
from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.reference import ReferencePhoto

DAY = date(2026, 10, 25)
HURGHADA = Place(27.2574, 33.8128, "Hurghada", "", "Egypt")


def _frames(n: int = 4) -> list[PlanFrame]:
    return [PlanFrame(key=f"h{i}", name=f"F{i + 1:02d}") for i in range(n)]


def _clock(plan: CapturePlan) -> list[str]:
    results, _ = plan.compute()
    return [results[f.key].local.strftime("%H:%M:%S") for f in plan.frames]


def _days(plan: CapturePlan) -> list[str]:
    results, _ = plan.compute()
    return [results[f.key].local.strftime("%m-%d %H:%M") for f in plan.frames]


def _photo(path: str, hh: int, mm: int, day: date = DAY, gps=(27.14, 33.88), offset=timedelta(hours=3)) -> ReferencePhoto:
    lat, lon = gps if gps else (None, None)
    return ReferencePhoto(path=path, local=datetime.combine(day, time(hh, mm)), offset=offset, latitude=lat, longitude=lon)


def _plan(n: int = 4, *photos: ReferencePhoto, reverse: bool = False) -> CapturePlan:
    plan = CapturePlan(_frames(n), reverse=reverse)
    plan.set_own(0, day=DAY, place=HURGHADA)
    plan.add_references(list(photos))
    return plan


def _stored(plan: CapturePlan) -> list[PlanFrame]:
    """The frames as they come back after Apply, in file order."""
    changes = plan.changes(plan.compute()[0])
    frames = [replace(f, metadata=replace(MetadataConfig(), **changes[f.key])) for f in plan.frames]
    return list(reversed(frames)) if plan.reverse else frames


class TestValuesOnFrames:
    def test_a_value_applies_to_the_frames_after_it(self) -> None:
        plan = _plan(4)
        assert [plan.values_at(i).day for i in range(4)] == [DAY] * 4
        assert plan.values_at(3).origins == {"day": "h0", "place": "h0"}

    def test_a_later_value_takes_over_from_its_frame(self) -> None:
        plan = _plan(4)
        plan.set_own(2, day=date(2026, 10, 26))
        assert _days(plan) == ["10-25 12:00", "10-25 12:01", "10-26 12:00", "10-26 12:01"]

    def test_clearing_a_value_follows_the_frame_before_again(self) -> None:
        plan = _plan(4)
        plan.set_own(2, day=date(2026, 10, 26))
        plan.clear_own(2, "day")
        assert plan.values_at(3).day == DAY and "h2" not in plan.own

    def test_a_step_set_on_a_frame_spaces_the_frames_from_there(self) -> None:
        plan = _plan(4)
        plan.set_own(2, step=timedelta(minutes=5))
        assert _clock(plan) == ["12:00:00", "12:01:00", "12:06:00", "12:11:00"]

    def test_a_chosen_zone_applies_from_its_frame(self) -> None:
        plan = _plan(2)
        plan.set_own(1, zone="Europe/Berlin")
        assert plan.zone_at(0) == "Africa/Cairo" and plan.zone_at(1) == "Europe/Berlin"


class TestSegments:
    def test_fresh_frames_form_one_segment(self) -> None:
        assert CapturePlan(_frames()).bounds() == [(0, 3)]

    def test_a_cut_keeps_the_values_of_the_new_segment(self) -> None:
        plan = _plan()
        plan.toggle_cut(1)
        assert plan.bounds() == [(0, 1), (2, 3)]
        assert plan.own_of(2) == {"day": DAY, "place": HURGHADA}

    def test_a_new_segment_takes_its_day_from_its_first_photo(self) -> None:
        giza = _photo("/r/giza.heic", 15, 20, day=date(2026, 10, 26), gps=(29.98, 31.13))
        plan = _plan(4, giza)
        plan.place(giza.path, 2)
        plan.toggle_cut(1)
        assert plan.values_at(2).day == date(2026, 10, 26) and plan.values_at(2).place == HURGHADA

    def test_joining_drops_the_values_the_cut_only_carried(self) -> None:
        plan = _plan()
        plan.toggle_cut(1)
        plan.toggle_cut(1)
        assert plan.bounds() == [(0, 3)] and "h2" not in plan.own

    def test_values_do_not_cross_into_the_next_segment(self) -> None:
        plan = _plan()
        plan.toggle_cut(1)
        plan.set_own(0, day=date(2026, 10, 24))
        assert plan.values_at(1).day == date(2026, 10, 24) and plan.values_at(2).day == DAY


class TestPlacing:
    def test_a_photo_between_two_frames_sets_their_times(self) -> None:
        photo = _photo("/r/IMG_2514.HEIC", 11, 9)
        plan = _plan(3, photo)
        assert plan.place(photo.path, 0)
        assert _clock(plan) == ["11:07:00", "11:11:00", "11:13:00"]
        assert plan.refs_at(0) == [photo] and plan.gap_of(photo.path) == 0

    def test_a_photo_has_one_place(self) -> None:
        photo = _photo("/r/a.heic", 11, 0)
        plan = _plan(3, photo)
        assert plan.place(photo.path, 0)
        assert not plan.place(photo.path, 1)
        assert plan.move(photo.path, 1) and plan.gap_of(photo.path) == 1
        plan.remove(photo.path)
        assert plan.gap_of(photo.path) is None and plan.place(photo.path, -1)

    def test_a_photo_without_a_time_cannot_be_placed(self) -> None:
        plan = _plan(2)
        plan.add_references([ReferencePhoto(path="/r/x.heic", error="no capture time")])
        assert not plan.place("/r/x.heic", 0)

    def test_reverse_holds_the_frames_in_film_order(self) -> None:
        plan = _plan(4, reverse=True)
        assert [f.key for f in plan.frames] == ["h3", "h2", "h1", "h0"]

    def test_turning_the_film_round_keeps_photos_between_the_same_frames(self) -> None:
        photo = _photo("/r/a.heic", 10, 0)
        plan = _plan(4, photo)
        plan.place(photo.path, 0)
        plan.toggle_cut(2)
        plan.set_reverse(True)
        assert [f.key for f in plan.frames] == ["h3", "h2", "h1", "h0"]
        assert plan.gap_of(photo.path) == 2 and set(plan.cuts) == {0}

    def test_a_cut_before_a_photo_gives_the_photo_to_the_next_segment(self) -> None:
        photo = _photo("/r/a.heic", 10, 0)
        plan = _plan(4, photo)
        plan.place(photo.path, 1)
        plan.toggle_cut(1, photos_before=0)
        assert plan.segment_of_photo(photo.path) == 1
        assert _clock(plan)[2:] == ["10:02:00", "10:04:00"]

    def test_a_cut_after_a_photo_keeps_the_photo_in_the_first_segment(self) -> None:
        photo = _photo("/r/a.heic", 10, 0)
        plan = _plan(4, photo)
        plan.place(photo.path, 1)
        plan.toggle_cut(1, photos_before=1)
        assert plan.segment_of_photo(photo.path) == 0 and plan.split_of(1) == 1
        assert _clock(plan) == ["09:56:00", "09:58:00", "12:00:00", "12:01:00"]

    def test_a_cut_moves_within_its_gap_and_joins_on_the_same_place(self) -> None:
        photo = _photo("/r/a.heic", 10, 0)
        plan = _plan(4, photo)
        plan.place(photo.path, 1)
        plan.toggle_cut(1, 0)
        plan.toggle_cut(1, 1)
        assert plan.split_of(1) == 1 and plan.bounds() == [(0, 1), (2, 3)]
        plan.toggle_cut(1, 1)
        assert plan.bounds() == [(0, 3)]

    def test_a_segment_holds_many_photos(self) -> None:
        photos = [_photo(f"/r/{n}.heic", 10, n) for n in range(4)]
        plan = _plan(5, *photos)
        for gap, photo in zip((0, 1, 1, 3), photos):
            plan.place(photo.path, gap)
        assert [p.path for _, p in plan.segment_photos(0)] == [p.path for p in photos]

    def test_typed_times(self) -> None:
        plan = _plan(3)
        plan.set_typed("h1", datetime(2026, 10, 25, 14, 0))
        assert _clock(plan) == ["13:58:00", "14:00:00", "14:02:00"]
        plan.remove_typed("h1")
        assert _clock(plan) == ["12:00:00", "12:01:00", "12:02:00"]

    def test_frames_without_a_day_give_no_changes(self) -> None:
        plan = CapturePlan(_frames(2))
        results, issues = plan.compute()
        assert NO_DAY in [i.code for i in issues]
        assert plan.changes(results) == {}


class TestChanges:
    def test_a_chosen_place_brings_its_names(self) -> None:
        plan = _plan(2, _photo("/r/a.heic", 10, 0))
        plan.place("/r/a.heic", 0)
        changes = plan.changes(plan.compute()[0])
        assert changes["h0"]["capture_timezone"] == "Africa/Cairo"
        assert (changes["h1"]["location_city"], changes["h1"]["location_country"]) == ("Hurghada", "Egypt")
        assert changes["h1"]["capture_anchor_ref"] == "a.heic"

    def test_a_photo_place_has_no_names(self) -> None:
        berlin = _photo("/r/b.heic", 10, 0, gps=(52.52, 13.40), offset=timedelta(hours=2))
        plan = CapturePlan(_frames(1))
        plan.set_own(0, day=DAY)
        plan.add_references([berlin])
        plan.place(berlin.path, 0)
        changes = plan.changes(plan.compute()[0])
        assert changes["h0"]["capture_timezone"] == "Europe/Berlin"
        assert changes["h0"]["location_city"] == "" and changes["h0"]["gps_latitude"] == 52.52

    def test_only_the_chosen_frames_are_written(self) -> None:
        plan = _plan(4)
        changes = plan.changes(plan.compute()[0], only={"h1", "h2"})
        assert set(changes) == {"h1", "h2"} and changes["h2"]["capture_date"] == "2026-10-25 12:02:00"


class TestOwnPlaces:
    def _framed(self, **metadata) -> list[PlanFrame]:
        return [replace(f, metadata=MetadataConfig(**metadata)) if f.key == "h0" else f for f in _frames(2)]

    def test_a_place_set_before_the_dialog_is_kept(self) -> None:
        plan = CapturePlan(self._framed(gps_latitude=29.98, gps_longitude=31.13, location_city="Giza"))
        plan.set_own(0, day=DAY)
        plan.add_references([_photo("/r/a.heic", 10, 0)])
        plan.place("/r/a.heic", 0)
        changes = plan.changes(plan.compute()[0])
        assert (changes["h0"]["gps_latitude"], changes["h0"]["location_city"]) == (29.98, "Giza")
        assert changes["h1"]["gps_latitude"] == 27.14

    def test_a_place_the_dialog_wrote_is_not_the_frames_own(self) -> None:
        plan = CapturePlan(_frames(1))
        plan.set_own(0, day=DAY)
        plan.add_references([_photo("/r/a.heic", 10, 0)])
        plan.place("/r/a.heic", 0)
        stored = _stored(plan)
        assert CapturePlan(stored).own_places == {}
        edited = [replace(f, metadata=replace(f.metadata, gps_latitude=30.0)) for f in stored]
        assert CapturePlan(edited).own_places == {"h0": (30.0, 33.88)}


class TestReferencePhotos:
    def test_removing_a_photo_takes_it_out_of_the_strip(self) -> None:
        plan = _plan(2, _photo("/r/a.heic", 10, 0))
        plan.place("/r/a.heic", 0)
        plan.remove_reference("/r/a.heic")
        assert plan.references == [] and plan.placed == {}

    def test_photos_sort_by_moment_and_filter_by_day(self) -> None:
        plan = CapturePlan(_frames(1))
        late, early = _photo("/r/b.heic", 14, 32), _photo("/r/a.heic", 11, 9)
        other_day = _photo("/r/c.heic", 9, 0, day=date(2026, 10, 26))
        plan.add_references([late, early, other_day, early])
        assert [p.path for p in plan.references] == ["/r/a.heic", "/r/b.heic", "/r/c.heic"]
        assert [p.path for p in plan.references_on(DAY)] == ["/r/a.heic", "/r/b.heic"]
        assert len(plan.references_on(None)) == 3


class TestReopen:
    def _applied(self) -> CapturePlan:
        plan = _plan(5, _photo("/r/IMG_2514.HEIC", 11, 9), _photo("/r/IMG_2600.HEIC", 15, 20, day=date(2026, 10, 26), gps=(29.98, 31.13)))
        plan.set_own(0, step=timedelta(minutes=5))
        plan.place("/r/IMG_2514.HEIC", 0)
        plan.place("/r/IMG_2600.HEIC", 2)
        plan.toggle_cut(2, photos_before=0)
        plan.set_typed("h2", datetime(2026, 10, 25, 15, 0))
        return plan

    def test_stored_values_give_the_same_plan_and_times(self) -> None:
        plan = self._applied()
        reopened = CapturePlan(_stored(plan))
        assert reopened.bounds() == plan.bounds() and reopened.split_of(2) == 0
        assert reopened.placed == {} and reopened.references == []
        assert reopened.typed == {"h2": datetime(2026, 10, 25, 15, 0)}
        assert reopened.own == plan.own
        assert _clock(reopened) == _clock(plan)

    def test_a_change_works_out_only_its_segment_again(self) -> None:
        plan = self._applied()
        reopened = CapturePlan(_stored(plan))
        assert reopened.is_kept(0) and reopened.is_kept(1)
        reopened.add_references([_photo("/r/new.heic", 15, 40, day=date(2026, 10, 26))])
        reopened.place("/r/new.heic", 3)
        assert reopened.is_kept(0) and not reopened.is_kept(1)
        results, _ = reopened.compute()
        assert _clock(reopened)[:3] == _clock(plan)[:3]
        # The segment steps 5 minutes, so the frame before the 15:40 photo is at 15:35.
        assert results["h3"].local == datetime(2026, 10, 26, 15, 35) and not results["h3"].kept

    def test_a_kept_frame_writes_back_what_it_had(self) -> None:
        stored = _stored(self._applied())
        reopened = CapturePlan(stored)
        changes = reopened.changes(reopened.compute()[0])
        for frame in stored:
            m = frame.metadata
            written = changes[frame.key]
            assert (written["capture_date"], written["capture_timezone"], written["gps_latitude"]) == (
                m.capture_date,
                m.capture_timezone,
                m.gps_latitude,
            )

    def test_a_reversed_roll_opens_reversed(self) -> None:
        reopened = CapturePlan(_stored(_plan(3, reverse=True)))
        assert reopened.reverse and [f.key for f in reopened.frames] == ["h2", "h1", "h0"]

    def test_a_version_1_entry_puts_its_segment_values_on_the_frame(self) -> None:
        entry = {"v": 1, "segment": {"day": "2026-10-25", "lat": 27.25, "lon": 33.81, "city": "Hurghada", "step": 300, "reverse": False}}
        frames = [replace(_frames(1)[0], metadata=MetadataConfig(capture_plan=json.dumps(entry)))]
        plan = CapturePlan(frames)
        assert plan.own["h0"] == {"day": DAY, "place": Place(27.25, 33.81, "Hurghada"), "step": timedelta(minutes=5)}

    def test_a_roll_without_a_stored_plan_opens_as_one_segment(self) -> None:
        frames = [
            replace(f, metadata=MetadataConfig(capture_date="2026-10-25 12:00", gps_latitude=27.25, gps_longitude=33.81))
            for f in _frames(3)
        ]
        plan = CapturePlan(frames)
        assert plan.bounds() == [(0, 2)] and plan.values_at(2).day == DAY


class TestSegmentOrder:
    def test_a_segment_that_starts_before_the_last_one_ends_gives_a_warning(self) -> None:
        plan = _plan(4)
        plan.toggle_cut(1)
        plan.set_typed("h2", datetime(2026, 10, 25, 9, 0))
        _, issues = plan.compute()
        assert [(i.code, i.frames, i.is_error) for i in issues] == [(SEGMENTS_OUT_OF_ORDER, ("h2", "h1"), False)]
