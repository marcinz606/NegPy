"""Assign Capture Times: plausible capture times for the frames of a segment, in film order.

Phone photos sit between frames (a reference), typed times sit on frames. Both are known moments;
the frames between them are spread evenly by position, and every time is a UTC instant."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable, Mapping, Optional
from zoneinfo import ZoneInfo

from negpy.features.metadata.capture import parse_capture_date
from negpy.features.metadata.timezone import resolve_local, zone_at

DEFAULT_START = time(12, 0)
STEP_WITHOUT_REFERENCES = timedelta(minutes=1)
STEP_AROUND_REFERENCES = timedelta(minutes=2)
LONG_GAP = timedelta(hours=6)
_ONE_SECOND = timedelta(seconds=1)

# Issue codes. The dialog turns each code into translated text. `frames` holds frame keys and
# reference names.
OUT_OF_ORDER = "out_of_order"  # error: a known moment is earlier than one before it on the film
NO_DAY = "no_day"  # error: a segment without known moments needs a day
NEXT_DAY = "next_day"  # warning: a frame leaves the day of its segment
DST_CHANGE = "dst_change"  # warning: a typed time falls in a daylight saving change
NO_ZONE = "no_zone"  # warning: no place or zone, so no offset is written
AFTER_SCAN = "after_scan"  # warning: a capture time is after the scan time
SEGMENTS_OUT_OF_ORDER = "segments_out_of_order"  # warning: a segment starts before the one before it ends
LONG_GAP_BETWEEN = "long_gap"  # warning: frames between two known moments far apart: (earlier, later, *frames)

SOURCE_BETWEEN = "between"
SOURCE_BEFORE = "before"
SOURCE_AFTER = "after"
SOURCE_START = "start"
SOURCE_TYPED = "typed"

# Where a frame's place comes from: the segment's chosen place, the frame's own place, or a photo.
PLACE_SEGMENT = "segment"
PLACE_FRAME = "frame"
PLACE_PHOTO = "photo"


@dataclass(frozen=True)
class Reference:
    """A phone photo placed in the film strip. Without an offset its local time is read in the zone
    of its own place, else of the segment."""

    name: str
    local: datetime
    offset: Optional[timedelta] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass(frozen=True)
class FrameSettings:
    """What applies to one frame: the day for timing by day alone, a chosen place, the chosen zone
    or the zone of the chosen place, and the time step to the frame before it."""

    day: Optional[date] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    zone: str = ""
    zone_fixed: bool = False
    step: Optional[timedelta] = None


@dataclass(frozen=True)
class Segment:
    """Consecutive frames in film order. `references` holds (position, reference): position p sits
    after p frames, so 0 is before the first frame and len(frames) after the last.

    Each frame's settings come from `frame_settings`, else from the segment fields. Frames with no
    known moment in the segment are timed by their day from `start`; a new day starts again there.
    A frame's place is its chosen place, else its own place in `places`, else the place of the
    nearest reference in time. With `zone_fixed` a frame takes `zone`, else the zone of its place."""

    frames: tuple[str, ...]
    day: Optional[date] = None
    zone: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    start: time = DEFAULT_START
    step: Optional[timedelta] = None
    references: tuple[tuple[int, Reference], ...] = ()
    typed: Mapping[str, datetime] = field(default_factory=dict)
    places: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    zone_fixed: bool = False
    frame_settings: Mapping[str, FrameSettings] = field(default_factory=dict)

    def settings(self, frame: str) -> FrameSettings:
        found = self.frame_settings.get(frame)
        if found is not None:
            return found
        return FrameSettings(self.day, self.latitude, self.longitude, self.zone, self.zone_fixed, self.step)


@dataclass(frozen=True)
class FrameTime:
    """The result for one frame. `local` is the wall-clock time in `zone`, or None after an error.
    `basis` names the known moments that set the time: reference names or typed frame keys."""

    frame: str
    local: Optional[datetime]
    offset: Optional[timedelta]
    zone: str
    latitude: Optional[float]
    longitude: Optional[float]
    source: str
    basis: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    place_source: str = ""
    kept: bool = False  # the result of the last Apply, shown again unchanged

    def capture_date_text(self) -> str:
        """The value for MetadataConfig.capture_date. The zone gives the offset at export."""
        return self.local.strftime("%Y-%m-%d %H:%M:%S") if self.local is not None else ""

    def metadata_fields(self) -> dict:
        """The MetadataConfig fields to write for this frame. Empty after an error."""
        if self.local is None:
            return {}
        return {
            "capture_date": self.capture_date_text(),
            "capture_timezone": self.zone,
            "gps_latitude": self.latitude,
            "gps_longitude": self.longitude,
            "capture_time_source": self.source,
            "capture_anchor_ref": " ".join(self.references),
        }


@dataclass(frozen=True)
class Issue:
    code: str
    frames: tuple[str, ...] = ()

    @property
    def is_error(self) -> bool:
        return self.code in (OUT_OF_ORDER, NO_DAY)


@dataclass
class _Item:
    """One place in film order: a frame, or a reference between frames."""

    key: str
    reference: Optional[Reference] = None
    instant: Optional[datetime] = None  # naive UTC, or naive local when no zone is known
    zone: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place_source: str = ""


def assign_times(segment: Segment, scan_times: Optional[Mapping[str, datetime]] = None) -> tuple[list[FrameTime], list[Issue]]:
    """Times for every frame of the segment, and the issues found. An error leaves `local` empty.
    A naive scan time is in the zone of the computer."""
    items = _items(segment)
    issues: list[Issue] = []
    _place_items(items, segment)
    no_zone = tuple(it.key for it in items if it.reference is None and not it.zone)
    if no_zone:
        issues.append(Issue(NO_ZONE, no_zone))

    for it in items:
        if it.reference is not None:
            it.instant = _reference_instant(it.reference, it.zone or segment.zone)
        elif it.key in segment.typed:
            it.instant, shifted = _wall_value(segment.typed[it.key], it.zone)
            if shifted:
                issues.append(Issue(DST_CHANGE, (it.key,)))
    known = [(p, it) for p, it in enumerate(items) if it.instant is not None]

    for (p_a, a), (p_b, b) in zip(known, known[1:]):
        if b.instant <= a.instant:
            issues.append(Issue(OUT_OF_ORDER, (b.key, a.key)))
        elif b.instant - a.instant > LONG_GAP:
            between = tuple(it.key for it in items[p_a + 1 : p_b] if it.reference is None)
            if between:
                issues.append(Issue(LONG_GAP_BETWEEN, (a.key, b.key, *between)))
    frame_items = [it for it in items if it.reference is None]
    if any(issue.is_error for issue in issues):
        return [_empty(it) for it in frame_items], issues

    if not known:
        no_day = tuple(it.key for it in frame_items if segment.settings(it.key).day is None)
        if no_day:
            issues.append(Issue(NO_DAY, no_day))
            return [_empty(it) for it in frame_items], issues
        values: dict[str, datetime] = {}
        day, value = None, None
        for it in frame_items:
            fs = segment.settings(it.key)
            if fs.day != day or value is None:
                day = fs.day
                value, shifted = _wall_value(datetime.combine(fs.day, segment.start), it.zone)
                if shifted:
                    issues.append(Issue(DST_CHANGE, (it.key,)))
            else:
                value = value + (fs.step or STEP_WITHOUT_REFERENCES)
            values[it.key] = value
        sources = {it.key: (SOURCE_START, ()) for it in frame_items}
    else:
        values, sources = _spread(items, known, lambda key: segment.settings(key).step or STEP_AROUND_REFERENCES)

    reference_names = {it.key for it in items if it.reference is not None}
    last: Optional[datetime] = None
    results = []
    for it in items:
        if it.reference is not None:
            last = it.instant if last is None or it.instant > last else last
            continue
        value = _whole_seconds(values[it.key])
        # Every frame gets its own second, after everything before it on the film.
        if last is not None and value <= last:
            value = last + _ONE_SECOND
        last = value
        if it.place_source == PLACE_PHOTO and it.key not in segment.typed:
            _nearest_in_time(it, value, items, segment)
        local, offset = _to_local(value, it.zone, _fallback_offset(value, items) if not it.zone else None)
        source, basis = sources[it.key]
        refs = tuple(b for b in basis if b in reference_names)
        results.append(FrameTime(it.key, local, offset, it.zone, it.latitude, it.longitude, source, basis, refs, it.place_source))

    if not known:
        moved = tuple(r.frame for r in results if r.local is not None and r.local.date() != segment.settings(r.frame).day)
        if moved:
            issues.append(Issue(NEXT_DAY, moved))
    if scan_times:
        late = tuple(r.frame for r in results if r.frame in scan_times and _after(r, scan_times[r.frame]))
        if late:
            issues.append(Issue(AFTER_SCAN, late))
    return results, issues


def day_of(capture_date: str) -> Optional[date]:
    """The day of a stored capture date, when it names one."""
    parsed = parse_capture_date(capture_date)
    if parsed is None or parsed.precision in ("year", "month"):
        return None
    return date.fromisoformat(parsed.text[:10])


def _items(segment: Segment) -> list[_Item]:
    """Frames and references in film order. References at one position stay in time order."""
    by_position: dict[int, list[Reference]] = {}
    for position, reference in segment.references:
        by_position.setdefault(max(0, min(position, len(segment.frames))), []).append(reference)
    items: list[_Item] = []
    for p in range(len(segment.frames) + 1):
        for reference in sorted(by_position.get(p, []), key=_reference_sort_key):
            items.append(_Item(reference.name, reference))
        if p < len(segment.frames):
            items.append(_Item(segment.frames[p]))
    return items


def _place_items(items: list[_Item], segment: Segment) -> None:
    """A reference keeps its own place and zone. A frame takes the segment's chosen place, else its
    own place, else, for now, the nearest reference with GPS on the strip; `_nearest_in_time` makes
    that final once the frame has its time. A typed time is read in the zone found here."""
    zones: dict[tuple[float, float], str] = {}

    def zone_of(lat: float, lon: float) -> str:
        if (lat, lon) not in zones:
            zones[(lat, lon)] = zone_at(lat, lon)
        return zones[(lat, lon)]

    with_gps = []
    for p, it in enumerate(items):
        ref = it.reference
        if ref is not None and ref.latitude is not None and ref.longitude is not None:
            it.latitude, it.longitude, it.zone = ref.latitude, ref.longitude, zone_of(ref.latitude, ref.longitude)
            with_gps.append((p, it))
    for p, it in enumerate(items):
        if it.reference is not None:
            continue
        fs = segment.settings(it.key)
        if fs.latitude is not None and fs.longitude is not None:
            it.latitude, it.longitude, it.zone, it.place_source = fs.latitude, fs.longitude, fs.zone, PLACE_SEGMENT
        elif it.key in segment.places:
            it.latitude, it.longitude = segment.places[it.key]
            it.zone, it.place_source = zone_of(it.latitude, it.longitude) or fs.zone, PLACE_FRAME
        elif with_gps:
            _, nearest = min(with_gps, key=lambda pair: (abs(pair[0] - p), pair[0]))
            it.latitude, it.longitude, it.zone, it.place_source = (
                nearest.latitude,
                nearest.longitude,
                nearest.zone or fs.zone,
                PLACE_PHOTO,
            )
        else:
            it.latitude, it.longitude, it.zone = None, None, fs.zone
        if fs.zone_fixed:
            it.zone = fs.zone


def _nearest_in_time(item: _Item, value: datetime, items: list[_Item], segment: Segment) -> None:
    """Give a frame the place and zone of the reference with GPS nearest to its time, the earlier on a tie."""
    candidates = [it for it in items if it.reference is not None and it.latitude is not None and it.instant is not None]
    if not candidates:
        return
    nearest = min(candidates, key=lambda it: (abs(it.instant - value), it.instant))
    fs = segment.settings(item.key)
    item.latitude, item.longitude = nearest.latitude, nearest.longitude
    item.zone = fs.zone if fs.zone_fixed else (nearest.zone or fs.zone)


def _fallback_offset(value: datetime, items: list[_Item]) -> Optional[timedelta]:
    """Without a zone, the offset of the reference nearest in time, so the local time still reads right."""
    with_offset = [it for it in items if it.reference is not None and it.reference.offset is not None and it.instant is not None]
    if not with_offset:
        return None
    return min(with_offset, key=lambda it: (abs(it.instant - value), it.instant)).reference.offset


def _spread(
    items: list[_Item], known: list[tuple[int, _Item]], step: Callable[[str], timedelta]
) -> tuple[dict[str, datetime], dict[str, tuple[str, tuple[str, ...]]]]:
    """Times from the known moments: even spacing by position between two, each frame's step
    outside them. Only frames lie before the first and after the last known moment."""
    values: dict[str, datetime] = {}
    sources: dict[str, tuple[str, tuple[str, ...]]] = {}
    (first_p, first), (last_p, last) = known[0], known[-1]
    for _, it in known:
        if it.reference is None:
            values[it.key], sources[it.key] = it.instant, (SOURCE_TYPED, ())
    value = first.instant
    for p in range(first_p - 1, -1, -1):
        it = items[p]
        value = value - step(items[p + 1].key if items[p + 1].reference is None else it.key)
        values[it.key], sources[it.key] = value, (SOURCE_BEFORE, (first.key,))
    value = last.instant
    for p in range(last_p + 1, len(items)):
        it = items[p]
        value = value + step(it.key)
        values[it.key], sources[it.key] = value, (SOURCE_AFTER, (last.key,))
    for (p_a, a), (p_b, b) in zip(known, known[1:]):
        for p in range(p_a + 1, p_b):
            it = items[p]
            if it.reference is None:
                values[it.key] = a.instant + (b.instant - a.instant) * (p - p_a) / (p_b - p_a)
                sources[it.key] = (SOURCE_BETWEEN, (a.key, b.key))
    return values, sources


def _reference_instant(reference: Reference, zone: str) -> datetime:
    """A reference's moment: from its own offset, else its local time in `zone`."""
    if reference.offset is not None:
        return _whole_seconds(reference.local - reference.offset)
    value, _ = _wall_value(reference.local, zone)
    return value


def _reference_sort_key(reference: Reference) -> datetime:
    return reference.local - reference.offset if reference.offset is not None else reference.local


def _wall_value(local: datetime, zone: str) -> tuple[datetime, bool]:
    """A wall-clock time as naive UTC, and whether it fell in a daylight saving change. Without a
    zone the local time stands in for the instant."""
    if not zone:
        return _whole_seconds(local), False
    tz = ZoneInfo(zone)
    shifted = local.replace(tzinfo=tz, fold=0).utcoffset() != local.replace(tzinfo=tz, fold=1).utcoffset()
    return _whole_seconds(resolve_local(local, zone).astimezone(timezone.utc).replace(tzinfo=None)), shifted


def _whole_seconds(value: datetime) -> datetime:
    return (value + timedelta(microseconds=500_000)).replace(microsecond=0)


def _to_local(value: datetime, zone: str, fallback: Optional[timedelta] = None) -> tuple[datetime, Optional[timedelta]]:
    """Local time and offset in `zone`. Without a zone there is no offset to write; `fallback`
    still turns the instant into the photographer's wall clock."""
    if not zone:
        return (value + fallback if fallback is not None else value), None
    local = value.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(zone))
    return local.replace(tzinfo=None), local.utcoffset()


def _empty(item: _Item) -> FrameTime:
    return FrameTime(item.key, None, None, item.zone, item.latitude, item.longitude, "")


def _after(result: FrameTime, scan: datetime) -> bool:
    """True when the frame's time is after its scan: instants for a frame with an offset, else the
    wall clock of the computer."""
    if result.local is None:
        return False
    if result.offset is not None:
        return result.local - result.offset > scan.astimezone(timezone.utc).replace(tzinfo=None)
    return result.local > scan.astimezone().replace(tzinfo=None)
