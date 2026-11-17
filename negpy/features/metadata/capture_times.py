"""Assign Capture Times: plausible capture times for the frames of a segment, in film order."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from negpy.features.metadata.capture import parse_capture_date
from negpy.features.metadata.timezone import resolve_local

DEFAULT_START = time(12, 0)
STEP_WITHOUT_ANCHORS = timedelta(minutes=1)
STEP_AROUND_ANCHORS = timedelta(minutes=2)
_ONE_SECOND = timedelta(seconds=1)

# Issue codes. The dialog turns each code into translated text.
ANCHORS_OUT_OF_ORDER = "anchors_out_of_order"  # error: a later frame has an earlier anchor time
NO_DAY = "no_day"  # error: a segment without anchors needs a day
NEXT_DAY = "next_day"  # warning: a frame leaves the day of its segment
DST_CHANGE = "dst_change"  # warning: a typed time falls in a daylight saving change
NO_ZONE = "no_zone"  # warning: no place or zone, so no offset is written
AFTER_SCAN = "after_scan"  # warning: a capture time is after the scan time


@dataclass(frozen=True)
class Anchor:
    """A frame pinned to a known moment. `offset` is None for a typed time or a photo without one."""

    local: datetime
    offset: Optional[timedelta] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    source: str = "anchor"  # "anchor" (from a reference photo) or "typed"
    ref: str = ""


@dataclass(frozen=True)
class Segment:
    """Consecutive frames in film order that share a day, a place and a zone."""

    frames: tuple[str, ...]
    day: Optional[date] = None
    zone: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    start: time = DEFAULT_START
    step: Optional[timedelta] = None
    anchors: Mapping[str, Anchor] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameTime:
    """The result for one frame. `local` is the wall-clock time in `zone`, or None after an error."""

    frame: str
    local: Optional[datetime]
    offset: Optional[timedelta]
    zone: str
    latitude: Optional[float]
    longitude: Optional[float]
    source: str
    anchor_ref: str = ""

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
            "capture_anchor_ref": self.anchor_ref,
        }


@dataclass(frozen=True)
class Issue:
    code: str
    frames: tuple[str, ...] = ()

    @property
    def is_error(self) -> bool:
        return self.code in (ANCHORS_OUT_OF_ORDER, NO_DAY)


def assign_times(segment: Segment, scan_times: Optional[Mapping[str, datetime]] = None) -> tuple[list[FrameTime], list[Issue]]:
    """Times for every frame of the segment, and the issues found. An error leaves `local` empty."""
    frames = segment.frames
    issues: list[Issue] = []
    zone = segment.zone
    if not zone:
        issues.append(Issue(NO_ZONE, frames))

    anchored = [(i, frames[i], segment.anchors[f]) for i, f in enumerate(frames) if f in segment.anchors]
    working: dict[int, datetime] = {}
    for i, frame, anchor in anchored:
        value, shifted = _anchor_value(anchor, zone)
        working[i] = value
        if shifted:
            issues.append(Issue(DST_CHANGE, (frame,)))

    for (i_a, f_a, _), (i_b, f_b, _) in zip(anchored, anchored[1:]):
        if working[i_b] <= working[i_a]:
            issues.append(Issue(ANCHORS_OUT_OF_ORDER, (f_b, f_a)))
    if any(issue.is_error for issue in issues):
        return [_empty(segment, f) for f in frames], issues

    if not anchored:
        if segment.day is None:
            issues.append(Issue(NO_DAY, frames))
            return [_empty(segment, f) for f in frames], issues
        base, shifted = _wall_value(datetime.combine(segment.day, segment.start), zone)
        if shifted:
            issues.append(Issue(DST_CHANGE, frames[:1]))
        step = segment.step or STEP_WITHOUT_ANCHORS
        values = [base + step * i for i in range(len(frames))]
        sources = ["day"] * len(frames)
    else:
        step = segment.step or STEP_AROUND_ANCHORS
        values, sources = _spread(len(frames), [(i, working[i]) for i, _, _ in anchored], step)
        for i, _, anchor in anchored:
            sources[i] = anchor.source

    values = _distinct(values)
    results = []
    for i, frame in enumerate(frames):
        local, offset = _to_local(values[i], zone)
        latitude, longitude = _place_for(i, anchored, segment)
        anchor = segment.anchors.get(frame)
        results.append(FrameTime(frame, local, offset, zone, latitude, longitude, sources[i], anchor.ref if anchor is not None else ""))

    if segment.day is not None:
        moved = tuple(r.frame for r in results if r.local is not None and r.local.date() != segment.day and r.frame not in segment.anchors)
        if moved:
            issues.append(Issue(NEXT_DAY, moved))
    if scan_times:
        late = tuple(r.frame for r in results if r.local is not None and r.frame in scan_times and r.local > scan_times[r.frame])
        if late:
            issues.append(Issue(AFTER_SCAN, late))
    return results, issues


def segment_bounds(day_and_zone: Sequence[tuple[Optional[date], str]]) -> list[tuple[int, int]]:
    """(first, last) frame positions of the segments that frames in film order form again: a
    segment is a run with the same day and zone. Places inside a segment can differ by anchor."""
    bounds: list[tuple[int, int]] = []
    for i, key in enumerate(day_and_zone):
        if bounds and day_and_zone[bounds[-1][0]] == key:
            bounds[-1] = (bounds[-1][0], i)
        else:
            bounds.append((i, i))
    return bounds


def day_of(capture_date: str) -> Optional[date]:
    """The day of a stored capture date, when it names one."""
    parsed = parse_capture_date(capture_date)
    if parsed is None or parsed.precision in ("year", "month"):
        return None
    return date.fromisoformat(parsed.text[:10])


def _anchor_value(anchor: Anchor, zone: str) -> tuple[datetime, bool]:
    """The working value of an anchor: a UTC instant with a zone, else the local time."""
    if not zone:
        return _whole_seconds(anchor.local), False
    if anchor.offset is not None:
        return _whole_seconds((anchor.local - anchor.offset).replace(tzinfo=timezone.utc)), False
    return _wall_value(anchor.local, zone)


def _wall_value(local: datetime, zone: str) -> tuple[datetime, bool]:
    """A typed wall-clock time as a working value, and whether it fell in a daylight saving change."""
    if not zone:
        return local, False
    tz = ZoneInfo(zone)
    shifted = local.replace(tzinfo=tz, fold=0).utcoffset() != local.replace(tzinfo=tz, fold=1).utcoffset()
    return resolve_local(local, zone).astimezone(timezone.utc), shifted


def _spread(count: int, anchors: list[tuple[int, datetime]], step: timedelta) -> tuple[list[datetime], list[str]]:
    """Values for all positions from the anchors: even spacing between two, `step` outside them."""
    values: list[datetime] = [anchors[0][1]] * count
    sources = [""] * count
    first_i, first = anchors[0]
    last_i, last = anchors[-1]
    for i in range(count):
        if i < first_i:
            values[i], sources[i] = first - step * (first_i - i), "before"
        elif i > last_i:
            values[i], sources[i] = last + step * (i - last_i), "after"
    for (i_a, a), (i_b, b) in zip(anchors, anchors[1:]):
        for i in range(i_a, i_b + 1):
            values[i] = _whole_seconds(a + (b - a) * (i - i_a) / (i_b - i_a))
            sources[i] = "between"
    return values, sources


def _distinct(values: list[datetime]) -> list[datetime]:
    """Whole seconds, strictly increasing: a frame never shares a time with the frame before it."""
    out: list[datetime] = []
    for value in values:
        value = _whole_seconds(value)
        if out and value <= out[-1]:
            value = out[-1] + _ONE_SECOND
        out.append(value)
    return out


def _whole_seconds(value: datetime) -> datetime:
    return (value + timedelta(microseconds=500_000)).replace(microsecond=0)


def _to_local(value: datetime, zone: str) -> tuple[datetime, Optional[timedelta]]:
    if not zone:
        return value, None
    local = value.astimezone(ZoneInfo(zone))
    return local.replace(tzinfo=None), local.utcoffset()


def _place_for(i: int, anchored: list[tuple[int, str, Anchor]], segment: Segment) -> tuple[Optional[float], Optional[float]]:
    """An anchor keeps its photo's GPS. Other frames take the nearest GPS anchor, the earlier one on
    a tie, else the place of the segment."""
    with_gps = [(j, a) for j, _, a in anchored if a.latitude is not None and a.longitude is not None]
    if not with_gps:
        return segment.latitude, segment.longitude
    _, anchor = min(with_gps, key=lambda item: (abs(item[0] - i), item[0]))
    return anchor.latitude, anchor.longitude


def _empty(segment: Segment, frame: str) -> FrameTime:
    return FrameTime(frame, None, None, segment.zone, segment.latitude, segment.longitude, "")
