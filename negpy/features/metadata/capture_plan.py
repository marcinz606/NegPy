"""The state of the Assign Capture Times dialog: the roll's frames in film order, segment cuts, the
values set on frames, reference photos placed between frames, and typed times. No Qt here, so every
rule has a plain test.

Gap g sits between frame g and frame g + 1 in film order; gap -1 is before the first frame and gap
len(frames) - 1 after the last. A cut sits in a gap, before or after each photo there: the photos
before it belong to the segment on the left, the rest to the segment on the right.

A value (day, place, zone, step) set on a frame applies to it and to the frames after it in its
segment, up to the next frame with its own value."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Sequence

from negpy.features.metadata.capture import parse_capture_date
from negpy.features.metadata.capture_times import (
    PLACE_FRAME,
    PLACE_SEGMENT,
    SEGMENTS_OUT_OF_ORDER,
    SOURCE_TYPED,
    FrameSettings,
    FrameTime,
    Issue,
    Reference,
    Segment,
    assign_times,
    day_of,
)
from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.reference import ReferencePhoto
from negpy.features.metadata.timezone import capture_zone, is_zone, resolve_local

PLAN_VERSION = 2
FIELDS = ("day", "place", "zone", "step")


@dataclass(frozen=True)
class PlanFrame:
    """One frame as the dialog sees it. `key` is the file hash. A rejected frame keeps its time slot,
    because it is a physical frame on the film."""

    key: str
    name: str
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    rejected: bool = False


@dataclass(frozen=True)
class Place:
    latitude: float
    longitude: float
    city: str = ""
    state: str = ""
    country: str = ""


@dataclass(frozen=True)
class FrameValues:
    """The values in effect at a frame. `origins` maps each set value to the key of the frame that set it."""

    day: Optional[date] = None
    place: Optional[Place] = None
    zone: str = ""
    step: Optional[timedelta] = None
    origins: dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        return getattr(self, name)


def photo_moment(photo: ReferencePhoto) -> datetime:
    """A sort key: the UTC moment when the photo has an offset, else its local time."""
    if photo.local is None:
        return datetime.max
    return photo.local - photo.offset if photo.offset is not None else photo.local


class CapturePlan:
    """The roll's frames in film order, cut into segments. `frames` comes in file order; with
    `reverse` the film ran the other way, so the plan holds them reversed. None takes the direction
    the frames stored."""

    def __init__(self, frames: Sequence[PlanFrame], reverse: Optional[bool] = None) -> None:
        given = list(frames)
        self.reverse = _stored_reverse(given)
        self.frames = list(reversed(given)) if self.reverse else given
        self.cuts: dict[int, int] = {}  # gap -> photos in it that belong to the segment on the left
        self.own: dict[str, dict[str, Any]] = {}  # frame key -> the values set on that frame
        self.placed: dict[str, int] = {}  # photo path -> gap
        self.typed: dict[str, datetime] = {}  # frame key -> local time
        self.references: list[ReferencePhoto] = []
        self.own_places: dict[str, tuple[float, float]] = {}  # frame key -> a place the user set on the Capture card
        self.kept: dict[str, FrameTime] = {}  # frame key -> its result from the last Apply
        self.released: set[str] = set()  # frames whose segment changed since the dialog opened
        self._load_stored_values()
        if reverse is not None:
            self.set_reverse(reverse)

    def set_reverse(self, reverse: bool) -> None:
        """Turn the film round. Photos and cuts stay between the same two frames; values stay on
        the frames that set them."""
        if reverse == self.reverse:
            return
        self.released.update(f.key for f in self.frames)
        n = len(self.frames)
        cuts = {n - 2 - g: len(self.refs_at(g)) - self._split(g) for g in self.cuts}
        self.frames.reverse()
        self.placed = {path: n - 2 - g for path, g in self.placed.items()}
        self.cuts, self.reverse = cuts, reverse

    def position_of(self, frame_key: str) -> int:
        return next((i for i, f in enumerate(self.frames) if f.key == frame_key), -1)

    # ── segments ──────────────────────────────────────────────────────────
    def bounds(self) -> list[tuple[int, int]]:
        """(first, last) frame positions of each segment."""
        bounds, start = [], 0
        for i in range(len(self.frames)):
            if i == len(self.frames) - 1 or i in self.cuts:
                bounds.append((start, i))
                start = i + 1
        return bounds

    def segment_at(self, position: int) -> int:
        return next(n for n, (a, b) in enumerate(self.bounds()) if a <= position <= b)

    def toggle_cut(self, position: int, photos_before: int = 0) -> None:
        """Cut in gap `position`, after its first `photos_before` photos. The same cut again joins
        the segments; another place in the same gap moves the cut. The first frame of a new
        segment keeps the values it had, and takes its day from its first photo."""
        if not 0 <= position < len(self.frames) - 1:
            return
        self._release(position, position + 1)
        k = max(0, min(photos_before, len(self.refs_at(position))))
        right_key = self.frames[position + 1].key
        if position in self.cuts:
            if self._split(position) != k:
                self.cuts[position] = k
                return
            del self.cuts[position]
            # A value the new segment's first frame only carried over now repeats the frame before it.
            carried = self.values_at(position)
            own = self.own.get(right_key, {})
            for name in [n for n in own if own[n] == carried.get(n)]:
                del own[name]
            if not own:
                self.own.pop(right_key, None)
            return
        carried = self.values_at(position + 1)
        n = self.segment_at(position)
        _, last = self.bounds()[n]
        photos = self.refs_at(position)[k:] + [p for g in range(position + 1, last + 1) for p in self.refs_at(g)]
        photos = sorted((p for p in photos if p.local is not None), key=photo_moment)
        self.cuts[position] = k
        own = self.own.setdefault(right_key, {})
        for name in FIELDS:
            if name not in own and carried.get(name) not in (None, ""):
                own[name] = carried.get(name)
        if photos:
            own["day"] = photos[0].local.date()
        if not own:
            self.own.pop(right_key, None)

    def split_of(self, gap: int) -> Optional[int]:
        """How many photos of a cut gap belong to the segment on the left, or None without a cut."""
        return self._split(gap) if gap in self.cuts else None

    def _split(self, gap: int) -> int:
        return max(0, min(self.cuts.get(gap, 0), len(self.refs_at(gap))))

    def segment_photos(self, segment: int) -> list[tuple[int, ReferencePhoto]]:
        """(gap, photo) for the photos of a segment, in film order."""
        first, last = self.bounds()[segment]
        out = []
        for gap in range(first - 1, last + 1):
            photos = self.refs_at(gap)
            if gap == first - 1 and first > 0:
                photos = photos[self._split(gap) :]
            if gap == last and last < len(self.frames) - 1:
                photos = photos[: self._split(gap)]
            out.extend((gap, p) for p in photos)
        return out

    def segment_of_photo(self, path: str) -> Optional[int]:
        return next((n for n in range(len(self.bounds())) if any(p.path == path for _, p in self.segment_photos(n))), None)

    def _release(self, *positions: int) -> None:
        """The segments of these frame positions changed: work their times out again."""
        for position in positions:
            if 0 <= position < len(self.frames):
                first, last = self.bounds()[self.segment_at(position)]
                self.released.update(f.key for f in self.frames[first : last + 1])

    def is_kept(self, segment: int) -> bool:
        """True while a segment shows the times of the last Apply: every frame has one, and nothing
        in the segment changed."""
        first, last = self.bounds()[segment]
        return all(f.key in self.kept and f.key not in self.released for f in self.frames[first : last + 1])

    # ── values on frames ──────────────────────────────────────────────────
    def values_at(self, position: int) -> FrameValues:
        """The values in effect at a frame: its own, else the nearest earlier frame's in its segment."""
        first, _ = self.bounds()[self.segment_at(position)]
        values: dict[str, Any] = {}
        origins: dict[str, str] = {}
        for frame in self.frames[first : position + 1]:
            for name, value in self.own.get(frame.key, {}).items():
                values[name], origins[name] = value, frame.key
        return FrameValues(values.get("day"), values.get("place"), values.get("zone", ""), values.get("step"), origins)

    def own_of(self, position: int) -> dict[str, Any]:
        return dict(self.own.get(self.frames[position].key, {}))

    def set_own(self, position: int, **values: Any) -> None:
        """Set values on a frame. They apply to the frames after it in its segment, up to the next
        frame with its own value."""
        assert set(values) <= set(FIELDS), values
        self.own.setdefault(self.frames[position].key, {}).update(values)
        self._release(position)

    def clear_own(self, position: int, *names: str) -> None:
        """The frame follows the frame before it again for these values."""
        key = self.frames[position].key
        own = self.own.get(key, {})
        for name in names:
            own.pop(name, None)
        if not own:
            self.own.pop(key, None)
        self._release(position)

    def zone_at(self, position: int) -> str:
        """The zone of the values at a frame: the chosen zone, else the zone of the chosen place."""
        v = self.values_at(position)
        return capture_zone(v.zone, v.place.latitude if v.place else None, v.place.longitude if v.place else None)

    # ── reference photos ──────────────────────────────────────────────────
    def add_references(self, photos: Sequence[ReferencePhoto]) -> None:
        """Add photos once each."""
        for photo in photos:
            if all(p.path != photo.path for p in self.references):
                self.references.append(photo)
        # Photos with a known moment first, in time order; the rest by local time.
        self.references.sort(key=lambda p: (0, p.instant, p.path) if p.instant else (1, p.local or datetime.max, p.path))

    def remove_reference(self, path: str) -> None:
        """Take a photo out of the list, and out of the strip."""
        self.remove(path)
        self.references = [p for p in self.references if p.path != path]

    def references_on(self, day: Optional[date]) -> list[ReferencePhoto]:
        """The reference photos taken on `day`, or all of them for None."""
        if day is None:
            return list(self.references)
        return [p for p in self.references if p.local is not None and p.local.date() == day]

    def photo(self, path: str) -> Optional[ReferencePhoto]:
        return next((p for p in self.references if p.path == path), None)

    def gap_of(self, path: str) -> Optional[int]:
        return self.placed.get(path)

    def refs_at(self, gap: int) -> list[ReferencePhoto]:
        """The photos in a gap, in time order."""
        photos = [self.photo(p) for p, g in self.placed.items() if g == gap]
        return sorted((p for p in photos if p is not None), key=photo_moment)

    def place(self, path: str, gap: int) -> bool:
        """Put a photo in a gap. A photo has one place: False when it has one, or has no time."""
        photo = self.photo(path)
        if photo is None or photo.local is None or path in self.placed or not -1 <= gap < len(self.frames):
            return False
        self.placed[path] = gap
        self._release(gap, gap + 1)
        return True

    def move(self, path: str, gap: int) -> bool:
        if path not in self.placed or not -1 <= gap < len(self.frames):
            return False
        self._release(self.placed[path], self.placed[path] + 1, gap, gap + 1)
        self.placed[path] = gap
        return True

    def remove(self, path: str) -> None:
        if path in self.placed:
            gap = self.placed.pop(path)
            self._release(gap, gap + 1)

    def set_typed(self, frame_key: str, local: datetime) -> None:
        self.typed[frame_key] = local
        self._release(self.position_of(frame_key))

    def remove_typed(self, frame_key: str) -> None:
        if self.typed.pop(frame_key, None) is not None:
            self._release(self.position_of(frame_key))

    # ── results ───────────────────────────────────────────────────────────
    def compute(self, scan_times: Optional[dict[str, datetime]] = None) -> tuple[dict[str, FrameTime], list[Issue]]:
        results: dict[str, FrameTime] = {}
        issues: list[Issue] = []
        for n, (first, last) in enumerate(self.bounds()):
            keys = [f.key for f in self.frames[first : last + 1]]
            if self.is_kept(n):
                results.update({k: self.kept[k] for k in keys})
                continue
            references = [
                (gap - first + 1, Reference(Path(photo.path).name, photo.local, photo.offset, photo.latitude, photo.longitude))
                for gap, photo in self.segment_photos(n)
                if photo.local is not None
            ]
            segment = Segment(
                frames=tuple(keys),
                references=tuple(references),
                typed={k: v for k, v in self.typed.items() if k in keys},
                places={k: self.own_places[k] for k in keys if k in self.own_places},
                frame_settings={self.frames[p].key: self._frame_settings(p) for p in range(first, last + 1)},
            )
            seg_results, seg_issues = assign_times(segment, scan_times)
            results.update({r.frame: r for r in seg_results})
            issues.extend(seg_issues)
        issues.extend(self._segment_order_issues(results))
        return results, issues

    def _frame_settings(self, position: int) -> FrameSettings:
        v = self.values_at(position)
        lat, lon = (v.place.latitude, v.place.longitude) if v.place else (None, None)
        return FrameSettings(v.day, lat, lon, capture_zone(v.zone, lat, lon), is_zone(v.zone), v.step)

    def changes(self, results: dict[str, FrameTime], only: Optional[set[str]] = None) -> dict[str, dict]:
        """MetadataConfig fields per frame key, for session.assign_capture_times. `only` limits the
        frames written; the others still count for the times."""
        out: dict[str, dict] = {}
        for first, last in self.bounds():
            for position in range(first, last + 1):
                frame = self.frames[position]
                result = results.get(frame.key)
                fields = result.metadata_fields() if result is not None else {}
                if not fields or (only is not None and frame.key not in only):
                    continue
                if not result.kept:
                    fields.update(self._names(position, frame, result))
                entry: dict[str, Any] = {"v": PLAN_VERSION, "place": result.place_source}
                entry["gps"] = [result.latitude, result.longitude] if result.latitude is not None and result.longitude is not None else None
                if position == first:
                    entry["segment"] = {"reverse": self.reverse, "split": self._split(position - 1) if position > 0 else 0}
                if frame.key in self.own:
                    entry["own"] = _own_entry(self.own[frame.key])
                fields["capture_plan"] = json.dumps(entry, separators=(",", ":"), sort_keys=True)
                out[frame.key] = fields
        return out

    def _names(self, position: int, frame: PlanFrame, result: FrameTime) -> dict:
        """Place names to go with the place: the chosen place's names, the frame's own names, or
        none for a photo's place, which has coordinates only."""
        if result.place_source == PLACE_SEGMENT:
            place = self.values_at(position).place
            if place is not None:
                return {"location_city": place.city, "location_state": place.state, "location_country": place.country}
        if result.place_source == PLACE_FRAME:
            m = frame.metadata
            return {"location_city": m.location_city, "location_state": m.location_state, "location_country": m.location_country}
        return {"location_city": "", "location_state": "", "location_country": ""}

    def _segment_order_issues(self, results: dict[str, FrameTime]) -> list[Issue]:
        """A warning for each segment whose first frame, in film order, comes before the last frame
        of the segment before it. Compares instants when both frames have an offset."""
        runs = [[f.key for f in self.frames[a : b + 1]] for a, b in self.bounds()]
        timed = [[results[k] for k in run if k in results and results[k].local is not None] for run in runs]
        issues = []
        for before, after in zip(timed, timed[1:]):
            if before and after and _moment(after[0], before[-1]) < _moment(before[-1], after[0]):
                issues.append(Issue(SEGMENTS_OUT_OF_ORDER, (after[0].frame, before[-1].frame)))
        return issues

    # ── re-open ───────────────────────────────────────────────────────────
    def _load_stored_values(self) -> None:
        """Rebuild cuts, the values set on frames and typed times from what the frames hold, and keep
        each frame's result from the last Apply. Photos are not kept: the dialog opens without them."""
        for position, frame in enumerate(self.frames):
            m = frame.metadata
            entry = _read_entry(m.capture_plan)
            segment = entry.get("segment")
            if segment is not None and position > 0:
                self.cuts[position - 1] = int(segment.get("split", 0) or 0)
            own = _own_from_entry(entry)
            if not entry and position == 0:
                own = {
                    name: value
                    for name, value in (("day", day_of(m.capture_date)), ("zone", m.capture_timezone))
                    if value and (name != "zone" or is_zone(value))
                }
            if own:
                self.own[frame.key] = own
            kept = _kept_result(frame.key, m, entry)
            if kept is not None:
                self.kept[frame.key] = kept
            if m.capture_time_source == SOURCE_TYPED:
                parsed = parse_capture_date(m.capture_date)
                if parsed is not None and parsed.local() is not None:
                    self.typed[frame.key] = parsed.local()
            if _is_own_place(m, entry):
                self.own_places[frame.key] = (m.gps_latitude, m.gps_longitude)


def _own_entry(own: dict[str, Any]) -> dict:
    out: dict[str, Any] = {}
    if own.get("day") is not None:
        out["day"] = own["day"].isoformat()
    place = own.get("place")
    if place is not None:
        out["place"] = {"lat": place.latitude, "lon": place.longitude, "city": place.city, "state": place.state, "country": place.country}
    if own.get("zone"):
        out["zone"] = own["zone"]
    if own.get("step"):
        out["step"] = int(own["step"].total_seconds())
    return out


def _own_from_entry(entry: dict) -> dict[str, Any]:
    """The values a frame set, from its stored entry. A version 1 entry held them on the segment."""
    data = entry.get("own") if entry.get("v") == PLAN_VERSION else entry.get("segment")
    if not isinstance(data, dict):
        return {}
    own: dict[str, Any] = {}
    try:
        if data.get("day"):
            own["day"] = date.fromisoformat(data["day"])
        place = data.get("place")
        if isinstance(place, dict):
            own["place"] = Place(place["lat"], place["lon"], place.get("city", ""), place.get("state", ""), place.get("country", ""))
        elif data.get("lat") is not None and data.get("lon") is not None:
            own["place"] = Place(data["lat"], data["lon"], data.get("city", ""), data.get("state", ""), data.get("country", ""))
        if data.get("zone"):
            own["zone"] = data["zone"]
        if data.get("step"):
            own["step"] = timedelta(seconds=int(data["step"]))
    except (KeyError, TypeError, ValueError):
        return {}
    return own


def _kept_result(key: str, m: MetadataConfig, entry: dict) -> Optional[FrameTime]:
    """A frame's result from the last Apply, from what it stores. None for a frame without one."""
    if not entry or not m.capture_time_source:
        return None
    parsed = parse_capture_date(m.capture_date)
    local = parsed.local() if parsed is not None else None
    if local is None:
        return None
    offset = resolve_local(local, m.capture_timezone).utcoffset() if is_zone(m.capture_timezone) else None
    return FrameTime(
        frame=key,
        local=local,
        offset=offset,
        zone=m.capture_timezone,
        latitude=m.gps_latitude,
        longitude=m.gps_longitude,
        source=m.capture_time_source,
        references=tuple(m.capture_anchor_ref.split()),
        place_source=entry.get("place", ""),
        kept=True,
    )


def _stored_reverse(frames: list[PlanFrame]) -> bool:
    """The direction the frames were last applied in."""
    for frame in frames:
        segment = _read_entry(frame.metadata.capture_plan).get("segment")
        if segment is not None:
            return bool(segment.get("reverse", False))
    return False


def _read_entry(text: str) -> dict:
    if not text:
        return {}
    try:
        entry = json.loads(text)
    except ValueError:
        return {}
    return entry if isinstance(entry, dict) and entry.get("v") in (1, PLAN_VERSION) else {}


def _is_own_place(m: MetadataConfig, entry: dict) -> bool:
    """True when the user set the frame's place: before Assign Capture Times wrote it, kept by it,
    or changed after it."""
    if m.gps_latitude is None or m.gps_longitude is None:
        return False
    if not entry or entry.get("place") == PLACE_FRAME:
        return True
    written = entry.get("gps")
    return written is not None and written != [m.gps_latitude, m.gps_longitude]


def _moment(frame: FrameTime, other: FrameTime) -> datetime:
    """The UTC instant when both frames have an offset, else the local time. Only timed frames come here."""
    local = frame.local or datetime.min
    if frame.offset is not None and other.offset is not None:
        return local - frame.offset
    return local
