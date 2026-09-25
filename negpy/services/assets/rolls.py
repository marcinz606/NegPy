"""Roll membership: a Roll is a named, navigable group of frames -- either a real
library folder recognized as a roll, or a virtual roll built by hand from whatever the
Film Strip currently holds (a search result, a hand-picked selection, extras added to a
folder roll that are not physically in that folder).

Edits are not stored here and are not scoped by roll by default: they stay in the edits
DB under each frame's own content hash, exactly as if no Roll existed. A Roll only
decides which files show up when you open it -- with two exceptions. A path a user has
explicitly forked (``forked_paths``) gets its own edit identity for that roll alone,
suffixed onto the frame's content hash (``roll_edit_hash``), the same convention
half-frame scans already use for their two halves. And roll-wide defaults, below, hold
a handful of film, rig and scanning facts that describe the roll rather than one
frame's own look.
"""

import os
import time
import uuid
from dataclasses import replace
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from negpy.features.metadata.models import GEAR_FIELDS, PROCESS_FIELDS, SCANNING_FIELDS
from negpy.features.process.models import neutral_axis_tuple
from negpy.services.assets.library import folder_counts

if TYPE_CHECKING:
    from negpy.domain.models import WorkspaceConfig

ROLLS_KEY = "rolls_by_id"
IMPORT_SOURCES_KEY = "roll_import_sources"
DISMISSED_FOLDERS_KEY = "dismissed_folder_rolls"
ROLL_PATH_SEP = "/"
DISCOVERY_FILTERS_KEY = "roll_discovery_filters"
DEFAULT_DISCOVERY_FILTERS = ("export",)
_FORK_SEP = "#roll:"


def _read(repo: Any) -> Dict[str, dict]:
    saved = repo.get_global_setting(ROLLS_KEY, default=None)
    return dict(saved) if isinstance(saved, dict) else {}


def _write(repo: Any, store: Dict[str, dict]) -> None:
    repo.save_global_setting(ROLLS_KEY, store)


def saved_rolls(repo: Any) -> Dict[str, dict]:
    """Every remembered roll, keyed by id."""
    return _read(repo)


def roll_for_id(repo: Any, roll_id: str) -> Optional[dict]:
    return _read(repo).get(roll_id)


def _folder_key(path: str) -> str:
    """*path* in the form two spellings of one folder share: separators, case on Windows."""
    return os.path.normcase(os.path.normpath(path))


def folder_roll_id_for_path(repo: Any, path: str) -> Optional[str]:
    """The id of the roll recognizing *path*, or None if not yet recognized."""
    key = _folder_key(path)
    for roll_id, entry in _read(repo).items():
        if entry.get("kind") == "folder" and _folder_key(entry.get("folder_path") or "") == key:
            return roll_id
    return None


def recognize_folder(repo: Any, path: str, name: str = "") -> str:
    """Mark *path* as a recognized folder roll. Idempotent: returns the existing id
    when the folder is already recognized, without touching its stored name."""
    dismissed = _dismissed_folders(repo)
    kept = [p for p in dismissed if _folder_key(p) != _folder_key(path)]
    if kept != dismissed:
        repo.save_global_setting(DISMISSED_FOLDERS_KEY, kept)
    existing = folder_roll_id_for_path(repo, path)
    if existing:
        return existing
    store = _read(repo)
    roll_id = uuid.uuid4().hex
    store[roll_id] = {
        "kind": "folder",
        "name": name or path.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1] or path,
        "folder_path": path,
        "extra_paths": [],
        "created_at": time.time(),
    }
    _write(repo, store)
    return roll_id


def _dismissed_folders(repo: Any) -> List[str]:
    saved = repo.get_global_setting(DISMISSED_FOLDERS_KEY, default=None)
    return [p for p in saved if isinstance(p, str)] if isinstance(saved, list) else []


def import_sources(repo: Any) -> List[str]:
    """Parents imported with Import Subfolders as Rolls, which a Library refresh walks again."""
    saved = repo.get_global_setting(IMPORT_SOURCES_KEY, default=None)
    return [p for p in saved if isinstance(p, str)] if isinstance(saved, list) else []


def discovery_filters(repo: Any) -> List[str]:
    """Folder-name filters discovery skips, with their subfolders."""
    saved = repo.get_global_setting(DISCOVERY_FILTERS_KEY, default=None)
    if not isinstance(saved, list):
        return list(DEFAULT_DISCOVERY_FILTERS)
    return [p for p in saved if isinstance(p, str)]


def set_discovery_filters(repo: Any, filters: List[str]) -> None:
    repo.save_global_setting(DISCOVERY_FILTERS_KEY, [f.strip() for f in filters if f.strip()])


def matches_discovery_filter(name: str, filters: Sequence[str]) -> bool:
    """A filter with ``*`` matches the whole name, one without matches any part; case is ignored."""
    name = name.casefold()
    return any(fnmatchcase(name, f.casefold()) if "*" in f else f.casefold() in name for f in filters)


def discover_roll_folders(parent_path: str, filters: Sequence[str]) -> List[str]:
    """Every folder at or under *parent_path* that holds images directly.

    The walk does not enter a roll folder, so its own subfolders (export output, for
    one) never become rolls. Hidden folders and folders matching *filters* are skipped.
    """
    found = []
    for dirpath, dirnames, _filenames in os.walk(os.path.normpath(parent_path)):
        if folder_counts(dirpath)[0]:
            found.append(dirpath)
            dirnames[:] = []
        else:
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and not matches_discovery_filter(d, filters))
    return found


def import_subfolders_as_rolls(repo: Any, parent_path: str, *, skip_dismissed: bool = False) -> List[str]:
    """Recognize every roll folder under *parent_path* and remember it as an import source.

    Idempotent per folder, so a re-run only creates the missing rolls. A roll is named by
    its path from the folder that holds *parent_path* ("20260901/kentmere_400_1").
    *skip_dismissed* leaves out folders whose roll was deleted.
    """
    parent_path = os.path.normpath(parent_path)
    paths = discover_roll_folders(parent_path, discovery_filters(repo))
    if not paths:
        return []
    sources = import_sources(repo)
    if _folder_key(parent_path) not in {_folder_key(p) for p in sources}:
        repo.save_global_setting(IMPORT_SOURCES_KEY, [*sources, parent_path])
    if skip_dismissed:
        dismissed = {_folder_key(p) for p in _dismissed_folders(repo)}
        paths = [p for p in paths if _folder_key(p) not in dismissed]
    # The name is a label, so it joins with "/" on every OS; ROLL_PATH_SEP splits it back.
    base = os.path.dirname(parent_path)
    return [recognize_folder(repo, path, ROLL_PATH_SEP.join(os.path.relpath(path, base).split(os.sep))) for path in paths]


def _under(key: str, folder_key: str) -> bool:
    return key == folder_key or key.startswith(folder_key.rstrip(os.sep) + os.sep)


def prune_filtered_rolls(repo: Any) -> int:
    """Drop folder rolls under an import source whose path from it matches a discovery
    filter; returns how many. Not a delete, so removing the filter brings them back."""
    filters = discovery_filters(repo)
    sources = [_folder_key(p) for p in import_sources(repo)]
    store = _read(repo)
    dropped = []
    for roll_id, entry in store.items():
        if entry.get("kind") != "folder":
            continue
        key = _folder_key(entry.get("folder_path") or "")
        for source in sources:
            if key != source and _under(key, source):
                if any(matches_discovery_filter(part, filters) for part in os.path.relpath(key, source).split(os.sep)):
                    dropped.append(roll_id)
                break
    for roll_id in dropped:
        del store[roll_id]
    if dropped:
        _write(repo, store)
    return len(dropped)


def delete_folder_rolls(repo: Any, folder: str) -> None:
    """Delete every folder roll at or under *folder*, and forget import sources there."""
    key = _folder_key(folder)
    ids = [
        roll_id
        for roll_id, entry in _read(repo).items()
        if entry.get("kind") == "folder" and _under(_folder_key(entry.get("folder_path") or ""), key)
    ]
    for roll_id in ids:
        delete_roll(repo, roll_id)
    sources = import_sources(repo)
    kept = [p for p in sources if not _under(_folder_key(p), key)]
    if kept != sources:
        repo.save_global_setting(IMPORT_SOURCES_KEY, kept)


def create_virtual_roll(repo: Any, name: str, member_paths: List[str]) -> str:
    store = _read(repo)
    roll_id = uuid.uuid4().hex
    store[roll_id] = {
        "kind": "virtual",
        "name": name,
        "member_paths": list(member_paths),
        "created_at": time.time(),
    }
    _write(repo, store)
    return roll_id


def add_extra_members(repo: Any, roll_id: str, paths: List[str]) -> None:
    """Extend a roll's membership, in one write: a folder roll's extra_paths, or a virtual
    roll's member_paths. Skips an unknown roll id and paths already members."""
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None:
        return
    key = "extra_paths" if entry["kind"] == "folder" else "member_paths"
    known = set(entry[key])
    new = [p for p in dict.fromkeys(paths) if p not in known]
    if new:
        entry[key] = [*entry[key], *new]
        _write(repo, store)


def roll_edit_hash(from_hash: str, roll_id: str) -> str:
    """The independent edit identity a fork of *from_hash* uses under *roll_id*.

    *from_hash* is whatever hash the asset currently resolves to -- the plain content
    hash for a whole frame, or an already-suffixed one (``#1``/``#2``) for a half-frame
    scan -- so each half forks to its own identity rather than collapsing together.
    """
    return f"{from_hash}{_FORK_SEP}{roll_id}"


def unforked_hash(file_hash: str) -> str:
    """*file_hash* with only a trailing ``#roll:<id>`` suffix removed, if present.

    Half-frame (``#1``/``#2``) and composite (``#stitch``/``#hdr``) hashes are left
    untouched -- those already carry their own independently-scoped edits and marks, and
    only a roll-fork is meant to share marks with whatever it was forked from.
    """
    idx = file_hash.find(_FORK_SEP)
    return file_hash[:idx] if idx != -1 else file_hash


def is_forked(repo: Any, roll_id: str, from_hash: str) -> bool:
    """Whether *from_hash* has its own edit under *roll_id*, rather than the shared one.

    Keyed on the asset's exact pre-fork hash, not its path: a half-frame scan's two
    halves have different hashes, so forking one never silently drags the other along.
    """
    entry = _read(repo).get(roll_id)
    return bool(entry) and from_hash in entry.get("forked_hashes", [])


def fork_edit(repo: Any, roll_id: str, from_hash: str, source_path: str, config: "WorkspaceConfig") -> str:
    """Give *from_hash* its own edit under *roll_id*, seeded from *config* (the shared
    edit at fork time). Returns the forked hash. Idempotent: re-forking an
    already-forked hash only re-seeds it -- callers fork once and edit the result from
    then on, so this never runs twice for the same hash in practice."""
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None:
        return from_hash
    forked = roll_edit_hash(from_hash, roll_id)
    if from_hash not in entry.get("forked_hashes", []):
        entry["forked_hashes"] = [*entry.get("forked_hashes", []), from_hash]
        _write(repo, store)
    repo.save_file_settings(forked, config, file_path=source_path)
    return forked


def unfork_edit(repo: Any, roll_id: str, from_hash: str) -> None:
    """Undo `fork_edit`: drop *from_hash* from *roll_id*'s forked hashes and delete the
    forked edit, its history and its work prints. The shared edit under *from_hash*
    itself is untouched."""
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is not None and from_hash in entry.get("forked_hashes", []):
        entry["forked_hashes"] = [h for h in entry["forked_hashes"] if h != from_hash]
        _write(repo, store)
    repo.delete_file_settings(roll_edit_hash(from_hash, roll_id))


def rolls_containing_path(repo: Any, path: str) -> List[str]:
    """Every roll *path* belongs to: under a folder roll's own folder or in its
    extra_paths, or listed in a virtual roll's member_paths. A cheap path comparison
    against the small rolls store, not a disk walk -- good enough to decide whether
    forking a path is meaningful (it is shared with at least one other roll)."""
    norm = os.path.normcase(os.path.abspath(path))
    out = []
    for roll_id, entry in _read(repo).items():
        if entry.get("kind") == "folder":
            folder = entry.get("folder_path", "")
            folder_norm = os.path.normcase(os.path.abspath(folder)) if folder else ""
            in_folder = bool(folder_norm) and norm.startswith(folder_norm + os.sep)
            if in_folder or path in entry.get("extra_paths", []):
                out.append(roll_id)
        elif path in entry.get("member_paths", []):
            out.append(roll_id)
    return out


def rename_roll(repo: Any, roll_id: str, name: str) -> None:
    store = _read(repo)
    if roll_id in store:
        store[roll_id]["name"] = name
        _write(repo, store)


def rename_folder_roll_disk(repo: Any, roll_id: str, new_name: str) -> Optional[str]:
    """Rename a folder roll's actual folder on disk to *new_name*, in its current
    parent directory, and update the roll's own folder_path to match. Returns the
    new path, or None (and nothing is touched) when the roll is not a folder roll,
    its folder is missing, a sibling is already named that, or the OS rename fails
    (no permission, a mount that refuses it). *new_name* equal to the folder's
    current basename is a no-op success, not a collision.
    """
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None or entry.get("kind") != "folder":
        return None
    old_path = entry.get("folder_path", "")
    if not old_path or not os.path.isdir(old_path):
        return None
    parent = os.path.dirname(old_path.rstrip("/\\"))
    new_path = os.path.join(parent, new_name)
    if os.path.normcase(os.path.abspath(new_path)) == os.path.normcase(os.path.abspath(old_path)):
        return old_path
    if os.path.exists(new_path):
        return None
    try:
        os.rename(old_path, new_path)
    except OSError:
        return None
    entry["folder_path"] = new_path
    _write(repo, store)
    return new_path


def delete_roll(repo: Any, roll_id: str) -> None:
    """Forget a roll. A deleted folder roll is not recognized again by a Library refresh."""
    store = _read(repo)
    entry = store.pop(roll_id, None)
    if entry is None:
        return
    _write(repo, store)
    path = entry.get("folder_path")
    if entry.get("kind") == "folder" and path:
        dismissed = _dismissed_folders(repo)
        if _folder_key(path) not in {_folder_key(p) for p in dismissed}:
            repo.save_global_setting(DISMISSED_FOLDERS_KEY, [*dismissed, path])


def all_rolls_sorted(repo: Any) -> List[tuple]:
    """(roll_id, entry) pairs for every roll, folder and virtual alike, name-sorted."""
    return sorted(_read(repo).items(), key=lambda pair: pair[1].get("name", "").casefold())


def virtual_rolls(repo: Any) -> List[tuple]:
    """(roll_id, entry) pairs for every virtual roll, name-sorted."""
    return [pair for pair in all_rolls_sorted(repo) if pair[1].get("kind") == "virtual"]


# --- Roll-wide defaults ----------------------------------------------------------
#
# The fields each Roll-tab card edits, as (WorkspaceConfig section, field names) keyed
# by the card that owns them -- the same grouping a per-card lock button unlocks. A
# card names one section; the stored `defaults` dict stays flat, which WorkspaceConfig's
# own flat key namespace already makes unambiguous. White/Black Point and their trims
# stay off this list: they are exposure choices that legitimately vary shot to shot
# within a roll, unlike these, which describe the rig or the roll's own shared baseline.
ROLL_DEFAULT_FIELDS: Dict[str, tuple] = {
    # Which film type the roll is, and whether it is already a finished positive --
    # edited and locked away from the roll exactly like every other card here, even
    # though a roll being one film type, scanned one way, means the common case is
    # every frame following it.
    "film": ("process", ("process_mode", "positive_source")),
    "sensor": (
        "process",
        (
            "linear_raw",
            "narrowband_scan",
            "sensor_profile",
            "sensor_matrix",
            "crosstalk_strength",
            "crosstalk_profile",
            "crosstalk_matrix",
            # Baked alongside the profile+matrix so the render can gate the unmix on it
            # without disk I/O -- travels with them, or another frame's roll-derived
            # crosstalk would be read back through its own, unrelated film process.
            "crosstalk_process",
            "hue_trim",
        ),
    ),
    "demosaic": ("process", ("demosaic_preview", "demosaic_export", "highlight_reconstruction")),
    "process": (
        "process",
        (
            "e6_normalize",
            "analysis_buffer",
            "luma_range_clip",
            "color_range_clip",
            # Film-base (Dmin) and Dmax corrections, per dye layer: a fact of the stock
            # and its development, not of one frame.
            "white_point_offset",
            "black_point_offset",
            "white_point_trim_red",
            "white_point_trim_green",
            "white_point_trim_blue",
            "black_point_trim_red",
            "black_point_trim_green",
            "black_point_trim_blue",
        ),
    ),
    # Which baseline this frame's bounds come from -- the roll's shared meter or its own
    # analysis. locked_floors/locked_ceils themselves stay Roll Analysis's own job to
    # spread (a metering run, not an edit).
    "baseline": ("process", ("use_luma_average", "use_color_average", "use_cast_average")),
    # The film edge, the rebate width and the format's shape are properties of the roll,
    # not of one frame. The rect autocrop finds from them is not: it stays each frame's own.
    "autocrop": ("geometry", ("autocrop_mode", "autocrop_offset", "autocrop_rebate_trim", "autocrop_ratio")),
    "lens": ("geometry", ("distortion_k1", "lens_distortion_from_metadata", "lens_ca_from_metadata")),
    # profile_id also has a rig-global fallback, applied upstream of roll defaults, so a
    # roll that names no profile of its own still gets the active one.
    "flatfield": ("flatfield", ("apply", "profile_id")),
    # Metadata describes the roll almost by definition: one camera, one stock, one
    # development, one scanning rig. capture_frame is unique to one frame, and
    # protect_original_metadata and description_fields are export decisions, so all three
    # stay off the list.
    "metadata_gear": ("metadata", GEAR_FIELDS),
    "metadata_capture": (
        "metadata",
        ("capture_date", "gps_latitude", "gps_longitude", "location_city", "location_state", "location_country"),
    ),
    "metadata_process": ("metadata", PROCESS_FIELDS),
    "metadata_scanning": ("metadata", SCANNING_FIELDS + ("capture_roll",)),
    "metadata_exposure": ("metadata", ("exposure_override",)),
}


def card_fields(card_key: str) -> tuple:
    """The field names *card_key* owns, without its section."""
    return ROLL_DEFAULT_FIELDS[card_key][1]


def config_value(value: Any) -> Any:
    """*value* read back from this store with its tuples restored. The store is JSON, so
    a tuple comes back as a list at every depth."""
    if isinstance(value, (list, tuple)):
        return tuple(config_value(v) for v in value)
    return value


def same_value(a: Any, b: Any) -> bool:
    """Whether a config value equals one read back from this store."""
    return config_value(a) == config_value(b)


def roll_defaults(repo: Any, roll_id: str) -> Dict[str, Any]:
    """The roll's own value for each field it has set at least once. A field absent
    here has no roll default yet -- the frame's own saved value is what is used,
    exactly as before roll defaults existed."""
    entry = roll_for_id(repo, roll_id)
    return dict(entry["defaults"]) if entry and entry.get("defaults") else {}


def set_roll_defaults(repo: Any, roll_id: str, **fields: Any) -> None:
    """Set one or more roll-default field values (ROLL_DEFAULT_FIELDS' names, or
    process_mode). Every member frame that has not locked the owning card away from
    the roll picks this up as soon as it is next loaded or rendered. No-op for an
    unknown roll id."""
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None:
        return
    defaults = dict(entry.get("defaults", {}))
    defaults.update(fields)
    entry["defaults"] = defaults
    _write(repo, store)


def frame_override_cards(repo: Any, roll_id: str, file_hash: str) -> set:
    """Which of ROLL_DEFAULT_FIELDS' card keys this frame has locked to its own value,
    away from the roll's defaults, within this roll."""
    entry = roll_for_id(repo, roll_id)
    if not entry:
        return set()
    return set(entry.get("frame_overrides", {}).get(file_hash, ()))


def clear_frame_overrides(repo: Any, roll_id: str, file_hash: str) -> None:
    """Unlock every card for one frame within one roll, so it follows the roll's
    defaults again. No-op for an unknown roll id or a frame with nothing locked."""
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None or file_hash not in entry.get("frame_overrides", {}):
        return
    entry["frame_overrides"] = {h: cards for h, cards in entry["frame_overrides"].items() if h != file_hash}
    _write(repo, store)


def set_frame_override(repo: Any, roll_id: str, file_hash: str, card_key: str, locked: bool) -> None:
    """Lock (locked=True) or unlock (False) one card for one frame within one roll.
    Locking freezes that card at the frame's current (usually roll-default) value;
    unlocking reverts it to whatever the roll currently says. No-op for an unknown
    roll id."""
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None:
        return
    overrides = dict(entry.get("frame_overrides", {}))
    cards = set(overrides.get(file_hash, ()))
    if locked:
        cards.add(card_key)
    else:
        cards.discard(card_key)
    if cards:
        overrides[file_hash] = sorted(cards)
    else:
        overrides.pop(file_hash, None)
    entry["frame_overrides"] = overrides
    _write(repo, store)


def resolve_roll_config(repo: Any, roll_id: Optional[str], file_hash: str, config: "WorkspaceConfig") -> "WorkspaceConfig":
    """Overlay this roll's defaults onto *config* for every card the frame has not
    locked to its own value. No roll, no defaults set yet, or every relevant card
    locked leaves *config* unchanged."""
    if roll_id is None:
        return config
    defaults = roll_defaults(repo, roll_id)
    if not defaults:
        return config
    locked_cards = frame_override_cards(repo, roll_id, file_hash)
    by_section: Dict[str, Dict[str, Any]] = {}
    for card_key, (section, field_names) in ROLL_DEFAULT_FIELDS.items():
        if card_key in locked_cards:
            continue
        for name in field_names:
            if name in defaults:
                by_section.setdefault(section, {})[name] = defaults[name]
    for section, updates in by_section.items():
        config = replace(config, **{section: replace(getattr(config, section), **updates)})
    return config


def section_push(repo: Any, roll_id: str, section_key: str) -> Dict[str, Any]:
    """What a frame-level card last pushed to this roll, field by field. Unlike
    ROLL_DEFAULT_FIELDS this never overlays onto a frame on open -- a look is a copy, not
    a binding -- it only records what the roll agreed on, so a card can say whether the
    frame in front of you still matches it."""
    entry = roll_for_id(repo, roll_id)
    pushes = entry.get("section_pushes", {}) if entry else {}
    return dict(pushes.get(section_key, {}))


def set_section_push(repo: Any, roll_id: str, section_key: str, values: Dict[str, Any]) -> None:
    """Records a whole-roll apply of *section_key*, merging into whatever it pushed
    before: applying two of a card's settings in two goes leaves both at the roll."""
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None:
        return
    pushes = dict(entry.get("section_pushes", {}))
    pushes[section_key] = {**pushes.get(section_key, {}), **values}
    entry["section_pushes"] = pushes
    _write(repo, store)


def roll_normalization(repo: Any, roll_id: str) -> Optional[Dict[str, Optional[tuple]]]:
    """The roll's saved Roll Analysis baseline (floors, ceils, cast, the pooled neutral
    ``axis`` or None, and the hashes of the frames that keep their own bounds, ``outliers``), or None if it
    has never been analyzed. Unlike roll_defaults, this is written only by Batch
    Analysis itself -- a metering run over the roll's files, not a per-frame edit --
    so a frame's own Use Luma/Color Average axes borrow it directly rather than
    through the lock/override machinery above."""
    entry = roll_for_id(repo, roll_id)
    saved = entry.get("normalization") if entry else None
    if not saved:
        return None
    return {
        "floors": tuple(saved["floors"]),
        "ceils": tuple(saved["ceils"]),
        "cast": tuple(saved["cast"]),
        "axis": _saved_axis(saved),
        "outliers": tuple(saved.get("outliers", ())),
    }


def _saved_axis(saved: dict) -> Optional[tuple]:
    axis = saved.get("axis")
    return neutral_axis_tuple(axis) if axis is not None else None


def set_roll_normalization(
    repo: Any,
    roll_id: str,
    floors: tuple,
    ceils: tuple,
    cast: tuple = (0.0, 0.0, 0.0),
    outliers: tuple = (),
    axis: Optional[tuple] = None,
) -> None:
    """Records a Roll Analysis result as the roll's own baseline, overwriting
    whatever was there. No-op for an unknown roll id."""
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None:
        return
    entry["normalization"] = {"floors": list(floors), "ceils": list(ceils), "cast": list(cast), "outliers": list(outliers), "axis": axis}
    _write(repo, store)


def roll_scenes(repo: Any, roll_id: Optional[str]) -> List[tuple]:
    """The roll's scenes as ``(scene_id, entry)`` pairs, in creation order: a scene's
    ordinal is its 1-based position here."""
    entry = roll_for_id(repo, roll_id) if roll_id else None
    return list((entry or {}).get("scenes", {}).items())


def _pull_members(scenes: Dict[str, dict], hashes: List[str]) -> None:
    """Removes *hashes* from every scene and drops a scene left empty: a frame is in at
    most one scene of a roll."""
    drop = set(hashes)
    for scene_id in list(scenes):
        members = [h for h in scenes[scene_id]["member_hashes"] if h not in drop]
        if members:
            scenes[scene_id] = {**scenes[scene_id], "member_hashes": members}
        else:
            del scenes[scene_id]


def _edit_scenes(repo: Any, roll_id: str, edit) -> Any:
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None:
        return None
    scenes = dict(entry.get("scenes", {}))
    result = edit(scenes)
    entry["scenes"] = scenes
    _write(repo, store)
    return result


def create_scene(repo: Any, roll_id: str, name: str, member_hashes: List[str]) -> Optional[str]:
    """Groups *member_hashes* (unforked content hashes) as a new scene, taking them out of
    any scene they were in. None for an unknown roll."""

    def edit(scenes: Dict[str, dict]) -> str:
        _pull_members(scenes, member_hashes)
        scene_id = uuid.uuid4().hex
        scenes[scene_id] = {"name": name, "member_hashes": list(dict.fromkeys(member_hashes)), "normalization": None}
        return scene_id

    return _edit_scenes(repo, roll_id, edit)


def add_to_scene(repo: Any, roll_id: str, scene_id: str, member_hashes: List[str]) -> None:
    def edit(scenes: Dict[str, dict]) -> None:
        if scene_id not in scenes:
            return
        target = scenes[scene_id]
        others = {sid: entry for sid, entry in scenes.items() if sid != scene_id}
        _pull_members(others, member_hashes)
        merged = {**target, "member_hashes": list(dict.fromkeys([*target["member_hashes"], *member_hashes]))}
        rebuilt = {sid: merged if sid == scene_id else others[sid] for sid in scenes if sid == scene_id or sid in others}
        scenes.clear()
        scenes.update(rebuilt)

    _edit_scenes(repo, roll_id, edit)


def remove_from_scenes(repo: Any, roll_id: str, member_hashes: List[str]) -> None:
    _edit_scenes(repo, roll_id, lambda scenes: _pull_members(scenes, member_hashes))


def rename_scene(repo: Any, roll_id: str, scene_id: str, name: str) -> None:
    def edit(scenes: Dict[str, dict]) -> None:
        if scene_id in scenes:
            scenes[scene_id] = {**scenes[scene_id], "name": name}

    _edit_scenes(repo, roll_id, edit)


def delete_scene(repo: Any, roll_id: str, scene_id: str) -> None:
    _edit_scenes(repo, roll_id, lambda scenes: scenes.pop(scene_id, None))


def scene_normalization(repo: Any, roll_id: str, scene_id: str) -> Optional[Dict[str, Optional[tuple]]]:
    """The scene's saved Scene Analysis baseline (floors, ceils, axis, outliers), or None before one."""
    saved = dict(roll_scenes(repo, roll_id)).get(scene_id, {}).get("normalization")
    if not saved:
        return None
    return {
        "floors": tuple(saved["floors"]),
        "ceils": tuple(saved["ceils"]),
        "axis": _saved_axis(saved),
        "outliers": tuple(saved.get("outliers", ())),
    }


def set_scene_normalization(
    repo: Any, roll_id: str, scene_id: str, floors: tuple, ceils: tuple, outliers: tuple = (), axis: Optional[tuple] = None
) -> None:
    def edit(scenes: Dict[str, dict]) -> None:
        if scene_id in scenes:
            saved = {"floors": list(floors), "ceils": list(ceils), "outliers": list(outliers), "axis": axis}
            scenes[scene_id] = {**scenes[scene_id], "normalization": saved}

    _edit_scenes(repo, roll_id, edit)


def scene_by_hash(repo: Any, roll_id: Optional[str]) -> Dict[str, tuple]:
    """``{unforked_hash: (ordinal, scene_id, name)}`` for every scene member of the roll."""
    return {h: (i, sid, entry["name"]) for i, (sid, entry) in enumerate(roll_scenes(repo, roll_id), 1) for h in entry["member_hashes"]}


def next_scene_name(repo: Any, roll_id: Optional[str]) -> str:
    taken = {entry["name"] for _sid, entry in roll_scenes(repo, roll_id)}
    n = len(taken) + 1
    while f"Scene {n}" in taken:
        n += 1
    return f"Scene {n}"


def resolve_roll_baseline(repo: Any, roll_id: str, file_hash: str, config: "WorkspaceConfig") -> "WorkspaceConfig":
    """A frame riding Use Luma/Color Average without a baseline of its own takes its
    scene's, else the roll's, neutral axis included: a frame loaded after the analysis ran
    still follows it. A frame that already carries one, or has Lock Bounds on, keeps it."""
    process = config.process
    if process.lock_bounds or process.is_locked_initialized or not (process.use_luma_average or process.use_color_average):
        return config
    scene = scene_by_hash(repo, roll_id).get(file_hash)
    saved = scene_normalization(repo, roll_id, scene[1]) if scene else roll_normalization(repo, roll_id)
    if not saved:
        return config
    source = f"scene:{scene[1]}" if scene else f"roll:{roll_id}"
    return replace(
        config,
        process=replace(
            process,
            locked_floors=saved["floors"],
            locked_ceils=saved["ceils"],
            locked_neutral_axis=saved["axis"],
            baseline_source=source,
        ),
    )


def baseline_label(repo: Any, process: Any) -> str:
    """What a frame's locked baseline was taken from, for display: “Roll …”, “Scene …” or
    “Frame …”. A baseline saved before sources were recorded names its roll_name, if any."""
    kind, _, ref = process.baseline_source.partition(":")
    if kind == "roll":
        entry = roll_for_id(repo, ref)
        return f"Roll “{entry['name']}”" if entry else "a deleted roll"
    if kind == "scene":
        for entry in _read(repo).values():
            scene = entry.get("scenes", {}).get(ref)
            if scene:
                return f"Scene “{scene['name']}”"
        return "a deleted scene"
    if kind == "frame":
        return f"Frame “{ref}”"
    return f"Roll “{process.roll_name}”" if process.roll_name else "a saved baseline"
