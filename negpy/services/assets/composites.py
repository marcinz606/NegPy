"""Composite membership: which files a stitch or an HDR merge is built from.

A composite is a user decision, not a fact about the files: nothing in a part scan
says which other scans it was stitched to, and no measurement can recover the choice.
So the grouping is remembered here, keyed by the composite's primary path, and it
lives until the user dissolves the composite — not until the file list changes.

Edits are not stored here. They stay in the edits DB under the composite's own
content hash (``stitch_hash`` / ``hdr_hash``), which is derived from the parts, so
re-forming the same composite finds its edit again.
"""

from typing import Any, Dict, Optional

from negpy.features.hdr.models import ANCHOR_EV_UNSET

COMPOSITES_KEY = "composites_by_path"

#: Legacy keys the store was seeded from: the open-file manifest carried the
#: registrations, so a composite lasted only as long as the file list that held it.
_LEGACY_KEYS = ("session_stitches", "session_hdr_merges")


def _read(repo: Any) -> Dict[str, dict]:
    saved = repo.get_global_setting(COMPOSITES_KEY, default=None)
    return dict(saved) if isinstance(saved, dict) else {}


def saved_composites(repo: Any) -> Dict[str, dict]:
    """Every remembered composite, keyed by primary path. Promotes the legacy
    session-manifest entries on first read."""
    store = _read(repo)
    if store:
        return store
    for key in _LEGACY_KEYS:
        legacy = repo.get_global_setting(key, default=None)
        if isinstance(legacy, dict):
            kind = "stitch" if key == "session_stitches" else "hdr"
            store.update({path: {**entry, "kind": kind} for path, entry in legacy.items() if isinstance(entry, dict)})
    if store:
        repo.save_global_setting(COMPOSITES_KEY, store)
    return store


def restore_maps(repo: Any) -> tuple:
    """The remembered composites split into the two ``(stitches, merges)`` maps that
    asset discovery re-attaches from."""
    store = saved_composites(repo)
    stitches = {path: entry for path, entry in store.items() if entry.get("kind") == "stitch"}
    merges = {path: entry for path, entry in store.items() if entry.get("kind") == "hdr"}
    return stitches, merges


def composite_entry(asset: dict) -> Optional[dict]:
    """The storable record of an asset's composite membership, or None when it is a
    plain frame. Settings that ride on the asset (align, render exposure) are part of
    the record: they are the composite's, not the edit's."""
    if asset.get("stitch_paths"):
        return {
            "kind": "stitch",
            "paths": list(asset["stitch_paths"]),
            "transforms": [list(t) for t in asset["stitch_transforms"]],
            "canvas": list(asset["stitch_canvas"]),
            "sizes": [list(s) for s in asset["stitch_sizes"]],
            "triplets": [list(t) for t in asset.get("stitch_triplets") or ()],
            "align": bool(asset.get("stitch_align", True)),
            "hash": asset["hash"],
            "process_mode": asset.get("process_mode", ""),
        }
    if asset.get("hdr_paths"):
        return {
            "kind": "hdr",
            "paths": list(asset["hdr_paths"]),
            "ratios": [float(r) for r in asset.get("hdr_ratios") or ()],
            "align": bool(asset.get("hdr_align", True)),
            "anchor": str(asset.get("hdr_anchor", "") or ""),
            "anchor_ev": float(asset.get("hdr_anchor_ev", ANCHOR_EV_UNSET)),
            "hash": asset["hash"],
            "process_mode": asset.get("process_mode", ""),
        }
    return None


def remember_composites(repo: Any, assets: Any) -> None:
    """Upsert the composites among ``assets``. An upsert, not a rewrite: assets holds
    one folder's frames, and composites in every other folder must survive loading it."""
    store = saved_composites(repo)
    updated = dict(store)
    for asset in assets:
        entry = composite_entry(asset)
        if entry:
            updated[asset["path"]] = entry
    if updated != store:
        repo.save_global_setting(COMPOSITES_KEY, updated)


def forget_composite(repo: Any, primary_path: Optional[str]) -> None:
    """Drop a composite, so its parts stay separate frames from now on."""
    store = saved_composites(repo)
    if primary_path in store:
        del store[primary_path]
        repo.save_global_setting(COMPOSITES_KEY, store)


def part_paths(entries: Any) -> set:
    """The non-primary files of the given composite records.

    A folder walk finds the parts as well as the primary, so discovery drops these:
    without it a stitch comes back beside the frames it is made of.
    """
    out = set()
    for entry in entries:
        out.update(p for p in entry.get("paths") or () if p)
        out.update(p for t in entry.get("triplets") or () for p in t if p)
    return out
