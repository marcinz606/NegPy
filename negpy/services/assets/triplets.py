"""Trichrome triplet membership, keyed by the red exposure's path, so a roll reopens without a raw read.

Each record holds the three content hashes: a file changed on disk since the grouping does not re-attach.
"""

from typing import Any, Dict

from negpy.services.assets.half_frame import base_hash

TRIPLETS_KEY = "triplets_by_path"


def saved_triplets(repo: Any) -> Dict[str, list]:
    """``{red_path: [green_path, blue_path, align, [red_hash, green_hash, blue_hash], by_hand]}``.
    A hash is empty when the grouping was made without it; a record without ``by_hand`` was grouped
    by Trichrome Mode."""
    saved = repo.get_global_setting(TRIPLETS_KEY, default=None)
    return dict(saved) if isinstance(saved, dict) else {}


def made_by_hand(record: Any) -> bool:
    """Whether the user grouped this triplet by hand. Such a triplet reopens with Trichrome Mode off too."""
    return len(record) > 4 and bool(record[4])


def applied_triplets(repo: Any, trichrome_mode: bool) -> Dict[str, list]:
    """The saved triplets a discovery re-attaches: all in Trichrome Mode, else those made by hand."""
    saved = saved_triplets(repo)
    return saved if trichrome_mode else {red: rec for red, rec in saved.items() if made_by_hand(rec)}


def triplet_record(asset: dict) -> list:
    # A roll fork or a half frame suffixes the asset's hash; discovery finds the file's own.
    hashes = [base_hash(asset.get("hash") or ""), asset.get("green_hash") or "", asset.get("blue_hash") or ""]
    return [asset["green_path"], asset["blue_path"], bool(asset.get("align", True)), hashes, bool(asset.get("triplet_by_hand"))]


def forget_triplets(repo: Any, red_paths: Any) -> None:
    """Drop the records for these primaries: a re-shot channel changes a member's hash on purpose,
    and the record's hashes would refuse the frame."""
    store = saved_triplets(repo)
    drop = set(red_paths)
    updated = {red: rec for red, rec in store.items() if red not in drop}
    if updated != store:
        repo.save_global_setting(TRIPLETS_KEY, updated)


def remember_triplets(repo: Any, assets: Any) -> None:
    """Upsert the triplets among ``assets``. A stored triplet that shares a file with a new
    one is dropped: one exposure belongs to one frame."""
    store = saved_triplets(repo)
    updated = dict(store)
    for asset in assets:
        # A stitch of triplets carries its primary part's pair too, under the stitch's hash.
        if asset.get("stitch_paths") or asset.get("hdr_paths") or not (asset.get("green_path") and asset.get("blue_path")):
            continue
        record = triplet_record(asset)
        members = {asset["path"], record[0], record[1]}
        for red, other in list(updated.items()):
            if red != asset["path"] and members & {red, other[0], other[1]}:
                del updated[red]
        updated[asset["path"]] = record
    if updated != store:
        repo.save_global_setting(TRIPLETS_KEY, updated)
