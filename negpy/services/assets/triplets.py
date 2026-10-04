"""Trichrome triplet membership, keyed by the red exposure's path, so a roll reopens without a raw read.

Each record holds the three content hashes: a file changed on disk since the grouping does not re-attach.
"""

from typing import Any, Dict

from negpy.services.assets.rolls import unforked_hash

TRIPLETS_KEY = "triplets_by_path"


def saved_triplets(repo: Any) -> Dict[str, list]:
    """``{red_path: [green_path, blue_path, align, [red_hash, green_hash, blue_hash]]}``.
    A hash is empty when the grouping was made without it."""
    saved = repo.get_global_setting(TRIPLETS_KEY, default=None)
    return dict(saved) if isinstance(saved, dict) else {}


def _record(asset: dict) -> list:
    # A roll fork suffixes the asset's hash; discovery finds the file's own.
    hashes = [unforked_hash(asset.get("hash") or ""), asset.get("green_hash") or "", asset.get("blue_hash") or ""]
    return [asset["green_path"], asset["blue_path"], bool(asset.get("align", True)), hashes]


def remember_triplets(repo: Any, assets: Any) -> None:
    """Upsert the triplets among ``assets``. A stored triplet that shares a file with a new
    one is dropped: one exposure belongs to one frame."""
    store = saved_triplets(repo)
    updated = dict(store)
    for asset in assets:
        # A stitch of triplets carries its primary part's pair too, under the stitch's hash.
        if asset.get("stitch_paths") or asset.get("hdr_paths") or not (asset.get("green_path") and asset.get("blue_path")):
            continue
        record = _record(asset)
        members = {asset["path"], record[0], record[1]}
        for red, other in list(updated.items()):
            if red != asset["path"] and members & {red, other[0], other[1]}:
                del updated[red]
        updated[asset["path"]] = record
    if updated != store:
        repo.save_global_setting(TRIPLETS_KEY, updated)
