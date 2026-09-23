"""Carries edits across the content-hash change that added interior sampling.

The old fingerprint was size + 1 MiB head + 1 MiB tail, so two same-size scans of one
frame — identical container header and trailer — shared a digest and only one of them
could ever be opened. ``file_hashes`` now samples the interior too, which also changed
the identity of every file over 2 MiB (files at or under it have no interior, so their
digest is untouched and they need nothing from this module).

Discovery puts the old digest on each asset as ``legacy_hash``; ``migrate_asset_hash``
rehomes that identity's edits onto the current one. Retire this module together with
the legacy slot of ``file_hashes`` and the ``legacy_hash`` asset key.
"""

from collections import Counter
from typing import Any, Dict, List

from negpy.domain.interfaces import IRepository
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def blank_ambiguous_legacy_hashes(assets: List[Dict[str, Any]]) -> None:
    """Drop the legacy digest from files that shared one — the very collision the
    interior sampling fixes. It cannot say whose edits it holds, and only one of the
    pair was ever openable, so migrating either way is a coin flip; both start clean."""
    counts = Counter(a["legacy_hash"] for a in assets if a.get("legacy_hash"))
    for a in assets:
        if counts[a.get("legacy_hash", "")] > 1:
            logger.warning("Ambiguous legacy hash for %s — starting it without migrated edits", a["path"])
            a["legacy_hash"] = ""


def migrate_asset_hash(repo: IRepository, asset: Dict[str, Any]) -> None:
    """Rehome an asset's edits from its legacy digest onto its current one.

    Call before the marks overlay and before anything reads the history index, so a
    migrated mark and history land in the same pass rather than one import later.
    """
    legacy = asset.get("legacy_hash")
    if legacy and legacy != asset["hash"]:
        repo.rehome_file_settings(legacy, asset["hash"], asset["path"])
