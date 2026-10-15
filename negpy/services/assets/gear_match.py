"""Suggest a camera, film stock, ISO and capture date for a newly imported roll from its
folder name.

Camera and film stock are matched against the gear catalog (bundled reference entries plus
whatever the caller has added as their own) two ways: as shared words ("gold" in
"08_penf_gold_marbella" against "Kodak Gold 200"), and as a squashed, delimiter-free run, for
the abbreviations scan-folder names tend to use ("penf" against "Olympus Pen F", "trix"
against "Kodak Tri-X 400"). More than one candidate matching the same field is treated as no
match: a wrong guess is worse than no guess.

ISO and capture date read the folder name directly, since they are plain numbers and dates
rather than catalog lookups. A matched film stock already carries its own canonical ISO
(applied by metadata_from_gear), so the standalone ISO guess here only fires when no stock
matched. Both guesses fall back to no match rather than picking between two plausible
readings, the same rule as gear matching.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from negpy.features.metadata.capture import parse_capture_date
from negpy.features.metadata.gear_models import GearLibrary
from negpy.services.assets import rolls
from negpy.services.assets.library import folder_label

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_TOKEN_LEN = 2
_MIN_SQUASH_LEN = 4

# Common still-film box speeds. A folder's own roll-sequence prefix ("05_...") or a film
# format number ("120") is never one of these, so the whitelist alone keeps them out without
# tracking token position.
_ISO_VALUES = frozenset({25, 32, 40, 50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 3200, 6400})
_DIGITS_RE = re.compile(r"\d+")
_DATE_RE = re.compile(r"(?<!\d)(\d{4})[-_. ]?(\d{2})[-_. ]?(\d{2})(?!\d)")


@dataclass(frozen=True)
class GearMatch:
    camera_id: str = ""
    film_stock_id: str = ""
    iso: Optional[int] = None
    capture_date: str = ""

    def any(self) -> bool:
        return bool(self.camera_id or self.film_stock_id or self.iso or self.capture_date)


def _infer_iso(folder_name: str) -> Optional[int]:
    candidates = {int(n) for t in _tokens(folder_name) for n in _DIGITS_RE.findall(t)}
    plausible = candidates & _ISO_VALUES
    return plausible.pop() if len(plausible) == 1 else None


def _infer_capture_date(folder_name: str) -> str:
    found = set()
    for year, month, day in _DATE_RE.findall(folder_name):
        parsed = parse_capture_date(f"{year}-{month}-{day}")
        if parsed is not None:
            found.add(parsed.text)
    return found.pop() if len(found) == 1 else ""


def _tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= _MIN_TOKEN_LEN}


def _squash(text: str) -> str:
    return "".join(_TOKEN_RE.findall(text.lower()))


def _match_strength(folder_tokens: set, folder_squashed: str, name_parts: list) -> int:
    """2 for a folder token that carries a digit (portra400, hp5) or the whole name as a run
    (penf); 1 for a shared word; 0 for none. A stock's speed is in its name, so a token with a
    digit names the stock, while a plain word may be the subject (street, gold, lucky)."""
    name = " ".join(p for p in name_parts if p)
    if not name:
        return 0
    candidate_tokens = _tokens(name)
    candidate_squashed = _squash(name)
    shared = folder_tokens & candidate_tokens
    runs = {t for t in folder_tokens if len(t) >= 3 and t in candidate_squashed}
    if any(any(c.isdigit() for c in t) for t in shared | runs):
        return 2
    if len(candidate_squashed) >= _MIN_SQUASH_LEN and candidate_squashed in folder_squashed:
        return 2
    return 1 if shared or runs else 0


def _best_match(folder_tokens: set, folder_squashed: str, items: list, name_parts) -> Optional[str]:
    scored = [(item.id, _match_strength(folder_tokens, folder_squashed, name_parts(item))) for item in items]
    best = max((s for _, s in scored), default=0)
    matches = [item_id for item_id, s in scored if s == best and s > 0]
    return matches[0] if len(matches) == 1 else None


def match_gear_for_folder(folder_name: str, library: GearLibrary) -> GearMatch:
    folder_tokens = _tokens(folder_name)
    folder_squashed = _squash(folder_name)
    camera_id = _best_match(folder_tokens, folder_squashed, library.cameras, lambda c: [c.make, c.model, c.display_name])
    film_stock_id = _best_match(
        folder_tokens, folder_squashed, library.film_stocks, lambda f: [f.manufacturer, f.stock_name, f.display_name]
    )
    return GearMatch(
        camera_id=camera_id or "",
        film_stock_id=film_stock_id or "",
        iso=None if film_stock_id else _infer_iso(folder_name),
        capture_date=_infer_capture_date(folder_name),
    )


def folder_name_for_active_context(state: Any, repo: Any) -> str:
    """The folder name to match gear against: the active folder roll's own folder,
    else the current frame's containing directory -- callers may have no roll active
    at all, from a plain Add Files/Add Folder load."""
    roll_id = state.active_roll_id
    if roll_id:
        entry = rolls.roll_for_id(repo, roll_id)
        if entry and entry.get("kind") == "folder":
            return folder_label(entry.get("folder_path", ""))
    src = state.selected_file_idx
    if src == -1 or src >= len(state.uploaded_files):
        return ""
    path = state.uploaded_files[src].get("path", "")
    return folder_label(os.path.dirname(path)) if path else ""
