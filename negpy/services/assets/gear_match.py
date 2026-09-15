"""Suggest a camera and film stock for a newly imported roll from its folder name.

Matches only against gear already in the user's own library -- nothing is ever invented
from free text. A folder name is compared to each candidate two ways: as shared words
("gold" in "08_penf_gold_marbella" against "Kodak Gold 200"), and as a squashed,
delimiter-free run, for the abbreviations scan-folder names tend to use ("penf" against
"Olympus Pen F", "trix" against "Kodak Tri-X 400"). More than one candidate matching the
same field is treated as no match: a wrong guess is worse than no guess.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from negpy.features.metadata.gear_models import GearLibrary
from negpy.services.assets import rolls
from negpy.services.assets.library import folder_label

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_TOKEN_LEN = 2
_MIN_SQUASH_LEN = 4


@dataclass(frozen=True)
class GearMatch:
    camera_id: str = ""
    film_stock_id: str = ""

    def any(self) -> bool:
        return bool(self.camera_id or self.film_stock_id)


def _tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= _MIN_TOKEN_LEN}


def _squash(text: str) -> str:
    return "".join(_TOKEN_RE.findall(text.lower()))


def _is_match(folder_tokens: set, folder_squashed: str, name_parts: list) -> bool:
    name = " ".join(p for p in name_parts if p)
    if not name:
        return False
    candidate_tokens = _tokens(name)
    if folder_tokens & candidate_tokens:
        return True
    candidate_squashed = _squash(name)
    if len(candidate_squashed) >= _MIN_SQUASH_LEN and candidate_squashed in folder_squashed:
        return True
    return any(len(t) >= 3 and t in candidate_squashed for t in folder_tokens)


def _best_match(folder_tokens: set, folder_squashed: str, items: list, name_parts) -> Optional[str]:
    matches = [item.id for item in items if _is_match(folder_tokens, folder_squashed, name_parts(item))]
    return matches[0] if len(matches) == 1 else None


def match_gear_for_folder(folder_name: str, library: GearLibrary) -> GearMatch:
    folder_tokens = _tokens(folder_name)
    folder_squashed = _squash(folder_name)
    camera_id = _best_match(folder_tokens, folder_squashed, library.cameras, lambda c: [c.make, c.model, c.display_name])
    film_stock_id = _best_match(
        folder_tokens, folder_squashed, library.film_stocks, lambda f: [f.manufacturer, f.stock_name, f.display_name]
    )
    return GearMatch(camera_id=camera_id or "", film_stock_id=film_stock_id or "")


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
