from __future__ import annotations

from typing import List, Optional, Tuple


def neighbor_indices(n_files: int, current_index: int) -> List[int]:
    """
    Returns actual list indices for previous and next file, in-bounds.
    """
    if n_files <= 0 or current_index < 0 or current_index >= n_files:
        return []
    out: List[int] = []
    if current_index > 0:
        out.append(current_index - 1)
    if current_index + 1 < n_files:
        out.append(current_index + 1)
    return out


def neighbor_paths_and_hashes(files: List[dict], current_index: int) -> List[Tuple[str, Optional[str]]]:
    """
    (path, hash) for prev/next neighbors; hash may be None.
    """
    ni = neighbor_indices(len(files), current_index)
    return [(files[i]["path"], files[i].get("hash")) for i in ni]


def prefetch_window(n_files: int, current_index: int, direction: int, ahead: int = 3, behind: int = 1) -> List[int]:
    """Indices to warm, biased toward the direction of travel.

    ``direction`` is +1 / -1 for forward / backward scrubbing (from the last two
    committed frames) or 0 when unknown. When the direction is known we warm the next
    ``ahead`` frames the user is heading toward first (they are the likely next landings)
    plus a couple ``behind`` for an immediate reversal; when unknown we fall back to a
    symmetric prev/next pair. Returned closest-first so the most imminent frame warms
    before the worker moves on. Out-of-range indices are dropped.
    """
    if n_files <= 0 or not (0 <= current_index < n_files):
        return []
    if direction not in (-1, 1):
        return [i for i in (current_index + 1, current_index - 1) if 0 <= i < n_files]
    out: List[int] = []
    for d in range(1, ahead + 1):
        i = current_index + direction * d
        if 0 <= i < n_files:
            out.append(i)
    for d in range(1, behind + 1):
        i = current_index - direction * d
        if 0 <= i < n_files:
            out.append(i)
    return out


def prefetch_paths_and_hashes(files: List[dict], indices: List[int]) -> List[Tuple[str, Optional[str]]]:
    """(path, hash) for the given indices, in order; hash may be None."""
    return [(files[i]["path"], files[i].get("hash")) for i in indices if 0 <= i < len(files)]
