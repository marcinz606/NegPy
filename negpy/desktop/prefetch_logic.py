from __future__ import annotations

from typing import List, Optional, Tuple


def display_neighbor_indices(display_order: List[int], current_index: int) -> List[int]:
    """
    Actual indices of the prev/next files in filmstrip (display) order.

    `display_order` is the display-ordered list of actual indices
    (AssetListModel.visible_actual_indices_ordered); `current_index` is an actual
    index. Mirrors next_file/prev_file so prefetch warms exactly the frames
    navigation lands on, not the raw uploaded_files (discovery-order) neighbours.
    """
    try:
        pos = display_order.index(current_index)
    except ValueError:
        return []
    out: List[int] = []
    if pos > 0:
        out.append(display_order[pos - 1])
    if pos + 1 < len(display_order):
        out.append(display_order[pos + 1])
    return out


def neighbor_paths_and_hashes(files: List[dict], display_order: List[int], current_index: int) -> List[Tuple[str, Optional[str]]]:
    """
    (path, hash) for the prev/next neighbours in display order; hash may be None.
    """
    ni = display_neighbor_indices(display_order, current_index)
    return [(files[i]["path"], files[i].get("hash")) for i in ni]
