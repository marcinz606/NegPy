from __future__ import annotations

from typing import List


def display_neighbor_indices(display_order: List[int], current_index: int) -> List[int]:
    """Actual indices of the previous and next visible assets."""
    try:
        position = display_order.index(current_index)
    except ValueError:
        return []
    neighbors: List[int] = []
    if position > 0:
        neighbors.append(display_order[position - 1])
    if position + 1 < len(display_order):
        neighbors.append(display_order[position + 1])
    return neighbors


def neighbor_assets(files: List[dict], display_order: List[int], current_index: int) -> List[dict]:
    """Previous and next assets in filmstrip order."""
    return [files[index] for index in display_neighbor_indices(display_order, current_index)]
