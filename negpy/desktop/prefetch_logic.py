from __future__ import annotations

from typing import List


def display_neighbor_indices(display_order: List[int], current_index: int, max_distance: int = 1) -> List[int]:
    """Actual indices of surrounding visible assets, nearest first (prev, next, prev-2, next+2, ...)."""
    try:
        position = display_order.index(current_index)
    except ValueError:
        return []
    neighbors: List[int] = []
    for step in range(1, max(1, int(max_distance)) + 1):
        if position - step >= 0:
            neighbors.append(display_order[position - step])
        if position + step < len(display_order):
            neighbors.append(display_order[position + step])
    return neighbors


def neighbor_assets(files: List[dict], display_order: List[int], current_index: int, max_distance: int = 1) -> List[dict]:
    """Surrounding assets in filmstrip order, nearest first."""
    return [files[index] for index in display_neighbor_indices(display_order, current_index, max_distance=max_distance)]
