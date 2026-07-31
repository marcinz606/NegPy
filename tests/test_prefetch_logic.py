from __future__ import annotations

from negpy.desktop.prefetch_logic import display_neighbor_indices, neighbor_paths_and_hashes


def test_display_neighbor_indices_identity_order() -> None:
    # Display order matches actual order: neighbours are current ± 1.
    assert display_neighbor_indices([0, 1, 2], 0) == [1]
    assert display_neighbor_indices([0, 1, 2], 1) == [0, 2]
    assert display_neighbor_indices([0], 0) == []


def test_display_neighbor_indices_sorted_order() -> None:
    # Discovery order [0,1,2] shown sorted as [2,0,1]: navigating from actual 0
    # lands on 2 (prev) and 1 (next), not the raw-order 1 (and nothing before).
    display = [2, 0, 1]
    assert display_neighbor_indices(display, 0) == [2, 1]  # middle of the strip
    assert display_neighbor_indices(display, 2) == [0]  # first shown: next only
    assert display_neighbor_indices(display, 1) == [0]  # last shown: prev only


def test_display_neighbor_indices_filtered_out_current() -> None:
    # Current file not in the display order (filtered out): no neighbours.
    assert display_neighbor_indices([0, 2], 1) == []


def test_neighbor_paths_and_hashes() -> None:
    files = [
        {"path": "/a", "hash": "ha"},
        {"path": "/b", "hash": "hb"},
        {"path": "/c", "hash": "hc"},
    ]
    # Sorted display order [/c, /a, /b]; from actual 0 (/a): prev /c, next /b.
    assert neighbor_paths_and_hashes(files, [2, 0, 1], 0) == [("/c", "hc"), ("/b", "hb")]
    assert set(neighbor_paths_and_hashes(files, [0, 1, 2], 1)) == {("/a", "ha"), ("/c", "hc")}
