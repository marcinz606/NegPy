from __future__ import annotations

from negpy.desktop.prefetch_logic import (
    neighbor_indices,
    neighbor_paths_and_hashes,
    prefetch_paths_and_hashes,
    prefetch_window,
)


def test_neighbor_indices() -> None:
    assert neighbor_indices(3, 0) == [1]
    assert neighbor_indices(3, 1) == [0, 2]
    assert neighbor_indices(1, 0) == []


def test_prefetch_window_forward_biases_ahead() -> None:
    # From index 5 heading forward: warm the next 3, then 1 behind, closest-first.
    assert prefetch_window(20, 5, direction=1, ahead=3, behind=1) == [6, 7, 8, 4]


def test_prefetch_window_backward_biases_ahead() -> None:
    assert prefetch_window(20, 5, direction=-1, ahead=3, behind=1) == [4, 3, 2, 6]


def test_prefetch_window_clamps_at_edges() -> None:
    # Near the end, forward frames run out; only in-range indices are returned.
    assert prefetch_window(7, 6, direction=1, ahead=3, behind=1) == [5]


def test_prefetch_window_unknown_direction_is_symmetric() -> None:
    assert prefetch_window(20, 5, direction=0) == [6, 4]


def test_prefetch_window_out_of_range_index() -> None:
    assert prefetch_window(0, 0, direction=1) == []
    assert prefetch_window(5, 9, direction=1) == []


def test_prefetch_paths_and_hashes() -> None:
    files = [{"path": f"/{i}", "hash": f"h{i}"} for i in range(4)]
    assert prefetch_paths_and_hashes(files, [2, 0, 9]) == [("/2", "h2"), ("/0", "h0")]


def test_neighbor_paths_and_hashes() -> None:
    files = [
        {"path": "/a", "hash": "ha"},
        {"path": "/b", "hash": "hb"},
        {"path": "/c", "hash": "hc"},
    ]
    assert neighbor_paths_and_hashes(files, 0) == [("/b", "hb")]
    assert set(neighbor_paths_and_hashes(files, 1)) == {("/a", "ha"), ("/c", "hc")}
