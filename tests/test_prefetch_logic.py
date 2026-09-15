from negpy.desktop.prefetch_logic import display_neighbor_indices, neighbor_assets


def test_neighbors_follow_visible_display_order() -> None:
    files = [{"path": f"/{index}.dng"} for index in range(5)]
    order = [4, 1, 3]

    assert display_neighbor_indices(order, 1) == [4, 3]
    assert neighbor_assets(files, order, 1) == [files[4], files[3]]


def test_neighbors_at_each_end_of_the_display_order() -> None:
    assert display_neighbor_indices([0, 1, 2], 0) == [1]
    assert display_neighbor_indices([0, 1, 2], 1) == [0, 2]
    assert display_neighbor_indices([0, 1, 2], 2) == [1]


def test_missing_selection_has_no_neighbors() -> None:
    assert display_neighbor_indices([0, 1], 9) == []
    assert neighbor_assets([], [], -1) == []
