"""AssetListModel.set_semantic_query: cosine-similarity ranking over cached
embeddings, with synthetic vectors -- no real CLIP model needed for this part."""

import numpy as np

from negpy.desktop.session import AppState, AssetListModel


def _files(n):
    return [{"name": f"file{i}.nef", "path": f"/roll/file{i}.nef", "hash": f"h{i}"} for i in range(n)]


def _state(n, embeddings=None):
    state = AppState()
    state.uploaded_files = _files(n)
    state.embeddings = embeddings or {}
    return state


def _unit(*coords):
    v = np.array(coords, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_ranks_by_similarity_most_relevant_first():
    state = _state(
        3,
        {
            "h0": _unit(1, 0),  # orthogonal to the query
            "h1": _unit(0.3, 0.9),  # close to the query
            "h2": _unit(0, 1),  # the query itself
        },
    )
    model = AssetListModel(state)

    model.set_semantic_query(_unit(0, 1))

    assert model.visible_actual_indices_ordered() == [2, 1]  # h0 below threshold, excluded


def test_a_file_with_no_cached_embedding_is_excluded_not_zero_scored():
    state = _state(2, {"h0": _unit(0, 1)})  # h1 never embedded
    model = AssetListModel(state)

    model.set_semantic_query(_unit(0, 1))

    assert model.visible_actual_indices_ordered() == [0]


def test_none_reverts_to_the_normal_sort():
    state = _state(2, {"h0": _unit(0, 1), "h1": _unit(1, 0)})
    model = AssetListModel(state)
    model.set_semantic_query(_unit(0, 1))

    model.set_semantic_query(None)

    assert model.visible_actual_indices_ordered() == [0, 1]  # name order restored
    assert model.semantic_query_active is False


def test_sheet_filter_still_applies_under_a_semantic_query():
    state = _state(2, {"h0": _unit(0, 1), "h1": _unit(0, 1)})
    state.uploaded_files[0]["keeper"] = True
    model = AssetListModel(state)
    model.set_sheet_filter("keepers")

    model.set_semantic_query(_unit(0, 1))

    assert model.visible_actual_indices_ordered() == [0]


def test_embeddings_updating_mid_batch_changes_the_ranking_on_refresh():
    state = _state(2, {"h0": _unit(0, 1)})
    model = AssetListModel(state)
    model.set_semantic_query(_unit(0, 1))
    assert model.visible_actual_indices_ordered() == [0]

    state.embeddings["h1"] = _unit(0, 1)
    model.refresh()

    assert set(model.visible_actual_indices_ordered()) == {0, 1}
