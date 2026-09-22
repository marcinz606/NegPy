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


def _cos(similarity):
    """Unit vector whose dot product with the query (0, 1) is exactly `similarity`."""
    return np.array([np.sqrt(1.0 - similarity**2), similarity], dtype=np.float32)


def test_ranks_by_similarity_most_relevant_first():
    """The threshold is relative to the whole candidate pool's own score spread, so it
    needs a real background to separate a match from -- a couple of near-identical
    scores can't stand out from each other the way a real search's rare match stands
    out from its library's own baseline."""
    background = {f"h{i}": _cos(c) for i, c in enumerate(np.linspace(0.08, 0.16, 28))}
    state = _state(
        30,
        {
            **background,
            "h28": _cos(0.32),  # close to the query
            "h29": _cos(0.35),  # closer still
        },
    )
    model = AssetListModel(state)

    model.set_semantic_query(_unit(0, 1))

    assert model.visible_actual_indices_ordered() == [29, 28]  # background below threshold, excluded


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
