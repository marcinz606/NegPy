"""generate_batch_embeddings: the async batch shape, persistence and callbacks --
get_thumbnail_worker and ClipModel are mocked, so no real decode or inference runs."""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from negpy.services.assets import embeddings as embed_service


def _file(hash_: str, name: str = "a.nef") -> dict:
    return {"path": f"/roll/{name}", "hash": hash_, "name": name}


def _run(files, asset_store=None, repo=None, **kwargs):
    return asyncio.run(embed_service.generate_batch_embeddings(files, asset_store or MagicMock(), repo or MagicMock(), **kwargs))


@pytest.fixture(autouse=True)
def _fake_thumbnail_and_model():
    vector = np.array([1.0, 0.0], dtype=np.float32)
    with (
        patch.object(embed_service, "get_thumbnail_worker", return_value=Image.new("RGB", (4, 4))) as thumb,
        patch.object(embed_service, "ClipModel") as model_cls,
    ):
        model_cls.return_value.embed_image.return_value = vector
        yield thumb, model_cls, vector


def test_embeds_every_file_and_persists_each_vector(_fake_thumbnail_and_model):
    _, _, vector = _fake_thumbnail_and_model
    repo = MagicMock()

    out = _run([_file("h1"), _file("h2")], repo=repo)

    assert set(out) == {"h1", "h2"}
    assert np.array_equal(out["h1"], vector)
    assert repo.save_embedding.call_count == 2
    repo.save_embedding.assert_any_call("h1", vector, embed_service.MODEL_VERSION, "/roll/a.nef")


def test_a_file_with_no_decodable_thumbnail_is_skipped(_fake_thumbnail_and_model):
    thumb, _, _ = _fake_thumbnail_and_model
    thumb.return_value = None
    repo = MagicMock()

    out = _run([_file("h1")], repo=repo)

    assert out == {}
    repo.save_embedding.assert_not_called()


def test_ready_callback_fires_once_per_embedded_file(_fake_thumbnail_and_model):
    seen = []
    _run([_file("h1"), _file("h2")], ready_callback=lambda h, v: seen.append(h))
    assert sorted(seen) == ["h1", "h2"]


def test_progress_callback_reaches_the_final_count(_fake_thumbnail_and_model):
    seen = []
    _run([_file("h1"), _file("h2"), _file("h3")], progress_callback=lambda current, name: seen.append(current))
    assert sorted(seen) == [1, 2, 3]


def test_empty_batch_returns_empty(_fake_thumbnail_and_model):
    assert _run([]) == {}


def test_is_cancelled_skips_files_but_keeps_the_batch_shape(_fake_thumbnail_and_model):
    """A file is only ever skipped before its own decode/embed step starts -- nothing
    raises, and files that already ran (none here, is_cancelled() is True from the
    start) are simply absent from the result rather than erroring the whole batch."""
    out = _run([_file("h1"), _file("h2")], is_cancelled=lambda: True)
    assert out == {}


def test_is_cancelled_none_runs_the_whole_batch(_fake_thumbnail_and_model):
    out = _run([_file("h1")], is_cancelled=lambda: False)
    assert set(out) == {"h1"}
