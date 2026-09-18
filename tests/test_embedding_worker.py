"""EmbeddingWorker.generate: the Qt-signal shell around generate_batch_embeddings --
chunked `partial` emission and a `finished` carrying the whole batch, mirroring
ThumbnailWorker's own contract."""

from unittest.mock import MagicMock, patch

import numpy as np

from negpy.desktop.workers import embedding as embedding_worker_module
from negpy.desktop.workers.embedding import EmbeddingWorker


def _file(hash_: str) -> dict:
    return {"path": f"/roll/{hash_}.nef", "hash": hash_, "name": f"{hash_}.nef"}


def test_finished_carries_every_embedded_file():
    vector = np.array([1.0, 0.0], dtype=np.float32)

    async def fake_batch(files, store, repo, progress_callback=None, ready_callback=None):
        for f in files:
            if ready_callback:
                ready_callback(f["hash"], vector)
        return {f["hash"]: vector for f in files}

    worker = EmbeddingWorker(MagicMock(), MagicMock())
    finished = MagicMock()
    error = MagicMock()
    worker.finished.connect(finished)
    worker.error.connect(error)

    with patch.object(embedding_worker_module, "logger"):
        with patch(
            "negpy.services.assets.embeddings.generate_batch_embeddings",
            side_effect=fake_batch,
        ):
            worker.generate([_file("h1"), _file("h2")])

    error.assert_not_called()
    finished.assert_called_once()
    (result,) = finished.call_args[0]
    assert set(result) == {"h1", "h2"}


def test_partial_emits_once_the_chunk_size_is_reached():
    vector = np.zeros(2, dtype=np.float32)
    files = [_file(f"h{i}") for i in range(embedding_worker_module._EMBED_CHUNK + 1)]

    async def fake_batch(batch_files, store, repo, progress_callback=None, ready_callback=None):
        for f in batch_files:
            if ready_callback:
                ready_callback(f["hash"], vector)
        return {f["hash"]: vector for f in batch_files}

    worker = EmbeddingWorker(MagicMock(), MagicMock())
    partial = MagicMock()
    worker.partial.connect(partial)

    with patch("negpy.services.assets.embeddings.generate_batch_embeddings", side_effect=fake_batch):
        worker.generate(files)

    assert partial.call_count == 1
    (chunk,) = partial.call_args[0]
    assert len(chunk) == embedding_worker_module._EMBED_CHUNK


def test_a_batch_failure_emits_error_not_finished():
    worker = EmbeddingWorker(MagicMock(), MagicMock())
    finished = MagicMock()
    error = MagicMock()
    worker.finished.connect(finished)
    worker.error.connect(error)

    async def boom(*a, **k):
        raise RuntimeError("onnx blew up")

    with patch.object(embedding_worker_module, "logger"):
        with patch("negpy.services.assets.embeddings.generate_batch_embeddings", side_effect=boom):
            worker.generate([_file("h1")])

    finished.assert_not_called()
    error.assert_called_once()
