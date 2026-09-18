"""download_clip_model / clip_model_ready / preprocessing and embedding math. The ONNX
sessions themselves are mocked -- these tests never run real inference or touch the
network."""

import io
import os
import urllib.request
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from negpy.services.assets import semantic_model as sm


@pytest.fixture(autouse=True)
def _sandbox_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sm.APP_CONFIG, "cache_dir", str(tmp_path))


class _FakeResponse:
    def __init__(self, data: bytes, headers: dict | None = None):
        self._buf = io.BytesIO(data)
        self.headers = headers or {"Content-Length": str(len(data))}
        self.status = 200

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_clip_model_ready_false_before_any_download():
    assert sm.clip_model_ready() is False


def test_download_writes_every_expected_file():
    payloads = {name: f"data-for-{name}".encode() for _, name in sm._ASSETS}

    def fake_urlopen(request, timeout=None):
        for path, name in sm._ASSETS:
            if path in request.full_url:
                return _FakeResponse(payloads[name])
        raise AssertionError(f"unexpected url {request.full_url}")

    with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
        sm.download_clip_model()

    assert sm.clip_model_ready() is True
    for _, name in sm._ASSETS:
        with open(sm._model_path(name), "rb") as fh:
            assert fh.read() == payloads[name]


def test_download_reports_progress_across_all_files():
    payloads = {name: b"x" * 100 for _, name in sm._ASSETS}
    seen = []

    def fake_urlopen(request, timeout=None):
        for path, name in sm._ASSETS:
            if path in request.full_url:
                return _FakeResponse(payloads[name])
        raise AssertionError

    with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
        sm.download_clip_model(on_progress=lambda done, total: seen.append((done, total)))

    assert seen
    final_done, final_total = seen[-1]
    assert final_done == final_total == 300


def test_download_cancelled_midway_leaves_no_ready_files():
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(b"x" * 100)

    with (
        patch.object(urllib.request, "urlopen", side_effect=fake_urlopen),
        pytest.raises(sm.ClipDownloadError),
    ):
        sm.download_clip_model(is_cancelled=lambda: True)

    assert sm.clip_model_ready() is False
    for _, name in sm._ASSETS:
        assert not os.path.isfile(sm._model_path(name))


def test_download_is_resumable_per_file():
    """A file already present (a prior run got partway through the set) is not
    re-fetched -- only the missing ones are."""
    os.makedirs(sm._model_dir(), exist_ok=True)
    first_path, first_name = sm._ASSETS[0]
    with open(sm._model_path(first_name), "wb") as fh:
        fh.write(b"already-here")

    requested = []

    def fake_urlopen(request, timeout=None):
        requested.append(request.full_url)
        return _FakeResponse(b"fetched")

    with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
        sm.download_clip_model()

    assert not any(first_path in url for url in requested)
    with open(sm._model_path(first_name), "rb") as fh:
        assert fh.read() == b"already-here"


def test_l2_normalize_unit_length():
    vec = np.array([3.0, 4.0], dtype=np.float32)
    normed = sm._l2_normalize(vec)
    assert abs(float(np.linalg.norm(normed)) - 1.0) < 1e-6


def test_l2_normalize_leaves_a_zero_vector_alone():
    vec = np.zeros(4, dtype=np.float32)
    assert np.array_equal(sm._l2_normalize(vec), vec)


def test_preprocess_image_shape_and_range():
    img = Image.new("RGB", (400, 300), color=(128, 64, 32))
    arr = sm._preprocess_image(img)
    assert arr.shape == (1, 3, sm.IMAGE_SIZE, sm.IMAGE_SIZE)
    assert arr.dtype == np.float32
    # A uniform-color image should preprocess to a uniform (per-channel) array.
    assert np.allclose(arr[0, 0], arr[0, 0, 0, 0])


def test_preprocess_image_handles_a_narrow_source():
    """The shorter edge governs the resize, so a very wide or very tall source still
    lands on the model's fixed square input instead of failing the centre crop."""
    img = Image.new("RGB", (50, 500), color=(10, 20, 30))
    arr = sm._preprocess_image(img)
    assert arr.shape == (1, 3, sm.IMAGE_SIZE, sm.IMAGE_SIZE)


def test_run_raises_a_clear_error_on_an_input_name_mismatch():
    fake_input = MagicMock()
    fake_input.name = "totally_unexpected_name"
    session = MagicMock()
    session.get_inputs.return_value = [fake_input]

    with pytest.raises(KeyError):
        sm._run(session, pixel_values=np.zeros((1, 3, 2, 2), dtype=np.float32))


def test_embed_image_uses_the_visual_session_and_normalizes():
    model = sm.ClipModel()
    fake_input = MagicMock()
    fake_input.name = "pixel_values"
    session = MagicMock()
    session.get_inputs.return_value = [fake_input]
    session.run.return_value = [np.array([[3.0, 4.0] + [0.0] * (sm.EMBEDDING_DIM - 2)], dtype=np.float32)]
    model._vision_session = session

    out = model.embed_image(Image.new("RGB", (224, 224)))

    assert out.shape == (sm.EMBEDDING_DIM,)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5


def test_embed_text_uses_the_tokenizer_and_text_session():
    model = sm.ClipModel()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = (np.zeros((1, 77), dtype=np.int64), np.ones((1, 77), dtype=np.int64))
    model._tokenizer = tokenizer

    fake_ids_input, fake_mask_input = MagicMock(), MagicMock()
    fake_ids_input.name, fake_mask_input.name = "input_ids", "attention_mask"
    session = MagicMock()
    session.get_inputs.return_value = [fake_ids_input, fake_mask_input]
    session.run.return_value = [np.array([[1.0] + [0.0] * (sm.EMBEDDING_DIM - 1)], dtype=np.float32)]
    model._text_session = session

    out = model.embed_text("a cat")

    tokenizer.encode.assert_called_once_with("a cat", sm.TEXT_CONTEXT_LENGTH)
    assert out.shape == (sm.EMBEDDING_DIM,)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5
