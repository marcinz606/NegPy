"""CLIP text/image embeddings for "search by meaning": one shared vector space where a
photo and a plain-language description of it land close together, so a query becomes a
nearest-neighbour lookup over cached image vectors instead of a keyword match.

The feature costs nothing until it is turned on: the ONNX sessions and tokenizer build
lazily, and the weights download on demand (download_clip_model) rather than shipping with
the app, which carries only the `onnxruntime` engine.

Model: Xenova/clip-vit-base-patch16 on Hugging Face, the quantized per-tower ONNX exports
plus the tokenizer's merges.txt. The preprocessing constants below come from that repo's
preprocessor_config.json, so a model swap needs new constants rather than new file names.
MODEL_VERSION names the cache directory for that reason: a retired model's download cannot
pass for the current one under the same generic filenames.
"""

from __future__ import annotations

import os
import urllib.request
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from PIL import Image

from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.version import get_app_version

if TYPE_CHECKING:
    from negpy.services.assets.clip_tokenizer import ClipTokenizer

logger = get_logger(__name__)

# Bumped whenever the model or preprocessing changes, so a retired model's cached
# embeddings are never compared against a new model's (repository.py's
# image_embeddings.model_version). It also names the download cache directory below.
MODEL_VERSION = "clip-vit-base-patch16-v1"
# Named once here, for the two surfaces that warn about the download before it starts.
MODEL_DOWNLOAD_SIZE = "about 150 MB"

_REPO = "Xenova/clip-vit-base-patch16"
_BASE_URL = f"https://huggingface.co/{_REPO}/resolve/main"
VISION_FILE = "vision_model_quantized.onnx"
TEXT_FILE = "text_model_quantized.onnx"
MERGES_FILE = "merges.txt"
# (published path, local filename)
_ASSETS = ((f"onnx/{VISION_FILE}", VISION_FILE), (f"onnx/{TEXT_FILE}", TEXT_FILE), (MERGES_FILE, MERGES_FILE))

_USER_AGENT = f"NegPy/{get_app_version()} (+https://github.com/marcinz606/NegPy)"
_CHUNK = 256 * 1024

# preprocessor_config.json: shortest-edge resize to this, then a centre crop of the
# same size, bicubic (PIL.Image.BICUBIC), rescaled to [0,1] then normalized.
IMAGE_SIZE = 224
_IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# config.json: text_config.max_position_embeddings.
TEXT_CONTEXT_LENGTH = 77

EMBEDDING_DIM = 512


class ClipDownloadError(Exception):
    pass


def _model_dir() -> str:
    return os.path.join(APP_CONFIG.cache_dir, "clip_model", MODEL_VERSION)


def _model_path(filename: str) -> str:
    return os.path.join(_model_dir(), filename)


def clip_model_ready() -> bool:
    """Every file the model needs is already on disk."""
    return all(os.path.isfile(_model_path(name)) for _, name in _ASSETS)


def download_clip_model(
    on_progress: Optional[Callable[[int, int], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> None:
    """Fetches every file clip_model_ready() checks for, atomically per file (a
    `.part` temp name, renamed only once complete) so a cancelled or interrupted
    download never leaves a file that reads as ready. Raises ClipDownloadError.

    `on_progress(done, total)` reports bytes across the whole set, not one file at a
    time, since the caller (a single progress bar) has no reason to know there are
    three of them.
    """
    os.makedirs(_model_dir(), exist_ok=True)
    sizes = {}
    for path, name in _ASSETS:
        dest = _model_path(name)
        sizes[name] = os.path.getsize(dest) if os.path.isfile(dest) else _content_length(f"{_BASE_URL}/{path}")
    total = sum(sizes.values())
    done = 0
    for path, name in _ASSETS:
        dest = _model_path(name)
        if os.path.isfile(dest):
            done += sizes[name]
            if on_progress is not None:
                on_progress(done, total)
            continue
        tmp = f"{dest}.part"
        try:
            request = urllib.request.Request(f"{_BASE_URL}/{path}", headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=30) as response, open(tmp, "wb") as out:
                while True:
                    if is_cancelled is not None and is_cancelled():
                        raise ClipDownloadError("Download cancelled.")
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(done, total)
        except ClipDownloadError:
            _discard(tmp)
            raise
        except Exception as exc:
            _discard(tmp)
            raise ClipDownloadError(f"Download failed: {exc}") from exc
        os.replace(tmp, dest)


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _content_length(url: str) -> int:
    """Best-effort size for the progress bar's total; 0 (unknown) is harmless."""
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=10) as response:
            return int(response.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


# CLIP cosine similarities run low even for a good match and cluster around a baseline
# that shifts with the query and the library, so a fixed cutoff cannot separate a match
# from noise. The cutoff is relative instead: how many standard deviations above this
# query's own mean score a candidate sits.
SIMILARITY_Z_SCORE = 3.0


def rank_by_similarity(query: np.ndarray, candidates: dict) -> list:
    """Keys of `candidates` (key -> L2-normalized vector) ranked by cosine similarity
    to `query`, most relevant first. A candidate scoring less than SIMILARITY_Z_SCORE
    standard deviations above this query's own mean score is excluded rather than kept
    at the bottom, so a search narrows instead of just reordering -- the one rule both
    the in-session and whole-library rankings share."""
    if not candidates:
        return []
    keys = list(candidates.keys())
    scores = np.stack([candidates[key] for key in keys]) @ query
    threshold = scores.mean() + SIMILARITY_Z_SCORE * scores.std()
    order = np.argsort(-scores)
    return [keys[i] for i in order if scores[i] >= threshold]


def _preprocess_image(image: Image.Image) -> np.ndarray:
    """PIL image -> (1, 3, 224, 224) float32, CLIP's own resize/crop/normalize."""
    img = image.convert("RGB")
    w, h = img.size
    scale = IMAGE_SIZE / min(w, h)
    img = img.resize((max(IMAGE_SIZE, round(w * scale)), max(IMAGE_SIZE, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - IMAGE_SIZE) // 2, (h - IMAGE_SIZE) // 2
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - _IMAGE_MEAN) / _IMAGE_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def _run(session, **named_arrays) -> list:
    """Feeds `named_arrays` to `session` by matching each of its own declared input
    names -- a naming mismatch against the model's actual ONNX graph then fails with a
    clear KeyError instead of silently running the wrong tensor into the wrong slot."""
    feed = {}
    for inp in session.get_inputs():
        if inp.name not in named_arrays:
            raise KeyError(f"CLIP ONNX model expects input '{inp.name}', not provided (have: {sorted(named_arrays)})")
        feed[inp.name] = named_arrays[inp.name]
    return session.run(None, feed)


class ClipModel:
    """Lazy CLIP inference: the ONNX sessions and tokenizer are constructed on first
    use, not at import or construction time, so holding an idle instance costs
    nothing. One instance is meant to be shared/reused (sessions are expensive to
    build, cheap to run repeatedly)."""

    def __init__(self) -> None:
        self._vision_session = None
        self._text_session = None
        self._tokenizer = None

    def _ensure_vision(self):
        if self._vision_session is None:
            import onnxruntime as ort

            self._vision_session = ort.InferenceSession(_model_path(VISION_FILE), providers=["CPUExecutionProvider"])
        return self._vision_session

    def _ensure_text(self):
        if self._text_session is None:
            import onnxruntime as ort

            self._text_session = ort.InferenceSession(_model_path(TEXT_FILE), providers=["CPUExecutionProvider"])
        return self._text_session

    def _ensure_tokenizer(self) -> "ClipTokenizer":
        if self._tokenizer is None:
            from negpy.services.assets.clip_tokenizer import ClipTokenizer

            self._tokenizer = ClipTokenizer(_model_path(MERGES_FILE))
        return self._tokenizer

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """L2-normalized (EMBEDDING_DIM,) vector -- similarity to embed_text's output
        is then a plain dot product."""
        pixel_values = _preprocess_image(image)
        outputs = _run(self._ensure_vision(), pixel_values=pixel_values)
        return _l2_normalize(np.asarray(outputs[0][0], dtype=np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        input_ids, attention_mask = self._ensure_tokenizer().encode(text, TEXT_CONTEXT_LENGTH)
        outputs = _run(self._ensure_text(), input_ids=input_ids, attention_mask=attention_mask)
        return _l2_normalize(np.asarray(outputs[0][0], dtype=np.float32))
