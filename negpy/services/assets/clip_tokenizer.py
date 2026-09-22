"""CLIP's own byte-level BPE tokenizer (OpenAI's reference algorithm, unchanged since
release and re-implemented identically across every CLIP port). Vendored rather than
pulled in from `transformers`/`tokenizers`, since the whole feature (semantic_model.py)
exists specifically to avoid a heavy ML dependency for one function.

The vocabulary is not read from a vocab.json: it is derived deterministically from
merges.txt in the same fixed order OpenAI built it in (256 byte-level base tokens, each
paired with an end-of-word variant, then one entry per merge rule in file order, then
the two special tokens last) -- this reproduces the exact token ids the model was
exported with without trusting a separately-scraped vocabulary file.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from typing import Tuple

import numpy as np

_START = "<|startoftext|>"
_END = "<|endoftext|>"


# GPT-2/CLIP's byte<->unicode mapping: every one of the 256 possible bytes gets a
# printable unicode stand-in, so BPE never has to merge across whitespace/control
# characters. Reversible via the inverse map (unused here -- only encoding is needed).
@lru_cache()
def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def _clean(text: str) -> str:
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


# ASCII contractions get their own token, as in the reference tokenizer. \w+ covers search
# queries without the `regex` package, one more dependency for non-Latin text this feature
# need not handle exactly.
_PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|[\w]+|[^\s\w]+""", re.IGNORECASE)


class ClipTokenizer:
    def __init__(self, merges_path: str) -> None:
        with open(merges_path, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        # First line is a version comment; a trailing blank line is dropped too.
        merges = [tuple(line.split()) for line in lines[1:] if line and len(line.split()) == 2]

        self._byte_encoder = _bytes_to_unicode()
        base = list(self._byte_encoder.values())
        vocab = base + [v + "</w>" for v in base] + ["".join(m) for m in merges] + [_START, _END]
        self._encoder = {tok: i for i, tok in enumerate(vocab)}
        self._bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self._cache = {_START: _START, _END: _END}
        self.start_id = self._encoder[_START]
        self.end_id = self._encoder[_END]

    def _bpe(self, token: str) -> str:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self._bpe_ranks.get(p, float("inf")))
            if bigram not in self._bpe_ranks:
                break
            first, second = bigram
            merged: list[str] = []
            i = 0
            while i < len(word):
                j = word.index(first, i) if first in word[i:] else -1
                if j == -1:
                    merged.extend(word[i:])
                    break
                merged.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        result = " ".join(word)
        self._cache[token] = result
        return result

    def _token_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for raw in re.findall(_PAT, _clean(text).lower()):
            piece = "".join(self._byte_encoder[b] for b in raw.encode("utf-8"))
            ids.extend(self._encoder[bpe] for bpe in self._bpe(piece).split(" "))
        return ids

    def encode(self, text: str, context_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """(input_ids, attention_mask), each shape (1, context_length) int64 -- CLIP's
        text tower always takes a fixed-length sequence: start/end tokens bracket the
        (possibly truncated) content, the rest is padded and masked out."""
        ids = [self.start_id] + self._token_ids(text)[: context_length - 2] + [self.end_id]
        mask = [1] * len(ids)
        pad = context_length - len(ids)
        if pad > 0:
            ids = ids + [0] * pad
            mask = mask + [0] * pad
        return np.asarray([ids], dtype=np.int64), np.asarray([mask], dtype=np.int64)
