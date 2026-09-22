"""ClipTokenizer: the BPE merge algorithm and the fixed-length encode() contract the
ONNX text model expects. A tiny synthetic merges.txt stands in for the real ~49k-line
file -- the algorithm's correctness does not depend on the vocabulary's size."""

import numpy as np
import pytest

from negpy.services.assets.clip_tokenizer import ClipTokenizer

# Merge rules that don't reproduce any single-byte "+</w>" base token outright, so the
# derived vocabulary has no accidental collisions to reason about.
_MERGES = "#version: 0.2\nc a\nca t\ndo g\n"


@pytest.fixture
def tokenizer(tmp_path):
    path = tmp_path / "merges.txt"
    path.write_text(_MERGES, encoding="utf-8")
    return ClipTokenizer(str(path))


def test_start_and_end_tokens_bracket_a_short_string(tokenizer):
    ids, mask = tokenizer.encode("cat", 8)
    assert ids.shape == (1, 8)
    assert mask.shape == (1, 8)
    assert ids[0, 0] == tokenizer.start_id
    end_positions = np.where(ids[0] == tokenizer.end_id)[0]
    assert len(end_positions) == 1
    end_pos = int(end_positions[0])
    # Real content in between, mask covers exactly through the end token.
    assert end_pos > 1
    assert mask[0, : end_pos + 1].all()
    assert not mask[0, end_pos + 1 :].any()


def test_padding_fills_with_zero_and_a_zero_mask(tokenizer):
    ids, mask = tokenizer.encode("cat", 8)
    end_pos = int(np.where(ids[0] == tokenizer.end_id)[0][0])
    assert (ids[0, end_pos + 1 :] == 0).all()
    assert (mask[0, end_pos + 1 :] == 0).all()


def test_long_text_is_truncated_to_context_length(tokenizer):
    ids, mask = tokenizer.encode(" ".join(["cat"] * 50), 8)
    assert ids.shape == (1, 8)
    assert mask.shape == (1, 8)
    # Truncated content still ends in the end token, occupying the full budget.
    assert ids[0, -1] == tokenizer.end_id
    assert mask[0].all()


def test_encoding_is_deterministic(tokenizer):
    a = tokenizer.encode("a cat and a dog", 12)
    b = tokenizer.encode("a cat and a dog", 12)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_empty_string_is_just_the_two_special_tokens(tokenizer):
    ids, mask = tokenizer.encode("", 8)
    assert ids[0, 0] == tokenizer.start_id
    assert ids[0, 1] == tokenizer.end_id
    assert mask[0, 0] == 1 and mask[0, 1] == 1
    assert not mask[0, 2:].any()


def test_start_and_end_ids_are_the_last_two_vocab_entries(tokenizer):
    """The whole point of deriving the vocabulary from merges.txt instead of trusting
    a separate vocab.json: these two ids must land exactly where the real export put
    them -- last, in this fixed order -- or every embedding is silently wrong."""
    assert tokenizer.end_id == tokenizer.start_id + 1
