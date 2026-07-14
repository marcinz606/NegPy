"""Regression contracts for the proven LS-5000 packed RGB4x + IR stream."""

from pathlib import Path

import numpy as np
import pytest

from negpy.infrastructure.scanners.ls5000_single_pass.packed import (
    FULL_IR_BYTE_OFFSET,
    FULL_RECORD_BYTES,
    FULL_RECORD_WORDS,
    FULL_RGB_SAMPLE_BYTE_OFFSETS,
    UNIT_WORDS,
    WIDTH,
    decode_core_records,
    decode_full_records,
    infer_record_geometry,
)


def _counter_train(words: np.ndarray, first_counter: int) -> None:
    words[0::2] = 0xAA55
    words[1::2] = (first_counter + np.arange(words.size // 2, dtype=np.uint32)) & 0xFFFF


def _synthetic_full_records(height: int = 3) -> tuple[np.ndarray, np.ndarray]:
    records = (height + 1) // 2
    base = (np.arange(records * 2 * WIDTH * 4, dtype=np.uint32).reshape(records * 2, WIDTH, 4) % 20_000).astype(np.uint16)
    wire_rows = np.concatenate(
        [base[:height, ::-1], np.zeros((records * 2 - height, WIDTH, 4), dtype=np.uint16)],
        axis=0,
    )
    paired = wire_rows.reshape(records, 2, WIDTH, 4)
    full = np.zeros((records, FULL_RECORD_WORDS), dtype=np.uint16)
    for sample, byte_offsets in enumerate(FULL_RGB_SAMPLE_BYTE_OFFSETS):
        for channel, byte_offset in enumerate(byte_offsets):
            unit = paired[..., channel].transpose(0, 2, 1).reshape(records, UNIT_WORDS)
            full[:, byte_offset // 2 : byte_offset // 2 + UNIT_WORDS] = unit + sample
    ir_unit = paired[..., 3].transpose(0, 2, 1).reshape(records, UNIT_WORDS)
    ir_start = FULL_IR_BYTE_OFFSET // 2
    full[:, ir_start : ir_start + UNIT_WORDS] = ir_unit
    for record in full:
        _counter_train(record[63_136 // 2 : 63_488 // 2], 0xE7FD)
        _counter_train(record[110_840 // 2 : 111_616 // 2], 0xD893)
        record[158_968 // 2 : 159_744 // 2] = record[ir_start : ir_start + 388]
        _counter_train(record[207_096 // 2 : 207_872 // 2], 0xD893)
    return base[:height], full


def test_core_decoder_preserves_channel_x_row_order_and_discards_surplus_row(tmp_path: Path) -> None:
    width, height, channels = 3, 3, 4
    records = 2
    core_words = width * channels * 2
    record_bytes = core_words * 2 + 10
    expected = np.arange(height * width * channels, dtype=np.uint16).reshape(height, width, channels)
    row_slots = np.concatenate([expected[:, ::-1], np.full((1, width, channels), 60_000, dtype=np.uint16)])
    wire = row_slots.reshape(records, 2, width, channels).transpose(0, 3, 2, 1).reshape(records, core_words)
    path = tmp_path / "prefix.bin"
    with path.open("wb") as stream:
        for record in wire:
            stream.write(record.astype(">u2").tobytes())
            stream.write(b"\xaa" * 10)

    decoded = decode_core_records(path, record_bytes=record_bytes, width=width, height=height, channels=channels)

    np.testing.assert_array_equal(expected, decoded)


def test_full_decoder_rounds_four_rgb_samples_and_keeps_one_ir_plane(tmp_path: Path) -> None:
    base, full = _synthetic_full_records()
    path = tmp_path / "full.bin"
    path.write_bytes(full.astype(">u2").tobytes())

    decoded, report = decode_full_records(path, height=3)

    expected = base.copy()
    expected[..., :3] += 2
    np.testing.assert_array_equal(expected, decoded)
    assert report["rgb_samples_decoded"] == 4
    assert report["rgb_average"] == "round-half-up uint16 average"
    assert report["ir_planes_transferred"] == 1
    assert report["padding_validated_records"] == 2


def test_full_decoder_fails_closed_on_padding_corruption(tmp_path: Path) -> None:
    _base, full = _synthetic_full_records()
    full[1, 63_136 // 2] ^= 1
    path = tmp_path / "corrupt.bin"
    path.write_bytes(full.astype(">u2").tobytes())

    with pytest.raises(ValueError, match="padding 0 counter train mismatch"):
        decode_full_records(path, height=3)


def test_geometry_inference_accepts_only_known_record_sizes(tmp_path: Path) -> None:
    path = tmp_path / "stream.bin"
    path.write_bytes(b"\x00" * (2 * FULL_RECORD_BYTES))
    assert infer_record_geometry(path, records=2) == FULL_RECORD_BYTES
    path.write_bytes(b"\x00" * 14)
    with pytest.raises(ValueError, match="neither"):
        infer_record_geometry(path, records=2)
