"""Regression contracts for fail-closed LS-5000 exposure metering."""

import json

import numpy as np
import pytest

from negpy.infrastructure.scanners.ls5000_single_pass.meter import (
    CHANNELS,
    DEFAULT_EXPOSURES,
    METER_PASS_BYTES,
    METER_ROWS,
    METER_TAIL_SAMPLES,
    METER_WIDTH,
    decode_meter_pass,
    evaluate_meter_sequence,
    observe_meter_pass,
    propose_next_exposures,
    verify_final_convergence,
)


def _payload(image: np.ndarray, *, tail_seed: int = 0) -> bytes:
    rows = np.zeros((METER_ROWS, 1280), dtype=">u2")
    rows[:, :1124] = image.transpose(0, 2, 1).reshape(METER_ROWS, 1124)
    tails = (np.arange(METER_ROWS * METER_TAIL_SAMPLES, dtype=np.uint32) + tail_seed) % 65536
    rows[:, 1124:] = tails.reshape(METER_ROWS, METER_TAIL_SAMPLES)
    return rows.tobytes()


def _textured(peaks: tuple[int, int, int, int] = (32_000, 30_000, 34_000, 29_000)) -> np.ndarray:
    yy, xx = np.mgrid[0:METER_ROWS, 0:METER_WIDTH]
    field = 0.06 + 0.94 * (0.53 * xx / (METER_WIDTH - 1) + 0.47 * yy / (METER_ROWS - 1))
    image = np.empty((METER_ROWS, METER_WIDTH, 4), dtype=np.uint16)
    for channel, peak in enumerate(peaks):
        image[:, :, channel] = np.round(900 + peak * field).astype(np.uint16)
    return image


def _rescale(
    image: np.ndarray,
    old: dict[str, int],
    new: dict[str, int],
    *,
    pedestal: float = 900.0,
) -> np.ndarray:
    result = np.empty_like(image)
    for index, channel in enumerate(CHANNELS):
        gain = new[channel] / old[channel]
        result[:, :, index] = np.clip(
            pedestal + (image[:, :, index].astype(np.float64) - pedestal) * gain,
            0,
            65_535,
        ).astype(np.uint16)
    return result


def test_meter_decode_preserves_channel_order_and_opaque_row_tail() -> None:
    payload = _payload(_textured(), tail_seed=123)

    decoded = decode_meter_pass(payload)

    assert len(payload) == METER_PASS_BYTES
    assert decoded.image.shape == (METER_ROWS, METER_WIDTH, 4)
    assert decoded.row_tail.shape == (METER_ROWS, METER_TAIL_SAMPLES)
    assert decoded.channels == CHANNELS
    assert decoded.to_bytes() == payload


@pytest.mark.parametrize("delta", [-2, -1, 2])
def test_meter_decode_rejects_any_length_corruption(delta: int) -> None:
    payload = _payload(_textured())
    corrupt = payload[:delta] if delta < 0 else payload + b"\x00" * delta
    with pytest.raises(ValueError, match="expected 1088000"):
        decode_meter_pass(corrupt)


def test_proposal_is_bounded_and_uses_a_lower_ir_target() -> None:
    proposal = propose_next_exposures(observe_meter_pass(_payload(_textured()), DEFAULT_EXPOSURES))

    assert proposal.accepted, proposal.to_dict()
    for channel in CHANNELS:
        assert DEFAULT_EXPOSURES[channel] < proposal.proposed_exposures[channel] <= 2 * DEFAULT_EXPOSURES[channel]
    assert proposal.channel_diagnostics["IR"]["target_fraction"] < proposal.channel_diagnostics["R"]["target_fraction"]


def test_width_only_ir_correlation_accepts_noise_but_refuses_one_transport_row_shift() -> None:
    rng = np.random.default_rng(20260713)
    _yy, xx = np.mgrid[0:METER_ROWS, 0:METER_WIDTH]
    row_structure = rng.normal(0.0, 1_050.0, size=(METER_ROWS, 1))
    column_structure = 180.0 * np.sin(xx * 2.0 * np.pi / 37.0)
    underlying_ir = 44_000.0 + row_structure + column_structure
    first_ir = 1_000.0 + underlying_ir + rng.normal(0.0, 195.0, size=underlying_ir.shape)
    exposure_gain = 1.1
    second_ir = 1_000.0 + underlying_ir * exposure_gain + rng.normal(0.0, 195.0, size=underlying_ir.shape)

    first_image = _textured((30_000,) * 4)
    first_image[:, :, 3] = np.clip(first_ir, 0, 65_535).astype(np.uint16)
    second_exposures = {channel: int(exposure * exposure_gain) for channel, exposure in DEFAULT_EXPOSURES.items()}
    second_image = _rescale(first_image, DEFAULT_EXPOSURES, second_exposures)
    second_image[:, :, 3] = np.clip(second_ir, 0, 65_535).astype(np.uint16)
    first = observe_meter_pass(_payload(first_image), DEFAULT_EXPOSURES)
    second = observe_meter_pass(_payload(second_image), second_exposures)

    aligned = propose_next_exposures(second, previous=first)
    assert aligned.accepted, aligned.to_dict()
    assert aligned.channel_diagnostics["IR"]["linearity"]["correlation_aggregation"] == {
        "axis": "width",
        "size": 9,
        "requires_all_raw_valid": True,
    }

    shifted_image = second_image.copy()
    shifted_image[:, :, 3] = np.roll(second_image[:, :, 3], 1, axis=0)
    shifted = propose_next_exposures(
        observe_meter_pass(_payload(shifted_image), second_exposures),
        previous=first,
    )
    assert ("low_correlation", "IR") in {(item.code, item.channel) for item in shifted.refusals}


def test_three_pass_sequence_converges_and_requires_exact_exposure_history() -> None:
    first_image = np.roll(_textured((31_000, 30_000, 32_000, 29_000)), shift=(141, 93), axis=(0, 1))
    first = observe_meter_pass(_payload(first_image), DEFAULT_EXPOSURES)
    first_proposal = propose_next_exposures(first)
    second_image = _rescale(first_image, DEFAULT_EXPOSURES, first_proposal.proposed_exposures)
    second = observe_meter_pass(_payload(second_image), first_proposal.proposed_exposures)
    second_proposal = propose_next_exposures(second, previous=first)
    third_image = _rescale(second_image, first_proposal.proposed_exposures, second_proposal.proposed_exposures)
    third = observe_meter_pass(_payload(third_image), second_proposal.proposed_exposures)

    final = verify_final_convergence(third, previous=second)
    result = evaluate_meter_sequence(
        [_payload(first_image), _payload(second_image), _payload(third_image)],
        [DEFAULT_EXPOSURES, first_proposal.proposed_exposures, second_proposal.proposed_exposures],
    )
    assert final.accepted, final.to_dict()
    assert result.accepted, result.to_dict()
    assert result.final_exposures == final.final_exposures
    json.dumps(result.to_dict(), allow_nan=False)

    mismatched = [dict(item) for item in [DEFAULT_EXPOSURES, first_proposal.proposed_exposures, second_proposal.proposed_exposures]]
    mismatched[1]["IR"] += 1
    refused = evaluate_meter_sequence(
        [_payload(first_image), _payload(second_image), _payload(third_image)],
        mismatched,
    )
    assert "exposure_history_mismatch" in {item.code for item in refused.refusals}


def test_final_acceptance_requires_fresh_previous_pass_linearity() -> None:
    observation = observe_meter_pass(_payload(_textured((24_000,) * 4)), DEFAULT_EXPOSURES)
    result = verify_final_convergence(observation)
    assert not result.accepted
    assert "missing_previous_linearity" in {item.code for item in result.refusals}
