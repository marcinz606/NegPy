#!/usr/bin/env python3
"""DLL-free evaluator for the two captured CML4 optimized transforms.

This program implements the integer runtime around two vendor-derived 32^3
CLUTs.  It intentionally treats the dumped CLUTs and 65,536-entry LUTs as
oracle data, verifies every asset hash before use, and never loads CML4.dll.

The evaluator covers the twelve stored events:

* Stage 1: NKLS5000_N + ramp8to16 -> NKLch (30,34,38,42,47,52)
* Stage 2: NKLch -> NKAdobe, no merged LUT (31,35,39,43,48,53)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
DEFAULT_LAB = PROJECT / "reverse_engineering/.work-cml-replay/lab"

STAGE1_EVENTS = (30, 34, 38, 42, 47, 52)
STAGE2_EVENTS = (31, 35, 39, 43, 48, 53)
GEOMETRY = {
    30: (104, 104), 31: (104, 104), 34: (54, 104), 35: (54, 104),
    38: (104, 72), 39: (104, 72), 42: (54, 72), 43: (54, 72),
    47: (104, 104), 48: (104, 104), 52: (54, 104), 53: (54, 104),
}
ROW_PIXELS = 104

ASSET_SHA256 = {
    "lch-atan-u16le.bin":
        "56b8ac82456941a0a8aad6d7de2c79b21785529037b504582731ce6abcd143b1",
    "lch-sincos-i16le.bin":
        "dc8e71681bc46e60e33448171865fc96770dc17d3cfd0db835f6173e6fed7a35",
    "lch-reciprocal-u16le.bin":
        "6959ef7deeb57dc96eb4653fe13b4d37fcb4250c28be15654b8e8dd236849042",
    "cml4-stage1-clut0.bin":
        "a2abbc1e76dc037e6b364a58b483ddf01b9dde7c0e072f85885cb1f8c9dcbf1c",
    "cml4-stage1-input-lut0.bin":
        "1a487c024ceaf83018c8ab0e405e9c12b25a0500a2ae6e3465e20b29638729d7",
    "cml4-stage1-output-lut0.bin":
        "fe87ce159ec126597f9fb605b57cebec9b0264da1ca16d1725efe34db2e4fd2b",
    "cml4-stage2-clut0.bin":
        "d14b7c76091552bf03899327ca6a7c74c712a0423e429de8f8cc8203d5c98da3",
    "cml4-stage2-input-lut0.bin":
        "60fa510492f2adad2be9b00107b3d7ed7188a798f29748c3401560b70be3a248",
    "cml4-stage2-output-lut0.bin":
        "b88574377d1a0cf47fe8806335641db2ca35197817cd055a4cd55141708c8291",
}


def verify_assets(directory: Path) -> None:
    for name, expected in ASSET_SHA256.items():
        path = directory / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"asset hash mismatch: {path}: {actual} != {expected}")


def load_event(directory: Path, event: int, kind: str) -> np.ndarray:
    width, height = GEOMETRY[event]
    return load_event_payload(directory, event, kind)[:, :width, :].reshape(-1, 3)


def load_event_payload(directory: Path, event: int, kind: str) -> np.ndarray:
    _, height = GEOMETRY[event]
    path = directory / f"event{event}-{kind}.bin"
    raw = np.fromfile(path, dtype="<u2")
    expected_words = height * ROW_PIXELS * 3
    if raw.size != expected_words:
        raise ValueError(f"unexpected event size: {path}: {raw.size} != {expected_words}")
    return raw.reshape(height, ROW_PIXELS, 3)


def lookup_three(lut: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    return np.column_stack([lut[c, pixels[:, c]] for c in range(3)]).astype(np.uint16)


def optimized_trilinear(clut: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Exact scalar expression of optimized kernel CML4.dll+0x11e30.

    Nodes are four padded u16s.  The MMX kernel halves every node, performs
    three q15 linear interpolations with arithmetic shifts, then doubles the
    result.  Coordinate multiplication by 0x1f001f maps u16 endpoints onto a
    32-point grid and yields a 15-bit fraction.
    """
    source = np.asarray(pixels, dtype=np.uint16).reshape(-1, 3)
    packed = source.astype(np.uint64) * np.uint64(0x1F001F) + np.uint64(0x10000)
    index = (packed >> np.uint64(32)).astype(np.int64)
    fraction = ((packed & np.uint64(0xFFFFFFFF)) >> np.uint64(17)).astype(np.int64)
    ix, iy, iz = index.T
    fx, fy, fz = fraction.T

    def node(dx: int, dy: int, dz: int) -> np.ndarray:
        # At the top endpoint the fraction is zero.  The DLL may read a padded
        # neighbor, but that value is multiplied by zero; clamping is exactly
        # equivalent and avoids an out-of-array access in the portable model.
        return (
            clut[
                np.minimum(ix + dx, 31),
                np.minimum(iy + dy, 31),
                np.minimum(iz + dz, 31),
            ].astype(np.int64)
            >> 1
        )

    def lerp(a: np.ndarray, b: np.ndarray, q15: np.ndarray) -> np.ndarray:
        return a + (((b - a) * q15[:, None]) >> 15)

    z0 = lerp(
        lerp(node(0, 0, 0), node(1, 0, 0), fx),
        lerp(node(0, 1, 0), node(1, 1, 0), fx),
        fy,
    )
    z1 = lerp(
        lerp(node(0, 0, 1), node(1, 0, 1), fx),
        lerp(node(0, 1, 1), node(1, 1, 1), fx),
        fy,
    )
    result = (lerp(z0, z1, fz) << 1) & 0xFFFF
    return result[:, :3].astype(np.uint16)


def trunc_q15(product: np.ndarray) -> np.ndarray:
    """Signed division by 32768, truncating toward zero (0x100041d0)."""
    return np.where(product < 0, -((-product) >> 15), product >> 15)


def lch_to_lab_codes(
    pixels: np.ndarray,
    sincos: np.ndarray,
    reciprocal: np.ndarray,
) -> np.ndarray:
    """Exact pre-CLUT Lch special case at CML4.dll+0x100041d0."""
    source = np.asarray(pixels, dtype=np.uint16).reshape(-1, 3)
    chroma = source[:, 1].astype(np.int64)
    hue = source[:, 2].astype(np.int64)
    used = np.minimum(chroma, reciprocal[hue].astype(np.int64))
    # Captured sincos records are [sin, cos].
    a = trunc_q15(sincos[hue, 1].astype(np.int64) * used)
    b = trunc_q15(sincos[hue, 0].astype(np.int64) * used)
    return np.column_stack(
        [
            source[:, 0],
            (32768 + a).astype(np.uint16),
            (32768 + b).astype(np.uint16),
        ]
    )


def hue_from_signed_ab(a: np.ndarray, b: np.ndarray, atan: np.ndarray) -> np.ndarray:
    """Exact ratio table, axes, and quadrant reconstruction in 0x10004290."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    aa, bb = np.abs(a), np.abs(b)
    high, low = np.maximum(aa, bb), np.minimum(aa, bb)
    ratio = np.zeros_like(high)
    nz = high != 0
    ratio[nz] = (low[nz] << 13) // high[nz]
    angle = atan[ratio].astype(np.int64)
    hue = np.zeros_like(a)
    a_major = aa >= bb
    q1 = (a > 0) & (b > 0)
    q2 = (a < 0) & (b > 0)
    q3 = (a < 0) & (b < 0)
    q4 = (a > 0) & (b < 0)
    hue[q1] = np.where(a_major[q1], angle[q1], 16384 - angle[q1])
    hue[q2] = np.where(a_major[q2], 32768 - angle[q2], 16384 + angle[q2])
    hue[q3] = np.where(a_major[q3], 32768 + angle[q3], 49152 - angle[q3])
    hue[q4] = np.where(a_major[q4], 65536 - angle[q4], 49152 + angle[q4])
    hue[(a == 0) & (b > 0)] = 16384
    hue[(a < 0) & (b == 0)] = 32768
    hue[(a == 0) & (b < 0)] = 49152
    return (hue & 0xFFFF).astype(np.uint16)


def rounded_integer_hypot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Round sqrt(a*a+b*b) to nearest, without host floating-point drift."""
    squared = a.astype(np.int64) ** 2 + b.astype(np.int64) ** 2
    floor_root = np.fromiter(
        (math.isqrt(int(value)) for value in squared),
        dtype=np.int64,
        count=squared.size,
    )
    # sqrt(integer) can never be exactly k+0.5, so no tie rule is needed.
    return floor_root + ((squared - floor_root * floor_root) > floor_root)


def lab_to_lch_codes(pixels: np.ndarray, atan: np.ndarray) -> np.ndarray:
    """Exact post-CLUT Lab special case at CML4.dll+0x10004290."""
    source = np.asarray(pixels, dtype=np.uint16).reshape(-1, 3)

    def rescale(code: np.ndarray) -> np.ndarray:
        value = (code.astype(np.uint64) - np.uint64(8192)) * np.uint64(65535)
        product = value * np.uint64(87384)  # ceil(2**32 / 49151)
        return (
            (product >> np.uint64(32))
            + ((product & np.uint64(0xFFFFFFFF)) >> np.uint64(31))
        ).astype(np.int64)

    a = rescale(source[:, 1]) - 32768
    b = rescale(source[:, 2]) - 32768
    chroma = rounded_integer_hypot(a, b)
    hue = hue_from_signed_ab(a, b, atan)
    return np.column_stack([source[:, 0], chroma, hue]).astype(np.uint16)


class CapturedTransforms:
    def __init__(self, assets: Path):
        verify_assets(assets)
        self.atan = np.fromfile(assets / "lch-atan-u16le.bin", dtype="<u2")
        self.sincos = np.fromfile(
            assets / "lch-sincos-i16le.bin", dtype="<i2"
        ).reshape(65537, 2)
        self.reciprocal = np.fromfile(
            assets / "lch-reciprocal-u16le.bin", dtype="<u2"
        )
        self.s1_clut = np.fromfile(
            assets / "cml4-stage1-clut0.bin", dtype="<u2"
        ).reshape(32, 32, 32, 4)
        self.s1_input = np.fromfile(
            assets / "cml4-stage1-input-lut0.bin", dtype="<u2"
        ).reshape(3, 65536)
        self.s1_output = np.fromfile(
            assets / "cml4-stage1-output-lut0.bin", dtype="<u2"
        ).reshape(3, 65536)
        self.s2_clut = np.fromfile(
            assets / "cml4-stage2-clut0.bin", dtype="<u2"
        ).reshape(32, 32, 32, 4)
        self.s2_input = np.fromfile(
            assets / "cml4-stage2-input-lut0.bin", dtype="<u2"
        ).reshape(3, 65536)
        self.s2_output = np.fromfile(
            assets / "cml4-stage2-output-lut0.bin", dtype="<u2"
        ).reshape(3, 65536)

    def stage1(self, source: np.ndarray) -> np.ndarray:
        working = lookup_three(self.s1_input, source)
        working = optimized_trilinear(self.s1_clut, working)
        working = lab_to_lch_codes(working, self.atan)
        return lookup_three(self.s1_output, working)

    def stage2(self, source: np.ndarray) -> np.ndarray:
        # CML4's Lch special-case deliberately bypasses the generated input
        # LUT for hue.  Its dumped plane 2 maps the top half down by one; using
        # it causes 4,729 mismatches across these events.
        working = source.copy()
        working[:, 0] = self.s2_input[0, working[:, 0]]
        working[:, 1] = self.s2_input[1, working[:, 1]]
        working = lch_to_lab_codes(working, self.sincos, self.reciprocal)
        working = optimized_trilinear(self.s2_clut, working)
        return lookup_three(self.s2_output, working)


def metric(predicted: np.ndarray, expected: np.ndarray) -> dict[str, int | float]:
    difference = predicted.astype(np.int64) - expected.astype(np.int64)
    absolute = np.abs(difference)
    return {
        "mismatched_u16": int(np.count_nonzero(difference)),
        "total_u16": int(difference.size),
        "max_abs": int(absolute.max(initial=0)),
        "mae": float(absolute.mean()),
    }


def evaluate(model: CapturedTransforms, lab: Path) -> dict:
    result: dict[str, dict] = {}
    every_predicted_active = []
    every_expected_active = []
    every_predicted_payload = []
    every_expected_payload = []
    for label, events, function in (
        ("stage1", STAGE1_EVENTS, model.stage1),
        ("stage2", STAGE2_EVENTS, model.stage2),
    ):
        predicted_all, expected_all = [], []
        predicted_payload_all, expected_payload_all = [], []
        per_event = {}
        for event in events:
            width, height = GEOMETRY[event]
            source = load_event(lab, event, "source")
            expected = load_event(lab, event, "expected")
            predicted = function(source)
            expected_payload = load_event_payload(lab, event, "expected")
            predicted_payload = np.zeros_like(expected_payload)
            predicted_payload[:, :width, :] = predicted.reshape(height, width, 3)
            per_event[str(event)] = {
                **metric(predicted, expected),
                "full_payload": metric(predicted_payload, expected_payload),
            }
            predicted_all.append(predicted)
            expected_all.append(expected)
            predicted_payload_all.append(predicted_payload)
            expected_payload_all.append(expected_payload)
        result[label] = {
            "per_event": per_event,
            "total": metric(np.concatenate(predicted_all), np.concatenate(expected_all)),
            "full_payload_total": metric(
                np.concatenate(predicted_payload_all),
                np.concatenate(expected_payload_all),
            ),
        }
        every_predicted_active.extend(predicted_all)
        every_expected_active.extend(expected_all)
        every_predicted_payload.extend(predicted_payload_all)
        every_expected_payload.extend(expected_payload_all)
    result["all_12_events"] = metric(
        np.concatenate(every_predicted_active), np.concatenate(every_expected_active)
    )
    result["all_12_full_payloads"] = {
        **metric(
            np.concatenate(every_predicted_payload),
            np.concatenate(every_expected_payload),
        ),
        "total_bytes": int(sum(value.nbytes for value in every_expected_payload)),
        "mismatched_bytes": int(
            np.count_nonzero(
                np.concatenate(every_predicted_payload).view(np.uint8)
                != np.concatenate(every_expected_payload).view(np.uint8)
            )
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=HERE)
    parser.add_argument("--lab", type=Path, default=DEFAULT_LAB)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = evaluate(CapturedTransforms(args.assets), args.lab)
    print(json.dumps(result, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return (
        0
        if result["all_12_events"]["mismatched_u16"] == 0
        and result["all_12_full_payloads"]["mismatched_u16"] == 0
        and result["all_12_full_payloads"]["mismatched_bytes"] == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
