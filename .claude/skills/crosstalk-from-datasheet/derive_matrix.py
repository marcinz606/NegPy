#!/usr/bin/env python3
"""Derive a NegPy spectral-crosstalk matrix from sampled dye spectral densities.

The hard arithmetic lives here so it stays deterministic: the caller only reads
values off the datasheet's dye-density curves and hands them over as a 3x3
leakage matrix ``A`` (rows = R/G/B measurement bands, cols = C/M/Y dyes).

Pipeline: column-normalize A so each dye's own band = 1 -> invert (M = A^-1,
giving a diagonal-dominant matrix with negative off-diagonals) -> rescale each
row so its diagonal = 1 (cosmetic; NegPy row-normalizes anyway) -> emit TOML.

Usage:
    derive_matrix.py --in readings.json [--name NAME] [--out film.toml]
    echo '{"name": "...", "readings": [[...],[...],[...]]}' | derive_matrix.py
    derive_matrix.py --selftest

Input JSON:
    {"name": "Kodak Foo 100", "readings": [[1.0, 0.08, 0.05],
                                           [0.06, 1.0, 0.10],
                                           [0.03, 0.07, 1.0]]}
"""

from __future__ import annotations

import argparse
import json
import sys

Matrix = list[list[float]]

ROUND = 4


def _validate(readings: object) -> Matrix:
    """Coerce input to a 3x3 float matrix or raise ValueError."""
    if not isinstance(readings, list) or len(readings) != 3:
        raise ValueError("readings must be a list of 3 rows")
    out: Matrix = []
    for row in readings:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("each row must have exactly 3 values")
        coerced: list[float] = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"non-numeric value: {v!r}")
            coerced.append(float(v))
        out.append(coerced)
    return out


def _normalize_columns(a: Matrix) -> Matrix:
    """Divide each dye column by its own-band (diagonal) value -> diagonal 1."""
    out: Matrix = [[0.0] * 3 for _ in range(3)]
    for j in range(3):
        peak = a[j][j]
        if peak == 0:
            raise ValueError(f"dye {j} has zero density in its own band")
        for i in range(3):
            out[i][j] = a[i][j] / peak
    return out


def _inverse3(m: Matrix) -> Matrix:
    """Analytic 3x3 inverse (pure stdlib). Raises on a singular matrix."""
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise ValueError("leakage matrix is singular (cannot invert)")
    inv = [
        [(e * i - f * h), (c * h - b * i), (b * f - c * e)],
        [(f * g - d * i), (a * i - c * g), (c * d - a * f)],
        [(d * h - e * g), (b * g - a * h), (a * e - b * d)],
    ]
    return [[inv[r][s] / det for s in range(3)] for r in range(3)]


def _rescale_rows(m: Matrix) -> Matrix:
    """Divide each row by its diagonal so the diagonal reads 1.0."""
    out: Matrix = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        diag = m[i][i]
        if diag == 0:
            raise ValueError(f"row {i} has zero diagonal after inversion")
        for j in range(3):
            out[i][j] = m[i][j] / diag
    return out


def derive(readings: object) -> Matrix:
    """Full pipeline: validated leakage readings -> crosstalk matrix."""
    a = _validate(readings)
    return _rescale_rows(_inverse3(_normalize_columns(a)))


def _round(m: Matrix) -> Matrix:
    return [[round(v, ROUND) for v in row] for row in m]


def to_toml(name: str | None, m: Matrix) -> str:
    """Render the gallery TOML (name optional, 3x3 matrix block)."""
    rows = ",\n".join(
        "  [" + ", ".join(f"{v:8.4f}" for v in row).strip() + "]" for row in m
    )
    head = f'name = "{name}"\n' if name else ""
    return f"{head}matrix = [\n{rows},\n]\n"


def _selftest() -> int:
    # Identity leakage -> identity correction (no off-diagonal terms).
    ident = derive([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert _round(ident) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], ident

    # Symmetric leakage -> diagonal 1, all off-diagonals negative (subtracts).
    m = derive([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]])
    assert all(m[i][i] == 1.0 for i in range(3)), m
    assert all(m[i][j] < 0 for i in range(3) for j in range(3) if i != j), m

    # Singular input must raise, not emit garbage.
    try:
        derive([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("singular matrix did not raise")

    print("selftest ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="infile", help="JSON input file (default: stdin)")
    p.add_argument("--name", help="display name (overrides JSON 'name')")
    p.add_argument("--out", help="write TOML here instead of stdout")
    p.add_argument("--selftest", action="store_true", help="run internal checks")
    args = p.parse_args(argv)

    if args.selftest:
        return _selftest()

    raw = open(args.infile).read() if args.infile else sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON: {exc}", file=sys.stderr)
        return 2

    try:
        matrix = _round(derive(data.get("readings")))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    name = args.name or data.get("name")
    toml = to_toml(name, matrix)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(toml)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(toml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
