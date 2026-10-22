"""
A print stock's color: a density-domain dye matrix, or a 3D LUT from the dye densities above
base to the print's color in linear Adobe RGB, media-relative. The LUTs are baked by
scripts/build_print_luts.py, one file per data source.
"""

import functools
import os
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numba import njit  # type: ignore

from negpy.features.lab.logic import LUM_B, LUM_G, LUM_R
from negpy.kernel.system.paths import get_resource_path

DyeMatrixRows = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]

LUT_DIR = pathlib.Path(get_resource_path(os.path.join("negpy", "features", "exposure", "print_luts")))


@dataclass(frozen=True)
class DyeMatrix:
    """D_rgb = M · D_dye above base; rows are normalized to sum 1 at use."""

    rows: DyeMatrixRows


@dataclass(frozen=True)
class DyeLut:
    """The LUT in print_luts/<key>.npz."""

    key: str


PrintColor = DyeMatrix | DyeLut


@dataclass(frozen=True, eq=False)
class PrintLut:
    """table: (N, N, N, 3) over C, M, Y dye density above base, 0..e_max per axis.
    flat_rows: the density 3×3 the Flat master uses in its place (see flat_matrix)."""

    key: str
    table: np.ndarray
    e_max: np.ndarray
    source: str
    flat_rows: Optional[np.ndarray] = None


def save_print_lut(path: pathlib.Path, table: np.ndarray, e_max: np.ndarray, source: str, flat_rows: Optional[np.ndarray] = None) -> None:
    extra = {} if flat_rows is None else {"flat": np.asarray(flat_rows, np.float64)}
    np.savez_compressed(path, lut=np.asarray(table, np.float32), e_max=np.asarray(e_max, np.float32), source=np.array(source), **extra)


@functools.lru_cache(maxsize=None)
def load_print_lut(key: str) -> PrintLut:
    path = LUT_DIR / f"{key}.npz"
    if not path.exists():
        raise FileNotFoundError(f"print LUT '{key}' not found at {path}")
    with np.load(path) as f:
        flat = np.array(f["flat"], np.float64) if "flat" in f.files else None
        if flat is not None:
            flat.flags.writeable = False
        return PrintLut(
            key, np.ascontiguousarray(f["lut"], np.float32), np.ascontiguousarray(f["e_max"], np.float32), str(f["source"]), flat
        )


@njit(inline="always")
def tetra_lookup(table: np.ndarray, e_max: np.ndarray, e0: float, e1: float, e2: float) -> Tuple[float, float, float]:
    """Tetrahedral interpolation, clamped to the table. exposure.wgsl's lut_tetra mirrors it."""
    n = table.shape[0] - 1
    x = min(max(e0 / e_max[0], 0.0), 1.0) * n
    y = min(max(e1 / e_max[1], 0.0), 1.0) * n
    z = min(max(e2 / e_max[2], 0.0), 1.0) * n
    i = min(int(x), n - 1)
    j = min(int(y), n - 1)
    k = min(int(z), n - 1)
    fx = x - i
    fy = y - j
    fz = z - k
    out = np.empty(3, dtype=np.float64)
    for c in range(3):
        c000 = table[i, j, k, c]
        c111 = table[i + 1, j + 1, k + 1, c]
        if fx >= fy:
            if fy >= fz:
                c100 = table[i + 1, j, k, c]
                c110 = table[i + 1, j + 1, k, c]
                out[c] = c000 + fx * (c100 - c000) + fy * (c110 - c100) + fz * (c111 - c110)
            elif fx >= fz:
                c100 = table[i + 1, j, k, c]
                c101 = table[i + 1, j, k + 1, c]
                out[c] = c000 + fx * (c100 - c000) + fz * (c101 - c100) + fy * (c111 - c101)
            else:
                c001 = table[i, j, k + 1, c]
                c101 = table[i + 1, j, k + 1, c]
                out[c] = c000 + fz * (c001 - c000) + fx * (c101 - c001) + fy * (c111 - c101)
        else:
            if fz >= fy:
                c001 = table[i, j, k + 1, c]
                c011 = table[i, j + 1, k + 1, c]
                out[c] = c000 + fz * (c001 - c000) + fy * (c011 - c001) + fx * (c111 - c011)
            elif fz >= fx:
                c010 = table[i, j + 1, k, c]
                c011 = table[i, j + 1, k + 1, c]
                out[c] = c000 + fy * (c010 - c000) + fz * (c011 - c010) + fx * (c111 - c011)
            else:
                c010 = table[i, j + 1, k, c]
                c110 = table[i + 1, j + 1, k, c]
                out[c] = c000 + fy * (c010 - c000) + fx * (c110 - c010) + fz * (c111 - c110)
    return out[0], out[1], out[2]


# Plain Python calls this twin: an inline kernel has no cache, so it would compile on every launch.
_tetra_lookup_cached = njit(cache=True)(tetra_lookup.py_func)


def lut_black_luminance(lut: PrintLut, d_min_rgb: Tuple[float, float, float], black_rgb: Tuple[float, float, float]) -> float:
    """Luminance of the print's black (density black_rgb), which black point compensation maps to 0."""
    rgb = _tetra_lookup_cached(lut.table, lut.e_max, *(b - d for b, d in zip(black_rgb, d_min_rgb)))
    return float(sum(w * 10.0**-d * v for w, d, v in zip((LUM_R, LUM_G, LUM_B), d_min_rgb, rgb)))


def flat_matrix(key: str) -> Optional[np.ndarray]:
    """The stock's color as a density 3×3 for the Flat master, which has no dye amounts: the rows
    build_print_luts.py fits to the LUT in CIEDE2000 and stores beside it. Rows sum to 1, so
    neutrals hold. None for a LUT baked without them."""
    rows = load_print_lut(key).flat_rows
    return None if rows is None else rows.astype(np.float64)
