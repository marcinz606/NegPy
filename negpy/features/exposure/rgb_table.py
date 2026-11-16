"""
A Camera Raw Look profile (.xmp with an RGBTable) as a print stock: the profile's 3D table applied to
Digital's print, with the profile's Amount as Stock Strength. An imported look is stored in the user's
looks folder as an npz named by its look ID (the table ID, suffixed by a table amount under 1), which is the
stock key after LOOK_KEY_PREFIX.
"""

import functools
import hashlib
import html
import os
import pathlib
import re
import struct
import zipfile
import zlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import negpy_core

from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.print_lut import PrintLut
from negpy.kernel.system.i18n import tr

LUT_NODES = 33
LOOK_KEY_PREFIX = "look:"
# A look ID: the table's MD5, with a table amount suffix. A stock key comes from an edit, so the ID is
# checked before it names a file.
_LOOK_ID = re.compile(r"[0-9A-F]{32}(-[0-9]{4})?")

# dng_rgb_table enums: primaries 0 sRGB, 1 Adobe RGB, 2 ProPhoto, 3 Display P3, 4 Rec. 2020;
# gamma 0 linear, 1 sRGB, 2 1.8, 3 2.2, 4 Rec. 2020; gamut 0 clip, 1 extend.
_D65 = (0.3127, 0.3290)
_PRIMARIES_XY = {
    0: (((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)), _D65),
    1: (((0.64, 0.33), (0.21, 0.71), (0.15, 0.06)), _D65),
    2: (((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)), (0.3457, 0.3585)),
    3: (((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)), _D65),
    4: (((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)), _D65),
}
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
GAMUT_EXTEND = 1


def _xyz(xy: Tuple[float, float]) -> np.ndarray:
    return np.array([xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]])


def _to_xyz_d65(primaries: Tuple[Tuple[float, float], ...], white: Tuple[float, float]) -> np.ndarray:
    """RGB to XYZ, Bradford-adapted to D65 for a space with another white."""
    p = np.stack([_xyz(xy) for xy in primaries], axis=1)
    m = p * np.linalg.solve(p, _xyz(white))
    adapt = np.linalg.inv(_BRADFORD) @ np.diag((_BRADFORD @ _xyz(_D65)) / (_BRADFORD @ _xyz(white))) @ _BRADFORD
    return adapt @ m


def _adobe_to(primaries: int) -> np.ndarray:
    return np.linalg.solve(_to_xyz_d65(*_PRIMARIES_XY[primaries]), _to_xyz_d65(*_PRIMARIES_XY[1]))


# dng_function_GammaEncode_1_8: x^(1/1.8) with a Hermite toe from slope 32 at 0 up to x1.
_G18_X1 = 8.2118790552e-4
_G18_Y1 = _G18_X1 ** (1 / 1.8)
_G18_S1 = (1 / 1.8) * _G18_X1 ** (1 / 1.8 - 1)


def _gamma18_encode(x: np.ndarray) -> np.ndarray:
    t = np.clip(x / _G18_X1, 0.0, 1.0)
    h10, h01, h11 = t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
    toe = h10 * _G18_X1 * 32.0 + h01 * _G18_Y1 + h11 * _G18_X1 * _G18_S1
    return np.where(x <= _G18_X1, toe, np.power(np.maximum(x, _G18_X1), 1 / 1.8))


_G18_GRID = np.linspace(0.0, 1.0, 65537)
_G18_ENC = _gamma18_encode(_G18_GRID)


def _signed(encode):
    return lambda x: np.sign(x) * encode(np.abs(x))


def _srgb_encode(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def _srgb_decode(y: np.ndarray) -> np.ndarray:
    return np.where(y <= 0.04045, y / 12.92, np.power((np.maximum(y, 0.04045) + 0.055) / 1.055, 2.4))


def _rec2020_encode(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0.018053968510807, 4.5 * x, 1.09929682680944 * np.power(np.maximum(x, 0.018), 0.45) - 0.09929682680944)


def _rec2020_decode(y: np.ndarray) -> np.ndarray:
    return np.where(y < 0.081242858298635, y / 4.5, np.power((np.maximum(y, 0.0812) + 0.09929682680944) / 1.09929682680944, 1 / 0.45))


_GAMMA = {
    0: (lambda x: x, lambda y: y),
    1: (_signed(_srgb_encode), _signed(_srgb_decode)),
    2: (_signed(_gamma18_encode), _signed(lambda y: np.interp(y, _G18_ENC, _G18_GRID))),
    3: (_signed(lambda x: np.power(x, 1 / 2.2)), _signed(lambda y: np.power(y, 2.2))),
    4: (_signed(_rec2020_encode), _signed(_rec2020_decode)),
}


def apply_rgb_table(rgb: np.ndarray, table: np.ndarray, space: Tuple[int, int, int], amount: float) -> np.ndarray:
    """An (M, 3) linear Adobe RGB array through one RGBTable at `amount` (1 = 100%).

    The Amount blends in the table's own encoding, as Camera Raw does. A clip table clips its
    input; an extend table carries a color past the table edge by its distance from that edge."""
    primaries, gamma, gamut = space
    encode, decode = _GAMMA[gamma]
    m = _adobe_to(primaries)
    x = rgb @ m.T
    if gamut != GAMUT_EXTEND:
        x = np.clip(x, 0.0, 1.0)
    enc = encode(x)
    inside = np.clip(enc, 0.0, 1.0)
    looked = negpy_core.apply_lut_f32(np.ascontiguousarray(inside.reshape(-1, 1, 3), np.float32), table).reshape(enc.shape)
    out = enc + (looked.astype(np.float64) - inside) * amount
    if gamut != GAMUT_EXTEND:
        out = np.clip(out, 0.0, 1.0)
    return decode(out) @ np.linalg.inv(m).T


def table_from_u16(values: np.ndarray) -> np.ndarray:
    """A (N, N, N, 3) uint16 RGBTable as the float32 table apply_lut_f32 reads."""
    return np.ascontiguousarray(values, np.float32) / np.float32(65535.0)


def look_on_digital(e: np.ndarray, table: np.ndarray, space: Tuple[int, int, int], amount: float) -> np.ndarray:
    """(M, 3) dye densities to the stock's color: the RGBTable on Digital's per-channel BPC print. With
    Y_b = Digital's black, the kernel's luminance BPC (T - Y_b) / (1 - Y_b) of this gives the table on that print."""
    y_b = 10.0 ** -float(EXPOSURE_CONSTANTS["d_max"])
    digital = np.clip((10.0**-e - y_b) / (1 - y_b), 0.0, 1.0)
    return apply_rgb_table(digital, table, space, amount) * (1 - y_b) + y_b


def look_table_lut(table: np.ndarray, space: Tuple[int, int, int], amount: float, n: int = LUT_NODES) -> Tuple[np.ndarray, np.ndarray]:
    """(table, e_max) of the print stock that is look_on_digital, on an n³ grid over 0..d_max."""
    d_max = float(EXPOSURE_CONSTANTS["d_max"])
    axis = np.linspace(0.0, d_max, n)
    e = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    out = look_on_digital(e.reshape(-1, 3), table, space, amount).reshape(e.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32), np.full(3, d_max, np.float32)


# ── Camera Raw Look profiles ─────────────────────────────────────────────────

# dng_big_table's string form: base85 of uint32 length + zlib, 4-byte little-endian words, low digit first.
_B85 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?`'|()[]{}@%$#"
_B85_VALUE = {c: i for i, c in enumerate(_B85)}


def b85_decode(text: str) -> bytes:
    out = bytearray()
    for i in range(0, len(text), 5):
        group = text[i : i + 5]
        value = sum(_B85_VALUE[c] * 85**k for k, c in enumerate(group))
        word = (value & 0xFFFFFFFF).to_bytes(4, "little")
        out += word if len(group) == 5 else word[: len(group) - 1]
    return bytes(out)


def b85_encode(data: bytes) -> str:
    out = []
    for i in range(0, len(data), 4):
        chunk = data[i : i + 4]
        value = int.from_bytes(chunk.ljust(4, b"\0"), "little")
        digits = []
        for _ in range(5):
            digits.append(_B85[value % 85])
            value //= 85
        out.append("".join(digits[: len(chunk) + 1]))
    return "".join(out)


@dataclass(frozen=True)
class RgbTable:
    """A decoded RGBTable: values (N, N, N, 3) uint16 indexed [r, g, b], (primaries, gamma, gamut), the Amount
    range and the table's strength at Amount 100%."""

    values: np.ndarray
    space: Tuple[int, int, int]
    amounts: Tuple[float, float]
    scale: float = 1.0


# A dng_rgb_table stream is a 16-byte header, the uint16 table, the 12-byte space and two doubles. The
# declared length is checked against the largest table (64³ nodes) before the stream is inflated.
_MAX_TABLE_NODES = 64
_MAX_STREAM = 44 + 2 * 3 * _MAX_TABLE_NODES**3


def parse_rgb_table(text: str) -> Tuple[str, RgbTable]:
    """(table ID, table) of a crs:Table_ value. The ID is the MD5 of the dng_rgb_table stream."""
    damaged = tr("The look table is damaged.")
    try:
        blob = b85_decode(text.strip())
        (size,) = struct.unpack_from("<I", blob)
        if not 44 <= size <= _MAX_STREAM:
            raise ValueError(damaged)
        inflate = zlib.decompressobj()
        payload = inflate.decompress(blob[4:], size)
        if len(payload) != size or inflate.unconsumed_tail:
            raise ValueError(damaged)
        kind, version, dims, n = struct.unpack_from("<4I", payload, 0)
    except (KeyError, ValueError, struct.error, zlib.error) as e:
        raise ValueError(damaged) from e
    if (kind, version, dims) != (1, 1, 3) or not 2 <= n <= _MAX_TABLE_NODES:
        raise ValueError(tr("The look table is not a 3D color table."))
    count = n**3 * 3
    try:
        deltas = np.frombuffer(payload, "<u2", count, 16).reshape(n, n, n, 3).astype(np.int64)
        primaries, gamma, gamut = struct.unpack_from("<3I", payload, 16 + 2 * count)
        lo, hi = struct.unpack_from("<2d", payload, 28 + 2 * count)
    except (ValueError, struct.error) as e:
        raise ValueError(damaged) from e
    if primaries not in _PRIMARIES_XY or gamma not in _GAMMA:
        raise ValueError(tr("The look table uses a color space NegPy does not know."))
    ident = (np.arange(n) * 0xFFFF + (n >> 1)) // (n - 1)
    nominal = np.stack(np.meshgrid(ident, ident, ident, indexing="ij"), -1)
    values = ((deltas + nominal) & 0xFFFF).astype(np.uint16)
    return hashlib.md5(payload).hexdigest().upper(), RgbTable(values, (primaries, gamma, gamut), (lo, hi))


def read_look_profile(text: str) -> Tuple[str, str, RgbTable]:
    """(name, look ID, table) of a Look profile .xmp, whose crs:RGBTable names its crs:Table_.

    The table's Amount range becomes (1, 1) when the profile has no Amount (crs:SupportsAmount False). A
    crs:RGBTableAmount is the table's strength at Amount 100%, so a table under one profile keeps its own ID."""
    ref = re.search(r'crs:RGBTable="([0-9A-Fa-f]{32})"', text)
    if not ref:
        raise ValueError(tr("The file is not a Look profile with a color table."))
    encoded = re.search(rf'crs:Table_{ref.group(1)}="([^"]*)"', text)
    if not encoded:
        raise ValueError(
            tr("This Look profile uses a color table from another file. NegPy imports only a profile that holds its own table.")
        )
    table_id, table = parse_rgb_table(encoded.group(1))
    if table_id != ref.group(1).upper():
        raise ValueError(tr("The look table is damaged."))
    scale = re.search(r'crs:RGBTableAmount="([0-9]*\.?[0-9]+)"', text)
    scale_value = min(max(float(scale.group(1)), 0.0), 1.0) if scale else 1.0
    fixed = re.search(r'crs:SupportsAmount="False"', text) is not None
    table = RgbTable(table.values, table.space, (1.0, 1.0) if fixed else table.amounts, scale_value)
    look_id = table_id if scale_value == 1.0 else f"{table_id}-{round(scale_value * 1000):04d}"
    name = re.search(r"<crs:Name>\s*<rdf:Alt>\s*<rdf:li[^>]*>([^<]*)<", text)
    return (html.unescape(name.group(1)).strip() if name else "") or table_id[:8], look_id, table


# ── Imported looks ───────────────────────────────────────────────────────────


def looks_dir() -> pathlib.Path:
    from negpy.kernel.system.config import APP_CONFIG

    return pathlib.Path(APP_CONFIG.looks_dir)


def import_look_profile(src_path: str) -> str:
    """Store a Look profile's table in the looks folder; returns its stock key. Raises ValueError."""
    with open(src_path, "r", encoding="utf-8", errors="replace") as f:
        name, look_id, table = read_look_profile(f.read())
    folder = looks_dir()
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / f"{look_id}.npz",
        rgb_table=table.values,
        space=np.array(table.space, np.int64),
        amounts=np.array(table.amounts, np.float64),
        scale=np.array(table.scale, np.float64),
        name=np.array(name),
        source=np.array(os.path.basename(src_path)),
    )
    load_look.cache_clear()
    list_looks.cache_clear()
    return LOOK_KEY_PREFIX + look_id


def _look_path(key: str) -> pathlib.Path | None:
    """The look's file in the looks folder, or None when the key is not a look ID."""
    if not key.startswith(LOOK_KEY_PREFIX) or not _LOOK_ID.fullmatch(key[len(LOOK_KEY_PREFIX) :]):
        return None
    return looks_dir() / f"{key[len(LOOK_KEY_PREFIX) :]}.npz"


def delete_look(key: str) -> None:
    """Delete an imported look's file. A frame that still names it prints on Digital."""
    path = _look_path(key)
    if path is None:
        return
    path.unlink(missing_ok=True)
    load_look.cache_clear()
    list_looks.cache_clear()
    _look_lut.cache_clear()


@dataclass(frozen=True)
class Look:
    name: str
    table: np.ndarray
    space: Tuple[int, int, int]
    amounts: Tuple[float, float]
    scale: float = 1.0


@functools.lru_cache(maxsize=None)
def load_look(key: str) -> Look | None:
    """An imported look by stock key, or None when the key is not an imported look or its file is gone or unreadable."""
    path = _look_path(key)
    if path is None or not path.is_file():
        return None
    try:
        with np.load(path) as f:
            space = tuple(int(v) for v in f["space"])
            lo, hi = (float(v) for v in f["amounts"])
            scale = float(f["scale"]) if "scale" in f.files else 1.0
            values = f["rgb_table"]
            name = str(f["name"])
    except (OSError, KeyError, ValueError, TypeError, zipfile.BadZipFile, zlib.error):
        return None
    n = values.shape[0] if values.ndim == 4 else 0
    if values.shape != (n, n, n, 3) or n < 2 or len(space) != 3 or space[0] not in _PRIMARIES_XY or space[1] not in _GAMMA:
        return None
    return Look(name, table_from_u16(values), (space[0], space[1], space[2]), (lo, hi), scale)


@functools.lru_cache(maxsize=None)
def list_looks() -> List[Tuple[str, str]]:
    """(stock key, name) of every imported look, by name."""
    folder = looks_dir()
    if not folder.is_dir():
        return []
    keys = [LOOK_KEY_PREFIX + p.stem for p in folder.glob("*.npz")]
    looks = [(k, load_look(k)) for k in keys]
    return sorted(((k, look.name) for k, look in looks if look is not None), key=lambda kn: kn[1].casefold())


def clamp_strength(look: Look, strength: float) -> float:
    """Strength inside the profile's Amount range, rounded to the slider step."""
    lo, hi = look.amounts
    return round(float(min(max(strength, lo), hi)), 3)


@functools.lru_cache(maxsize=16)
def _look_lut(key: str, strength: float) -> PrintLut:
    look = load_look(key)
    assert look is not None
    table, e_max = look_table_lut(look.table, look.space, strength * look.scale)
    return PrintLut(key, table, e_max, look.name)


def look_print_lut(key: str, strength: float) -> PrintLut | None:
    """The imported look's LUT at a Strength, or None when the look is gone. Every Strength keeps the
    stock key, so the GPU replaces that stock's one texture."""
    look = load_look(key)
    return None if look is None else _look_lut(key, clamp_strength(look, strength))
