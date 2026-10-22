"""Film edge barcodes as module bits read left to right; 1 is a bar, which prints light."""

DX_MODULE = 0.406
DX_DATA_TRACK = (0.0, 1.0)
DX_CLOCK_TRACK = (1.0, 2.2)

# SMPTE ST 254 KEYKODE: Code 128 subset C, 123 modules of 0.1905 mm (23.432 mm).
KEYKODE_MODULE = 0.1905
KEYKODE_MODULES = 123
KEYKODE_TRACK = (0.25, 1.75)
# SMPTE ST 254 frame index: a short bar every 4 perforations, clear of the keys.
FRAME_INDEX_LENGTH = 1.5
FRAME_INDEX_TRACK = (1.1, 1.35)
# The human key's first letter names the maker; the barcode carries the maker's SMPTE number.
_KEYKODE_MAKERS = {"A": 1, "E": 2, "K": 2, "V": 2, "F": 3, "I": 4}

# Code 128 bar and space widths of values 0-105; 106 is the stop.
_CODE128 = (
    "212222 222122 222221 121223 121322 131222 122213 122312 132212 221213 "
    "221312 231212 112232 122132 122231 113222 123122 123221 223211 221132 "
    "221231 213212 223112 312131 311222 321122 321221 312212 322112 322211 "
    "212123 212321 232121 111323 131123 131321 112313 132113 132311 211313 "
    "231113 231311 112133 112331 132131 113123 113321 133121 313121 211331 "
    "231131 213113 213311 213131 311123 311321 331121 312113 312311 332111 "
    "314111 221411 431111 111224 111422 121124 121421 141122 141221 112214 "
    "112412 122114 122411 142112 142211 241211 221114 413111 241112 134111 "
    "111242 121142 121241 114212 124112 124211 411212 421112 421211 212141 "
    "214121 412121 111143 111341 131141 114113 114311 411113 411311 113141 "
    "114131 311141 411131 211412 211214 211232 2331112"
).split()
_START_C, _STOP = 105, 106


def dx_bits(number: int, half: bool, product: tuple[int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(clock, data) tracks of a 31-module DX film edge barcode (ISO 1007, zxing layout)."""
    p1, p2 = product

    def bits(value: int, count: int) -> list[int]:
        return [(value >> shift) & 1 for shift in range(count - 1, -1, -1)]

    payload = [0, *bits(p1, 7), 0, *bits(p2, 4), *bits(number % 64, 6), int(half), 0]
    parity = sum(payload) % 2
    data = (1, 0, 1, 0, 1, *payload, parity, 0, 1, 0, 1)
    clock = (1, 1, 1, 1, 1, *[i % 2 for i in range(23)], 1, 1, 1)
    return clock, data


def code128c_bits(digits: str) -> tuple[int, ...]:
    """Code 128 subset C of an even count of digits, with its mod-103 check symbol."""
    values = [_START_C] + [int(digits[i : i + 2]) for i in range(0, len(digits), 2)]
    check = (values[0] + sum(i * value for i, value in enumerate(values[1:], 1))) % 103
    bits: list[int] = []
    for value in [*values, check, _STOP]:
        for k, width in enumerate(_CODE128[value]):
            bits += [1 - k % 2] * int(width)
    return tuple(bits)


def keykode_bits(code: str, roll: str, foot: int, perforations: int) -> tuple[int, ...]:
    """KEYKODE barcode: maker, product, the key's 6-digit roll prefix and 4-digit foot count, perforation offset."""
    maker = _KEYKODE_MAKERS.get(code[:1], 2)
    # A stand-in product number from the film letter; the real one is in the maker's own table.
    product = (ord(code[1:2] or "A") - ord("A")) % 100
    return code128c_bits(f"{maker:02d}{product:02d}{roll}{foot % 10000:04d}{perforations:02d}")
