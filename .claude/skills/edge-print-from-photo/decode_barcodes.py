"""Decode DX and KEYKODE (Code 128) barcodes in a photo, scan or NegPy render of a film edge.

    uv run --no-project --with zxing-cpp --with pillow python -I .claude/skills/edge-print-from-photo/decode_barcodes.py IMAGE

zxing-cpp is not a NegPy dependency, hence --no-project. Each barcode is tried as is (a negative on a light table:
dark bars) and inverted (a NegPy render: light bars), at 1x to 4x, because phone photos often decode only enlarged.
"""

import sys

import zxingcpp
from PIL import Image, ImageOps

FORMATS = (zxingcpp.BarcodeFormat.DXFilmEdge, zxingcpp.BarcodeFormat.Code128)


def main() -> None:
    image = Image.open(sys.argv[1]).convert("L")
    found: dict[tuple[str, str], str] = {}
    for scale in (1, 2, 3, 4):
        big = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS) if scale > 1 else image
        for name, variant in (("as is", big), ("inverted", ImageOps.invert(big))):
            for result in zxingcpp.read_barcodes(variant, formats=FORMATS, try_rotate=True):
                found.setdefault((str(result.format), result.text), f"{name} at {scale}x")
    if not found:
        raise SystemExit("no barcode decoded; crop to the rebate and try a sharper image")
    for (fmt, text), how in sorted(found.items()):
        print(f"{fmt:24} {text:20} ({how})")


if __name__ == "__main__":
    main()
