"""Render a film stock's edge print as NegPy draws it, and list every item in roll millimeters.

    uv run python .claude/skills/edge-print-from-photo/render_edge.py film-hp5-400 --out /tmp/hp5.png
    uv run python .claude/skills/edge-print-from-photo/render_edge.py film-hp5-400 --edge '{"text": "HP5 PLUS"}'

`--edge` replaces the stock's `edge` object for this render only, to try values before editing the catalog.
Frame windows print mid-gray so the strip reads like the photo; the item list is what to compare measurements to.
"""

import argparse
import json
from dataclasses import replace

import numpy as np
from PIL import Image

from negpy.features.metadata.gear_models import FilmColorType, FilmEdge, FilmStock, GearLibrary
from negpy.services.assets.gear import GearProfiles
from negpy.services.export.contact_sheet import palette_for
from negpy.services.export.contact_sheet_roll import SheetLook
from negpy.services.film_edge.edge import EdgeCode, EdgeText, edge_items, edge_style
from negpy.services.film_edge.geometry import FRAME_SIZES_120, SheetFormat, film_geometry
from negpy.services.film_edge.layouts import LAYOUTS
from negpy.services.film_edge.paint import paint_strip

PALETTES = {FilmColorType.COLOR_NEGATIVE: "color", FilmColorType.COLOR_SLIDE: "slide", FilmColorType.BW_SLIDE: "slide"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stock_id")
    parser.add_argument("--edge", help="JSON edge object that replaces the stock's own")
    parser.add_argument("--format", default="35mm", choices=[f.value for f in SheetFormat])
    parser.add_argument("--frame-size", default="6×6", choices=list(FRAME_SIZES_120))
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--px-per-mm", type=float, default=10.0)
    parser.add_argument("--out", default="edge_print.png")
    args = parser.parse_args()

    stocks = GearProfiles._read_bundled_list("film_stocks.json", FilmStock)
    library = GearLibrary(film_stocks=stocks)
    stock = library.get_film_stock(args.stock_id)
    if stock is None:
        raise SystemExit(f"no bundled stock {args.stock_id!r}")
    if args.edge:
        stock = replace(stock, edge=FilmEdge.from_dict(json.loads(args.edge)))
    fmt = SheetFormat(args.format)
    palette = PALETTES.get(stock.color_type, "bw")
    style = edge_style(stock, library)
    style = replace(style, dx=fmt != SheetFormat.MEDIUM and palette in LAYOUTS[style.layout].dx)
    geometry = film_geometry(fmt, args.frame_size)

    s = args.px_per_mm
    length = args.frames * geometry.pitch
    pal = palette_for(SheetLook(palette, 0, style))
    canvas = np.full((int(round(geometry.width * s)), int(round(length * s)), 3), pal.no_film, np.uint8)
    paint_strip(canvas, 0.0, 0.0, geometry, style, args.frames, 0.0, length, s, pal)
    top = geometry.image_top * s
    for i in range(args.frames):
        left = (geometry.frame_center(i) - geometry.frame_along / 2) * s
        canvas[int(top) : int(top + geometry.frame_across * s), int(left) : int(left + geometry.frame_along * s)] = 96
    Image.fromarray(canvas).save(args.out)

    print(f"{stock.id}: {style}")
    print(f"strip {length:.1f} mm, frame pitch {geometry.pitch} mm, frame 1 centered at {geometry.frame_center(0)} mm -> {args.out}")
    for item in sorted(edge_items(geometry, style, args.frames, 0.0, length), key=lambda i: i.x):
        if isinstance(item, EdgeText):
            print(f"  {item.band:6} x={item.x:7.2f} cap={item.cap:.2f} {item.align:6} {item.text!r} {item.lead}{item.trail}{item.under}")
        elif isinstance(item, EdgeCode):
            print(f"  code   x={item.x:7.2f} length={item.length:.2f} tracks={[span for span, _bits in item.tracks]}")


if __name__ == "__main__":
    main()
