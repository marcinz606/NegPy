"""Read a datasheet's characteristic curves from the PDF's vector paths into print_curves/.

Dev tool, not shipped. Kodak draws the curves as paths, so the points are exact; a raster plot
needs WebPlotDigitizer instead. The plot frame comes from the page's drawings (the axis lines
or the frame rectangle), in page points, with the axis values at its edges.

    uv run --with pymupdf python scripts/digitize_datasheet_curves.py PDF PAGE OUT.csv \
        --frame X0 Y0 X1 Y1 --log-e LEFT RIGHT --density BOTTOM TOP [--order grb]

    E-4020 (Ultra Endura):   PAGE 6  --frame 97.8 84.1 283.0 268.0  --log-e -3 0  --density 0 3
    E-7020 (Ektacolor Edge): PAGE 2  --frame 368.4 87.0 555.6 269.3 --log-e -4 1  --density 0 4

--order names the curves from the highest top to the lowest, as the plot labels them.
"""

import argparse

import numpy as np
import pymupdf


def bezier(p0, p1, p2, p3, n=12):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def curve_paths(page, frame):
    """Stroked paths inside the frame as point lists, without the frame's own lines."""
    x0, y0, x1, y1 = frame
    out = []
    for d in page.get_drawings():
        r = d["rect"]
        if (d.get("width") or 0.0) < 0.9 or r.x0 < x0 - 1 or r.x1 > x1 + 1 or r.y0 < y0 - 1 or r.y1 > y1 + 1:
            continue
        pts = []
        for item in d["items"]:
            if item[0] == "l":
                pts += [np.array(item[1]), np.array(item[2])]
            elif item[0] == "c":
                pts += list(bezier(*[np.array(p) for p in item[1:5]]))
        if not pts:
            continue
        pts = np.array(pts)
        span_x, span_y = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        if span_x > 0.05 and not (span_y < 0.05 and span_x > 100):
            out.append(pts)
    return out


def chain(paths):
    """Join the pieces a sheet splits one curve into: each starts where another ends."""
    paths = sorted(paths, key=lambda p: p[0, 0])
    chains = []
    while paths:
        c = paths.pop(0)
        while True:
            nxt = next((i for i, p in enumerate(paths) if np.linalg.norm(p[0] - c[-1]) < 0.3), None)
            if nxt is None:
                break
            c = np.vstack([c, paths.pop(nxt)])
        chains.append(c)
    return [c for c in chains if np.ptp(c[:, 0]) > 40]


def main() -> None:
    ap = argparse.ArgumentParser(description="Digitize a datasheet's characteristic curves from its PDF.")
    ap.add_argument("pdf")
    ap.add_argument("page", type=int)
    ap.add_argument("out")
    ap.add_argument("--frame", type=float, nargs=4, required=True)
    ap.add_argument("--log-e", type=float, nargs=2, required=True)
    ap.add_argument("--density", type=float, nargs=2, required=True)
    ap.add_argument("--order", default="grb")
    args = ap.parse_args()

    x0, y0, x1, y1 = args.frame
    chains = chain(curve_paths(pymupdf.open(args.pdf)[args.page], args.frame))
    if len(chains) != 3:
        raise SystemExit(f"found {len(chains)} curves, not three: check the frame")
    curves = []
    for c in chains:
        x = args.log_e[0] + (c[:, 0] - x0) / (x1 - x0) * (args.log_e[1] - args.log_e[0])
        d = args.density[0] + (y1 - c[:, 1]) / (y1 - y0) * (args.density[1] - args.density[0])
        o = np.argsort(x, kind="stable")
        curves.append((x[o], d[o]))
    curves.sort(key=lambda c: -c[1].max())
    named = dict(zip(args.order, curves))
    xs = np.linspace(max(c[0].min() for c in curves), min(c[0].max() for c in curves), 121)
    table = np.column_stack([xs] + [np.interp(xs, *named[ch]) for ch in "rgb"])
    np.savetxt(args.out, table, delimiter=",", fmt="%.4f", header="log_e,r,g,b", comments="")
    print(f"feet {table[:, 1:].min(0).round(3)}  tops {table[:, 1:].max(0).round(3)}")


if __name__ == "__main__":
    main()
