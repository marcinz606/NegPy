"""Fit NegPy paper-profile constants to digitized datasheet characteristic curves.

Dev tool, not shipped. Turns published D-logH curves into `PaperProfile` constants for
negpy/features/exposure/papers.py.

Workflow
--------
1. Digitize the datasheet's characteristic curves to CSV (two columns: log_e,density).
   Kodak and Ilford PDFs draw them as vector paths, which PyMuPDF's page.get_drawings()
   reads exactly; for a raster plot use WebPlotDigitizer (https://automeris.io/).
2. Fit:
       # B&W: a grade family, one shared paper shape
       uv run --with scipy python scripts/fit_paper_profile.py bw g0.csv g1.csv ... g5.csv
       # RA4: one paper's Status A R, G, B curves
       uv run --with scipy python scripts/fit_paper_profile.py ra4 r.csv g.csv b.csv
3. Copy the printed PaperProfile kwargs into PAPER_PROFILES. Reject the fit if the RMS
   residual exceeds ~0.05 D: the parametric family cannot represent that paper.

The fit renders through the app's own print_curve (default sliders, paper white on, no
black point compensation), so it fits exactly what the engine draws. The slope is not
free: the engine sets k = grade_contrast_scale / (R/100) density per log E, so each curve's
slope comes from its measured ISO R. A free slope trades against the knee sharpness and
converges on soft knees under a steep line, which renders flat at the user's grade.
"""

import argparse
import dataclasses
import sys

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    sys.exit("scipy required: run via `uv run --with scipy python scripts/fit_paper_profile.py ...`")

from negpy.features.exposure import papers
from negpy.features.exposure.logic import default_grade_range, grade_coupled_shape, grade_to_slope, print_curve
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.process.models import ProcessMode

RNG = default_grade_range()
# |profile + Snap + Snap trim| must stay below 1, the kernel's monotonicity bound.
MIDTONE_GAMMA_LIMIT = 0.25


def iso_r(x, d, d_max=None):
    """ISO 6846 range. d_max is the paper's: a soft grade may not reach it on the plot."""
    hi = 0.9 * (d.max() if d_max is None else d_max)
    return 100.0 * (np.interp(hi, d, x) - np.interp(d.min() + 0.04, d, x))


def render(shape, d_min, kind, grade, x, x0, gain=1.0):
    d_max, toe, shoulder, mg, gw = shape
    papers.PAPER_PROFILES["_fit"] = papers.PaperProfile(
        label="fit",
        kind=kind,
        d_min=d_min,
        d_max=d_max,
        toe_sharpness_base=toe,
        shoulder_sharpness_base=shoulder,
        paper_midtone_gamma=mg,
        paper_gamma_width=gw,
    )
    mode = ProcessMode.BW if kind == "bw" else ProcessMode.C41
    exp = dataclasses.replace(ExposureConfig(), paper_profile="_fit", paper_dmin=True, paper_black=True, grade=grade)
    k = grade_to_slope(grade, RNG)
    toe_eff, shoulder_eff = grade_coupled_shape(k, exp.toe, exp.shoulder)
    curve = print_curve(exp, k * gain, 0.0, mode, toe=toe_eff, shoulder=shoulder_eff)
    xn = ((x - x0) / RNG).astype(np.float32)
    return np.asarray(curve(xn.reshape(1, -1, 1).repeat(3, 2)), dtype=np.float64)[0, :, 0]


def fit(kind, curves):
    """One paper shape shared by every curve; d_min is the measured base, x0 is per curve."""
    d_min = float(np.mean([d.min() for _, d in curves]))
    top = max(d.max() for _, d in curves)
    grades = [iso_r(x, d, top) for x, d in curves]
    lo_r, hi_r = EXPOSURE_CONSTANTS["iso_r_min"], EXPOSURE_CONSTANTS["iso_r_max"]
    if any(not lo_r < g < hi_r for g in grades):
        sys.exit(f"ISO R {[round(g) for g in grades]} outside the Grade range {lo_r:.0f}-{hi_r:.0f}: drop those curves")
    x0s = [float(np.interp(0.7, d, x)) for x, d in curves]
    lo = [1.5, 0.5, 0.5, -MIDTONE_GAMMA_LIMIT, 0.2] + [-10.0] * len(curves)
    hi = [3.0, 20.0, 20.0, MIDTONE_GAMMA_LIMIT, 2.0] + [10.0] * len(curves)

    def residuals(p):
        return np.concatenate([render(p[:5], d_min, kind, g, x, p[5 + i]) - d for i, ((x, d), g) in enumerate(zip(curves, grades))])

    # The knee sharpness has local minima; restart from a small grid.
    starts = [[top, a, a, mg, 0.6] + x0s for mg in (-0.15, 0.0, 0.2) for a in (2.0, 6.0, 12.0)]
    best = min((least_squares(residuals, s, bounds=(lo, hi)) for s in starts), key=lambda r: r.cost)
    parts = np.split(best.fun, np.cumsum([len(x) for x, _ in curves])[:-1])
    return best.x[:5], d_min, grades, [float(np.sqrt(np.mean(r**2))) for r in parts]


def channel_gain(shape, d_min, grade, x, d):
    """One channel's slope multiplier (PaperProfile.channel_gamma) under the shared shape."""
    r = least_squares(
        lambda p: render(shape, d_min, "ra4", grade, x, p[1], p[0]) - d,
        [1.0, float(np.interp(0.7, d, x))],
        bounds=([0.5, -10.0], [2.0, 10.0]),
    )
    return float(r.x[0])


def load(path):
    a = np.loadtxt(path, delimiter=",")
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit NegPy paper-profile constants to digitized datasheet curves.")
    ap.add_argument("kind", choices=("bw", "ra4"))
    ap.add_argument("csvs", nargs="+", help="bw: one CSV per grade; ra4: R, G, B")
    args = ap.parse_args()

    curves = [load(p) for p in args.csvs]
    if args.kind == "ra4" and len(curves) != 3:
        sys.exit("ra4 takes exactly three curves: R, G, B")
    xs = np.linspace(max(x.min() for x, _ in curves), min(x.max() for x, _ in curves), 150)
    chans = [np.interp(xs, x, d) for x, d in curves]
    shape, d_min, grades, rms = fit(args.kind, [(xs, np.mean(chans, axis=0))] if args.kind == "ra4" else curves)

    d_max, toe, shoulder, mg, gw = shape
    total = float(np.sqrt(np.mean(np.square(rms))))
    print(f"RMS residual: {total:.4f} D  ({'OK' if total <= 0.05 else 'POOR FIT'})  per curve {[round(r, 3) for r in rms]}")
    print(f"ISO R per curve: {[round(g) for g in grades]}")
    print("PaperProfile kwargs:")
    print(f"    d_max={d_max:.2f},\n    d_min={d_min:.3f},")
    print(f"    toe_sharpness_base={toe:.2f},\n    shoulder_sharpness_base={shoulder:.2f},")
    print(f"    paper_midtone_gamma={mg:.3f},\n    paper_gamma_width={gw:.2f},")
    if args.kind == "ra4":
        g = [channel_gain(shape, d_min, grades[0], xs, d) for d in chans]
        print(f"    channel_gamma=({g[0] / g[1]:.3f}, 1.0, {g[2] / g[1]:.3f}),")
        tint = [float(d.min()) - d_min for d in chans]
        print(f"    base_tint_cmy=({tint[0]:.3f}, {tint[1]:.3f}, {tint[2]:.3f}),")


if __name__ == "__main__":
    main()
