"""Fit NegPy paper-profile constants to digitized datasheet characteristic curves.

Dev tool, not shipped. Turns published D-logH curves (Ilford MG, Kodak Endura,
Fuji Crystal Archive, ...) into `PaperProfile` constants for
negpy/features/exposure/papers.py, replacing hand-tuned values with a fit.

Workflow
--------
1. Open the paper datasheet PDF, screenshot the characteristic curve.
2. Digitize it with WebPlotDigitizer (https://automeris.io/) — axes:
   x = relative log exposure, y = density. Export CSV (two columns: log_e,density).
3. Fit one curve:
       uv run --with scipy python scripts/fit_paper_profile.py mgrc_grade2.csv
   or a VC grade family (shared paper shape, per-curve slope/pivot):
       uv run --with scipy python scripts/fit_paper_profile.py --family g00.csv g0.csv ... g5.csv
4. Copy the printed PaperProfile kwargs into PAPER_PROFILES. Reject the fit if
   RMS residual exceeds ~0.05 D (the parametric family can't represent that
   paper; see docs/PIPELINE.md before reaching for a LUT).

Model: the engine's print-curve stage-2 ("paper") family — straight line
slope·(x−pivot), tanh midtone gamma, double softplus toe/shoulder bounds —
identical math to CharacteristicCurve/logic._apply_print_curve_kernel with the
shape constants freed:  k, x0 per curve (nuisance: datasheet x-axis is
arbitrary-origin logE);  d_min, d_max, a_hl, a_sh, midtone_gamma, gamma_width
per paper.
"""

import argparse
import sys

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    sys.exit("scipy required: run via `uv run --with scipy python scripts/fit_paper_profile.py ...`")

# Shape parameter vector per paper: (d_min, d_max, a_hl, a_sh, midtone_gamma, gamma_width)
_SHAPE_GUESS = np.array([0.06, 2.3, 3.0, 4.0, 0.15, 0.6])
_SHAPE_LO = np.array([0.0, 1.5, 0.5, 0.5, 0.0, 0.2])
_SHAPE_HI = np.array([0.3, 3.0, 20.0, 20.0, 0.8, 2.0])
# Per-curve nuisance: (k, x0)
_CURVE_GUESS = np.array([5.0, 0.5])
_CURVE_LO = np.array([0.5, -5.0])
_CURVE_HI = np.array([30.0, 5.0])


_ANCHOR_TARGET_DENSITY = 0.74  # EXPOSURE_CONSTANTS["anchor_target_density"]


def _inv_softplus(y):
    # inverse of log(1+e^x), stable for large y
    return y + np.log(-np.expm1(-y))


def _v_star(d_min, d_max, a_hl, a_sh):
    """Mirror of logic._reference_linear_value with the shape params freed."""
    t = _ANCHOR_TARGET_DENSITY
    v1 = d_max - _inv_softplus(a_sh * (d_max - t)) / a_sh
    return d_min + _inv_softplus(a_hl * (v1 - d_min)) / a_hl


def paper_density(x, k, x0, d_min, d_max, a_hl, a_sh, midtone_gamma, gamma_width):
    """CharacteristicCurve with toe/shoulder sliders at 0 and free shape params."""
    v = k * (x - x0)
    v = v + midtone_gamma * gamma_width * np.tanh((v - _v_star(d_min, d_max, a_hl, a_sh)) / gamma_width)
    v1 = d_min + np.logaddexp(0.0, a_hl * (v - d_min)) / a_hl
    return d_max - np.logaddexp(0.0, a_sh * (d_max - v1)) / a_sh


def _residuals(theta, curves):
    d_min, d_max, a_hl, a_sh, mg, gw = theta[:6]
    res = []
    for i, (x, d) in enumerate(curves):
        k, x0 = theta[6 + 2 * i : 8 + 2 * i]
        res.append(paper_density(x, k, x0, d_min, d_max, a_hl, a_sh, mg, gw) - d)
    return np.concatenate(res)


def fit(curves):
    theta0 = np.concatenate([_SHAPE_GUESS] + [_CURVE_GUESS] * len(curves))
    lo = np.concatenate([_SHAPE_LO] + [_CURVE_LO] * len(curves))
    hi = np.concatenate([_SHAPE_HI] + [_CURVE_HI] * len(curves))
    return least_squares(_residuals, theta0, bounds=(lo, hi), args=(curves,))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csvs", nargs="+", help="CSV file(s): two columns, log_e,density")
    ap.add_argument("--family", action="store_true", help="fit all CSVs with one shared paper shape (VC grade family)")
    args = ap.parse_args()

    datasets = []
    for path in args.csvs:
        arr = np.loadtxt(path, delimiter=",")
        datasets.append((path, arr[:, 0].astype(float), arr[:, 1].astype(float)))

    groups = [datasets] if args.family else [[d] for d in datasets]
    for group in groups:
        curves = [(x, d) for _, x, d in group]
        r = fit(curves)
        d_min, d_max, a_hl, a_sh, mg, gw = r.x[:6]
        rms = float(np.sqrt(np.mean(r.fun**2)))
        names = ", ".join(name for name, _, _ in group)
        print(f"\n== {names}")
        print(f"RMS residual: {rms:.4f} D  ({'OK' if rms <= 0.05 else 'POOR FIT — parametric family may not represent this paper'})")
        print("PaperProfile kwargs:")
        print(f"    d_min={d_min:.3f}, d_max={d_max:.3f},")
        print(f"    shoulder_sharpness_base={a_hl:.2f}, toe_sharpness_base={a_sh:.2f},")
        print(f"    paper_midtone_gamma={mg:.3f}, paper_gamma_width={gw:.3f},")
        for i, (name, x, d) in enumerate(group):
            k, x0 = r.x[6 + 2 * i : 8 + 2 * i]
            print(f"    # {name}: slope k={k:.2f} pivot={x0:.3f} (nuisance; grade/pivot come from the engine)")
        if args.family and len(group) > 1:
            print("    # Grade-family trend: regress the per-curve k values against any per-grade")
            print("    # shape refit to calibrate grade_coupled_shape (toe/shoulder/midtone-gamma vs slope).")


if __name__ == "__main__":
    main()
