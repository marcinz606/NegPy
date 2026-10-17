"""Estimate crosstalk readings for Kodak films whose datasheets plot only a midscale neutral.

Dev tool, not shipped. A consumer datasheet draws a midscale-neutral and a minimum-density curve,
not separate dyes. This fits their difference with Kodak's Vision3 dye curves, shifting each dye in
wavelength, and samples the fitted dyes at 650/550/450 nm: the `readings` that
.claude/skills/crosstalk-from-datasheet/derive_matrix.py turns into a gallery matrix. The Vision3
curves are D-min subtracted, so they carry Vision3's masking couplers; the estimate assumes the
film's masking matches.

Data (scripts/crosstalk_dyes/):
    kodak_vision3_dyes.csv  mean of the H-1-5203/5207/5213/5219 dye curves, spektrafilm's traces
                            (CC BY-SA 4.0, see NOTICE.md)
    <film>_neutral.csv      midscale neutral and minimum density, our read of the sheet's vector
                            curves: E-4046 Ektar 100, E-7022 Gold 200, E-7023 Ultramax 400,
                            E-4051/E-4050/E-4040 Portra 160/400/800, the Verita 200D brochure.
                            Every file here is fitted.

Usage:
    uv run python scripts/fit_crosstalk_estimates.py
"""

import json
import pathlib

import numpy as np

DATA = pathlib.Path(__file__).with_name("crosstalk_dyes")
BANDS = (650, 550, 450)
WL = np.arange(400, 701, 1.0)
FIT = (WL >= 420) & (WL <= 680)


def levenberg_marquardt(residual, x, iters=200):
    damping = 1e-3
    for _ in range(iters):
        r0 = residual(x)
        jac = np.stack([(residual(x + h) - r0) / 1e-6 for h in np.eye(len(x)) * 1e-6], 1)
        a, g = jac.T @ jac, jac.T @ r0
        while damping < 1e9:
            step = np.linalg.solve(a + damping * np.diag(np.diag(a) + 1e-12), -g)
            if (residual(x + step) ** 2).sum() < (r0**2).sum():
                x, damping = x + step, max(damping / 3, 1e-8)
                break
            damping *= 4
        else:
            break
        if np.abs(step).max() < 1e-9:
            break
    return x


def table(name):
    a = np.genfromtxt(DATA / f"{name}.csv", delimiter=",", names=True)
    return {k: np.interp(WL, a["wavelength_nm"], a[k]) for k in (a.dtype.names or ())[1:]}


def shifted(dyes, shifts):
    return np.stack([np.interp(WL - s, WL, d) for d, s in zip(dyes, shifts)], 1)


def main():
    t = table("kodak_vision3_dyes")
    dyes = (t["cyan"], t["magenta"], t["yellow"])
    for path in sorted(DATA.glob("*_neutral.csv")):
        film = path.stem.removesuffix("_neutral")
        n = table(path.stem)
        target = (n["midscale"] - n["minimum"])[FIT]
        p = levenberg_marquardt(lambda q: shifted(dyes, q[3:])[FIT] @ q[:3] - target, np.array([0.7, 0.7, 0.7, 0.0, 0.0, 0.0]))
        fitted = shifted(dyes, p[3:])
        rms = float(np.sqrt(np.mean((fitted[FIT] @ p[:3] - target) ** 2)))
        readings = [[round(float(fitted[b - 400, j]), 4) for j in range(3)] for b in BANDS]
        print(json.dumps({"name": film, "readings": readings}))
        print(f"    fit RMS {rms:.4f} D, dye shifts C/M/Y {np.round(p[3:], 1).tolist()} nm")


if __name__ == "__main__":
    main()
