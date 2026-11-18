"""Derive a print stock's per-layer maximum density (PaperProfile.d_max_drop) from its datasheet.

Dev tool, not shipped. A datasheet plots Status A densities, which each read all three dyes.
The kernel's densities are dye amounts on the stock's neutral axis (build_print_luts.py's
gray shaper), so the curves' tops are solved through the stock's spectral dyes for the three
dye amounts, then read back on that axis. d_max_drop is each layer's distance below the
deepest one.

    uv run python scripts/fit_d_max_drop.py [--cie-dir DIR]

Inputs: print_curves/<profile>.csv (log_e,r,g,b: the datasheet's Status A curves, from a
trace or digitize_datasheet_curves.py),
print_dyes/status_a.csv (the Status A responsivities) and the stock's spectral dye CSV.
A stock needs the curves and the dyes of the same datasheet family.
"""

import argparse
import pathlib

import build_print_luts as B
import numpy as np

CURVES = pathlib.Path(__file__).with_name("print_curves")
# Profile key -> its spectral dye CSV in print_dyes/.
STOCKS = {
    "kodak_portra_endura": "kodak_ra4_e4021.csv",
    "kodak_supra_endura": "kodak_ra4_e4021.csv",
    "kodak_endura": "kodak_ra4_e4070.csv",
    "kodak_ultra_endura": "kodak_ra4_e4070.csv",
    "kodak_ektacolor_edge": "kodak_ra4_e4070.csv",
}


def status_a_responsivity():
    a = np.genfromtxt(B.DATA / "status_a.csv", delimiter=",", names=True)
    return np.stack([np.interp(B.WL, a["wavelength_nm"], a[c]) for c in "rgb"], 1)


def layer_tops(profile, dye_csv, cie_dir, resp):
    """The three layers' maximum density on the stock's neutral axis, in kernel units."""
    xyz, w = B.spectral_xyz(dye_csv, B.viewing_light(cie_dir, "D50"), B.cie_table(cie_dir, "cmf", 3))
    shaper = B.gray_shaper(xyz, w, 3.0)
    a = np.genfromtxt(B.DATA / dye_csv, delimiter=",", names=True)
    dyes = np.stack([np.interp(B.WL, a["wavelength_nm"], a[k]) for k in ("cyan", "magenta", "yellow")], 1)

    def status_a(amounts):
        return -np.log10((10.0 ** -(dyes @ amounts)) @ resp / resp.sum(0))

    curves = np.genfromtxt(CURVES / f"{profile}.csv", delimiter=",", names=True)
    d = np.stack([curves[c] for c in "rgb"], 1)
    target = d[-1] - d.min(0)
    amounts = target.copy()
    for _ in range(60):
        f0 = status_a(amounts) - target
        jac = np.stack([(status_a(amounts + h) - target - f0) / 1e-5 for h in np.eye(3) * 1e-5], 1)
        amounts = np.clip(amounts - np.linalg.solve(jac, f0), 0.0, 30.0)
    e = np.linspace(0.0, 3.0, 601)
    return np.interp(amounts / w, shaper(e), e)


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive PaperProfile.d_max_drop from datasheet Status A curves.")
    ap.add_argument("--cie-dir", type=pathlib.Path, default=pathlib.Path.home() / ".cache" / "negpy" / "cie")
    args = ap.parse_args()
    resp = status_a_responsivity()
    for profile, dye_csv in STOCKS.items():
        tops = layer_tops(profile, dye_csv, args.cie_dir, resp)
        drop = tops.max() - tops
        print(f"{profile}: tops {tops.round(2)}  d_max_drop=({drop[0]:.2f}, {drop[1]:.2f}, {drop[2]:.2f})")


if __name__ == "__main__":
    main()
