"""Bake NegPy's print-stock LUTs into negpy/features/exposure/print_luts/<key>.npz.

Dev tool, not shipped. A LUT maps the three dye densities above base (C, M, Y = the print kernel's
R, G, B density) to the print's color in linear Adobe RGB, media-relative. Every LUT keeps the curve's
luminance on its gray axis (Y of LUT(e, e, e) = 10^-e), so the tone tools that model the print as
achromatic 10^-D stay true, and grays drift only in chroma and hue.

Sources:
    spectral   a datasheet's spectral dye density curves (scripts/print_dyes/, 5 nm) as END
               amounts, T(λ) = 10^-(Σ amount·dye(λ)), CIE 1931 2° under D50 (paper) or the Kinoton
               75P xenon projector (print film), Bradford to D65. The END weights make density 1.0
               a visual neutral; one shaper on all three dyes holds the gray axis luminance.
    icc look   an RGB look profile applied to Digital's print as NegPy applies an Input ICC, baked
               so that the kernel with its default black point compensation reproduces it.
    frontier   an RGB printer profile of a minilab that exposes each layer with its own laser, such as
               Fujifilm's Frontier-PD profiles. Each device channel drives one dye, so the profile's
               gray axis (the device RGB that prints a* = b* = 0 at each visual density) gives three
               1D curves; the LUT reads A2B1 through them. The paper white and black of the profile
               give the stock's d_min, base tint and d_max.

The CIE tables and the xenon spectrum are downloaded once into --cie-dir.

Usage:
    uv run python scripts/build_print_luts.py [--key KEY ...] [--icc PATH --as KEY] [--against KEY] [--profiles DIR]
        [--cie-dir DIR] [--check-only]
"""

import argparse
import pathlib
import re
import struct
import urllib.request

import imagecodecs
import numpy as np

from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.print_lut import LUT_DIR, save_print_lut, tetra_lookup
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE, ColorSpaceRegistry
from negpy.infrastructure.display.icc_profile import extract_primaries_matrix, is_matrix_trc_profile
from negpy.kernel.image.logic import apply_primaries_transform

REPO = pathlib.Path(__file__).resolve().parents[1]
N = 33
GAMMA = 563 / 256

DATA = pathlib.Path(__file__).with_name("print_dyes")
CIE_URL = "https://files.cie.co.at/Publications-datasets/"
CIE_FILES = {"cmf": "CIE_xyz_1931_2deg.csv", "D50": "CIE_std_illum_D50.csv", "D65": "CIE_std_illum_D65.csv"}
COLOUR_LIGHTS = "https://raw.githubusercontent.com/colour-science/colour/v0.4.6/colour/colorimetry/datasets/light_sources/sds.py"
WL = np.arange(380, 781, 5.0)

# LUT key -> (dye csv, viewing light, source). fuji_af3 is the cross-check for the Frontier DPII LUT.
SPECTRAL = {
    "kodak_e4070": ("kodak_ra4_e4070.csv", "D50", "Kodak E-4070 spectral dye density, D50"),
    "kodak_e4021": ("kodak_ra4_e4021.csv", "D50", "Kodak E-4021 spectral dye density (spektrafilm trace), D50"),
    "kodak_2383": ("kodak_2383_h1_2383.csv", "K75P", "Kodak H-1-2383 spectral dye density, Kinoton 75P xenon"),
    "kodak_2393": ("kodak_2393_h1_2393.csv", "K75P", "Kodak H-1-2393t spectral dye density (spektrafilm trace), Kinoton 75P xenon"),
    "fuji_af3": ("fujicolor_ca_af3_198e.csv", "D50", "Fujifilm AF3-198E spectral dye density, D50"),
}
# LUT key -> an ICC look profile in the repo (redistributable).
LOOKS = {"pakon": "tests/fixtures/RGBScan.icc"}
# LUT key -> a minilab printer profile, read from --profiles (not in the repo).
FRONTIER = {
    "fuji_ca_dpii": "Fuji_Frontier-PD_CA-DPII_V3.icm",
    "fuji_ca_hd": "Fuji_Frontier-PD_CA-HD_v3a.icm",
    "fuji_ca_pearl": "Fuji_Frontier-PD_CA-Pearl_V3.icm",
}
# A Frontier stock's d_max adds the paper base to the profile's black, so its LUT reaches past device black.
FRONTIER_HEADROOM = 0.3
SHIPPED = ("kodak_e4070", "kodak_e4021", "kodak_2383", "kodak_2393", *LOOKS, *FRONTIER)

ADOBE_XY = ((0.64, 0.33), (0.21, 0.71), (0.15, 0.06))
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def xy_to_xyz(x, y):
    return np.array([x / y, 1.0, (1 - x - y) / y])


W65 = xy_to_xyz(0.3127, 0.3290)
P = np.stack([xy_to_xyz(*p) for p in ADOBE_XY], 1)
ADOBE_TO_XYZ = P * np.linalg.solve(P, W65)


def cie_table(cie_dir, name, cols):
    path = cie_dir / CIE_FILES[name]
    if not path.exists():
        cie_dir.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(CIE_URL + CIE_FILES[name], headers={"User-Agent": "Mozilla/5.0"})
        path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    a = np.genfromtxt(path, delimiter=",")
    return np.stack([np.interp(WL, a[:, 0], a[:, k]) for k in range(1, cols + 1)], 1).squeeze()


def viewing_light(cie_dir, name):
    if name != "K75P":
        return cie_table(cie_dir, name, 1)
    path = cie_dir / "kinoton_75p.csv"
    if not path.exists():
        req = urllib.request.Request(COLOUR_LIGHTS, headers={"User-Agent": "Mozilla/5.0"})
        src = urllib.request.urlopen(req, timeout=60).read().decode()
        block = src[src.index('"Kinoton 75P": {') :]
        rows = re.findall(r"(\d+):\s*([0-9.eE+-]+)", block[: block.index("}")])
        cie_dir.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{w},{v}\n" for w, v in rows))
    a = np.genfromtxt(path, delimiter=",")
    return np.interp(WL, a[:, 0], a[:, 1])


def lab(xyz):
    r = xyz / W65
    f = np.where(r > (6 / 29) ** 3, np.cbrt(r), r / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def de2000(l1, l2):
    """CIEDE2000 (CIE 142-2001), kL = kC = kH = 1."""
    L1, a1, b1 = np.moveaxis(l1, -1, 0)
    L2, a2, b2 = np.moveaxis(l2, -1, 0)
    cb = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(cb**7 / (cb**7 + 25**7)))
    a1, a2 = (1 + g) * a1, (1 + g) * a2
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    h1, h2 = np.degrees(np.arctan2(b1, a1)) % 360, np.degrees(np.arctan2(b2, a2)) % 360
    dh = h2 - h1
    dh = np.where(c1 * c2 == 0, 0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    hs = h1 + h2
    hbar = np.where(c1 * c2 == 0, hs, np.where(np.abs(h1 - h2) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    lbar, cbar = (L1 + L2) / 2, (c1 + c2) / 2
    t = (
        1
        - 0.17 * np.cos(np.radians(hbar - 30))
        + 0.24 * np.cos(np.radians(2 * hbar))
        + 0.32 * np.cos(np.radians(3 * hbar + 6))
        - 0.20 * np.cos(np.radians(4 * hbar - 63))
    )
    sl = 1 + 0.015 * (lbar - 50) ** 2 / np.sqrt(20 + (lbar - 50) ** 2)
    sc, sh = 1 + 0.045 * cbar, 1 + 0.015 * cbar * t
    rt = -2 * np.sqrt(cbar**7 / (cbar**7 + 25**7)) * np.sin(np.radians(60 * np.exp(-(((hbar - 275) / 25) ** 2))))
    dl, dc, dhh = L2 - L1, c2 - c1, 2 * np.sqrt(c1 * c2) * np.sin(np.radians(dh / 2))
    return np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dhh / sh) ** 2 + rt * (dc / sc) * (dhh / sh))


def spectral_xyz(csv, light, cmf):
    """Dye amounts -> media-relative XYZ (D65-adapted) of the datasheet dyes, and the END weights that
    make amount 1.0 of each a visual neutral of density 1.0."""
    a = np.genfromtxt(DATA / csv, delimiter=",", names=True)
    dyes = np.stack([np.interp(WL, a["wavelength_nm"], a[k]) for k in ("cyan", "magenta", "yellow")], 1)
    norm = (light * cmf[:, 1]).sum()
    white = light @ cmf / norm
    adapt = np.linalg.inv(BRADFORD) @ np.diag((BRADFORD @ W65) / (BRADFORD @ (white / white[1]))) @ BRADFORD

    def xyz(amounts):
        t = 10.0 ** (-(amounts @ dyes.T))
        return ((t * light) @ cmf / norm / white[1]) @ adapt.T

    w, target = np.ones(3), lab(W65 * 0.1)[0]
    for _ in range(50):
        f0 = lab(xyz(w)) - [target, 0, 0]
        jac = np.stack([(lab(xyz(w + h)) - [target, 0, 0] - f0) / 1e-6 for h in np.eye(3) * 1e-6], 1)
        w = w - np.linalg.solve(jac, f0)
    return xyz, w


def adobe(xyz):
    return xyz @ np.linalg.inv(ADOBE_TO_XYZ).T


def to_lab(rgb):
    return lab(rgb @ ADOBE_TO_XYZ.T)


def nodes(e_max, n=N):
    axes = [np.linspace(0.0, m, n) for m in e_max]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1)


def gray_shaper(xyz, w, top):
    """f with Y(xyz(w·f(g))) = 10^-g on 0..top, by bisection per level; applied per dye."""
    g = np.linspace(0.0, top, 513)
    lo, hi = np.zeros_like(g), np.full_like(g, 20.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        dark = xyz(mid[:, None] * w)[:, 1] < 10.0**-g
        hi, lo = np.where(dark, mid, hi), np.where(dark, lo, mid)
    return lambda e: np.interp(e, g, (lo + hi) / 2)


def spectral_lut(csv, light_name, cie_dir, e_top=3.0):
    """(table, e_max, source, model): model maps (M, 3) dye densities to unclipped linear Adobe RGB."""
    xyz, w = spectral_xyz(csv, viewing_light(cie_dir, light_name), cie_table(cie_dir, "cmf", 3))
    f = gray_shaper(xyz, w, e_top)

    def model(e):
        return adobe(xyz(f(e) * w))

    e_max = np.full(3, e_top)
    source = next(v[2] for v in SPECTRAL.values() if v[0] == csv)
    return np.clip(model(nodes(e_max)), 0, 1).astype(np.float32), e_max.astype(np.float32), source, model


def header(icc):
    """Device class, color space, connection space, creator, description, and whether it carries a targ table."""
    cls, space, pcs, creator = (icc[a : a + 4].decode("latin-1").strip("\x00 ") for a in (12, 16, 20, 80))
    desc, targ = "", False
    for i in range(struct.unpack(">I", icc[128:132])[0]):
        sig, off, size = struct.unpack(">4sII", icc[132 + 12 * i : 144 + 12 * i])
        targ |= sig == b"targ"
        if sig != b"desc":
            continue
        if icc[off : off + 4] == b"mluc":  # v4: UTF-16BE records; take the first
            length, start = struct.unpack(">II", icc[off + 20 : off + 28])
            desc = icc[off + start : off + start + length].decode("utf-16-be", "replace")
        else:
            desc = icc[off + 12 : off + size].split(b"\x00")[0].decode("latin-1", "replace")
    return cls, space, pcs, creator, desc.strip(), targ


def profile_on_adobe(rgb, icc):
    """An RGB profile applied to linear Adobe RGB as NegPy applies an Input ICC: a matrix/TRC profile by
    its primaries in linear light, any other through lcms at 16 bit, relative colorimetric with BPC."""
    enc = np.clip(rgb, 0, 1) ** (1 / GAMMA)
    src_to_xyz = extract_primaries_matrix(icc) if is_matrix_trc_profile(icc) else None
    if src_to_xyz is not None:
        out = apply_primaries_transform(enc.reshape(-1, 1, 3).astype(np.float32), src_to_xyz)
        return np.asarray(out, dtype=np.float64).reshape(rgb.shape) ** GAMMA
    adobe_icc = pathlib.Path(ColorSpaceRegistry.get_icc_path(WORKING_COLOR_SPACE)).read_bytes()
    u16 = np.round(enc * 65535).astype(np.uint16).reshape(-1, 1, 3)
    out = imagecodecs.cms_transform(u16, icc, adobe_icc, colorspace="RGB", outcolorspace="RGB", intent=1, flags=0x2000)
    return (out.reshape(rgb.shape) / 65535.0) ** GAMMA


def icc_look_lut(icc, source):
    """The look on Digital's print (Paper White off): with Y_b = Digital's black, the kernel's luminance BPC
    (T − Y_b) / (1 − Y_b) of LUT(e) gives exactly the profile applied to Digital's per-channel BPC print."""
    d_max = float(EXPOSURE_CONSTANTS["d_max"])
    y_b = 10.0**-d_max

    def model(e):
        digital = np.clip((10.0**-e - y_b) / (1 - y_b), 0, 1)
        return profile_on_adobe(digital, icc) * (1 - y_b) + y_b

    e_max = np.full(3, d_max)
    return np.clip(model(nodes(e_max)), 0, 1).astype(np.float32), e_max.astype(np.float32), source, model


D50 = xy_to_xyz(0.3457, 0.3585)
D50_TO_D65 = np.linalg.inv(BRADFORD) @ np.diag((BRADFORD @ W65) / (BRADFORD @ D50)) @ BRADFORD


def lab_to_xyz(L, white):
    fy = (L[..., 0] + 16) / 116
    f = np.stack([fy + L[..., 1] / 500, fy, fy - L[..., 2] / 200], -1)
    return np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * white


def printer_lab(icc, rgb):
    """A2B1 (relative colorimetric, no BPC): device RGB in 0..1 -> Lab relative to the paper white, D50."""
    u16 = np.round(np.clip(rgb, 0, 1) * 65535).astype(np.uint16).reshape(-1, 1, 3)
    out = imagecodecs.cms_transform(
        u16, icc, imagecodecs.cms_profile("lab4"), colorspace="RGB", outcolorspace="Lab", outdtype="f4", intent=1
    )
    return np.asarray(out, np.float64).reshape(rgb.shape)


def tag_xyz(icc, name):
    for i in range(struct.unpack(">I", icc[128:132])[0]):
        sig, off, _ = struct.unpack(">4sII", icc[132 + 12 * i : 144 + 12 * i])
        if sig == name.encode():
            return np.array(struct.unpack(">3i", icc[off + 8 : off + 20])) / 65536.0
    raise SystemExit(f"the profile has no {name} tag")


def frontier_lut(icc, source, levels=257):
    """(table, e_max, source, model, tone) from a laser minilab printer profile."""
    white = printer_lab(icc, np.ones(3))
    if abs(white[0] - 100) > 0.5:
        raise SystemExit(f"device white reads L* {white[0]:.1f}, not the paper white")
    # Densities are relative to the device white, which a profile may place a little under L* 100.
    y_white = lab_to_xyz(white, D50)[1]
    d_black = -np.log10(lab_to_xyz(printer_lab(icc, np.zeros(3)), D50)[1] / y_white)
    d = np.linspace(0.0, d_black, levels)
    target = np.stack([lab(W65 * y_white * 10.0 ** -d[:, None])[:, 0], np.zeros_like(d), np.zeros_like(d)], -1)
    ramp = np.linspace(0, 1, 1025)
    ramp_l = printer_lab(icc, ramp[:, None].repeat(3, 1))[:, 0]
    rgb = np.interp(target[:, 0], ramp_l, ramp)[:, None].repeat(3, 1)
    for _ in range(30):
        f0 = printer_lab(icc, rgb) - target
        jac = np.stack([(printer_lab(icc, rgb + h) - target - f0) / 1e-3 for h in np.eye(3) * 1e-3], -1)
        rgb = np.clip(rgb - np.linalg.solve(jac + np.eye(3) * 1e-9, f0[..., None])[..., 0], 0, 1)
    miss = np.linalg.norm(printer_lab(icc, rgb) - target, axis=-1)
    top = int(np.nonzero(miss < 0.5)[0].max())
    d, rgb = d[: top + 1], rgb[: top + 1]
    for k in range(3):
        rgb[:, k] = np.minimum.accumulate(rgb[:, k])
    # Past the darkest neutral the gray axis runs straight to device black, at the curve's luminance.
    d_ext = np.linspace(d[-1], d_black, 33)[1:]
    l_ext = lab(W65 * y_white * 10.0 ** -d_ext[:, None])[:, 0]
    lo, hi = np.zeros_like(d_ext), np.ones_like(d_ext)
    for _ in range(40):
        t = (lo + hi) / 2
        dark = printer_lab(icc, t[:, None] * rgb[-1])[:, 0] < l_ext
        lo, hi = np.where(dark, t, lo), np.where(dark, hi, t)
    d, rgb = np.concatenate([d, d_ext]), np.vstack([rgb, ((lo + hi) / 2)[:, None] * rgb[-1]])

    def model(e):
        device = np.stack([np.interp(e[..., k], d, rgb[:, k]) for k in range(3)], -1)
        # Denser than device black (Paper White off reads e up to d_max): luminance only, toward 0.
        beyond = np.maximum(e - d_black, 0.0).mean(-1, keepdims=True)
        return adobe(lab_to_xyz(printer_lab(icc, device), D50) @ D50_TO_D65.T) / y_white * 10.0**-beyond

    wt, bk = tag_xyz(icc, "wtpt"), tag_xyz(icc, "bkpt")
    base = -np.log10(adobe(wt @ D50_TO_D65.T))
    tone = {
        "d_min": float(base.mean()),
        "base_tint_cmy": tuple(float(v) for v in base - base.mean()),
        "black_above_base": float(-np.log10(bk[1] / wt[1])),
    }
    e_max = np.full(3, d_black + FRONTIER_HEADROOM)
    return np.clip(model(nodes(e_max)), 0, 1).astype(np.float32), e_max.astype(np.float32), source, model, tone


def cross_check(name, model_a, model_b, top):
    e = np.random.default_rng(1).uniform(0, top, (4000, 3))
    d = de2000(to_lab(np.clip(model_a(e), 0, 1)), to_lab(np.clip(model_b(e), 0, 1)))
    near = np.ptp(e, axis=1) < 0.3
    print(f"   against {name}: dE00 mean {d.mean():.2f} / p95 {np.percentile(d, 95):.2f}; near-neutral mean {d[near].mean():.2f}")


def report(key, table, e_max, model):
    rng = np.random.default_rng(0)
    e = rng.uniform(0, 1, (2000, 3)) * e_max
    looked = np.array([tetra_lookup(table, e_max, *p) for p in e])
    d = de2000(to_lab(np.clip(looked, 0, 1)), to_lab(np.clip(model(e), 0, 1)))
    print(f"{key}: interpolation dE00 mean {d.mean():.3f} / p95 {np.percentile(d, 95):.3f}")
    g = np.linspace(0, float(e_max.min()), 9)[1:]
    gray = np.array([tetra_lookup(table, e_max, v, v, v) for v in g])
    L = to_lab(gray)
    y = gray @ ADOBE_TO_XYZ[1]
    print("   gray axis  D " + " ".join(f"{v:5.2f}" for v in g))
    print("            C* " + " ".join(f"{v:5.2f}" for v in np.hypot(L[:, 1], L[:, 2])))
    print("             h " + " ".join(f"{v:5.0f}" for v in np.degrees(np.arctan2(L[:, 2], L[:, 1])) % 360))
    print(f"   gray luminance error, max |log10 Y + D|: {np.abs(np.log10(np.maximum(y, 1e-9)) + g).max():.4f}")


def main():
    ap = argparse.ArgumentParser(description="Bake NegPy print-stock LUTs.")
    ap.add_argument("--key", nargs="+", default=list(SHIPPED), help="LUT keys to build (default: every shipped one)")
    ap.add_argument("--icc", type=pathlib.Path, help="an RGB profile to bake")
    ap.add_argument("--as", dest="as_key", help="the LUT key for --icc")
    ap.add_argument("--against", help="a spectral key to compare the built LUT with")
    ap.add_argument("--profiles", type=pathlib.Path, default=pathlib.Path.home() / "Pobrane" / "papers", help="the Frontier profiles")
    ap.add_argument("--cie-dir", type=pathlib.Path, default=pathlib.Path.home() / ".cache" / "negpy" / "cie")
    ap.add_argument("--check-only", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    jobs = []
    if args.icc:
        if not args.as_key:
            ap.error("--icc needs --as KEY")
        jobs.append((args.as_key, args.icc))
    else:
        profiles = {k: REPO / f for k, f in LOOKS.items()} | {k: args.profiles / f for k, f in FRONTIER.items()}
        jobs += [(k, profiles.get(k)) for k in args.key]
    for key, icc_path in jobs:
        if icc_path is None:
            table, e_max, source, model = spectral_lut(*SPECTRAL[key][:2], args.cie_dir)
        else:
            icc = icc_path.read_bytes()
            cls, space, pcs, creator, desc, targ = header(icc)
            print(f"{icc_path.name}: class {cls}, space {space}, PCS {pcs}, creator '{creator}', '{desc}'")
            if targ:
                raise SystemExit(
                    "CHARACTERIZATION: the profile carries a targ measurement table, so it removes one device's color and holds no look"
                )
            if space != "RGB":
                raise SystemExit(f"{space} profile: only an RGB profile makes a print stock")
            if cls == "prtr":
                table, e_max, source, model, tone = frontier_lut(icc, f"{desc or icc_path.name} printer profile")
                tint = tuple(round(v, 4) for v in tone["base_tint_cmy"])
                print(f"   tone: d_min {tone['d_min']:.3f}, base_tint_cmy {tint}, black {tone['black_above_base']:.3f} D above base")
            else:
                table, e_max, source, model = icc_look_lut(icc, desc or icc_path.name)
        report(key, table, e_max, model)
        if args.against:
            cross_check(args.against, model, spectral_lut(*SPECTRAL[args.against][:2], args.cie_dir)[3], float(e_max.min()))
        if args.check_only or key not in SHIPPED and not args.icc:
            continue
        LUT_DIR.mkdir(exist_ok=True)
        save_print_lut(LUT_DIR / f"{key}.npz", table, e_max, source)
        print(f"   wrote {LUT_DIR / f'{key}.npz'}")


if __name__ == "__main__":
    main()
