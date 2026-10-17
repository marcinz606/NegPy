---
name: print-stock-from-icc
description: "Use when the user wants an ICC profile turned into a NegPy print stock (paper profile), for example a minilab look profile such as Pakon RGBScan.icc, a Frontier or Noritsu printer profile, or a PaperRGB profile, or asks how closely a print stock can reproduce an ICC. Keywords: ICC, .icc, .icm, print stock, paper profile, print LUT, DyeLut, Frontier, Pakon, minilab, prtr, Input ICC."
---

# Print stock from ICC

A color print stock carries its dyes as a 3D LUT: `PaperProfile.print_color = DyeLut(key)`, read from
`negpy/features/exposure/print_luts/<key>.npz`. `scripts/build_print_luts.py` bakes it from a profile, so the
job is: check that the profile qualifies, bake, read the report, ship. Worked examples: *Digital (Pakon
Minilab)* (`pakon`, a look profile) and the three Fujicolor Crystal Archive stocks (`fuji_ca_*`, Frontier
printer profiles).

## First: is it already a stock?

Look for the profile's file name in `LOOKS` and `FRONTIER` in `scripts/build_print_luts.py`, and for its
description in the LUT sources:

    uv run python -c "import numpy as np, glob; [print(f, np.load(f)['source']) for f in sorted(glob.glob('negpy/features/exposure/print_luts/*.npz'))]"

If it is there, find the stock with `grep -nF 'DyeLut("KEY")' negpy/features/exposure/papers.py`, tell the
user its label, and stop. If it is another version of a shipped profile (same paper, other revision), ask the
user whether it replaces that stock's LUT (a render change: bump `THUMBNAIL_RENDER_VERSION`) or becomes a
second stock. Run `--check-only` on both files and give the user both reports to decide on. A second stock
gets its own key with the revision in it (`fuji_ca_hd_v3a`).

## Does the profile qualify?

Run `uv run python scripts/build_print_luts.py --icc PATH --as KEY --check-only` from the repo root, with
the profile kept outside the repo. It prints the header, picks the method by device class, and stops on a
`targ` table or a non-RGB profile.

| Profile | Verdict |
|---|---|
| RGB look or minilab output space (class `spac`, `scnr`, `mntr`, and no sign of a characterization below), applied to a finished positive | **ICC look** method: the LUT is the profile applied to Digital's print |
| RGB printer profile (`prtr`) of a photographic minilab that exposes each paper layer with its own narrow light: Frontier, Noritsu QSS, Durst Lambda, a laser or LED printer | **Frontier** method: the paper as that printer measured it. A printer profile is always made from a printed target; that is a measurement of the paper, not a characterization to decline |
| Inkjet, dye-sub or thermal printer, or any CMYK profile | decline: not a dye print on photo paper |
| A scanner or camera profile with a `targ` table, a target named in its description, or made by the user from a target | decline: a characterization of one input device; it removes that device's color and holds no look |
| An ICC look whose report shows a gray luminance error over 0.02 D | decline: it carries a tone curve, which would stack on NegPy's. A gray tint (C*) is fine; the LUT keeps it |
| Modelled from datasheet dyes (spektrafilm, PaperRGB LUTs) | do not ship (data rule) |

**After a decline**, say where the profile belongs. A scanner or camera characterization goes in Export ›
Input ICC for raw scans from that device. A look with a tone curve can only be applied whole, as Input ICC
with Soft Proof. An inkjet profile is an export profile for that printer.

**Data rule.** NegPy ships color data from manufacturer datasheets, standards, papers, or the profile of a
real machine. A profile modelled from datasheet dyes is model output.

## Steps

1. **Bake:** `uv run python scripts/build_print_luts.py --icc PATH --as KEY` writes `print_luts/KEY.npz`.
2. **Read the report:**
   - interpolation ΔE00 mean under 0.1;
   - gray luminance error under 0.02 D;
   - gray C* (the stock's gray drift).
   - The Frontier method also prints `tone`: `d_min`, `base_tint_cmy`, and the black in D above base.
3. **Cross-check a paper** when `SPECTRAL` in the script has its datasheet dyes: add `--against <key>`
   (as Crystal Archive DPII against `fuji_af3`). Hues should agree within a few degrees. If near-neutral
   colors differ by more than about 1.5 ΔE00, show the user the numbers before shipping.
4. **Ship** (next section). Run `uv run pytest tests/test_print_luts.py tests/test_paper_profiles.py` and
   `uv run ruff check` on changed files, not the full pipeline. Then the `verify` skill: pick the stock on a
   real frame and check that it renders.

## Ship

- **`scripts/build_print_luts.py`:** add the key to `LOOKS` (a profile that may be redistributed, copied to
  `tests/fixtures/`) or to `FRONTIER` (a file name read from `--profiles`). Then the plain rebuild of every
  LUT includes it.
- **Names:** the LUT key names the data source (`fuji_ca_hd`), the stock key follows its family in
  `papers.py` (`fuji_crystal_hd`), and the label is the maker's product name unless the user gives one.
- **`negpy/features/exposure/papers.py`:** a `PaperProfile` with `kind="ra4"`, the label,
  `print_color=DyeLut(KEY)`, and one comment naming the profile:
  - ICC look: `medium="digital"`, no tonal keys (Digital's curve).
  - Frontier: `medium="paper"`. Set `d_min` and `base_tint_cmy` from `tone`, and `d_max = d_min + black`.
    Copy the knees and midtone from the closest stock of the same paper family, and tell the user which.
  - Dict order is menu order. The dropdown puts a header above each run of one `medium`, so place the stock
    next to its family; `test_crystal_archive_tones_from_frontier` pins the Crystal Archive order.
- **Tests:** `test_color_stocks_are_luts` and `test_bundled_lut_*` cover the entry. For an ICC look in
  `tests/fixtures/`, add it to `ICC_STOCKS` in `tests/test_paper_profiles.py`.
- **Docs:** the Print stock bullet of `docs/USER_GUIDE.md`, the combo tooltip (`sidebar/paper.py`) if it
  names the family, and the bundled list under "Print stocks" in `docs/PIPELINE.md`. Grep those files and
  `NOTICE.md` for the family's name: a count ("the three Crystal Archive stocks") or a list changes too.
- **`NOTICE.md`:** credit the profile's source. Keep a profile the user may not redistribute out of the repo;
  ship only the LUT.
- **Render version:** a new stock changes no existing render, so no `THUMBNAIL_RENDER_VERSION` bump and no
  migration. A new LUT for an existing stock changes renders: bump it.

## Reading the numbers

| ΔE00 | What a viewer sees |
|---|---|
| under 1 | nothing, even on flat patches side by side |
| 1 to 2 | a trained eye, on flat patches side by side; rarely in a photo |
| 2 to 3 | visible side by side |
| 3 and over | visible at a glance side by side |

## Common mistakes

| Mistake | Fix |
|---|---|
| Declining a minilab `prtr` profile as "not a look" | It is the paper, measured; use the Frontier method |
| Baking a printer profile with the ICC-look method | The script picks the method by class; do not force it |
| Copying a vendor's `.icm` into the repo | Keep it outside and pass `--profiles` or `--icc`; ship the `.npz` |
| Running the script with `python -I` | It drops `scripts/` from the import path; run the command as written |
| Placing a paper stock among the Digital stocks | The dropdown groups runs of one `medium` |
