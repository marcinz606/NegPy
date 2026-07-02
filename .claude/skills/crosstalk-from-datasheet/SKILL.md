---
name: crosstalk-from-datasheet
description: "Use when the user supplies a film-stock spec sheet, datasheet, or spectral data (PDF / URL / numbers) and wants a NegPy spectral-crosstalk .toml matrix for the Process panel's Crosstalk control, or asks to calibrate / derive channel-unmixing for a specific film or scanner. Keywords: crosstalk, datasheet, spec sheet, film stock, dye density, spectral, separation, unmixing, toml."
---

# Crosstalk from datasheet

## Overview

A color negative's three dye layers each leak density into channels they shouldn't
("unwanted absorption"), muddying color. NegPy's **Crosstalk** control (Process panel)
unmixes them with a 3×3 matrix applied to the raw negative densities. This skill turns a
film's published **spectral dye-density** curves into that matrix: read the unwanted
absorptions, invert them, write a `.toml`.

**Core principle:** the matrix is the *inverse* of the dye-leakage matrix. You only read
values off curves and judge data quality — `derive_matrix.py` does the arithmetic.

Read `docs/CROSSTALK.md` and `crosstalk/README.md` once for the file format and the
density-domain math NegPy applies (`d=-log10(rgb_negative)`, `d_out=M·d`, pre-normalization);
NegPy **row-normalizes** `M` and blends with identity by the Crosstalk strength (0–1), so
only the off-diagonal/diagonal *ratios* matter — absolute row scale is irrelevant.

## Workflow

1. **Acquire the source.**
   - Local PDF → `Read` it with `pages` (**vision** — the curves are pixels; a text fetch
     will not see them).
   - URL → PDF → `curl` it to the scratchpad, then `Read` the pages as images.
   - HTML page → `WebFetch` for any numeric spectral tables; if it only embeds a plot,
     download the image and `Read` it.
   - Raw spectral numbers / CSV → use them directly (Tier 1).

2. **Classify the data tier** (the one decision that's easy to get wrong):

   ```dot
   digraph tier {
     "Separated C/M/Y dye-density curves\n(or numeric spectral data)?" [shape=diamond];
     "Only aggregate neutral dye-density\n(one combined curve)?" [shape=diamond];
     "Tier 1: derive directly" [shape=box];
     "Tier 2: estimate, label (approx)" [shape=box];
     "Tier 3: do NOT fabricate" [shape=box];
     "Separated C/M/Y dye-density curves\n(or numeric spectral data)?" -> "Tier 1: derive directly" [label="yes"];
     "Separated C/M/Y dye-density curves\n(or numeric spectral data)?" -> "Only aggregate neutral dye-density\n(one combined curve)?" [label="no"];
     "Only aggregate neutral dye-density\n(one combined curve)?" -> "Tier 2: estimate, label (approx)" [label="yes"];
     "Only aggregate neutral dye-density\n(one combined curve)?" -> "Tier 3: do NOT fabricate" [label="no"];
   }
   ```

   Most *consumer* datasheets (e.g. Kodak Gold 200) are **Tier 2**: the spectral-dye-density
   plot is one *aggregate* "midscale neutral" curve, not separated dyes. Spectral-*sensitivity*
   curves are **not** dye-density and must never be used as a substitute.

3. **Read the leakage values into matrix `A`** (rows = R/G/B measurement bands, cols = C/M/Y
   dyes). Dye→channel: **cyan↔Red, magenta↔Green, yellow↔Blue**. Sample each dye's diffuse
   density at band centers **R≈650 / G≈550 / B≈450 nm** (datasheets use Status M). Each dye's
   own-band value is the diagonal; the other two are its unwanted absorption (positive).
   Reading a graph by eye is ±~0.05 density — fine, the result gets row-normalized.

   - **Tier 1:** read all nine values off the separated curves.
   - **Tier 2:** diagonal = own band; estimate the off-diagonals from the visible secondary
     humps in the aggregate curve plus known dye chemistry (magenta's unwanted *blue* and some
     *red*; cyan a little *green/blue*; yellow nearly clean). Name it `"… (approx)"`.
   - **Tier 3:** stop. State that the datasheet lacks dye-density data. Optionally emit a
     conservative starter close to NegPy's Default, named `"… (approx, starter)"`, and suggest
     chart-based calibration instead. Never invent precision.

4. **Run the deriver:**
   ```bash
   echo '{"name":"Kodak Gold 200 (approx)","readings":[[1.0,0.06,0.02],[0.04,1.0,0.05],[0.02,0.12,1.0]]}' \
     | python3 .claude/skills/crosstalk-from-datasheet/derive_matrix.py --out <film>.toml
   ```
   `readings` is `A` row-major (R/G/B band rows). It column-normalizes, inverts, and rescales
   so the diagonal reads `1.0`; it **exits non-zero** on a singular or misshaped input rather
   than emitting a bad matrix.

5. **Review & self-validate** — never trust the loader's silence (it skips malformed files
   without error):
   - 3×3 numeric, no booleans.
   - Diagonal `1.0`; off-diagonals **negative** (a correction *subtracts* contamination) and
     small — flag anything `|·| > 0.2` as a likely misread/transpose.
   - Confirm it actually loads:
     ```bash
     uv run python -c "from negpy.services.assets.crosstalk import CrosstalkProfiles as C; \
       r=C._parse_file('<film>.toml'); assert r, 'INVALID — would be skipped'; print('ok', r[0])"
     ```

6. **Write the file** — pick the destination by intent:
   - **Contribute / bundle** → repo gallery `crosstalk/<film>.toml`.
   - **Personal use now** → `<Documents>/NegPy/crosstalk/<film>.toml` (resolve the Documents
     path via `get_default_user_dir()` in `negpy/kernel/system/paths.py`).

## Quick reference

| | |
|---|---|
| TOML keys | `matrix` (required, 3×3 numeric, no bool); `name` (optional; `"Default"` reserved) |
| Convention | rows = **output** channels, cols = **input** channels (R/G/B) |
| Band centers | R 650 nm · G 550 nm · B 450 nm (Status M) |
| Dye→channel | cyan→R · magenta→G · yellow→B |
| `A` shape | rows = bands (R/G/B), cols = dyes (C/M/Y); diagonal = own-band peak |
| Default (fallback) | `[1.0,-0.05,-0.02, -0.04,1.0,-0.08, -0.01,-0.1,1.0]` |

## Common mistakes

- **Transposing rows/cols** — `A` is bands×dyes; the output is out-channels×in-channels. Keep
  R/G/B order throughout.
- **Positive off-diagonals in the output** — wrong sign; a correction must *subtract* leakage.
- **Using sensitivity curves as dye density** — they measure different things; not interchangeable.
- **Over-claiming precision** — an aggregate (Tier 2) curve gives an estimate; say so in the name.
- **Forgetting it's density-domain** — read *density* off the dye-density plot, not transmittance.

## Worked example — Kodak Gold 200 (Tier 2)

`Read` the datasheet PDF pages as images. The CURVES section has Characteristic, Spectral-
Sensitivity, and Spectral-Dye-Density plots — but the dye-density plot is a single *aggregate*
"midscale neutral" curve → **Tier 2**. Diagonals = `1.0`; estimate off-diagonals from the
curve's secondary absorptions + dye chemistry (magenta's unwanted blue is the biggest term):

```
A (rows R/G/B, cols C/M/Y):   R:[1.0,  0.06, 0.02]
                              G:[0.04, 1.0,  0.05]
                              B:[0.02, 0.12, 1.0 ]
```

Run the deriver → emitted TOML (note `name` flags it approximate):

```toml
name = "Kodak Gold 200 (approx)"
matrix = [
  [1.0000,  -0.0579,  -0.0171],
  [-0.0390,   1.0000,  -0.0492],
  [-0.0152,  -0.1191,   1.0000],
]
```

The strong `B,−0.119(G)` term is the magenta-dye unwanted-blue correction. Validate it loads,
then write it to `crosstalk/kodak_gold_200.toml` (contribute) or the Documents folder (personal).
