# Paper-profile fits (2026-07-03)

Digitized D-logH curves from the vendor PDFs in `papers/` (vector-art extraction;
MG IV RC chart is raster — only Dmax was measured off it). CSVs here are the
digitized points (`log_e,density`), `FIT_LOG.txt` the raw fit_paper_profile.py
output, `verify_digitized.png` the extraction sanity plot.

Applied to `negpy/features/exposure/papers.py`:

| profile | source | fit | RMS |
|---|---|---|---|
| ilford_fb_classic | fb_*.csv, grades 00–5 family | shape as printed | 0.028 D |
| ilford_mg_rc | FB shape + raster Dmax 2.20 (plateau rows 31–38, 90 px/D, D0 row 232) | — | — |
| foma_fomatone | fomatone_s*.csv, grades 0–4 family | d_max raised 1.83→2.0 (glossy table; chart is semi-glossy) | 0.017 D |
| foma_fomabrom | fomabrom_s*.csv, grades 0–5 family | d_max 1.97→2.0 (chart's own “Dmax=2,0”) | 0.014 D |
| kodak_endura | endura_g.csv shape; R/B slope refits with G shape fixed | channel_gamma = k/k_G = (9.46, 8.27, 8.52)/8.27 → (1.14, 1.0, 1.03) | G 0.005; R 0.105 (shared-shape limit) |
| fuji_crystal | none — Fuji publishes no D-logH (Type II / DPII / AF3-198E all spectral-only) | unchanged estimate | — |

Grade naming: `fomatone_s0..s4` / `fomabrom_s0..s5` are ordered by fitted mid-slope
(softest → hardest), not by the datasheet grade labels (identity is nuisance for a
family fit).

## Per-grade slopes (for future grade_coupled_shape calibration)

FB Classic (grade → k): 00 1.20, 0 1.55, 1 2.06, 2 2.66, 3 3.36, 4 4.30, 5 4.74
Fomatone (s0→s4): 0.69, 0.81, 0.93, 1.08, 1.25
Fomabrom (s0→s5): 1.18, 1.38, 1.61, 1.93, 2.00, 2.31

Fit-degeneracy notes: FB mg fit → 0.0 at bound (relaxing bounds: mg −0.49/gw 2.0,
RMS gain only 0.0014 D — rejected). Fomatone mg 0.80 at bound is pure mg×gw
degeneracy (relaxed fit mg 1.48/gw 0.52, identical RMS). Endura R/G/B one-shape
family fit: RMS 0.059, rejected — channels really differ (Dmax 2.75/2.52/2.44).
