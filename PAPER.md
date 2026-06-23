# Per-channel Dmax for paper profiles (Endura colour)

## Context

Kodak Endura's datasheet characteristic curves (E-4070) show R/G/B that are
**equal at Dmin and on the rise**, diverging **only at the shoulder**:

- Dmax: **R 2.74 / G 2.51 / B 2.44** (spread 0.30 D), Dmin ~0.09 shared.

So Endura's colour is a **shadow-only crossover**: deepest shadows go cool (R runs
densest → less red), midtones/highlights stay neutral. Today `kodak_endura` is
rendered fully neutral because the only colour knobs we have can't express this:

- `channel_gamma` (per-channel slope) makes a *symmetric* crossover — cool shadows
  **and** warm highlights (verified: `(1.05,1,0.97)` → shadow B−R +0.015 but
  highlight −0.017). Endura highlights are neutral, so this adds a wrong warm tint.
- `base_tint_cmy` shifts the whole tone scale uniformly.

Faithful fix = **per-channel Dmax**: it bends only the toe asymptote (deepest
shadows) per channel, leaving the shared rise/Dmin neutral — exactly the curve.

## Design

Add an optional per-channel Dmax that overrides the scalar `d_max` for the toe
(paper-black) bound only. The achromatic `d_max` stays the reference for pivot /
`_reference_linear_value` / the chart, so **neutral midtones stay neutral** — only
the shadow asymptote diverges.

1. **`PaperProfile`** (`negpy/features/exposure/papers.py`)
   - Add `dmax_rgb: Optional[Tuple[float,float,float]] = None`. `None` → achromatic
     (use scalar `d_max`). Set `kodak_endura` → `(2.74, 2.51, 2.44)` with scalar
     `d_max=2.51` (G, the neutral reference). Not in `_TONAL_KEYS` (threaded
     directly, like `channel_gamma`/`base_tint_cmy`).

2. **CPU kernel** (`logic.py::_apply_print_curve_kernel`)
   - `d_max` becomes a length-3 array. `d_max_eff` (and the toe `toe_height` /
     negative-toe branch) is currently computed once before the channel loop —
     move it **inside** the `for ch` loop, indexing `d_max[ch]`.
   - `apply_characteristic_curve` builds the 3-vector from
     `paper.dmax_rgb or (c["d_max"],)*3` and passes it.
   - `_reference_linear_value` / `compute_pivot` keep the **scalar** `d_max`
     (achromatic neutral reference) — do NOT make them per-channel.

3. **GPU** (`exposure.wgsl` + `gpu_engine.py::_build_exposure_uniforms`)
   - Shader `d_max: f32` → `vec3<f32>` in `ExposureUniforms` (check the WGSL toe
     math is already vec3 per-channel; if scalar, vectorise the `d_max_eff` step).
   - Pack three Dmax floats instead of one; keep struct 16-byte alignment
     (`v_star` stays the achromatic scalar). Mirror the CPU `dmax_rgb or *3`.

4. **Chart** (`charts.py`) — achromatic, unchanged (uses scalar `d_max`).

## Files

- `negpy/features/exposure/papers.py` (field + Endura value)
- `negpy/features/exposure/logic.py` (kernel + `apply_characteristic_curve`)
- `negpy/services/rendering/gpu_engine.py` + `features/exposure/shaders/exposure.wgsl`
- `tests/test_paper_profiles.py`, `tests/test_pipeline_parity.py`

## Verification

- Numeric: neutral grey ramp through Endura → **shadow B−R > 0 (cool), midtone &
  highlight B−R ≈ 0**. Confirms shadow-only crossover (channel_gamma failed the
  highlight-neutral check; this must pass it).
- Default profile + non-`dmax_rgb` papers render byte-for-byte unchanged
  (`dmax_rgb=None` path).
- CPU↔GPU parity: add an Endura `test_paper_profile_dmax_rgb` case.
- `make all` green; `make run` eyeball Endura — cool deep shadows, neutral skin/
  highlights.

## Notes / caveats

- Effect is subtle (0.30 D spread only at the absolute Dmax / deepest black).
- Only Endura has clean R/G/B curves; Fuji has no published D-logE curve, so it
  keeps the `channel_gamma` + `base_tint` estimate. Other RA4 papers can adopt
  `dmax_rgb` later if curves are digitised.
