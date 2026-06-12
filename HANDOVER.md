# HANDOVER — branch `def-conv` (default conversion refinements)

State as of 2026-06-12. All checks green: `make all` (lint + ty + pytest) and `uv run pytest tests/test_pipeline_parity.py` (20 passed + 1 known xfail: GPU chroma denoise ignores radius).

## What this branch does

Reworks the default negative→print conversion to be paper-physical. The exposure stage now models:

1. **ISO-R grade**: `ExposureConfig.grade` stores ISO R directly (slider 50–180, inverted: right = harder, default 115). `grade_to_slope` (logic.py): `slope = ln(81) · negative_range / (R/100)`, clamped [2, 11]. Legacy 0–5 grades migrate in `ExposureConfig.__post_init__` via `R = 150 − 20·G` (values ≤ 5 are unambiguously legacy) so saved edits keep their look. W/S shortcuts step ∓10 R (`keyboard_shortcuts.py`).
2. **Paper density scale** (all in `EXPOSURE_CONSTANTS`, models.py — single source of truth):
   - `d_max: 2.3` — physical paper black (real glossy RA-4/fiber).
   - `curve_asymptote: 4.6` — the logistic's virtual ceiling. Curve = `d_min + (asymptote − d_min)·sigmoid(k·(x − pivot))`, then **soft-clamped at d_max** via `D −= softplus(β·(D − d_max))/β`, `β = dmax_shoulder: 8.0`. Rationale: real paper curves are asymmetric — straight-line portion projects past Dmax, abrupt saturation at paper black. A symmetric L=d_max logistic starves film-toe-compressed shadows into gray (verified on `samples/raw0004.dng`). Clamp applies AFTER the toe lever.
   - `d_min: 0.06` behind `ExposureConfig.paper_dmin` toggle ("Paper White" button, exposure sidebar); effective d_min = 0.0 when off.
   - `toe_onset_density: 1.2` — absolute D (NOT a fraction of d_max); toe slider = density-domain shadow lever, anchored at D=0 with tangent removed (highlight-invariant).
3. **Fixed calibrated pivot** (NO auto-exposure): `compute_pivot(slope, density, d_min)` solves so the assumed reference tone (`assumed_anchor: 0.7` normalized) prints at `anchor_target_density: 0.85`. Grade rotates contrast around that tone; Density slider shifts ±0.2 around it. Deterministic: same sliders → same exposure.
4. sRGB OETF output encode; unclamped normalization (extremes roll off through the curve); B&W collapses to luminance before the curve.

## History you must know (don't undo / re-propose)

- **LATD auto-exposure metering was built and then FULLY removed** at user request (classic subject failure on skewed histograms; user prefers deterministic baseline + Density slider). Removal touched: `LogNegativeBounds` (no anchor field), `ProcessConfig` (no local/locked_anchor), `normalization.py` analyze, both engines, controller, session sticky settings, roll worker signal `(tuple, tuple)`, repository (roll DB `anchor` column unused but left in old DBs — don't reference it), presets, session_panel, charts. `tests/test_auto_exposure.py` replaced by `tests/test_exposure_pivot.py`.
- Also previously rejected: shared density scale (slope factors stay GLOBAL, never per-channel), enlarger flare control. See memory files in `~/.claude/projects/-home-marcin-code-darkroom-py/memory/`.
- d_max/d_min/asymptote/targets were hand-tuned by the user several times — treat current constant values as user taste; change only on request.

## Key files

- `negpy/features/exposure/models.py` — `ExposureConfig` (+ grade migration `__post_init__`, `paper_dmin`), `EXPOSURE_CONSTANTS`.
- `negpy/features/exposure/logic.py` — fused numba kernel (curve + toe/shoulder + soft clamp + sRGB), `LogisticSigmoid` (chart reference, identical math), `grade_to_slope`, `compute_pivot`, stable `_expit` (`exp(−logaddexp(0,−x))`).
- `negpy/features/exposure/shaders/exposure.wgsl` + `gpu_engine.py` `_upload_unified_uniforms` (~line 750) — GPU mirror; uniform struct carries d_max, d_min, toe_onset, asymptote, shoulder_beta. **Any curve change must be made in kernel + LogisticSigmoid + WGSL together**, parity test enforces it.
- `negpy/desktop/view/sidebar/exposure.py` — grade slider (inverted), Paper White button.
- `negpy/desktop/view/widgets/charts.py` — H&D chart; gets slope/pivot from `session_panel.py`, falls back to same helpers.

## Verification workflow

- Gate: `make all` (never run pytest/ruff/ty directly — always via make / `uv run`).
- CPU↔GPU: `uv run pytest tests/test_pipeline_parity.py` (needs GPU; passes locally).
- Real-image probe (used throughout): decode `samples/raw0004.dng` with rawpy (`user_flip=0, gamma=(1,1), no_auto_bright=True`), run `DarkroomEngine().process(img, WorkspaceConfig(), "probe")`, check percentiles. Current default output on that sample: p0.5 ≈ 0.018, p50 ≈ 0.19, p90 ≈ 0.81 (sample is a pathological backlit scene — median sits high in its normalized range; user rides Density slider for such frames).
- Visual: `make run`, compare against `samples/raw0004.jpg` (export from the now-removed metering era — darker mids expected now).

## Open threads / candidates (discussed, not committed)

- Specular highlights stretch the normalized range when `drange_clip = 0` (floor = absolute min); a robust percentile floor would stabilize range/slope. User controls via D-Range Clip slider for now.
- Color-accuracy candidates user showed interest in: printing-density **crosstalk matrix** (3×3 in log space) and **auto shadow-neutral** (measure cast in densest ~2%, auto-set existing shadow CMY offsets). Neither started.
- Shadow color casts (dense-end channel misalignment) remain a known annoyance; crosstalk/auto-neutral above are the planned attacks.
- CHANGELOG 0.25.0 (`docs/CHANGELOG.md`) is up to date with all of the above.

## Conventions

- Caveman mode active for chat replies (terse); code/commits/docs written normally.
- `WorkspaceConfig` is frozen; mutate via `dataclasses.replace`. New config fields serialize automatically via flat-dict dataclass fields.
- Normalization direction is counter-intuitive: normalized 0 = densest negative = scene highlight = print white side; 1 = thinnest = scene shadow = print black side. floors/ceils naming explained in memory `normalization_conventions.md`.
- Plan/spec docs go in project root with short UPPERCASE names (like this file).
- Pyright inline diagnostics are noisy (param-name overrides, frombuffer overloads, `self.layout` MethodType) — `make all` (ruff + ty) is the actual gate.
