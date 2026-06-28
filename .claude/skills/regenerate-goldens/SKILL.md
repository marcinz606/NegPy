---
name: regenerate-goldens
description: "Use when a deliberate look/pipeline change has shifted NegPy's golden / characterization test values and they need updating — e.g. after changing the working-space TRC or primaries, the print curve, a lab/toning default (saturation, vibrance), or any default that alters rendered output, and a snapshot test now fails. Keywords: golden, goldens, characterization test, snapshot, regenerate, update expected values, test_scene_linear_relocation, test_characteristic_curve, _GOLDEN, look drift."
---

# Regenerate goldens

## Overview

A few tests pin **rendered output** as hardcoded "golden" values — they exist to catch
*unintended* drift in the look. When you change the pipeline on purpose (a default, the
TRC, primaries, a curve), those snapshots go stale and must be re-baselined.

**Core principle — this is the one thing to get right:** only regenerate a golden when the
change that moved it was **intentional**. A failing golden after an unrelated edit is a
*real regression* — fix the code, do not bless the new numbers. Regenerating to silence a
surprise defeats the entire point of the test.

## Which tests carry rendered goldens

- **`tests/test_scene_linear_relocation.py`** — `_GOLDEN`: full-engine output (6 pixels ×
  default / `expo_dark` / `expo_cmy`). Depends on the whole creative chain, so it shifts on
  almost any look change: working TRC/primaries, print curve, lab/toning/finish defaults
  (saturation, vibrance, etc.).
- **`tests/test_characteristic_curve.py::test_default_curve_shape`** — `golden`: the default
  print curve sampled at x = 0, .25, .5, .75, 1 (exposure kernel only). Shifts on TRC or
  print-curve changes, **not** on lab/saturation.

Not rendered snapshots (do **not** use this skill for them):
- `test_working_oetf.py` — values are defined by the TRC *formula*; hand-edit if you change
  the curve.
- `test_lab_colorspace.py` / `test_lab_logic.py` — assert structural properties / explicit
  inputs, not default-look pixels. A failure there usually means a real math change.
- Parity tests (`test_pipeline_parity.py`) — relative CPU↔GPU; default-value changes apply
  to both sides, so they don't need regeneration.

## Workflow

1. **Confirm the change was intentional.** If you can't name why the look moved, stop and
   investigate — that's a regression, not a golden update.

2. **See what actually fails** so you only touch real snapshots:
   ```bash
   make test
   ```
   Expect `test_scene_linear_relocation` and/or `test_characteristic_curve` if the change is
   look-affecting. Anything else failing is a separate signal — read it, don't auto-bless.

3. **Print the new values:**
   ```bash
   uv run python .claude/skills/regenerate-goldens/regenerate.py
   ```
   It imports each test's own image / sample-point / curve helpers, so those stay in sync;
   only the relocation config list is mirrored in the script — if you add or remove a config
   or sample point in the test, update the script to match.

4. **Paste** the printed blocks over `_GOLDEN` (relocation) and `golden` (characteristic
   curve). Keep the surrounding comment/docstring accurate (which TRC, which default).

5. **Re-run to confirm green** and eyeball the magnitude of the shift — a tiny tweak that
   moved a pixel by 0.4 is a red flag the change did more than intended:
   ```bash
   make test
   ```

6. If a *non-golden* test (lab_logic L* tolerance, etc.) also moved, decide per-test whether
   the new behaviour is correct before loosening a bound — don't widen tolerances to pass.
