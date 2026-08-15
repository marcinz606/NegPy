# Narrowband/ICC export failure on unnotarized macOS arm64 builds

Handoff note for continuing the root-cause hunt on real Apple Silicon hardware.
The symptom is already worked around on this branch (see below); this file is
about finding *why* it happens, so the workaround can eventually be removed.

## Symptom

- Flat (and Print) exports with Narrowband Scan on, or any custom Input/Output
  ICC profile, silently ship without color management on some Apple Silicon
  Macs — toggling the profile on/off makes no difference to the exported file.
- `negpy.log` (`~/Documents/NegPy/negpy.log`) shows:
  `CMS transformation failed: could not import name 'cms_transform' from 'imagecodecs'`
- Confirmed on the official 0.49.0 release: same frame, same version — fails
  on an M1 MacBook Pro, works correctly on an Intel Mac Mini (2018).

## What's fixed already on this branch (masks the symptom, not the cause)

1. `negpy/desktop/controller.py` (`55d3fc8`) — `request_export`/
   `request_batch_export` resolved Narrowband's implicit ICC profile using the
   pre-flatten (Print) `render_intent`, so an E-6-with-Normalize-off
   transparency transfer's ICC suppression leaked into Flat exports. Fixed by
   resolving after `flat_master_config()`. **This is a separate, real bug,
   unrelated to the arm64 issue below** — found while chasing the original
   report, worth keeping regardless of how the arm64 investigation ends.
2. `negpy/services/rendering/image_processor.py` (`befdfd2`) —
   `_apply_color_management_u16` now catches the specific `ImportError`
   (`imagecodecs.DelayedImportError`) raised when `cms_transform` isn't
   available, and falls back to the existing PIL/LUT-based
   `_apply_color_management_u16_rgb` at a 65³ grid, instead of silently
   no-op'ing. **This is the workaround** — it should mask the arm64 symptom
   regardless of root cause, at the cost of a LUT approximation instead of
   lcms2's exact per-pixel transform.
3. `build.py` (`571893a`) — added `codesign_macos_app()`, a
   `codesign --force --deep -s -` pass over the whole `.app` after PyInstaller
   builds it. **Unconfirmed whether this changes anything** — see below.

## What's ruled out

Downloaded and statically compared the real 0.49.0 arm64 and x86_64 DMGs
(`gh release download 0.49.0 --repo marcinz606/NegPy`):

- The `imagecodecs==2026.6.6` wheel on PyPI genuinely bundles `_cms.abi3.so` +
  `liblcms2.2.dylib` for **both** arm64 and x86_64 — not a missing-wheel-codec
  problem.
- Both packaged `.app` bundles have identical file layout, identical
  `@rpath`/`LC_RPATH` resolution from `_cms.abi3.so` to `liblcms2.2.dylib`,
  and both pass `codesign --verify --deep --strict` cleanly with zero errors.
- Re-running `codesign --force --deep -s -` on the arm64 `.app` produced a
  **byte-identical** signature on `_cms.abi3.so` compared to what already
  ships. So the build.py re-sign is unlikely to change runtime behavior by
  itself — it's a free, harmless thing to keep, not a confirmed fix.

## Leading theory (unconfirmed — needs real hardware to test)

`.github/workflows/release.yml` has no codesign/notarize step at all — the
app ships with PyInstaller's automatic ad-hoc signature only: no Developer ID,
not notarized. Apple Silicon's AMFI/Gatekeeper enforcement is stricter than
Intel's for unnotarized, ad-hoc-only code, and imagecodecs resolves every
codec — including `_cms` — lazily via `importlib.import_module` the first
time it's actually used, not at process start (see
`imagecodecs/imagecodecs.py: __getattr__`, wraps failures in
`DelayedImportError(ImportError)`). A dlopen refused by AMFI at that later,
lazy point would explain: no crash, no Gatekeeper prompt (those only fire for
the top-level `.app` at first launch), identical static file structure to the
working Intel build, but a different runtime outcome. Not proven — this
machine is x86_64 and cannot execute arm64 code to confirm.

## Next diagnostic steps (needs the M1)

1. `make build` (not `make run` — dev mode runs straight from the venv,
   bypassing PyInstaller entirely, so it cannot reproduce a packaging bug at
   all) on this branch, install the resulting `.app`, and reproduce the
   original failure. Check `negpy.log`: it should now show the new fallback
   warning (`imagecodecs CMS codec unavailable...falling back to the
   LUT-based ICC transform`) instead of the old silent `CMS transformation
   failed`.
2. To confirm or kill the AMFI/Gatekeeper theory: right when a `cms_transform`
   failure happens (or would have, pre-fallback), run:
   ```
   log show --predicate 'eventMessage contains "cms" or process == "amfid"' --last 10m
   spctl -a -vv /Applications/NegPy.app/Contents/Frameworks/imagecodecs/_cms.abi3.so
   ```
   A denial in either output confirms AMFI/Gatekeeper as the cause.
3. If confirmed: the durable fix is notarizing the release build (Apple
   Developer ID certificate + `notarytool submit --wait` + `stapler staple`,
   wired into `release.yml`). Bigger lift — paid Developer account + CI
   secrets — worth a separate discussion once confirmed, not before.
4. If AMFI/Gatekeeper is *not* the cause: worth checking whether a different
   `imagecodecs` version behaves differently on this exact machine, and
   whether `xattr -cr` on the installed `.app` (stripping the quarantine flag
   before first launch) changes anything — that would still implicate
   Gatekeeper/quarantine, just via a different mechanism than AMFI.

## Commits on this branch so far

- `55d3fc8` — Narrowband/flat-export ICC `render_intent` ordering fix
  (separate real bug, not arm64-specific)
- `befdfd2` — LUT fallback when `imagecodecs.cms_transform` is unavailable
- `571893a` — build.py deep re-sign (unconfirmed impact)

## If the root cause gets found

Once confirmed, this file should either get folded into a proper fix (and
deleted), or — if notarization turns out to be the answer but isn't done
immediately — turned into a tracked follow-up instead of living here
indefinitely.
