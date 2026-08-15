# Narrowband/ICC export failure on unnotarized macOS arm64 builds

**Root cause confirmed 2026-08-15 on real Apple Silicon hardware (M1 Pro,
macOS 15.7.9) — see "Root cause" below.** The symptom is already worked
around on this branch (see below); the remaining work is the durable fix
(notarization), tracked as a follow-up, not blocking this branch.

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

## Root cause (confirmed on M1 Pro, macOS 15.7.9)

`.github/workflows/release.yml` has no codesign/notarize step at all — the
app ships with PyInstaller's automatic ad-hoc signature only: no Developer ID,
not notarized. Apple Silicon's AMFI/Gatekeeper enforcement is stricter than
Intel's for unnotarized, ad-hoc-only code, and imagecodecs resolves every
codec — including `_cms` — lazily via `importlib.import_module` the first
time it's actually used, not at process start (see
`imagecodecs/imagecodecs.py: __getattr__`, wraps failures in
`DelayedImportError(ImportError)`). A dlopen refused by AMFI at that later,
lazy point explains: no crash, no Gatekeeper prompt (those only fire for
the top-level `.app` at first launch), identical static file structure to the
working Intel build, but a different runtime outcome.

Confirmed by reproducing on real M1 hardware:

1. `make build` on this branch, installed the resulting `.app`, triggered a
   Flat export with an ICC profile. `negpy.log` showed the new fallback
   warning instead of the old silent failure:
   ```
   WARNING ...image_processor: imagecodecs CMS codec unavailable on this
   build (could not import name 'cms_transform' from 'imagecodecs');
   falling back to the LUT-based ICC transform for this export
   ```
2. `spctl -a -vv .../Contents/Frameworks/imagecodecs/_cms.abi3.so` rejects
   the binary outright (`codesign -dvvv` confirms `Signature=adhoc`,
   `TeamIdentifier=not set`).
3. `log show --last 10m --predicate '(eventMessage CONTAINS "cms") OR
   (process == "amfid")'` (note: `log` is a zsh builtin — use `/usr/bin/log`
   explicitly, or the shell reports a spurious `too many arguments`) shows
   AMFI denying the exact file at the exact moment the fallback fires:
   ```
   09:28:30.121791 kernel: (AppleMobileFileIntegrity) AMFI:
     '.../imagecodecs/_cms.abi3.so' is adhoc signed.
   09:28:30.147701 amfid: .../imagecodecs/_cms.abi3.so not valid:
     Error Domain=AppleMobileFileIntegrityError Code=-423 "The file is
     adhoc signed or signed by an unknown certificate chain"
   ```
   Timestamps match the `negpy.log` warning to the second. AMFI is the cause.

## Why this never showed up in dev mode

Confirmed dev mode (`make run`, i.e. `uv run python desktop.py`) does not
reproduce the bug at all:

```
uv run python -c "from imagecodecs import cms_transform"   # imports cleanly
```

The `_cms.abi3.so` file itself is the same kind of signature either way —
plain ad-hoc, no Team ID, in both the venv copy and the packaged copy
(`codesign -dvvv` on each shows identical `flags=0x2(adhoc)`,
`TeamIdentifier=not set`). The difference is the signature of the process
*loading* it:

- Dev mode: `uv`'s managed CPython interpreter —
  `flags=0x20002(adhoc,linker-signed)`.
- Packaged app: PyInstaller's `NegPy` bootloader executable —
  `flags=0x2(adhoc)`, no `linker-signed`.

`linker-signed` is a flag the OS linker (`ld`) stamps on binaries it builds
locally on that machine; AMFI treats those as implicitly trusted and is
lenient about the ad-hoc extension modules they dlopen. A binary re-signed
after the fact with plain `codesign -s -` — which is what PyInstaller's
bootloader gets, and what `build.py`'s re-sign step does — never carries that
flag, and doesn't get the same leniency: AMFI enforces the strict
chain-to-Apple check on everything it lazily dlopens.

Practical consequence: any contributor running `make run`, or launching
`desktop.py` from an IDE, will never see this bug regardless of chip
architecture — that's the normal dev workflow, and it's likely why this
went unreported until a user hit it on a downloaded release build.

## Free workaround tested and ruled out

Before accepting the $99/year Developer ID cost, tested whether the
`com.apple.security.cs.disable-library-validation` entitlement — a free,
ad-hoc-compatible opt-out of *Library Validation* — would satisfy AMFI here:

```
codesign --force --deep -s - --entitlements entitlements.plist NegPy.app
```
(entitlements.plist: `com.apple.security.cs.disable-library-validation` = true)

Confirmed present on the re-signed executable (`codesign -d --entitlements -`),
relaunched, reproduced the export. No change — `negpy.log` and `log show`
recorded the identical rejection at the identical moment:

```
kernel: AMFI: '.../_cms.abi3.so' is adhoc signed.
amfid: .../_cms.abi3.so not valid: Error Domain=AppleMobileFileIntegrityError
  Code=-423 "The file is adhoc signed or signed by an unknown certificate chain"
```

Doesn't help because it's the wrong gate: Library Validation governs whether a
dylib's Team ID must match the loading executable's; disabling it waives that
match. Code `-423` is AMFI refusing the code outright because its signature
doesn't chain to Apple's root at all — ad-hoc and free self-signed certs both
fail that identically, and no entitlement waives it. The only certificate that
chains to Apple is a Developer ID Application certificate, issued only to paid
Apple Developer Program members. Signing with one (without going as far as
full notarization) would likely clear this specific check, since it's a
chain-of-trust check, not the separate Gatekeeper notarization-ticket check —
but both live behind the same $99/year membership, and skipping notarization
would leave the "unidentified developer" Gatekeeper prompt for anyone
downloading the app from GitHub Releases. So there's no cheaper path in
practice: full notarization is the right target regardless.

## Follow-up: notarize the release build (not done on this branch)

The durable fix is notarizing the release build (Apple Developer ID
certificate + `notarytool submit --wait` + `stapler staple`, wired into
`release.yml`). Bigger lift — paid Developer account + CI secrets — tracked
as separate follow-up work, not part of this branch. Until it lands, the LUT
fallback (`befdfd2`) is what ships: correct output, slightly lower fidelity
than lcms2's exact per-pixel transform.

## Commits on this branch so far

- `55d3fc8` — Narrowband/flat-export ICC `render_intent` ordering fix
  (separate real bug, not arm64-specific)
- `befdfd2` — LUT fallback when `imagecodecs.cms_transform` is unavailable
- `571893a` — build.py deep re-sign (unconfirmed impact)

## Status

Root cause confirmed; notarization (the durable fix) is not being done as
part of this branch. Tracked as
[#1](https://github.com/thetalkingdrum/NegPy/issues/1). This file now stands
as the record behind that issue rather than a live investigation — delete it
once #1 lands.
