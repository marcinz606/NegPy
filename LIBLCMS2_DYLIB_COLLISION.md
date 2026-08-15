# liblcms2.2.dylib collision breaks ICC export in the packaged macOS app

**Root cause confirmed and fixed 2026-08-15.** This started as an
investigation into an apparent AMFI/code-signing/notarization problem on
Apple Silicon (see git history of this file for that earlier, incorrect
diagnosis). It is not that. The real cause is a mundane PyInstaller dylib
de-duplication bug, and it is now fixed in `build.py`.

## Symptom

- Flat (and Print) exports with Narrowband Scan on, or any custom Input/Output
  ICC profile, silently ship without color management on some Apple Silicon
  Macs — toggling the profile on/off makes no difference to the exported file.
- `negpy.log` (`~/Documents/NegPy/negpy.log`) shows:
  `CMS transformation failed: could not import name 'cms_transform' from 'imagecodecs'`
- Confirmed on the official 0.49.0 release: same frame, same version — fails
  on an M1 MacBook Pro, works correctly on an Intel Mac Mini (2018).

## What's fixed already on this branch

1. `negpy/desktop/controller.py` (`55d3fc8`) — `request_export`/
   `request_batch_export` resolved Narrowband's implicit ICC profile using the
   pre-flatten (Print) `render_intent`, so an E-6-with-Normalize-off
   transparency transfer's ICC suppression leaked into Flat exports. Fixed by
   resolving after `flat_master_config()`. **A separate, real bug**,
   unrelated to everything below.
2. `negpy/services/rendering/image_processor.py` (`befdfd2`) —
   `_apply_color_management_u16` (`image_processor.py:1248`) now catches the
   specific `ImportError` (`imagecodecs.DelayedImportError`) raised when
   `cms_transform` isn't available, and falls back to the existing
   PIL/LUT-based `_apply_color_management_u16_rgb` at a 65³ grid, instead of
   silently no-op'ing. **Worth keeping regardless** — a real fallback is
   better than a silent no-op even with the real cause fixed below, in case
   something else ever makes the codec unavailable.
3. `build.py` (`571893a`) — a `codesign --force --deep -s -` pass over the
   whole `.app` after PyInstaller builds it. **Confirmed to have no effect on
   this bug** (see "Ruled out" below) — kept anyway as free and harmless.
4. `build.py` (this fix) — `fix_lcms2_dylib_collision()`, described below.

## Root cause

`cv2`, `PIL`, `rawpy`, and `imagecodecs` each vendor their own build of
`liblcms2.2.dylib`. PyInstaller's `--collect-all` collapses same-named
dylibs collected from different wheels into one canonical
`Contents/Frameworks/liblcms2.2.dylib`, with the other packages' copies
becoming `@rpath` references pointing at it (in the shipped 0.49.0 build,
that canonical file was a symlink to **rawpy's** copy). PyInstaller gives no
guarantee it keeps the copy a given consumer actually needs.

`imagecodecs`'s `_cms.abi3.so` has `LC_RPATH @loader_path/..`, so
`@rpath/liblcms2.2.dylib` resolves to that canonical file — not to
`imagecodecs`'s own paired copy at
`Contents/Frameworks/imagecodecs/__dot__dylibs/liblcms2.2.dylib`. rawpy's
vendored lcms2 build is missing a symbol `_cms.abi3.so` needs:

```
$ python3 -c "import ctypes; ctypes.CDLL('NegPy.app/Contents/Frameworks/imagecodecs/_cms.abi3.so')"
OSError: dlopen(.../imagecodecs/_cms.abi3.so, 0x0006): Symbol not found: _cmsChannelsOfColorSpace
  Referenced from: <...> .../imagecodecs/_cms.abi3.so
  Expected in:     <...> .../rawpy/__dot__dylibs/liblcms2.2.dylib
```

`imagecodecs`'s lazy resolver (`imagecodecs/imagecodecs.py: __getattr__`)
catches that `ImportError` and re-raises it as a generic
`DelayedImportError`, discarding the real dlopen error — so
`image_processor.py` only ever sees
`could not import name 'cms_transform' from 'imagecodecs'`, with no hint
that the actual problem is a missing symbol in the wrong `liblcms2.2.dylib`.
This is why the symptom looked identical to a codec-unavailable /
code-signing problem: the observable error text is the same regardless of
cause.

Nothing about this is arm64-specific in principle — it's a build-time
dependency collision that could in theory resolve either way on any
platform PyInstaller collects multiple vendored copies of the same dylib
name for. It happened to resolve to the broken copy on this arm64 build and
(per the original bug report) to a working copy on the Intel build, likely
because the wheels involved vendor different lcms2 builds per architecture.

## Fix

`build.py`: `fix_lcms2_dylib_collision()`, run after PyInstaller's build and
before `codesign_macos_app()` (must run first so the corrected symlink is
covered by the final signature). Finds `imagecodecs`'s own bundled
`liblcms2.2.dylib` and repoints the canonical
`Contents/Frameworks/liblcms2.2.dylib` symlink at it, so every consumer's
`@rpath` resolution lands on the copy `_cms.abi3.so` actually needs.

Verified on the M1 Pro used for this investigation:
- `ctypes.CDLL()` on the rebuilt `_cms.abi3.so` now loads without error.
- `codesign --verify --deep --strict` on the rebuilt `.app` passes cleanly.
- The canonical symlink now reads
  `liblcms2.2.dylib -> imagecodecs/__dot__dylibs/liblcms2.2.dylib`.

## What's ruled out (the earlier, incorrect investigation)

The original hypothesis was that AMFI refuses the app's ad-hoc, unnotarized
code signature at `_cms.abi3.so`'s lazy `dlopen`. That was wrong. Kept here
so the dead end isn't re-walked:

- **AMFI log correlation is not evidence of a block.** The
  `Error Domain=AppleMobileFileIntegrityError Code=-423 "adhoc signed or
  unknown certificate chain"` line that appeared at the same timestamp as
  the failure is routine noise logged for *every* ad-hoc-signed file AMFI
  evaluates — including the main `NegPy` executable and
  `libpython3.13.dylib` itself, both of which obviously loaded (the app was
  running). Timestamp correlation only shows the code was evaluated then,
  not refused.
- **A minimal PyInstaller repro (only `imagecodecs`, no `cv2`/`PIL`/`rawpy`)
  never reproduces the failure** — not as a bare ad-hoc binary, not after
  the same `--deep` re-sign `build.py` applies, not launched as a proper
  `.app` via `open` (the LaunchServices path). Its canonical
  `liblcms2.2.dylib` symlink correctly points at `imagecodecs`'s own copy,
  because there's no other package present to collide with — consistent
  with the real cause, not with an AMFI/signing theory.
- **The `com.apple.security.cs.disable-library-validation` entitlement
  (free, ad-hoc-compatible) changed nothing** — identical rejection,
  identical timestamp, after re-signing with it applied. In hindsight this
  is expected regardless of cause: the build never passes
  `--options runtime`, so Library Validation was never enabled on the
  process, making the entitlement a no-op either way.
- Static comparison of the official 0.49.0 arm64 and x86_64 DMGs
  (`gh release download 0.49.0 --repo marcinz606/NegPy`) showed identical
  file layout and `@rpath` structure, both passing
  `codesign --verify --deep --strict` — this was read as "packaging is
  fine," but in hindsight neither of those checks can detect a dylib
  version collision, only a broken signature.
- Dev mode (`make run` / `uv run python -c "from imagecodecs import
  cms_transform"`) never reproduces the bug — correctly identified as
  significant, but for the wrong reason. It's not that the process loading
  it is more trusted; it's that the dev venv has no `cv2`/`PIL`/`rawpy`
  dylib collision at all — `imagecodecs`'s own copy is the only
  `liblcms2.2.dylib` involved, so there's nothing to resolve incorrectly.

**Notarization is not required to fix this bug.** It may still be worth
doing someday for the general Gatekeeper/quarantine download experience,
but that's an unrelated, unevidenced motivation now — not a fix for
anything diagnosed here. [Issue #1](https://github.com/thetalkingdrum/NegPy/issues/1),
which tracked notarization as the fix, has been closed as not applicable.

## Status

Fixed on this branch. Worth a follow-up: check whether other same-named
dylibs vendored by multiple bundled packages (e.g. `libjpeg`, `libpng`,
`libz`, `libtiff` — several are shared across `cv2`, `PIL`, `rawpy`,
`imagecodecs`, `tifffile`) have a similar risk of PyInstaller picking an
incompatible copy. No evidence any of them are currently broken; this is a
speculative risk, not a known bug, and untested — not fixed here.
