# liblcms2.2.dylib collision breaks ICC export in the packaged macOS app

**Root cause confirmed and fixed 2026-08-15.** This started as an investigation into an
apparent AMFI / code-signing / notarization problem on Apple Silicon. See the git history of
this file for that earlier, incorrect diagnosis. The real cause is a mundane PyInstaller
dylib de-duplication bug, now fixed in `build.py`.

## Symptom

- Flat and Print exports with Narrowband Scan on, or with any custom Input/Output ICC
  profile, silently ship without color management on some Apple Silicon Macs. Toggling the
  profile makes no difference to the exported file.
- `negpy.log` (`~/Documents/NegPy/negpy.log`) shows:
  `CMS transformation failed: could not import name 'cms_transform' from 'imagecodecs'`
- Confirmed on the official 0.49.0 release. Same frame, same version: it fails on an M1
  MacBook Pro and works on an Intel Mac Mini (2018).

## What is already fixed on this branch

1. `negpy/desktop/controller.py` (`55d3fc8`). `request_export` and `request_batch_export`
   resolved Narrowband's implicit ICC profile using the pre-flatten (Print) `render_intent`,
   so the ICC suppression of an E-6 transparency transfer with Normalize off leaked into
   Flat exports. Now resolved after `flat_master_config()`. **A separate, real bug**,
   unrelated to everything below.
2. `negpy/services/rendering/image_processor.py` (`befdfd2`).
   `_apply_color_management_u16` (`image_processor.py:1248`) now catches the specific
   `ImportError` (`imagecodecs.DelayedImportError`) raised when `cms_transform` is
   unavailable, and falls back to the existing PIL/LUT `_apply_color_management_u16_rgb` at
   a 65³ grid instead of silently doing nothing. **Worth keeping regardless.** A real
   fallback beats a silent no-op, in case something else ever makes the codec unavailable.
3. `build.py` (`571893a`). A `codesign --force --deep -s -` pass over the whole `.app` after
   PyInstaller builds it. **Confirmed to have no effect on this bug** (see "Ruled out"
   below). Kept anyway as free and harmless.
4. `build.py` (this fix). `fix_lcms2_dylib_collision()`, described below.

## Root cause

`cv2`, `PIL`, `rawpy` and `imagecodecs` each vendor their own build of `liblcms2.2.dylib`.
PyInstaller's `--collect-all` collapses same-named dylibs from different wheels into one
canonical `Contents/Frameworks/liblcms2.2.dylib`. The other packages' copies become
`@rpath` references that point at it. In the shipped 0.49.0 build that canonical file was a
symlink to **rawpy's** copy. PyInstaller gives no guarantee that it keeps the copy a given
consumer needs.

`imagecodecs`'s `_cms.abi3.so` has `LC_RPATH @loader_path/..`, so `@rpath/liblcms2.2.dylib`
resolves to that canonical file, not to `imagecodecs`'s own paired copy at
`Contents/Frameworks/imagecodecs/__dot__dylibs/liblcms2.2.dylib`. rawpy's vendored lcms2
build is missing a symbol that `_cms.abi3.so` needs:

```
$ python3 -c "import ctypes; ctypes.CDLL('NegPy.app/Contents/Frameworks/imagecodecs/_cms.abi3.so')"
OSError: dlopen(.../imagecodecs/_cms.abi3.so, 0x0006): Symbol not found: _cmsChannelsOfColorSpace
  Referenced from: <...> .../imagecodecs/_cms.abi3.so
  Expected in:     <...> .../rawpy/__dot__dylibs/liblcms2.2.dylib
```

`imagecodecs`'s lazy resolver (`imagecodecs/imagecodecs.py: __getattr__`) catches that
`ImportError` and re-raises it as a generic `DelayedImportError`, which discards the real
dlopen error. `image_processor.py` therefore sees only
`could not import name 'cms_transform' from 'imagecodecs'`, with no hint that the problem is
a missing symbol in the wrong `liblcms2.2.dylib`. That is why the symptom looked identical
to a codec-unavailable or code-signing problem: the error text is the same whatever the
cause.

Nothing about this is arm64-specific. The x86_64 *wheels* were checked first
(`uv pip install --target ... --python-platform x86_64-apple-darwin`, pure download and
inspection, no Intel hardware needed). rawpy's x86_64 `liblcms2.2.dylib` is **also** missing
`_cmsChannelsOfColorSpace`; PIL's and cv2's x86_64 copies have it. That shows the *risk*
exists on both architectures, but not what the real shipped Intel build did. Inferring
"Intel got lucky" from wheel contents alone would have been the same shape of guess that had
already cost this investigation twice, so the real artefacts were checked instead: the
official 0.49.0 arm64 and x86_64 DMGs (`gh release download 0.49.0 --repo
marcinz606/NegPy`), each one's actual canonical symlink.

- Shipped arm64: `Contents/Frameworks/liblcms2.2.dylib` → `rawpy/.dylibs/liblcms2.2.dylib`,
  which is missing the symbol. This matches the reported M1 failure exactly.
- Shipped x86_64: `Contents/Frameworks/liblcms2.2.dylib` →
  `imagecodecs/.dylibs/liblcms2.2.dylib`, imagecodecs' own copy, which has the symbol. This
  matches the reported "works on Intel" exactly.

Confirmed, not inferred. The original bug report was accurate on both counts, and the reason
is what it looks like: PyInstaller's same-basename collision resolution picked a different
copy per architecture, and only the arm64 pick was broken. The underlying risk, that rawpy's
copy is incomplete, is architecture-independent. Whether it bites you is down to build-time
pick order, which is not something to rely on.

## Fix

`build.py`: `fix_lcms2_dylib_collision()`, run after PyInstaller's build and before
`codesign_macos_app()`. It must run first so the final signature covers the corrected
symlink. It finds imagecodecs' own bundled `liblcms2.2.dylib` and repoints the canonical
`Contents/Frameworks/liblcms2.2.dylib` symlink at it, so every consumer's `@rpath`
resolution lands on the copy `_cms.abi3.so` needs.

Verified on the M1 Pro used for this investigation, across two rebuilds (the initial fix,
then the hardened version below):

- `ctypes.CDLL()` on the rebuilt `_cms.abi3.so` loads without error.
- `codesign --verify --deep --strict` on the rebuilt `.app` passes cleanly.
- The canonical symlink now reads `liblcms2.2.dylib -> imagecodecs/__dot__dylibs/liblcms2.2.dylib`.
- End to end, once per rebuild: force-quit any running instance, relaunch the freshly
  rebuilt app, re-run the export. `negpy.log` shows no fallback warning and no error both
  times, so the real lcms2 transform runs, not the LUT approximation.

`fix_lcms2_dylib_collision()` fails the build (`raise`, not `print` and `return`) if
imagecodecs' own copy cannot be found unambiguously. It also verifies the
`_LCMS2_CONSUMER_GLOBS` symbol closure described below, and fails the build on a gap. A
follow-up review pointed out that the first version of this fix only warned and shipped
anyway on a miss, which is the same silent-failure shape as the bug itself.

## Did repointing the symlink break the other consumers?

Four packages vendor their own `liblcms2.2.dylib` copy (`cv2`, `PIL`, `rawpy`,
`imagecodecs`), and they all resolve through the shared canonical symlink. Before this fix
it pointed at rawpy's copy, so `cv2` and `PIL` were also getting rawpy's build. Repointing
it at imagecodecs' copy could in principle have fixed one consumer by breaking another, on
the unverified assumption that imagecodecs' lcms2 build is a strict superset. That was
checked directly:

- Every real (non-symlink) `.so` and `.dylib` in the bundle that links `liblcms2.2.dylib`
  (`otool -L`): 7 files. `PIL/_imagingcms.cpython-313-darwin.so`; `rawpy/libraw_r.dylib`,
  `rawpy/libraw_r.25.dylib` and `rawpy/libraw_r.25.0.0.dylib` (three filenames for the same
  versioned library, which PyInstaller duplicates rather than symlinks);
  `imagecodecs/_cms.abi3.so` and `imagecodecs/_jpeg2k.abi3.so`; and cv2's own
  `libjxl_cms.0.11.1.dylib`, of which the top-level copy is a symlink.
- For each, its undefined `_cms*` symbols (`nm -u`) were diffed against the new canonical's
  exports (`nm -gU`): **no gaps for any of the 7.**
- `LC_ID_DYLIB` compatibility versions: all four original copies, and every consumer's
  recorded `LC_LOAD_DYLIB` requirement, declare compatibility version `3.0.0`. Only
  `current version` differs, because lcms2 keeps its ABI compatibility version fixed across
  releases. There is no `dyld` version-based rejection risk independent of symbols.
- Runtime `ctypes.CDLL()` after the fix on `PIL/_imagingcms*.so`, `rawpy/libraw_r.dylib` and
  `imagecodecs/_jpeg2k.abi3.so`: all load without error.

This symbol-closure check is now permanent, inside `fix_lcms2_dylib_collision()` itself
(`_LCMS2_CONSUMER_GLOBS`). A future imagecodecs or lcms2 version that is *not* a superset
fails the build instead of shipping a silently broken consumer.

## Why the smoke test is scoped to lcms2, not every bundled dylib

A general post-build check was considered: `ctypes.CDLL()` every `.so` and `.dylib` in the
bundle, and fail on any error. It would be a durable catch-all for this whole bug class,
since `libjpeg`, `libpng`, `libz` and `libtiff` are vendored by several of the same packages
and carry the same theoretical risk. The prototype found a real, pre-existing, unrelated
failure: `numba/np/ufunc/omppool.cpython-313-darwin.so` fails to load with
`Library not loaded: @rpath/libomp.dylib`, because PyInstaller does not bundle
`libomp.dylib` at all. It is visible as a build-time warning independent of this
investigation: "Library not found: could not resolve '@rpath/libomp.dylib'". numba's own
OpenMP-parallel pool is optional and presumably has a runtime fallback, and there is no
evidence it affects NegPy.

Note also that this check must run through `uv run python`, the project's own venv
interpreter and the exact one PyInstaller freezes. A bare system `python3` false-positives
on legitimate CPython extensions such as numpy's `_multiarray_umath`, whose missing Python
C-API symbols only resolve against a matching `libpython`.

A blanket check would have made this PR fail on that pre-existing, unrelated gap, which is
out of scope here. The hardening is therefore scoped to the `liblcms2.2.dylib` consumers
this fix touches. The general version, which enumerates every same-basename dylib collision
with differing content hashes, symbol-closure-checks every consumer and fails the build on
any gap, is still worth building and would have caught the `libomp.dylib` gap too. It is a
follow-up, not part of this fix.

## What is ruled out (the earlier, incorrect investigation)

The original hypothesis was that AMFI refuses the app's ad-hoc, unnotarized code signature
at `_cms.abi3.so`'s lazy `dlopen`. That was wrong. It is kept here so the dead end is not
re-walked.

- **AMFI log correlation is not evidence of a block.** The
  `Error Domain=AppleMobileFileIntegrityError Code=-423 "adhoc signed or unknown certificate
  chain"` line that appeared at the same timestamp as the failure is routine noise. AMFI
  logs it for *every* ad-hoc-signed file it evaluates, including the main `NegPy` executable
  and `libpython3.13.dylib`, both of which obviously loaded, since the app was running.
  Timestamp correlation only shows that the code was evaluated then, not that it was
  refused.
- **A minimal PyInstaller repro never reproduces the failure.** With only `imagecodecs` and
  no `cv2`, `PIL` or `rawpy`, it fails neither as a bare ad-hoc binary, nor after the same
  `--deep` re-sign `build.py` applies, nor launched as a proper `.app` through `open` (the
  LaunchServices path). Its canonical `liblcms2.2.dylib` symlink correctly points at
  imagecodecs' own copy, because no other package is present to collide with. That is
  consistent with the real cause, not with an AMFI or signing theory.
- **The `com.apple.security.cs.disable-library-validation` entitlement changed nothing.**
  It is free and ad-hoc-compatible, and re-signing with it applied gave an identical
  rejection at an identical timestamp. In hindsight this is expected whatever the cause: the
  build never passes `--options runtime`, so Library Validation was never enabled on the
  process, and the entitlement is a no-op either way.
- **Static comparison of the official DMGs was read wrongly.** The 0.49.0 arm64 and x86_64
  DMGs (`gh release download 0.49.0 --repo marcinz606/NegPy`) showed identical file layout
  and `@rpath` structure, and both passed `codesign --verify --deep --strict`. That was read
  as "packaging is fine". Neither check can detect a dylib version collision, only a broken
  signature.
- **Dev mode never reproduces the bug**, and that was correctly identified as significant,
  but for the wrong reason. `make run` and
  `uv run python -c "from imagecodecs import cms_transform"` work not because the loading
  process is more trusted, but because the dev venv has no `cv2`/`PIL`/`rawpy` dylib
  collision at all. imagecodecs' own copy is the only `liblcms2.2.dylib` involved, so there
  is nothing to resolve incorrectly.

**Notarization is not required to fix this bug.**
[Issue #1](https://github.com/thetalkingdrum/NegPy/issues/1), which tracked notarization as
the fix, is closed as not applicable. The unnotarized `.dmg` still triggers Gatekeeper's
"unidentified developer" warning on download, which is a real, separate, unfixed UX cost. It
is tracked on its own as [issue #2](https://github.com/thetalkingdrum/NegPy/issues/2), so
that it does not get folded back into a future bug investigation the way it was here.

## Status

Fixed and hardened on this branch:

- `fix_lcms2_dylib_collision()` repoints the canonical symlink, verifies symbol closure for
  every known consumer, and raises rather than warns on any problem, including an ambiguous
  or missing imagecodecs copy.
- `image_processor.py`'s `ImportError` handler probes the codec path with `ctypes.CDLL()`
  and logs the real `OSError` before it falls back, so a future regression here is
  diagnosable from `negpy.log` alone instead of costing another multi-day investigation.

Follow-up, done: `check_bundled_dylib_collisions()` generalizes the symbol-closure check to
every same-basename dylib collision under `Contents/Frameworks`, not just liblcms2. It scans
the whole bundle rather than a curated list of risky basenames. A manual pass that guessed
which libraries might collide, by comparing vendored version strings across `cv2`, `PIL`,
`rawpy` and `imagecodecs`, missed a real one: three packages vendor byte-different copies of
`libjpeg.8.3.2.dylib`, which is invisible unless you check for repeated exact filenames
instead of reasoning about version numbers. The scan also found real collisions on
`libpng16.16.dylib` and `libtiff.6.dylib` (`cv2` against `PIL`), benign today because no
symbols are missing. Confirmed on the real local build: it passes clean, and it correctly
raises when a canonical symlink is forced to point at a copy missing a symbol a sibling
consumer needs.

Still not done, deliberately: the `libomp.dylib` and numba gap. It is a different failure
mode, PyInstaller not bundling the library at all rather than a collision between bundled
copies, so it stays outside what this check looks for. There is still no evidence it affects
NegPy, since numba's OpenMP-parallel pool is optional with a runtime fallback. It is not
worth fixing speculatively.
