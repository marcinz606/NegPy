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

Nothing about this is arm64-specific. First checked the x86_64 *wheels*
(`uv pip install --target ... --python-platform x86_64-apple-darwin`, no
Intel hardware needed — pure download/inspection): rawpy's x86_64
`liblcms2.2.dylib` is **also** missing `_cmsChannelsOfColorSpace`; PIL's
and cv2's x86_64 copies have it. That shows the *risk* exists on both
architectures, but not what the real shipped Intel build actually did —
inferring "Intel got lucky" from wheel contents alone would have been the
same shape of guess that cost this investigation twice already, so
checked the real thing instead: downloaded the official 0.49.0
arm64 *and* x86_64 DMGs (`gh release download 0.49.0 --repo
marcinz606/NegPy`) and inspected each one's actual canonical symlink.

- Shipped arm64: `Contents/Frameworks/liblcms2.2.dylib` →
  `rawpy/.dylibs/liblcms2.2.dylib` — missing the symbol. Matches the
  reported M1 failure exactly.
- Shipped x86_64: `Contents/Frameworks/liblcms2.2.dylib` →
  `imagecodecs/.dylibs/liblcms2.2.dylib` — imagecodecs' own copy, has the
  symbol. Matches the reported "works on Intel" exactly.

Confirmed, not inferred: the original bug report was accurate on both
counts, and the reason is exactly what it looks like — PyInstaller's
same-basename collision resolution picked a different copy per
architecture, and only the arm64 pick happened to be broken. The
underlying risk (rawpy's copy is incomplete) is architecture-independent;
whether it bites you is down to build-time pick order, which is not
something to rely on.

## Fix

`build.py`: `fix_lcms2_dylib_collision()`, run after PyInstaller's build and
before `codesign_macos_app()` (must run first so the corrected symlink is
covered by the final signature). Finds `imagecodecs`'s own bundled
`liblcms2.2.dylib` and repoints the canonical
`Contents/Frameworks/liblcms2.2.dylib` symlink at it, so every consumer's
`@rpath` resolution lands on the copy `_cms.abi3.so` actually needs.

Verified on the M1 Pro used for this investigation, across two rebuilds
(the initial fix, then the hardened version below):
- `ctypes.CDLL()` on the rebuilt `_cms.abi3.so` now loads without error.
- `codesign --verify --deep --strict` on the rebuilt `.app` passes cleanly.
- The canonical symlink now reads
  `liblcms2.2.dylib -> imagecodecs/__dot__dylibs/liblcms2.2.dylib`.
- End-to-end, once per rebuild: force-quit any running instance, relaunched
  the freshly rebuilt app, re-ran the export. `negpy.log` shows no fallback
  warning and no error both times — the real lcms2 transform runs, not the
  LUT approximation.

`fix_lcms2_dylib_collision()` also fails the build (`raise`, not a
`print`+`return`) if imagecodecs' own copy can't be found unambiguously, or
verifies — and fails the build on — the `_LCMS2_CONSUMER_GLOBS` symbol
closure described below. A follow-up review pointed out the first version
of this fix only warned and shipped anyway on a miss, which is the same
silent-failure shape as the bug itself.

## Did repointing the symlink break the other consumers?

Four packages vendor their own `liblcms2.2.dylib` copy (`cv2`, `PIL`,
`rawpy`, `imagecodecs`), and the canonical symlink they all resolve through
is shared — before this fix it pointed at rawpy's copy, so `cv2`'s and
`PIL`'s own dylibs were also getting rawpy's build. Repointing it at
imagecodecs' copy could in principle have fixed one consumer by breaking
another, on the unverified assumption that imagecodecs' lcms2 build is a
strict superset. Checked directly rather than assumed:

- Found every real (non-symlink) `.so`/`.dylib` in the bundle that links
  `liblcms2.2.dylib` (`otool -L`): 7 files —
  `PIL/_imagingcms.cpython-313-darwin.so`; `rawpy/libraw_r.dylib`,
  `rawpy/libraw_r.25.dylib`, and `rawpy/libraw_r.25.0.0.dylib` (three
  filenames for the same versioned library, PyInstaller duplicates rather
  than symlinks them); `imagecodecs/_cms.abi3.so` and
  `imagecodecs/_jpeg2k.abi3.so`; and `cv2`'s own `libjxl_cms.0.11.1.dylib`
  (the top-level copy of that name is a symlink to this one).
- For each, diffed its undefined `_cms*` symbols (`nm -u`) against the new
  canonical's exports (`nm -gU`): **no gaps for any of the 7.**
- `LC_ID_DYLIB` compatibility versions: all four original copies, and every
  consumer's recorded `LC_LOAD_DYLIB` requirement, declare compatibility
  version `3.0.0` (only `current version` differs — lcms2 keeps its ABI
  compatibility version fixed across releases). No `dyld` version-based
  rejection risk independent of symbols.
- Runtime `ctypes.CDLL()` on `PIL/_imagingcms*.so`, `rawpy/libraw_r.dylib`,
  and `imagecodecs/_jpeg2k.abi3.so` post-fix: all load without error.

This symbol-closure check is now permanent, in `fix_lcms2_dylib_collision()`
itself (`_LCMS2_CONSUMER_GLOBS`), so a future imagecodecs/lcms2 version that
*isn't* a superset fails the build instead of shipping a silently broken
consumer.

## Why the smoke test is scoped to lcms2, not every bundled dylib

Considered a general post-build check — `ctypes.CDLL()` every `.so`/`.dylib`
in the bundle, fail on any error — as a durable catch-all for this whole bug
class (`libjpeg`/`libpng`/`libz`/`libtiff` are vendored by multiple of the
same packages and carry the same theoretical risk). Prototyped it and found
a real, pre-existing, unrelated failure:
`numba/np/ufunc/omppool.cpython-313-darwin.so` fails to load —
`Library not loaded: @rpath/libomp.dylib` — because PyInstaller doesn't
bundle `libomp.dylib` at all (visible as a build-time warning independent of
this investigation: "Library not found: could not resolve
'@rpath/libomp.dylib'"). numba's own OpenMP-parallel pool is optional and
presumably has a runtime fallback; this is not evidenced to affect NegPy.
Also worth noting: this check must run via `uv run python` — the project's
own venv interpreter, the exact one PyInstaller freezes — not a bare system
`python3`, which false-positives on legitimate CPython extensions like
numpy's `_multiarray_umath` (missing Python C-API symbols that only resolve
against a matching `libpython`).

A blanket check would have made this PR fail on that pre-existing,
unrelated gap, which is out of scope here. Scoped the hardening to the
`liblcms2.2.dylib` consumers this fix actually touches instead. The general
version — enumerate every same-basename dylib collision with differing
content hashes, symbol-closure-check every consumer, fail the build on any
gap — is still worth building, and would have caught the `libomp.dylib` gap
too, but as its own follow-up, not bundled into this fix.

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

**Notarization is not required to fix this bug.** [Issue #1](https://github.com/thetalkingdrum/NegPy/issues/1),
which tracked notarization as the fix, has been closed as not applicable.
The unnotarized `.dmg` still triggers Gatekeeper's "unidentified developer"
warning on download for end users, which is a real, separate, unfixed UX
cost — tracked on its own as
[issue #2](https://github.com/thetalkingdrum/NegPy/issues/2) so it doesn't
get folded back into a future bug investigation the way it was here.

## Status

Fixed and hardened on this branch:
- `fix_lcms2_dylib_collision()` repoints the canonical symlink, verifies
  symbol closure for every known consumer, and raises (doesn't warn) on any
  problem — including an ambiguous or missing imagecodecs copy.
- `image_processor.py`'s `ImportError` handler now probes the codec path
  with `ctypes.CDLL()` and logs the real `OSError` before falling back, so
  a future regression in this area is diagnosable from `negpy.log` alone
  instead of requiring a multi-day investigation like this one.

Follow-up, done: `check_bundled_dylib_collisions()` generalizes the
symbol-closure check to every same-basename dylib collision under
`Contents/Frameworks`, not just liblcms2. It scans the whole bundle rather
than a curated list of risky basenames — a manual pass guessing which
libraries might collide (by comparing vendored version strings across
`cv2`/`PIL`/`rawpy`/`imagecodecs`) missed a real one: three packages vendor
byte-different copies of `libjpeg.8.3.2.dylib`, invisible unless you check
for repeated exact filenames rather than reasoning about version numbers.
Also found real (currently benign — no missing symbols today) collisions on
`libpng16.16.dylib` and `libtiff.6.dylib` (`cv2` vs `PIL`). Confirmed on the
real local build: passes clean, and correctly raises when a canonical
symlink is forced to point at a copy missing a symbol a sibling consumer
needs.

Still not done, deliberately: the `libomp.dylib`/numba gap. It's a
different failure mode (PyInstaller not bundling the library at all, not a
collision between multiple bundled copies), stays outside what this check
looks for, and there is still no evidence it affects NegPy (numba's
OpenMP-parallel pool is optional with a runtime fallback) — not worth
fixing speculatively.
