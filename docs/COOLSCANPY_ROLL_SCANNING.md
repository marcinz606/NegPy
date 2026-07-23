# Roll scanning with coolscanpy

This page describes NegPy's optional integration with
[coolscanpy](https://github.com/rohanpandula/coolscanpy), a standalone
Python library for whole-roll scanning on a Nikon Coolscan LS-5000 with a
roll feeder. coolscanpy talks to the scanner directly over USB and does not
need SANE for the accepted color-negative path. NegPy consumes coolscanpy as
an optional dependency that the rest of the application never requires.

## What it adds

The plain Scan panel in NegPy scans one frame at a time through SANE. A
roll feeder changes the shape of the problem. You preview the whole roll
in one transport pass, check or nudge each frame's spacing, and then run a
batch fine scan across a list of slots without reloading the film. That
whole-roll workflow is what coolscanpy implements: preview, spacing
correction, approval of any frame the transport could not place
automatically, and batch scanning with per-frame receipts. This
integration exposes all of it inside NegPy, including a desktop panel.

Color-negative roll scanning is the accepted path. Conventional silver
black-and-white fine scanning remains explicitly refused by this pinned
release rather than silently taking a different capture route; the current
sidebar opens rolls as color negative.

## Install

Nothing here is required to run NegPy. The feature is entirely absent
until coolscanpy is importable, and every entry point checks for that
before doing anything else.

If you use uv, sync the new dependency group the same way you already
sync the `scanner` group for python-sane or the `camera` group for
gphoto2:

```
uv sync --group coolscan-roll
```

That group pins the reviewed immutable commit
`e42c68fb53b74aee9fc29eae432d092de351001d`, built on the fixed-size
Nikon frame-table repair and the merged streaming
finalization from [PR #1](https://github.com/rohanpandula/coolscanpy/pull/1).
The pin includes exact USB-topology ownership, six-strip leading-edge
handling, Nikon density/exact-builder evidence, and Digital ICE acquisition
evidence, retained caller-owned attempt evidence, terminal-tail-safe
short-strip mapping, and complete continuation-frame meter-layout receipts
(required for multi-frame Nikon-exact publication and for NegPy's
six-frame capture-evidence validation). Its complete Coolscan suite passed before the pin moved, and
`build.py` independently rejects a stale API or capture-bundle hash before
packaging. The pin can move to a tagged release once those APIs are published.

The local macOS build may use the default ad-hoc code-signing identity (`-`)
for on-machine integrity and testing. That is not publisher identity and does
not make the DMG a notarized release. A distributable build must use a real
Developer ID Application identity through `NEGPY_CODESIGN_IDENTITY`, pass the
same offline frozen smoke, and then complete Apple's notarization and stapling
workflow. `build.py` currently signs and verifies the app bundle but does not
claim to perform that final notarization step.

The hardened Developer ID build carries one runtime exception:
`com.apple.security.cs.allow-unsigned-executable-memory`. Numba's llvmlite
runtime allocates anonymous read/write pages and later changes them to
read/execute; LLVM 20 does not use `MAP_JIT` for that path, so Apple's
`allow-jit` entitlement is not applicable. The build intentionally does not
disable library validation, executable-page protection, or DYLD protections.
PyInstaller applies the entitlement to the initial bundle seal, the libusb
post-processing step reapplies it explicitly, and the build reads the signed
entitlement and hardened-runtime flag back before running the offline smoke.
That smoke explicitly executes llvmlite's anonymous RW-to-RX allocation check,
so a signed build fails before release if the JIT exception is absent.

If you manage your own environment instead, install the package directly:

```
pip install "coolscanpy @ git+https://github.com/rohanpandula/coolscanpy.git@e42c68fb53b74aee9fc29eae432d092de351001d"
```

The accepted color-negative roll workflow has no SANE dependency.
coolscanpy's separate plain single-frame `scan()` path may require SANE; see
the coolscanpy README for that setup.

## Hardware support

coolscanpy, and by extension this integration, has been validated against
one specific setup: a Nikon Super Coolscan 5000 ED (LS-5000) running
firmware 1.03, with an SA-21 roll feeder wired for SA-30 compatibility.
Live hardware validation on 2026-07-18 confirmed USB enumeration, device
open, and a full roll preview with correct spacing and manual-review
flagging. Fine scanning and infrared capture through the roll engine have
not yet been re-run live since coolscanpy was extracted into its own
package. That is the remaining validation step on the coolscanpy side, not
something specific to this NegPy integration.

Every other Coolscan model and every roll feeder other than the SA-21/SA-30
combination above is untested. The transport protocol is not assumed to be
LS-5000-only where it does not have to be, but nothing beyond that one
combination has scanned real film. Windows has never run the roll engine
against hardware.

## Durable live-acceptance evidence

The six-frame command-line acceptance requires `--attempts-root`. Point it at
an existing, empty, non-symlink directory owned by this one run. It must be
different from `--output-dir`, and the run receipt must live outside both
directories. NegPy passes that directory to
`coolscanpy.Device.roll(..., attempts_root=...)`; unlike Coolscan's default
temporary workspace, its worker journals and durable acquisition evidence
survive a successful close and remain available for audit. Fine-scan
`capture.bin` streams are verified scratch: Coolscan hashes and finalizes each
one, then intentionally deletes it after the frame has been consumed.

The final run receipt records these files under `capture_evidence`. NegPy
hashes a complete path/size/SHA-256 inventory, caps its canonical manifest at
128 KiB, and binds it to exactly one completed slots-1-through-6 batch. A
successful binding requires the hashed batch job, the session journal's
`batch_job_sha256`, all six frame journals, every fine-capture hash and byte
count recorded before scratch deletion, all six
durable meter sidecars and parent acknowledgements, the first-frame preview,
transport table, and frame map, exact USB topology, and the sealed
plan/engine/bundle identities. Extra attempt trees, incomplete or inconsistent
frame evidence, mixed sessions, and post-hash mutations are fatal. If evidence
changes during success or failure finalization, the receipt is downgraded
truthfully to `retained: false` instead of preserving an earlier success claim.

This caller-owned attempt tree is scanner-transport evidence for the whole
live run. It is separate from the per-frame movable Digital ICE acquisition
archive described below; neither archive substitutes for the other.

## Structure

The camera capture route (Scanlight light source plus a tethered camera,
under `negpy/infrastructure/capture/` and `negpy/services/capture/`) lives
beside NegPy's single-frame `ScannerBackend` protocol, not inside it,
because it is a different acquisition workflow with its own hardware
lifecycle. Roll scanning follows the same shape rather than sitting inside
the plain-SANE scanner packages:

- `negpy/infrastructure/roll/coolscanpy_roll.py` is the hardware adapter.
  It is the only module that imports coolscanpy, and it translates
  coolscanpy's typed exceptions to plain `RuntimeError`s the way
  `SaneBackend` already does for the plain scanner (see Error handling
  below). `negpy/infrastructure/roll/settings.py` holds the sidebar's
  persisted settings, `RollScanSettings`, mirroring `ScannerSettings` and
  `ScanlightSettings`.
- `negpy/services/roll/service.py` defines `RollScanningService`, which
  orchestrates the coolscanpy device and roll lifecycle and writes results
  to disk across the three output tiers described below. It never imports
  coolscanpy itself, only the adapter above.
- `negpy/infrastructure/roll/repair.py` is the Tier-2 repair engine seam:
  an `available()` check and a `register_engine()` entry point. The
  `fauxice_bridge` registers the portable Digital ICE engine when its pinned
  core runtime is installed. The macOS parity build treats a missing
  registration as a frozen-smoke failure rather than silently disabling
  repair. See "Output tiers" below.
- `negpy/services/roll/positive.py` renders Tier 3. It calls
  `ImageProcessor.run_pipeline`, the same in-memory entry point NegPy's own
  export pipeline uses after decoding a file, directly on a Tier-2 buffer.
- `negpy/services/roll/exact_color.py` is the separate fail-closed seam for
  portable Nikon-exact color. It contains no builder tables or CML4 math.
- `negpy/services/roll/portable_builder.py` is an evidence/replay bridge. It
  applies three pre-F builder LUTs captured and validated together by the
  Stage-3 Windows oracle in the pinned order
  `F[B_c(i)]`, using the 131,072-byte fixed LS5000.md3 post-F table. It
  hashes both the repaired source and computed CML Stage-1 input. It is not
  the final macOS-native per-scan builder.
- `negpy/services/roll/portable_cms.py` is the production adapter for the
  verified DLL-free CML4 Stage-1/Stage-2 evaluator. It loads nine pinned
  binary tables (2,506,760 bytes total), verifies every SHA-256 before use,
  and evaluates full frames in bounded chunks. Its integer evaluator is a
  byte-identical copy of the independently validated oracle source.
- `negpy/desktop/workers/roll_worker.py` defines `RollWorker`, a `QObject`
  moved to its own thread, mirroring `CaptureWorker`. It opens a device's
  roll extension lazily the first time a preview or batch scan names it,
  and holds that reservation open across calls instead of reopening it
  each time.
- `negpy/desktop/view/sidebar/coolscan_roll.py` defines
  `CoolscanRollSidebar`, described below.

## What this integration writes

A batch scan can produce up to three tiers of output per slot, plus one
receipt. Each tier is a separate on/off setting in the sidebar, and any
combination is valid. A new profile enables all three, selects Hybrid repair,
and selects Nikon-exact color; existing saved choices are preserved. See
"Output tiers" below for what each tier is and what happens when a tier cannot
be produced.

The filename pattern is the same Jinja2 template the plain Scan panel
uses, with `date` and `seq` variables. For roll scanning, `seq` is the
frame's physical slot number rather than an incrementing counter. Slot
numbers are already a stable identity for a given roll, so re-scanning one
bad slot overwrites that slot's old files instead of accumulating a second
copy beside them. This holds for every tier.

Only a written Tier 1 (unrepaired) RGB TIFF is handed to NegPy's asset
discovery once a batch scan finishes. Tier 2 and Tier 3 are written to
disk when selected, but neither is opened as a NegPy asset automatically.
The receipt is not opened as an asset either. The scanner-native Digital ICE
prepass and infrared-validity array are not part of that Tier-1 TIFF pair.
They are consumed from the frame-bound acquisition while the scan is being
written and retained in the movable acquisition archive described below.
Hybrid output retains its disclosure mask and verified receipt, but that is
provenance for the result, not a replacement input from which a later repair
can be reconstructed.

### Movable Digital ICE acquisition archive

When Tier 1 includes infrared, NegPy also tries to retain the complete input
set needed to reconstruct that frame's `RepairAcquisition`. The frame receipt
records the result under `outputs.repair_acquisition_evidence`. A successful
retention has `retained: true`, `replayable: true`, and schema
`negpy.dice-acquisition-replay-v1`.

The retained set contains:

- `acquisition-binding.json`, represented by a `binding` object with `path`,
  `bytes`, and `sha256`;
- `prepass.rgbi16.npy` and `ir-validity.npy`, listed under `artifacts` as the
  scanner-native prepass and validity map; and
- the Tier-1 RGB and infrared TIFFs, listed under `sources` in
  `upright-storage` orientation.

Artifact and source rows include relative paths as well as their content and
layout bindings. The binding's `replay.requires` list names all five required
inputs: `storage_rgb_tiff`, `storage_ir_tiff`, `prepass_rgbi`, `ir_validity`,
and `acquisition_provenance`. Replay stacks the upright-storage RGB and
infrared planes, then applies `rot90(k=-1)` to reconstruct the scanner-native
main RGBI array. The loader re-derives the producer acquisition identity and
evidence hash before returning it.

`load_repair_acquisition_evidence` accepts the retained binding file. Relative
source paths make the archive movable when the whole output archive—including
the Tier-1 TIFFs and its hidden evidence directories—is moved together. Moving
only the binding directory, or only the TIFF pair, is incomplete. After a
move, pass `acquisition-binding.json` at its new location; the capture-time
absolute `binding.path` in the outer frame receipt does not relocate itself.
The contract describes its authenticity as `integrity-bound-not-signed`:
hashes and canonical producer bindings detect changed or mismatched inputs,
but do not claim a cryptographic publisher signature.

Retention is fail-soft for the irreplaceable Tier-1 output. If any unique
input cannot be validated or retained, the RGB/IR TIFF pair can still be
published while `retained: false`, `replayable: false`, and an `unavailable`
status explain why repair replay was withheld. That TIFF pair remains useful
as the scanner master, but is still insufficient by itself for parity repair.

## Output tiers

Tier 1 is the unrepaired capture: the scanner-linear RGB and its aligned
infrared plane, exactly as coolscanpy returned them. This is the archival
master, and the only tier the scanner itself can reproduce. It is written
as `<basename>.tif` plus `<basename>_IR.tif`, the same infrared suffix the
plain Scan panel already uses. It defaults on.

Tier 2 is the repaired capture: Tier 1 with infrared-guided dust and
scratch repair applied, still scanner-linear and still a negative. It is
written as `<basename>_repaired.tif` plus `<basename>_repaired_IR.tif`.
That infrared sidecar is Tier 1's own infrared plane, unchanged, not a
repaired version of it. Repair consumes infrared to find defects; it does
not produce a new infrared image. Keeping the original infrared plane is
still useful archival evidence, but it is not sufficient for a later parity
repair without the bound prepass, validity data, and acquisition provenance.

Repair runs in one of two modes. Exact mode uses the pinned portable Digital
ICE runtime over the bound scanner-native main pass, 285-dpi prepass, and
infrared-validity evidence. Hybrid mode additionally routes severe
zero-signal regions, where the infrared channel gives no usable reading, to
the separately pinned inpainting runtime. The receipt records the selected
backend, resolved mode, hashes, and (for hybrid) disclosure evidence. Runtime
depends heavily on backend, frame geometry, and the hybrid routing/model, so
the integration does not promise a fixed time or make a blanket
byte-identical claim for every hybrid environment.

Tier 3 is the positive. The Roll Scanning service and its persisted
**Positive color** sidebar choice default to `nikon-exact` for the Nikon C-41
parity workflow. `negpy-approximate` runs only when explicitly selected and
remains visibly labeled as a preview path: Tier 2 is inverted through NegPy's
own negative-to-positive rendering pipeline, the same pipeline a freshly
imported negative reaches the first time it is opened in NegPy. It is never
claimed or labeled as Nikon Scan parity.

The service also exposes a fail-closed `nikon-exact` mode. NegPy ships two
separate DLL-free adapters: `PortableStage1Builder`, which applies either a
validated Stage-3 replay receipt or a freshly derived native receipt, and
`PortableCMSOnEvaluator` for the two captured CML4 stages. Neither uses a Wine
process, CML4 DLL, scanner, VM, or external repository at runtime.

The native receipt boundary accepts the 97-dpi density result only as an
explicit pair on the returned frame: `Frame.nikon_density_evidence` plus
`Frame.nikon_density_ownership`, mirrored by
`Frame.receipt.nikon_density_ownership`. The ownership receipt binds the same
reservation and batch, preview bytes and preview identity, transport table,
reviewed and fresh registration fingerprints, frame attempt, one-based batch
index, and selected slot. A new preview or reservation, re-registration, film
movement, eject/refeed, changed transport identity, missing field, or mismatch
between the frame and its public receipt makes exact color unavailable. There
is deliberately no generic Roll/session evidence cache that can be promoted
later.

The pinned Coolscan release gives each C-41 frame
`nikon_exact_builder_evidence`, which binds its settled 285-dpi analyzer
raster and final exposure triplet to the same ownership pair. `Frame`
construction revalidates the ownership, density, builder, and Digital ICE
bindings before NegPy can consume them. A clean build verifies that API and
the sealed Coolscan capture-bundle hash before packaging, so a stale pin cannot
silently publish an app that lacks the native exact path. The Stage-3 replay
route remains independently usable with an explicit validated replay receipt.

The Stage-3 replay bridge is deliberately not presented as the production
builder architecture. The macOS-native path derives fresh pre-F LUTs for each
acquisition from coolscanpy's settled, frame-bound meter and analyzer evidence.
Reusing a previously captured Windows pre-F set is useful for evidence-bound
replay and regression testing only.

The builder receipt can only be created by the file-backed trusted loader. It
uses stable, non-symlink reads and requires the complete PASS summary, an empty
error list, the exact 15-file artifact inventory, the pinned LS5000.md3 module
and resource, and complete observer source/executable provenance. The immutable
builder envelope binds the raw Stage-3 PASS report and its
SHA-256, the three 131,072-byte pre-F LUT blobs and their hashes, and the
pinned fixed post-F LUT identity. The builder computes the Stage-1 input only
after Fauxice repair; that future full-frame hash is output evidence in the
builder-application receipt, not something the earlier prescan receipt is
asked to predict. The CMS receipt separately binds that computed Stage-1
input and final output. NegPy independently checks each receipt and content
binding before writing. Missing, malformed, unattested, or tampered evidence
produces an `unavailable` receipt entry; it never falls back to the
approximate renderer under an exact label.

The frozen app and wheel also carry the original 4,014-byte portable-oracle
validation receipt, not only a copied summary of its result. Startup checks it
against SHA-256
`edf6f3f89158810f1de4ce3b4ff8938326bc50e1b3035af59af472258e7d95e8`
and a closed schema before the CMS adapter can exist. The production CMS
receipt derives and binds the verified totals: 12 events, 265,440 active
16-bit values with zero mismatches, and 698,880 full-payload bytes with zero
mismatched bytes.

Native same-acquisition builder evidence is an acquisition artifact, not a
side effect of rendering Tier 3. Whenever a frame carries it, NegPy validates
and stages the canonical density receipt, frame-ownership receipt, analyzer
raster, combined builder-evidence JSON, and three derived pre-F blobs before
attempting repair or the positive. The frame sidecar records that result under
`outputs.native_color_evidence`, including when Tier 3 was not selected or
failed. A successful native exact positive points at that already retained
artifact. A successful replay exact positive instead retains its raw Stage-3
JSON and three pre-F blobs as part of the exact result. Every retained path,
byte size, and SHA-256 is recorded, and the frame receipt is published last.
The transaction rolls back errors observed by the process; it is not described
as a power-loss durability guarantee. These color-builder artifacts do not
replace Digital ICE's bound 285-dpi prepass, infrared-validity data, or repair
provenance, so Tier 1 TIFF and IR files alone still cannot reconstruct a later
parity repair.

A successful exact positive also embeds Nikon Scan's exact 492-byte
`Nikon Adobe RGB 4.0.0.3000` profile in TIFF tag 34675. The source-embedded
profile is checked against SHA-256
`a8d0d753bd6129357cc2647435ce675e8637a679eb526fa180fba460874ce1d3`
before use, and the sidecar binds its name, byte size, and hash. Unrepaired,
repaired, and `negpy-approximate` TIFFs keep their existing untagged behavior.
The exact sidecar also records the finished TIFF's file hash, decoded-pixel
hash, geometry, bit depth, planar layout, and embedded-profile hash after
reopening it. Here “exact Nikon color” means the color-stage RGB values and
ICC identity are exact; it does not claim that NegPy's TIFF compression,
secondary pages, or unrelated application metadata are byte-identical to a
Nikon Scan-written TIFF container.

The positive is written as `<basename>_positive.tif`. Tier 3 always
derives from Tier 2's result in memory, never from a Tier 1 or Tier 2 file
already on disk, and never from Tier 1 directly. Selecting Tier 3 without
Tier 2 still runs repair; the repaired result is just not written to disk
on its own. Tier 3 is enabled in the first-run parity defaults; a saved user
choice to disable it is preserved.

Repair, when it can run, always runs before inversion: capture, then
repair, then invert. That way Tier 3 benefits from whatever Tier 2 was
able to fix, rather than inverting an uncorrected frame.

The first-run parity defaults enable all three tiers, choose Hybrid repair,
and choose Nikon-exact color. Tier 1 remains enabled because it is the archival
scanner output. Repaired and Positive run while the frame-bound 285-dpi
prepass, infrared-validity data, acquisition provenance, and native
color-builder evidence are still available. Existing saved settings continue
to win over these defaults, and users may opt out of derived tiers to reduce
compute or storage. A later Tier-1-only reprocessing job must not invent the
missing evidence or label its result exact.

The macOS parity build includes the pinned portable Digital ICE core and the
bridge auto-registers it. Hybrid additionally needs its separately pinned
companion interpreter/model manifest. `write_frame` still checks registration
and evidence before attempting Tier 2, and records a plain unavailable status
instead of losing lower tiers. Since Tier 3 depends on Tier 2's in-memory
result, it degrades the same way whenever repair is unavailable, even if Tier
2 itself was not selected for writing. Tier 1 still writes when selected,
regardless of Tier-2 or Tier-3 failure.

The receipt records, for every tier, whether it was written, and a status
explaining why not when it was not. A successful Tier 2 write also records
the repair engine's name and version, along with which mode ran. A
successful approximate Tier 3 write records its color-mode label, rendering
path, process mode, render intent, and whether auto exposure was on. A
successful exact write instead records the input/output content hashes, the
bound ICC profile identity, and both embedded receipt payloads plus their
SHA-256 bindings. Every Tier 3
receipt entry also carries the Tier 2 provenance that fed it, so it stays
self-contained even when Tier 2 was not itself written to disk.

Writing every tier is not free of storage cost. At 4000 dpi, one frame's
three tiers together take up roughly half a gigabyte on disk. A long roll
scanned at every tier adds up quickly, so plan storage accordingly.

## Error handling

coolscanpy raises a typed exception hierarchy rooted at
`PyCoolscanError`, covering things like a roll that no longer matches its
last preview, a slot that needs manual approval before it can be scanned,
and a requested safe stop. NegPy's existing scanner code has no typed
exception vocabulary of its own for coolscanpy's own failures;
`SaneBackend` reports every failure as a plain `RuntimeError` with a
human-readable message, and that message is what ends up in a status
label. This integration follows the same convention: every coolscanpy
exception is flattened to a `RuntimeError` at the boundary, with the
original exception preserved as `__cause__` for any caller that wants to
branch on it.

`RollScanningService` itself raises one exception type of its own,
`RollScanningError`, for a lifecycle misuse that has nothing to do with
coolscanpy: calling `open_roll()` twice without closing the first
reservation, or calling `preview()`/`scan_many()` before `open_roll()` at
all. It subclasses `RuntimeError` and mirrors `CaptureError` on the camera
route, which exists for the same reason on that side.

`Roll.safe_stop()` lets the frame already in flight finish and only
refuses the next one, by raising `SafeStopRequested`. `RollWorker`
distinguishes that deliberate stop from a real failure with
`coolscanpy_roll.is_safe_stop()`, which checks whether a caught
`RuntimeError`'s `__cause__` is coolscanpy's `SafeStopRequested`. A stop
reported this way ends the batch scan the same way a cancelled plain scan
already does: quietly, not as an error.

## The integration point

Everything that touches coolscanpy itself lives in one file,
`negpy/infrastructure/roll/coolscanpy_roll.py`. `open_roll()` in that
file is the single place a device handle gets resolved for the roll
workflow. It currently calls `coolscanpy.open()` directly. Everything
else, including `negpy/services/roll/service.py`, the worker, and the
sidebar, only ever talks to the `RollHandle` that function returns.

That module is marked in a comment as the intended target for a future
change. NegPy's maintainer has a generic SANE-based coolscan route planned
upstream, which is expected to add a real backend-selection seam to
`ScannerService` (today `ScannerService._get_backend()` just hardcodes
`SaneBackend()`, with no selection mechanism at all). When that seam
lands, re-pointing this roll integration at it should only require
changing how `open_roll()` resolves a device. The roll workflow built on
top of it, the exception translation in the same file, the whole of
`service.py`, and the worker and sidebar built on top of that, should not
need to change.

## The desktop panel

`CoolscanRollSidebar` lives in the Scan tab as a third collapsible
section, "Roll Scanning", next to "Scanner (SANE)" and "Camera Scanning".
It is built unconditionally and hidden behind a setup hint when
coolscanpy is not importable, the same way `ScanlightSidebar` handles a
missing gphoto2.

The panel covers device selection, a whole-roll preview rendered as a
thumbnail contact sheet, a spacing-offset spinner and an approve button
for whichever slot is selected, and a batch scan of every selected slot
with a progress bar and a Safe Stop button. Select All Frames makes a
full-roll batch explicit instead of requiring every thumbnail to be
clicked individually.

Eject Roll uses the same proven SANE `scanimage --eject` transport action
as NegPy's ordinary scanner panel. It first closes the coolscanpy roll
reservation so the direct SANE command never competes for the scanner.
Successful ejection discards the contact sheet because the film's
registration is no longer valid. If the command returns an error, the
panel deliberately disables preview and further eject attempts for the
rest of that app session: the physical result is uncertain and a blind
retry could move film twice.

The Output section has three checkboxes, one per tier, plus a repair-mode
dropdown that governs Tier 2. Any combination of the three checkboxes is
valid, and at least one must be checked before Scan Selected enables.
Unchecking Unrepaired shows a warning below the checkboxes: Tier 1 is the
only tier the scanner itself can reproduce, so turning it off is a real
tradeoff, not just another setting.

Safe Stop calls
`Roll.safe_stop()` through the worker, which is coolscanpy's own name for
this action and its own contract: the frame already in flight always
finishes, and only the next one is refused. A Scan/Stop toggle button, the
shape the plain Scan and Camera Scanning panels use for their own cancel
action, would imply the click stops the frame in progress. Safe Stop
cannot promise that, so it gets a name and a button that say what it
actually does.

The panel keeps a roll reservation open for the life of the app session,
including across a tab switch. It closes the current reservation only
when the user picks a different device, opening the new one in its place.
Eject Roll also closes it immediately before handing transport control to
SANE. There is no separate close-without-eject button. This differs from
the camera route, where the tethered session is deliberately released
once neither the live-view window nor the calibration window is open,
because some camera bodies get stuck in a tethered-capture state if a
session is left open past the window that used it. coolscanpy's roll
reservation has no equivalent failure mode, so there is nothing to
protect against by closing it early.

The panel does not yet expose a material picker and opens a roll as color
negative. Coolscanpy's B&W batch route is implemented and covered offline,
but its first live macOS validation with conventional silver B&W film remains
outstanding. Exposing that choice in NegPy waits on that hardware acceptance,
not on a missing fine-scan implementation.

A contact sheet thumbnail is a raw scanner preview frame, not a NegPy
pipeline buffer, so it is converted to a displayable image with a small
local helper rather than going through NegPy's color-managed
`ImageConverter`. That converter assumes a working-space buffer a
scanner's preview array is not.
