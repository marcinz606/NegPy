# Roll scanning with coolscanpy

This page describes NegPy's optional integration with
[coolscanpy](https://github.com/rohanpandula/coolscanpy), a standalone
Python library for whole-roll scanning on a Nikon Coolscan LS-5000 with a
roll feeder. coolscanpy talks to the scanner directly over USB and does not
need SANE installed. NegPy consumes it the same way it consumes
python-sane or gphoto2 elsewhere in this package: as an optional dependency
that the rest of the application never requires.

## What it adds

The plain Scan panel in NegPy scans one frame at a time through SANE. A
roll feeder changes the shape of the problem. You preview the whole roll
in one transport pass, check or nudge each frame's spacing, and then run a
batch fine scan across a list of slots without reloading the film. That
whole-roll workflow is what coolscanpy implements: preview, spacing
correction, approval of any frame the transport could not place
automatically, and batch scanning with per-frame receipts. This
integration exposes all of it inside NegPy, including a desktop panel.

Coverage is narrower than the plain Scan panel. Only color negative film
scans through coolscanpy's roll engine end to end today. Black and white
negative previews and approves normally, but batch fine scanning it raises
`NotImplementedError` inside coolscanpy itself. That limitation is
documented in the coolscanpy README and is unchanged by this integration.

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

If you manage your own environment instead, install the package directly:

```
pip install coolscanpy
```

coolscanpy has no SANE dependency for the roll workflow. See the
coolscanpy README if you also want its plain single-frame `scan()` path,
which does need SANE.

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
  an `available()` check and a `register_engine()` entry point, with no
  engine registered by default. See "Output tiers" below.
- `negpy/services/roll/positive.py` renders Tier 3. It calls
  `ImageProcessor.run_pipeline`, the same in-memory entry point NegPy's own
  export pipeline uses after decoding a file, directly on a Tier-2 buffer.
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
combination is valid. See "Output tiers" below for what each tier is, why
the defaults are set the way they are, and what happens when a tier cannot
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
The receipt is not opened as an asset either. The infrared confidence mask
coolscanpy also returns per frame is not written to disk at all. Nothing
downstream currently reads it, and adding a file format for it before
there is a consumer would be guessing at a convention rather than
following one.

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
not produce a new infrared image. Keeping the original lets a later repair
pass, run under a different mode, start from the same evidence.

Repair runs in one of two modes. Exact mode heals only the pixels the
infrared channel confidently flags as a defect. Hybrid mode additionally
routes severe zero-signal regions, places where the infrared channel gives
no usable reading at all, to an inpainting model. Both modes are
deterministic: the same frame and the same mode always repair to the same
result. That is what makes Tier 2 worth caching. Once a frame has been
repaired under a given mode, repairing it again produces an identical
file, so there is no need to redo the work. Exact mode is expected to take
about 10 seconds per frame. Hybrid mode is expected to take 70 to 210
seconds per frame, since inpainting is far slower than the exact-match
heal.

Tier 3 is the positive: Tier 2 inverted through NegPy's own
negative-to-positive rendering pipeline, the same pipeline a freshly
imported negative reaches the first time it is opened in NegPy. It is
written as `<basename>_positive.tif`. Tier 3 always derives from Tier 2's
result in memory, never from a Tier 1 or Tier 2 file already on disk, and
never from Tier 1 directly. Selecting Tier 3 without Tier 2 still runs
repair; the repaired result is just not written to disk on its own. Tier 3
defaults off.

Repair, when it can run, always runs before inversion: capture, then
repair, then invert. That way Tier 3 benefits from whatever Tier 2 was
able to fix, rather than inverting an uncorrected frame.

The defaults follow from an asymmetry between the tiers. Tier 1 is the
only tier the scanner can produce, so losing it is permanent. Tier 2 is
expensive to compute but, once computed under a given mode, never needs
recomputing. Tier 3 is cheap to compute, and NegPy's color rendering is
still being tuned, so a Tier 3 file written today is expected to look
different from what the same negative renders to later. Treat a Tier 3
file as a current preview, not as a finished edit. Tier 1 defaults on
because turning it off risks losing data that only the scanner can
reproduce. Tier 2 and Tier 3 default off because both can be regenerated
from Tier 1 at any time, once a repair engine is registered.

No repair engine ships with this integration today. `write_frame` checks
whether one is registered before attempting Tier 2, and records a plain
status in the receipt instead of raising when none is available. Since
Tier 3 always depends on Tier 2's result, Tier 3 degrades the same way
whenever repair is unavailable, even if Tier 2 itself was not selected for
writing. Tier 1 still writes when selected, regardless of what happens to
Tier 2 or Tier 3. If NegPy's own rendering pipeline fails for any reason,
Tier 3 degrades on its own, and Tier 1 and Tier 2 are unaffected.

The receipt records, for every tier, whether it was written, and a status
explaining why not when it was not. A successful Tier 2 write also records
the repair engine's name and version, along with which mode ran. A
successful Tier 3 write also records which rendering path produced it. It
records the process mode and render intent that were used, and whether
auto exposure was on, so a later regeneration or audit has something
concrete to compare against. A Tier 3 receipt entry also carries the Tier
2 provenance that fed it, so it stays self-contained even when Tier 2 was
not itself written to disk.

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
with a progress bar and a Safe Stop button.

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
There is no separate close button. This differs from the camera route,
where the tethered session is deliberately released once neither the
live-view window nor the calibration window is open, because some camera
bodies get stuck in a tethered-capture state if a session is left open
past the window that used it. coolscanpy's roll reservation has no
equivalent failure mode, so there is nothing to protect against by
closing it early.

The panel does not expose a material picker. It always opens a roll as
color negative, matching the one material coolscanpy's roll engine
fine-scans end to end today. A black and white picker would only ever
support preview, not the batch scan, as the coverage note above already
explains, so adding one now would be a control with no working action
behind it.

A contact sheet thumbnail is a raw scanner preview frame, not a NegPy
pipeline buffer, so it is converted to a displayable image with a small
local helper rather than going through NegPy's color-managed
`ImageConverter`. That converter assumes a working-space buffer a
scanner's preview array is not.
