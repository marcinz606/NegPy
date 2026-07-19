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
  to disk. It never imports coolscanpy itself, only the adapter above.
- `negpy/desktop/workers/roll_worker.py` defines `RollWorker`, a `QObject`
  moved to its own thread, mirroring `CaptureWorker`. It opens a device's
  roll extension lazily the first time a preview or batch scan names it,
  and holds that reservation open across calls instead of reopening it
  each time.
- `negpy/desktop/view/sidebar/coolscan_roll.py` defines
  `CoolscanRollSidebar`, described below.

## What this integration writes

A batch scan produces, per slot, three files in the configured output
folder:

- a 16-bit TIFF holding the scanner-linear RGB image
- a 16-bit TIFF holding the infrared plane, named with an `_IR` suffix
  next to the RGB file, matching the suffix NegPy's plain scan writer
  already uses for infrared
- a JSON file with a `_receipt` suffix, holding the full scan receipt
  coolscanpy returns for that frame: exposure, clipping and focus
  telemetry, transport-smear assessment, and the fingerprints the frame was
  checked against

The filename pattern is the same Jinja2 template the plain Scan panel
uses, with `date` and `seq` variables. For roll scanning, `seq` is the
frame's physical slot number rather than an incrementing counter. Slot
numbers are already a stable identity for a given roll, so re-scanning one
bad slot overwrites that slot's old files instead of accumulating a second
copy beside them.

Only the RGB TIFF is handed to NegPy's asset discovery once a batch scan
finishes. The IR sidecar and the receipt are written but never opened as
an asset. The infrared confidence mask coolscanpy also returns per frame
is not written to disk at all. Nothing downstream currently reads it, and
adding a file format for it before there is a consumer would be guessing
at a convention rather than following one.

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
with a progress bar and a Safe Stop button. Safe Stop calls
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
