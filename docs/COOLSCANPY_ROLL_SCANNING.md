# Roll scanning with coolscanpy

This page describes NegPy's optional integration with
[coolscanpy](https://github.com/rohanpandula/coolscanpy), a standalone
Python library for whole-roll scanning on a Nikon Coolscan LS-5000 with a
roll feeder. coolscanpy talks to the scanner directly over USB and does not
need SANE installed. NegPy consumes it the same way it consumes
python-sane or gphoto2: as an optional dependency that the rest of the
application never requires.

## What it adds

The plain Scan panel in NegPy scans one frame at a time through SANE. A
roll feeder changes the shape of the problem: you preview the whole roll in
one transport pass, check or nudge each frame's spacing, and then run a
batch fine scan across a list of slots without reloading the film. That
whole-roll workflow, preview, spacing correction, approval of any frame the
transport could not place automatically, and batch scanning with per-frame
receipts, is what coolscanpy implements and what this integration exposes
inside NegPy.

Coverage is narrower than the plain Scan panel. Only color negative film
scans through coolscanpy's roll engine end to end today. Black and white
negative previews and approves normally, but batch fine scanning it raises
`NotImplementedError` inside coolscanpy itself; that limitation is
documented in the coolscanpy README and is unchanged by this integration.

## Install

Nothing here is required to run NegPy. The feature is entirely absent
until coolscanpy is importable, and every entry point checks for that
before doing anything else.

If you use uv, sync the new dependency group the same way you already
sync the `scanner` group for python-sane:

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
package; that is the remaining validation step on the coolscanpy side, not
something specific to this NegPy integration.

Every other Coolscan model and every roll feeder other than the SA-21/SA-30
combination above is untested. The transport protocol is not assumed to be
LS-5000-only where it does not have to be, but nothing beyond that one
combination has scanned real film. Windows has never run the roll engine
against hardware.

## What this integration writes

A batch scan produces, per slot, three files in the configured output
folder:

- a 16-bit TIFF holding the scanner-linear RGB image
- a 16-bit TIFF holding the infrared plane, named with an `_IR` suffix
  next to the RGB file, matching the suffix NegPy's plain scan writer
  already uses for infrared
- a JSON file with an `_receipt` suffix, holding the full scan receipt
  coolscanpy returns for that frame: exposure, clipping and focus
  telemetry, transport-smear assessment, and the fingerprints the frame was
  checked against

The filename pattern is the same Jinja2 template the plain Scan panel
uses, with `date` and `seq` variables. For roll scanning, `seq` is the
frame's physical slot number rather than an incrementing counter. Slot
numbers are already a stable identity for a given roll, so re-scanning one
bad slot overwrites that slot's old files instead of accumulating a second
copy beside them.

The infrared confidence mask coolscanpy also returns per frame is not
written to disk by this integration. Nothing downstream currently reads
it, and adding a file format for it before there is a consumer would be
guessing at a convention rather than following one.

## Error handling

coolscanpy raises a typed exception hierarchy rooted at
`PyCoolscanError`, covering things like a roll that no longer matches its
last preview, a slot that needs manual approval before it can be scanned,
and a requested safe stop. NegPy's existing scanner code has no typed
exception vocabulary of its own; `SaneBackend` reports every failure as a
plain `RuntimeError` with a human-readable message, and that message is
what ends up in the Scan panel's status label. This integration follows
the same convention: every coolscanpy exception is flattened to a
`RuntimeError` at the boundary, with the original exception preserved as
`__cause__` for any caller that wants to branch on it.

One case is worth calling out for whoever builds the interactive panel
described below. `Roll.safe_stop()` lets the frame already in flight
finish and only refuses the next one, by raising `SafeStopRequested`. The
existing plain-scan worker treats a cancellation the same way, checking
its own cancel flag and returning quietly instead of reporting an error.
A roll-scanning worker should do the same for a `RuntimeError` whose
`__cause__` is `SafeStopRequested`: that is an expected stop, not a
failure.

## The integration point

Everything that touches coolscanpy itself lives in one file,
`negpy/infrastructure/scanners/coolscanpy_roll.py`. `open_roll()` in that
file is the single place a device handle gets resolved for the roll
workflow; it currently calls `coolscanpy.open()` directly. Everything else,
including `negpy/services/scanning/roll_service.py`, only ever talks to the
`RollHandle` that function returns.

That module is marked in a comment as the intended target for a future
change. NegPy's maintainer has a generic SANE-based coolscan route planned
upstream, which is expected to add a real backend-selection seam to
`ScannerService` (today `ScannerService._get_backend()` simply hardcodes
`SaneBackend()`, with no selection mechanism at all). When that seam
lands, re-pointing this roll integration at it should only require changing
how `open_roll()` resolves a device. The roll workflow built on top of it,
the exception translation in the same file, and the whole of
`roll_service.py`, should not need to change.

## What is not built yet

This integration ships the backend and service layer: device discovery,
roll preview, spacing correction, approval, batch scanning, and file
writing, all covered by hardware-free tests. It does not ship a contact
sheet panel with rendered thumbnail images in the desktop GUI. Building
that panel from scratch, before the maintainer's own SANE-based route
defines how a Coolscan device is meant to surface in the UI, risked adding
exactly the kind of large, hard-to-review, foreign-feeling change this
integration is meant to avoid.

A future panel can follow the existing Scan panel closely. Look at
`negpy/desktop/view/sidebar/scan.py` for the sidebar structure and its
`_sane_available()` availability gate, `negpy/desktop/workers/scan_worker.py`
for the background-thread pattern (a `QObject` moved to its own `QThread`,
progress and result reported back over Qt signals), and
`negpy/desktop/controller.py` for how that worker's signals get wired into
the rest of the application. A roll panel would follow the same shape,
calling `RollScanningService` instead of `ScannerService`, gated behind
`coolscanpy_roll.available()` the same way the plain panel gates itself
behind SANE. The simplest first version would not need thumbnail images at
all: a plain list of slots with their approval state, backed by
`Roll.preview()`, is enough to let someone pick which slots to scan.
