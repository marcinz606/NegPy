# Coolscan / SANE film scanning

The **Scan** tab drives SANE-compatible film scanners directly — pick a device, set the
resolution and depth, and the finished scan lands in NegPy as a scanner-linear negative
master, ready for inversion and editing. It is built around the Nikon Coolscan family
(LS-50 / Coolscan V, LS-5000) but works with any SANE film scanner; Plustek, pieusb and
other transparency units keep scanning exactly as before, and controls a device does not
support simply stay hidden.

---

## Device

Pick your scanner from the **Device** dropdown; **Refresh** re-enumerates after you plug
one in or load film. NegPy lists only devices that report a film / transparency source.

**Eject** appears beside the device when the scanner exposes an eject control (Coolscan
feeders do). Use it to release the strip at the end of a session — a Coolscan feeder
auto-parks a few minutes after a session closes, and a parked feeder needs a power cycle
to recover mid-roll.

## Resolution and depth

**DPI** offers the resolutions the device reports, defaulting to the highest ("Best
quality"). **Depth** offers the device's real sample depths. A Coolscan V reports 8 and
**14** bits, not 16 — NegPy defaults to the deepest available and captures the full range
(coolscan3 ships 14-bit samples in a 16-bit container and rescales them to fill it, so a
14-bit scan is not two stops dark).

## Frame selection and range batch

When the scanner reports a strip or roll feeder, a **Frames** range appears. Its bounds
come live from the adapter: an SA-21 strip reports **6**, an SA-30 roll reports **40**.
Set a **from–to** range to scan the whole holder or any sub-range (e.g. `2–4`); a single
frame is just `from == to`. Each frame is scanned in its own pass and written with its
frame number in the filename. **Stop** finishes the current frame, then halts — frames
already scanned stay imported.

A flatbed or a scanner without a frame feeder shows no Frames control and scans once.

## Scan window

When the film's frames sit slightly off the scanner's fixed frame positions, every scan
picks up the black inter-frame gap and a sliver of the neighbour. **Set scan window…**
(shown with the Frames control) fixes this: it runs a fast low-resolution preview of one
frame, and you drag a rectangle over the actual picture area. That one window is reused
for every frame in the scan or range batch — the offset is constant across a strip, so a
single window lands them all.

Drag to draw the window, drag a corner to resize, or drag inside it to move it. **Clear**
reverts to the scanner's full default frame. The window persists across sessions.

The dialog also has a **Frame offset (mm)** — a feed-axis nudge applied to every frame.
Raise it and Preview again to push the inter-frame gap off the top of the frame *without*
cropping (preferred when the frames are simply shifted, not oversized). Offset and window
compose: the offset repositions, the window trims. Both persist across sessions.

## Autofocus and auto-exposure

**Autofocus** (on by default) focuses before each scan — film is rarely perfectly flat,
and some Coolscans power up at an uncalibrated focus position. **Auto-exposure** meters in
hardware before the scan; it is available only on devices that expose it.

## Infrared dust channel

With **IR** enabled, the scanner captures an infrared plane alongside RGB. NegPy writes it
as an `<name>_IR.tif` sidecar (plus an `<name>_IR_VALID.tif` mask), and the **Retouch**
panel's IR dust repair uses it after import. Scan chromogenic (C-41 / dye) film with IR;
conventional silver black-and-white blocks infrared, so an IR map cannot tell dust from
the photograph — leave IR off for it.

**Infrared needs a patched `coolscan3` driver.** Stock sane-backends compiles the infrared
option out for every Coolscan, so the IR toggle stays greyed out on a stock system. Run
`make sane-rgbi-help` for the one-time, three-line local patch, then launch with
`make run-ir` — nothing system-wide is touched. Without it, everything except IR works
normally.

## Output

Scans are written as 16-bit **TIFF** or a linear **DNG** (same scanner-linear pixels; the
format does not bake in inversion — use the Process panel for that). The **Filename**
field is a Jinja2 template with `{{ date }}` and `{{ seq }}`; in a range batch `seq` is the
frame number. Each finished scan is imported and selected automatically.
