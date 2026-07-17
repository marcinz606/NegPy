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
feeders do). NegPy also returns the strip automatically once a range batch finishes, so you
seldom need it — but press it any time to release the strip on demand rather than waiting
for the feeder to auto-park a few minutes after a session closes (a parked feeder needs a
power cycle to recover mid-roll).

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

## Scan window and strip preview

When the film's frames sit slightly off the scanner's fixed frame positions, every scan
picks up the black inter-frame gap and a sliver of the neighbour. **Preview strip…** (shown
with the Frames control) fixes this and more: it previews the strip frame by frame and lets
you set the picture area per frame and choose exactly which frames to scan.

Each frame gets its own tile. **Preview** scans that one frame at low resolution; **Preview
all** walks the strip and previews every frame in turn (one at a time — the scanner handles
a single frame per pass). On a previewed frame, drag to draw its window, drag a corner to
resize, or drag inside it to move it — each frame keeps its own window. Tick **Scan** on the
frames you want and leave the rest unticked to skip them (e.g. preview 1–6 but scan only
1, 2, 4, 6). **Clear all** drops every window (full frames); **Use** applies your choices.
Windows, selection and offset all persist across sessions.

A single **Offset** slider (top of the dialog, 0–4 mm) is a feed-axis nudge applied to every
frame: raise it to push the inter-frame gap off the top *without* cropping (preferred when
the frames are simply shifted, not oversized), then Preview again to see it. Offset and
windows compose — the offset repositions, each window trims.

Setting a selection here takes over from the **Frames** range; **Clear all** hands control
back to the simple from–to range.

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
