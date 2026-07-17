# Camera Scanning

The **Camera Scanning** tab captures negatives with a tethered camera and imports them
straight into NegPy — no separate capture app, no shuffling folders. It has two modes,
and it picks between them on its own depending on what hardware it finds.

---

## What it does

**Normal camera scanning.** One exposure of the frame under whatever light you use,
imported as an ordinary RAW and processed like any other file. This needs nothing but
a supported camera.

**Narrowband RGB scanning.** With an RGB [Scanlight](https://github.com/jackw01/scanlight)
connected, the light flashes red, then green, then blue while the camera takes one
exposure per channel. The three RAWs are handed to NegPy's existing **RGB Scan** merge,
which sub-pixel-aligns them and assembles one frame before inversion.

Why three shots beat one: a single broadband exposure lets each dye layer contaminate
the neighbouring channels, and the green Bayer filter — the broadest of the three —
catches leakage from the red and blue light at once. Lighting one narrow band at a time
removes that crosstalk by construction, and lets every channel use the full dynamic
range of the sensor instead of sharing it.

---

## What you need

| | |
|---|---|
| **Camera** | Any body [libgphoto2 can drive](http://gphoto.org/proj/libgphoto2/support.php) with live view and remote capture. Bodies missing from that list often still work through the generic PTP driver — an a7C II does. |
| **python-gphoto2** | An optional dependency, free software (LGPL). `pip install gphoto2`. |
| **Scanlight** *(optional)* | Only needed for narrowband RGB. Without it, normal camera scanning still works. |

**NegPy runs perfectly well without any of this.** If python-gphoto2 isn't installed, the
Camera Scanning tab shows a one-line setup hint and every other part of NegPy is
unaffected. Nothing proprietary is involved: libgphoto2 is LGPL, and no vendor SDK is
bundled, linked or required.

> ⚠️ **macOS and Linux only.** libgphoto2 has no Windows build, so there are no Windows
> wheels and the tab cannot connect there.

---

## Setup

```bash
uv sync --group camera     # or: pip install gphoto2
```

That is the whole install. libgphoto2 ships inside the wheel; nothing else to download,
build or place anywhere. (Packaging the app yourself? Run that same command before
`make build` — the build then bundles libgphoto2's camera drivers. Skip it and the
packaged app just shows the setup hint.)

Then put the camera in **PC Remote** mode and plug it in over USB. It is detected
automatically — there is no address to type, no login, no pairing.

> On macOS, quit **Preview**, **Photos** and **Image Capture** first. The system's
> ImageCapture daemons hand a PTP camera to whichever of those apps is open, and
> libgphoto2 is then locked out.

---

## Scanning

**Frame and focus.** Open **Live View & Scan**. Click anywhere on the image to aim the
camera's *hardware* focus magnifier at that spot; click again to return to the full
frame. ISO, shutter and aperture are adjustable live from the toolbar in white-light and
normal (camera-only) scanning; with a calibrated RGB preset they're hidden there and
locked to the preset instead (see **Presets**), so the scan can't drift. A control the
body cannot offer — aperture on a lens with no electronic diaphragm, which is most
enlarging and macro glass — is simply greyed out.

**Calibrate (RGB mode).** Set the ISO and aperture you'll scan with, then press **+**
next to the preset dropdown, place the small rectangle on the clear film base — the rebate
strip between frames is ideal — name the preset and run it. Calibration meters that patch
and solves one shared shutter plus a per-channel LED level so each channel lands just
under clipping, and records the ISO and aperture alongside them. That highlight matters:
the clear base is what becomes the *black point* after inversion, so a clip guard checks
the raw Bayer photosites and backs the exposure off if any channel saturates. Save it
once per film stock and reuse it.

**Presets.** A selected preset is shown read-only — RGB levels, ISO, shutter and aperture —
and the scan forces that exposure on the body before every frame, so a bumped dial can't
falsify the result. To build one by hand instead of calibrating, pick **Create a manual
preset…** from the dropdown: the sliders and exposure steppers unlock, dial them in, then
press the save (floppy) button to name and store it. (White is only the white-light
preset's channel — the Scanlight can't light it together with RGB.)

**Scan.** Pick an output folder and a preset, then press **Scan** for each frame. Files
land in a per-roll subfolder, auto-numbered, and are imported and merged automatically —
so you see the inverted positive a moment after the shutter. **Retake** re-shoots the
current frame without advancing the counter.

**Narrowband Scan.** Scans lit by narrowband RGB LEDs render more saturated than
white-light scans: each channel samples its dye near the absorption peak, so the natural
spectral overlap of broadband light is missing. The **Narrowband** toggle in the Process
panel (next to **Linear RAW**) corrects this by applying the bundled RGBScan input
profile to the preview and every export. An explicit **Input ICC** chosen in Export
settings takes precedence while set.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No camera found, or the session won't open on macOS | An ImageCapture app is holding the body. Only one program may claim a PTP camera. | **Quit Preview, Photos and Image Capture**, then retry. Preview is the usual culprit — it grabs the camera silently. |
| No camera found, and nothing else is running | The body isn't in PC Remote mode, or it's a mass-storage/MTP connection. | Set the camera's USB connection mode to **PC Remote**. |
| `[-10] Timeout reading from or writing to the port`, and no other program is holding it | A program crashed while connected. The *camera* still thinks the session is open and refuses a new one. | Power-cycle the camera, or unplug and replug the cable. Nothing on the computer will fix it. |
| Live view is black | The body dropped out of PC Remote, or the lens cap is on. | Power-cycle the camera. |
| Capture says the camera returned JPEG instead of RAW | The camera's image-quality setting is JPEG or RAW+JPEG selected the processed file. | Set image quality to **RAW only**, then retry. |
| The aperture stepper is greyed out | The lens has no electronic diaphragm. | Expected — set the aperture on the lens itself. |
| A setting snaps back to its old value | Property writes are asynchronous; the body needs a moment. | NegPy already polls until the value lands and logs a warning if it never does. If it never does, that setting is not writable in the body's current mode (try **M**). |
| Scanlight not detected | Wrong USB-C port. | The Scanlight has two: only one is the **data** port (the other is power-only). Use the data port. |

---

## Notes and limitations

- **USB only.** libgphoto2 can reach some cameras over the network, but not Sony bodies;
  the tab is a tethered-USB workflow.
- **Only Sony bodies are tested** — that is the hardware on hand. Nothing in NegPy assumes
  a vendor, though: every control is looked up rather than assumed. `iso` and
  `shutterspeed` are named the same everywhere; the aperture is `f-number` on Sony and
  Panasonic but `aperture` on Canon, Nikon, Fujifilm, Olympus and Sigma. The RAW suffix
  comes from the camera, and the still is taken into memory rather than onto a card (Canon
  and Nikon default to the card, and will not shoot without one). Reports from other brands
  are very welcome.
- **The focus magnifier depends on the vendor.** Sony packs the zoom ratio and the target
  point into one property, so a click both magnifies *and* aims. Canon (`eoszoom`) and
  Nikon (`liveviewimagezoomratio`) split them, and their coordinate space is unknown here,
  so a click magnifies where the body already looks. Every other body has no magnifier at
  all, and the feature disables itself.
- **Tested on macOS.** The Python is portable and libgphoto2 is a Linux-first project, so
  Linux should be at least as good — but it is unverified.
- **Speed.** A three-shot RGB triplet takes roughly six seconds on an a7C II over USB.
  Almost all of that is the body's per-image transfer latency, not the file size. Stills
  are taken inside the running live-view session, so there is no per-frame reconnect.
- **Credit.** The R/G/B sequencing and exposure-calibration approach come from
  [rohanpandula/TriRGB](https://github.com/rohanpandula/TriRGB); the light is
  [jackw01/scanlight](https://github.com/jackw01/scanlight); the camera is driven through
  [python-gphoto2](https://github.com/jim-easterbrook/python-gphoto2). The narrowband
  approach follows Flückiger et al.'s work on trichromatic film scanning.
