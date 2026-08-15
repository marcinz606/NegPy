# Camera Scanning

The **Camera Scanning** tab captures negatives with a tethered camera and imports them
straight into NegPy. There is no separate capture app and no folder shuffling. It has two
modes, and it selects between them from the hardware it finds.

---

## What it does

**Normal camera scanning.** One exposure of the frame under whatever light you use,
imported as an ordinary RAW and processed like any other file. This needs only a supported
camera.

**Narrowband RGB scanning.** With an RGB [Scanlight](https://github.com/jackw01/scanlight)
connected, the light flashes red, then green, then blue, and the camera takes one exposure
per channel. NegPy's **RGB Scan** merge sub-pixel-aligns the three RAWs and assembles one
frame before inversion.

Three shots beat one because a single broadband exposure lets each dye layer contaminate
the neighbouring channels. The green Bayer filter is the broadest of the three, so it
catches leakage from the red and the blue light at once. One narrow band at a time removes
that crosstalk by construction, and every channel gets the full dynamic range of the sensor
instead of sharing it.

---

## What you need

| | |
|---|---|
| **Camera** | Any body [libgphoto2 can drive](http://gphoto.org/proj/libgphoto2/support.php) with live view and remote capture. A body missing from that list often still works through the generic PTP driver. An a7C II does. |
| **python-gphoto2** | An optional dependency, free software (LGPL). `pip install gphoto2`. |
| **Scanlight** *(optional)* | Needed for narrowband RGB only. Normal camera scanning works without it. |

**NegPy runs well without any of this.** Without python-gphoto2, the Camera Scanning tab
shows a one-line setup hint, and every other part of NegPy is unaffected. Nothing
proprietary is involved: libgphoto2 is LGPL, and no vendor SDK is bundled, linked or
required.

> ⚠️ **macOS and Linux only.** libgphoto2 has no Windows build, so there are no Windows
> wheels and the tab cannot connect there.

---

## Setup

```bash
uv sync --group camera     # or: pip install gphoto2
```

That is the whole install. libgphoto2 ships inside the wheel, so there is nothing else to
download, build or place. If you package the app yourself, run that same command before
`make build`. The build then bundles libgphoto2's camera drivers. Skip it and the packaged
app shows the setup hint.

Then put the camera in **PC Remote** mode and plug it in over USB. NegPy detects it
automatically. There is no address to type, no login and no pairing.

> On macOS, quit **Preview**, **Photos** and **Image Capture** first. The system's
> ImageCapture daemons hand a PTP camera to whichever of those apps is open, and
> libgphoto2 is then locked out.

---

## Scanning

**Frame and focus.** Open **Live View & Scan**. Click anywhere on the image to aim the
camera's *hardware* focus magnifier at that spot. Click again to return to the full frame.
In white-light and normal (camera-only) scanning, you can set ISO, shutter and aperture
live from the toolbar. With a calibrated RGB preset those controls are hidden and locked to
the preset instead (see **Presets**), so the scan cannot drift. A control the body cannot
offer is greyed out. Aperture on a lens with no electronic diaphragm is the usual case, and
that is most enlarging and macro glass.

**Calibrate (RGB mode).** Set the ISO and the aperture you will scan with. Press **+**
beside the preset dropdown, place the small rectangle on the clear film base, name the
preset and run it. The rebate strip between frames is an ideal target. Calibration meters
that patch and solves one shared shutter plus a per-channel LED level, so each channel
lands just under clipping. It records the ISO and the aperture with them.

That highlight matters, because the clear base becomes the *black point* after inversion. A
clip guard therefore checks the raw Bayer photosites and backs the exposure off if any
channel saturates. Save the preset once per film stock and reuse it.

If the target is unreachable at your exposure, the run stops at the probe. A pop-up says
which way to adjust: over-exposed means close the aperture or lower the ISO, under-exposed
means open up or raise the ISO. **No preset is saved.** Adjust and calibrate again in the
window that stayed open.

**Presets.** A selected preset is shown read-only, with its RGB levels, ISO, shutter and
aperture. The scan forces that exposure on the body before every frame, so a bumped dial
cannot falsify the result. To build a preset by hand instead, pick **Create a manual
preset…** from the dropdown. The sliders and the exposure steppers unlock. Dial them in,
then press the save (floppy) button to name and store the preset. White is the white-light
preset's channel only, because the Scanlight cannot light it together with RGB.

**Scan.** Pick an output folder and a preset, then press **Scan** for each frame. Files
land in a per-roll subfolder, auto-numbered, and are imported and merged automatically, so
the inverted positive appears a moment after the shutter. **Retake** re-shoots the current
frame without advancing the counter.

**Narrowband Scan.** Scans lit by narrowband RGB LEDs render more saturated than
white-light scans, because each channel samples its dye near the absorption peak and the
natural spectral overlap of broadband light is missing. The **Narrowband** toggle in the
Process panel, beside **Linear RAW**, corrects this with the bundled RGBScan input profile,
in the preview and in every export. An explicit **Input ICC** in the Export settings takes
precedence while it is set.

### Sensor calibration

A single-shot scan under a narrowband RGB source can come out with hues that no slider
fixes. Yellows drifting orange is the classic sign. The cause is **sensor crosstalk**: the
camera's color-filter passbands overlap the source's bands, so the green pixel sees the
blue LED and some red, and every channel carries a share of its neighbours. It is a fixed
property of your sensor and light pair, independent of the film.

To correct it, photograph the bare light three times with no film in the holder: red only,
green only, blue only. Use the same settings you scan with, exposed just below clipping.
Then open the **Calibration** panel, find *Trichrome Calibration*, press the calibrate
button, pick the three captures, name the profile and save it. The selected profile un-mixes
every scan with a 3×3 matrix in the linear domain, before inversion. Profiles are TOML
files in the `NegPy/sensor` folder. Re-run **Batch Analysis** after you change the profile.

Do not use it on RGB-triplet (trichrome) scans. They are crosstalk-free by construction,
because each channel comes from its own single-light exposure, and NegPy skips the
correction automatically for triplet assets. A dedicated film scanner has no sensor
crosstalk either: a Coolscan's mono CCD reads one LED at a time. Its residual color error
is film-dye crosstalk, which the density-domain **Crosstalk** matrix handles (see
[CROSSTALK.md](CROSSTALK.md)).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| The camera dot shows **"in use"** (amber hint) | An ImageCapture app holds the body. Only one program may claim a PTP camera. | **Quit Preview, Photos and Image Capture.** Preview is the usual culprit, because it grabs the camera silently. NegPy reconnects on its own once the camera is free. |
| No camera found, and nothing else is running | The body is not in PC Remote mode, or it is a mass-storage/MTP connection. | Set the camera's USB connection mode to **PC Remote**. |
| `[-10] Timeout reading from or writing to the port`, and no other program holds it | A program crashed while connected. The *camera* still thinks the session is open and refuses a new one. | Power-cycle the camera, or unplug and replug the cable. Nothing on the computer fixes it. |
| Live view is black | The body dropped out of PC Remote, or the lens cap is on. | Power-cycle the camera. |
| The scan window opens with **"no live view"** instead of a preview | libgphoto2's entry for this body has no preview capability. Either the body genuinely lacks it (Sony a6000), or it is connected in MTP mode, where no body has it. | If the message names MTP, set the camera's USB mode to **PC Remote** and reconnect. If not, this is expected. Scanning works normally, but you must set framing and focus on the camera, and calibration is unavailable because it aims through the live view. |
| Capture says the camera returned JPEG instead of RAW | The camera's image-quality setting is JPEG, or RAW+JPEG selected the processed file. | Set image quality to **RAW only**, then retry. |
| The aperture stepper is greyed out | The lens has no electronic diaphragm. | Expected. Set the aperture on the lens itself. |
| A setting snaps back to its old value | Property writes are asynchronous, so the body needs a moment. | NegPy polls until the value lands and logs a warning if it never does. If it never does, that setting is not writable in the body's current mode. Try **M**. |
| The Scanlight is not detected | Wrong USB-C port. | The Scanlight has two ports and only one carries data. The other is power only. Use the data port. |

---

## Notes and limitations

- **USB only.** libgphoto2 can reach some cameras over the network, but not Sony bodies.
  The tab is a tethered-USB workflow.
- **Only Sony bodies are tested**, because that is the hardware on hand. Nothing in NegPy
  assumes a vendor: every control is looked up rather than assumed. `iso` and
  `shutterspeed` are named the same everywhere. The aperture is `f-number` on Sony and
  Panasonic, and `aperture` on Canon, Nikon, Fujifilm, Olympus and Sigma. The RAW suffix
  comes from the camera, and the still is taken into memory rather than onto a card. Canon
  and Nikon default to the card and will not shoot without one. Reports from other brands
  are very welcome.
- **The focus magnifier depends on the vendor.** Sony packs the zoom ratio and the target
  point into one property, so a click both magnifies *and* aims. Canon (`eoszoom`) and
  Nikon (`liveviewimagezoomratio`) split them, and their coordinate space is unknown here,
  so a click magnifies where the body already looks. Every other body has no magnifier at
  all, and the feature disables itself.
- **Tested on macOS.** The Python is portable and libgphoto2 is a Linux-first project, so
  Linux should be at least as good. This is unverified.
- **Speed.** A three-shot RGB triplet takes about six seconds on an a7C II over USB. Almost
  all of that is the body's per-image transfer latency, not the file size. Stills are taken
  inside the running live-view session, so there is no per-frame reconnect.
- **Credit.** The R/G/B sequencing and the exposure-calibration approach come from
  [rohanpandula/TriRGB](https://github.com/rohanpandula/TriRGB). The light is
  [jackw01/scanlight](https://github.com/jackw01/scanlight). The camera is driven through
  [python-gphoto2](https://github.com/jim-easterbrook/python-gphoto2). The narrowband
  approach follows Flückiger et al.'s work on trichromatic film scanning.
