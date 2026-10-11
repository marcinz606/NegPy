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
per channel. NegPy's **Trichrome** merge sub-pixel-aligns the three RAWs and assembles one
frame before inversion.

Three shots beat one because a single broadband exposure lets each dye layer contaminate
the neighboring channels. The green Bayer filter is the broadest of the three, so it
catches leakage from the red and the blue light at once. One narrow band at a time removes
that crosstalk by construction, and every channel gets the full dynamic range of the sensor
instead of sharing it.

**Single-capture narrowband scanning.** An RGB preset can instead light red, green and blue
together for one exposure. The frame is imported as an ordinary RAW, and its roll is
recorded with Trichrome Mode off. It takes one shutter actuation per frame and has no registration between channels.
The channels overlap on the sensor, so pair it with a
[sensor calibration](#sensor-calibration) profile.

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

> On macOS a PTP camera is claimed by the system camera daemon as soon as nothing else holds
> it, and it does not let go again for minutes. NegPy therefore takes the claim the moment the
> Camera Scanning panel sees the body and keeps it: while that panel is open the camera belongs
> to NegPy, and Preview, Photos, Image Capture and other tethering software cannot use it.
> Unplugging the cable hands it back.
>
> What wakes that daemon is any application asking for a camera, and it does not have to be one
> you opened. A background sync client is the common case, silently and repeatedly: check for
> those first if the camera reads as in use with nothing visible running.

---

## Scanning

**Frame and focus.** Open **Live View**. Click anywhere on the image to aim the
camera's *hardware* focus magnifier at that spot. Click again to return to the full frame.
The **Focus meter** under the image reads live sharpness against the best value since the
view last changed. A click changes the view. So does the body's own magnifier: Sony's MF
Assist zooms in when the focus ring turns, and **Focus Magnif. Time** zooms back out after
2 s or 5 s. Set it to **No Limit** to keep the zoom while you focus. Turn the focus ring past
best focus, then back until it reads **at peak**. It works on every body with live view.
In white-light and normal (camera-only) scanning, you can set ISO, shutter and aperture
live from the toolbar. With a calibrated RGB preset those controls are hidden and locked to
the preset instead (see **Presets**), so the scan cannot drift. A control the body cannot
offer is grayed out. Aperture on a lens with no electronic diaphragm is the usual case, and
that is most enlarging and macro glass.

A thin border marks the edge of the captured frame.

**Histogram** and **Zebras** on the toolbar read the preview frame. The histogram shows R, G
and B under the image; the zebras hatch pixels at the top (red, 99% and above) and the bottom
(blue, 1% and below) of the preview range. The preview is the body's JPEG, which clips before
the RAW does, so a zebra marks a region close to the limit with some RAW headroom left. On a
negative the clear base must stay under the top; on a slide the picture's highlights must.

**Meter (white-light and camera-only scanning).** Press **Meter** to take one RAW probe at the
current exposure. The preset's film type decides what must not clip. On **White Light, B&W
Negative**, or a color negative on the AUTO preset, the clear base, read in the rebate around
the picture, is placed at 90% of the RAW range in the brightest channel, the same target
Calibrate uses. On **White Light, Slide** the picture's brightest highlights are. The AUTO
preset classifies the probe the way an import does, so a slide it misreads is metered as a
negative until you pick the Slide preset. A capture with no rebate, or a rebate no brighter
than the picture, meters the picture's highlights either way.

The shutter moves to the body's nearest step at or under that exposure, as a counted exposure
write: Scan waits until the body confirms it. The status line reads, for example,
`Meter: negative, base at 62% → 1/15 (+0.5 st)`, the stops being the move that step makes.
A clipped base, or highlights clipped in the picture's brightest 0.1%, cannot be measured: the
shutter steps down two stops and the line asks for another probe, which then reads them. The
line also names the share of the picture that clipped. A target past the body's fastest step is
reported with the remedy, close the aperture or lower the ISO, and nothing is written. Under the
Scanlight, Meter needs the shutter on a step between 1/250 s and 2 s, the span Calibrate solves
on (a shorter exposure integrates too few of the LED's PWM pulses); camera-only scanning meters
on the body's whole ladder and lets the probe decide the film type, as its import does. It waits
while a scan, a calibration or another probe runs. A negative's base is the same across a
roll, so one reading serves it; a slide's highlights differ on every frame, so meter each
one. After a probe the zebras switch to the RAW: they hatch where the RAW will clip at the
shutter the Meter set, predicted from the probe (the RAW is linear, so a level scales exactly
with the exposure; a pixel the probe itself clipped stays hatched until a probe reads it). They
hold while the ISO, shutter, aperture and light are the metered ones and the preview is still
the same picture; a film advance, a reframe or the focus magnifier hands them back to the
preview's zebras. The blue shadow hatching stays the preview's. The map is used only when the
first preview at the metered exposure frames the same picture as the probe: the same aspect, and
the same picture by rank correlation of thumbnails, which the JPEG's tone curve cannot change.
A body whose live view crops or reframes the RAW fails it, and the status line says
`RAW zebras off:` with the reason. In-camera lens correction moves the preview's edges by less
than one block of the map and is not checked. A calibrated RGB preset owns its exposure, and the button is hidden with the exposure
steppers.

**Calibrate (RGB mode).** Set the ISO and the aperture you will scan with. Press **+**
beside the preset dropdown, place the small rectangle on the clear film base, name the
preset and run it. The rebate strip between frames is an ideal target. Calibration meters
that patch and solves one shared shutter plus a per-channel LED level, so each channel
lands just under clipping. It records the ISO and the aperture with them.

The dropdown beside the name picks the capture mode. **Triplet** solves each channel under
its own LED. **Single Capture** solves the three levels together for one exposure with all
three LEDs lit: each sensor channel also reads the neighboring LEDs, so the run measures
that overlap and lowers the levels to match. If the overlap leaves no levels that balance
the channels, the run stops and says so. Calibrate a triplet preset instead.

With **Create Sensor Profile** on, a Single Capture run also saves a
[sensor calibration](#sensor-calibration) profile under the preset's name, measured from the
same exposures. A roll scanned with the preset takes that profile and turns Linear RAW on.
Turn the toggle off to keep a profile you made yourself.

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
**Capture mode** marks the manual preset as a triplet or a single capture. Then press the
save (floppy) button to name and store the preset. White is the white-light
presets' channel only, because the Scanlight cannot light it together with RGB. The three
white-light presets differ in film type alone: AUTO classifies each capture, **B&W Negative**
and **Slide** fix it, for the import and for Meter alike.

**Scan.** Pick an output folder and a preset, then press **Scan** for each frame. Files
land in a per-roll subfolder, auto-numbered, and are imported and merged automatically, so
the inverted positive appears a moment after the shutter. A single-capture preset writes
one file per frame. **Retake** re-shoots the current frame without advancing the counter. The **Delay between exposures** control adds a pause
between red/green/blue captures so older bodies can finish flushing the previous shot before
the next one arrives; this avoids the USB/PTP lockups that some cameras trigger when they are
bombarded with a new capture command too quickly.

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

There are three ways to build a profile, and they give near-identical corrections.
[Sensor profile workflows](#sensor-profile-workflows) puts each one in a full scan:

- **With a preset.** A Single Capture preset calibrated with **Create Sensor Profile** on saves a
  profile under the preset's name, measured through the film base. Rolls scanned with the
  preset take it automatically. Use this if you scan with Single Capture presets.
- **Capture from Camera…** With the camera tethered, a Scanlight connected and no film in
  the holder, open the **Calibration** panel, find *Single-Shot Narrowband Calibration*,
  press the calibrate button, name the profile and press **Capture from Camera…**, then
  confirm. NegPy lights each LED in turn, sets the shutter itself and saves the profile.
  It uses the ISO and aperture the camera is set to, and needs no live view. One profile
  serves every film stock. Use this for a rig you scan with outside the presets.
- **From files.** Photograph the bare light three times with no film in the holder: red
  only, green only, blue only, exposed just below clipping. In the same dialog, pick the
  three captures, name the profile and save it. Use this for a light NegPy cannot control.

A profile measured through the film base and one measured on the bare light differ
slightly, because the base tints each LED's light. The leak also changes a little with
the density of the picture, so no single profile is exact, and neither method is the more
accurate one. One profile for a sensor and light pair is enough.

The selected profile un-mixes every scan with a 3×3 matrix in the linear domain, before
inversion. Profiles are TOML files in the `NegPy/sensor` folder. Re-run **Roll Analysis**
after you change the profile.

### Sensor profile workflows

A profile is built in one of two places: the preset calibration that **+** opens in the
**Preset & Light** panel, or the **Calibration** panel on the Roll tab. They differ in who assigns the profile to
the roll. A preset that made its own profile assigns it on every scan. A profile from the
Calibration panel is assigned by you, once per roll.

| Your setup | Workflow | Profile built in | Assigned by |
|---|---|---|---|
| Tethered camera and Scanlight, calibrated preset | [A](#a-the-preset-makes-the-profile) | Preset calibration | The preset |
| Tethered camera and Scanlight, calibrated preset, one profile for every preset | [B](#b-a-calibrated-preset-and-a-separate-profile) | Calibration panel | You |
| Tethered camera and Scanlight, manual preset | [C](#c-a-manual-preset) | Calibration panel | You |
| Files shot without NegPy | [D](#d-files-not-scanned-with-negpy) | Calibration panel | You |

All four apply to single-shot narrowband frames only. A Triplet preset needs no profile.

#### A. The preset makes the profile

Use this when you scan with calibrated Single Capture presets and want nothing more to
set. It is the default.

Once per film stock:

1. Load the film. In the **Preset & Light** panel, press **+** beside the preset
   dropdown. The calibration window opens with its own live view.
2. Set the ISO and the aperture you scan with, click the clear film base and name the
   preset after the stock.
3. Pick **Single Capture**, leave **Create Sensor Profile** on and press **Calibrate & Save**.

For every roll of that stock, this one and later ones:

1. Pick the preset in the dropdown.
2. Scan the roll.

What follows from it:

- The preset and its profile are stored, so there is nothing to calibrate again for the
  next roll of the same stock. Calibrate again when the camera, the lens, the light or
  the film stock changes.
- The preset holds the profile's name, and the profile is a file in the `NegPy/sensor`
  folder. With that file deleted, the preset still scans and the roll gets no correction.

- Every scan makes the preset's profile the roll's own and turns Linear RAW on. A profile
  you pick by hand for that roll is replaced at the next scan.
- The profile has the preset's name. Calibrating a preset again under the same name writes
  the profile again, and so does any other profile saved under that name.
- Each preset has its own profile, measured through that film's base.
- The profile stays with the preset's rolls. It is not carried to frames opened or
  scanned another way.

#### B. A calibrated preset and a separate profile

Use this when one profile is to serve every preset and film stock on the rig, when you
already have a profile for this sensor and light, or when a preset calibration reports
that its exposures were too dim to measure one.

1. Build the profile once for the rig. Take the film out of the holder. In the
   **Calibration** panel, turn **Linear RAW** on, press the calibrate button, name the
   profile and press **Capture from Camera…**. To build it from three bare-light files
   you shot yourself instead, pick them in the same dialog as in workflow D, step 1.
2. Load the film. Calibrate the preset as in workflow A, with **Create Sensor Profile** off.
3. Scan the first frame. In the **Calibration** panel, check that **Linear RAW** is on,
   pick the profile under **Profile**, and press **Roll** on the panel header.
4. Scan the rest of the roll. New frames take the roll's profile.
5. Re-run **Roll Analysis** if frames were scanned before step 3.

What follows from it:

- The preset sets nothing on the roll. A roll with no profile picked is not corrected.
- The profile is measured on the bare light, so it holds no film base. Its correction is
  close to a preset's, not identical.
- There is one profile to keep current. Build it again when the camera or the light
  changes.
- The profile is a Calibration setting, so it carries to the next frames you open. Set
  **Profile** to *None* on a roll scanned under another light.

#### C. A manual preset

A manual preset is dialed in by hand. Nothing is metered, so it cannot make a profile,
and a single-shot roll scanned with it has no correction until you pick one.

1. Build the profile as in workflow B, step 1, or use one the rig already has.
2. In the **Preset & Light** panel, pick **Create a manual preset…** from the preset
   dropdown, set the levels and the exposure, set **Capture mode** to **Single Capture**
   and save the preset.
3. Scan the first frame, then assign the profile to the roll as in workflow B, step 3.

What follows from it is the same as in workflow B. The profile does not depend on the LED
levels or the shutter, so one profile serves a manual preset at any levels.

#### D. Files not scanned with NegPy

Use this for single-shot RAW files shot to the card, with other tethering software, or
under an RGB light NegPy cannot control.

1. Build the profile for the camera and light that shot the files. If that rig is a
   camera NegPy can tether and a Scanlight, use **Capture from Camera…** as in workflow B.
   If not, photograph the bare light three times with no film in the holder: red only,
   green only, blue only, as RAW, exposed just below clipping. In the **Calibration**
   panel, turn **Linear RAW** on, press the calibrate button, pick the three files, name
   the profile and press **Compute and Save**.
2. Open the roll. In the **Calibration** panel, turn **Linear RAW** and **Narrowband** on.
3. Pick the profile under **Profile** and press **Roll** on the panel header.
4. Re-run **Roll Analysis**.

What follows from it:

- The files must be camera RAW. The profile needs Linear RAW, and it is grayed out
  without it.
- A profile fits one sensor and light pair. Files from another camera or another light
  need their own.
- A light that cannot show one color at a time gives no way to measure a profile.
- The profile carries to the next frames you open, as in workflow B.

### When no profile applies

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
| The camera dot shows **"in use"** (amber hint) | Another program holds the body, and only one program may claim a PTP camera. On macOS the holder is the system camera daemon, and it is woken by any application that watches for cameras. Often that application is a **background sync client** with no window and no notification: Google Drive running as a login item has been confirmed to do it, and Dropbox and similar tools behave the same way. | Quit the background app, or remove it from **System Settings → General → Login Items** if it starts itself. Unplugging and reconnecting the cable frees the camera too, but only until that app asks again. Preview, Photos and Image Capture do the same when one of them is genuinely open. NegPy reconnects by itself once the camera is free. |
| No camera found, and nothing else is running | The body is not in PC Remote mode, or it is a mass-storage/MTP connection. | Set the camera's USB connection mode to **PC Remote**. |
| `[-10] Timeout reading from or writing to the port`, and no other program holds it | A program crashed while connected. The *camera* still thinks the session is open and refuses a new one. A timeout that clears on its own is retried silently and never reaches you, so seeing this means it did not clear. | Power-cycle the camera, or unplug and replug the cable. Nothing on the computer fixes it. |
| Live view is black | The body dropped out of PC Remote, or the lens cap is on. | Power-cycle the camera. |
| The scan window opens with **"no live view"** instead of a preview | libgphoto2's entry for this body has no preview capability. Either the body genuinely lacks it (Sony a6000), or it is connected in MTP mode, where no body has it. | If the message names MTP, set the camera's USB mode to **PC Remote** and reconnect. If not, this is expected. Scanning works normally, but you must set framing and focus on the camera, and calibration is unavailable because it aims through the live view. |
| Capture says the camera returned JPEG instead of RAW | The camera's image-quality setting is JPEG, or RAW+JPEG selected the processed file. | Set image quality to **RAW only**, then retry. |
| The aperture stepper is grayed out | The lens has no electronic diaphragm. | Expected. Set the aperture on the lens itself. |
| A setting snaps back to its old value | Property writes are asynchronous, so the body needs a moment. | NegPy polls until the value lands and logs a warning if it never does. If it never does, that setting is not writable in the body's current mode. Try **M**. |
| The Scanlight is not detected | Wrong USB-C port. | The Scanlight has two ports and only one carries data. The other is power only. Use the data port. |
| The Scanlight is not detected on Linux, and the data port is right | Your user cannot open the serial port; most distributions give it to the `dialout` group. | `sudo usermod -aG dialout $USER`, then log out and back in. |

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
  all, and the feature disables itself. A body that stops streaming while magnified (the
  Nikon D3300) returns to full frame, and its magnifier stays off for the session. On all of
  these, focus with the Focus meter.
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
