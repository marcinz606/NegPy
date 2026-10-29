# Camera Scanning

The **Scan** tab, with **Scanner** set to **Camera**, captures negatives with a tethered camera and imports them
into NegPy. It has two modes and picks one from the hardware it finds.

## What it does

**Normal camera scanning.** One exposure of the frame under any light, imported as an
ordinary RAW. This needs only a supported camera.

**Trichrome scanning.** With a [Scanlight](https://github.com/jackw01/scanlight)
connected, the light flashes red, then green, then blue, and the camera takes one exposure
per channel. NegPy's **Trichrome** merge sub-pixel-aligns the three RAWs into one frame
before inversion. One narrow band at a time removes the crosstalk between dye layers, and
each channel gets the full dynamic range of the sensor.

**Single Exposure scanning.** A narrowband preset can instead light red, green and blue
together for one exposure: one shutter actuation per frame and no registration. The frame
imports as an ordinary RAW, and its roll is recorded with Trichrome Mode off. The channels
overlap on the sensor, so pair it with a [sensor calibration](#sensor-calibration) profile.

## What you need

| | |
|---|---|
| **Camera** | Any body [libgphoto2 can drive](http://gphoto.org/proj/libgphoto2/support.php) with live view and remote capture. A body missing from that list often works through the generic PTP driver (an a7C II does). |
| **python-gphoto2** | Optional dependency, LGPL. `pip install gphoto2`. No vendor SDK is bundled, linked or required. |
| **Scanlight** *(optional)* | Needed for narrowband light only. |

Without python-gphoto2, the Scan tab's Camera scanner shows a one-line setup hint and the rest
of NegPy works as usual.

> **macOS and Linux only.** libgphoto2 has no Windows build, so there are no Windows
> wheels and the tab cannot connect there.

## Setup

```bash
uv sync --group camera     # or: pip install gphoto2
```

libgphoto2 ships inside the wheel. If you package the app yourself, run the same command
before `make build` to bundle libgphoto2's camera drivers; without it, the packaged app
shows the setup hint.

Put the camera in **PC Remote** mode and plug it in over USB. NegPy detects it.

> On macOS the system camera daemon claims a free PTP camera and keeps it for minutes, so
> NegPy claims the body as soon as the Scan tab's Camera scanner sees it. While that scanner is
> open, Preview, Photos, Image Capture and other tethering software cannot use the camera.
> Unplug the cable to release it. If the camera reads as in use with nothing visible
> running, see [Troubleshooting](#troubleshooting).

## Scanning

**Frame and focus.** Open **Live View**. Click the image to aim the camera's *hardware*
focus magnifier at that spot; click again for the full frame. The **Focus meter** under the
image compares live sharpness with the best value since the view last changed. A click
changes the view, and so does the body's own magnifier: Sony's MF Assist zooms in when the
focus ring turns, and **Focus Magnif. Time** zooms back out after 2 s or 5 s (set it to
**No Limit** to keep the zoom). Turn the focus ring past best focus, then back until the
meter reads **at peak**. It works on every body with live view.

In white-light and camera-only scanning, set ISO, shutter and aperture live from the
toolbar. A calibrated narrowband preset hides and locks them (see **Presets**). A control the body
cannot offer is grayed out, usually aperture on a lens with no electronic diaphragm (most
enlarging and macro glass). A thin border marks the edge of the captured frame.

**Histogram** and **Zebras** on the toolbar read the preview frame. The histogram shows R, G
and B. The zebras hatch the top (red, 99% and above) and the bottom (blue, 1% and below) of
the preview range. The preview is the body's JPEG, which clips before the RAW, so a zebra
leaves some RAW headroom. On a negative the clear base must stay under the top; on a slide
the picture's highlights must.

**Meter.** Press **Meter** in Live View, or the **Aperture Priority Meter** gauge beside
CAMERA EXPOSURE on the card, to take one RAW probe at the current exposure. The card's gauge
opens Live View when it is closed. On **White Light, B&W Negative**, **White Light, Color
Negative**, or a color negative on the AUTO preset, the clear base (read in the rebate around
the picture) goes to 90% of the range in the brightest channel, the same target as Calibrate.
A negative with no clear base in frame, or a base the probe cannot read unclipped, meters its
thinnest picture instead and says so. A white-light frame imports with the camera's white
balance, so that gain is in its target. On **White Light, Slide** the
picture's brightest highlights go there. AUTO classifies the probe as an import does, so it
meters a misread slide as a negative until you pick the Slide preset. With no rebate, or a
rebate no brighter than the picture, Meter uses the picture's highlights.

The shutter moves to the body's nearest step at or under that exposure, as a counted
exposure write: Scan waits until the body confirms it. The status line reads, for example,
`Meter: negative, base at 62% → 1/15 (+0.5 st)` (the stops are the move that step makes),
and gives the share of the picture that clipped. A base the probe cannot read unclipped
meters the picture's thin end and the line says `(base unreadable)`. A picture with two
percent or more of its pixels saturated steps down two stops and the line asks for another
probe; fewer saturated pixels are reported and left out of the level. A probe under 0.05% of
the range in every channel is reported as too dark, and nothing is written. For a target
past the body's fastest step, the line says to close the aperture or lower the ISO, and
writes nothing.

Under the Scanlight, Meter needs the shutter between 1/250 s and 2 s, the span Calibrate
solves on. Camera-only
scanning meters on the body's whole ladder and lets the probe decide the film type. Meter
waits while a scan, a calibration or another probe runs. One reading serves a roll of
negatives; meter each slide.

After a probe the zebras switch to the RAW: they hatch where the RAW will clip at the
metered shutter, predicted from the linear probe (a pixel the probe clipped stays hatched
until a probe reads it). They hold while the ISO, shutter, aperture and light are the
metered ones and the preview shows the same picture. A film advance, a reframe or the focus
magnifier returns to the preview's zebras. The blue shadow hatching always comes from the
preview. The map is used only when the first preview at the metered exposure has the same
aspect as the probe and matches it by rank correlation of thumbnails. A body whose live
view crops or reframes the RAW fails this check, and the status line says `RAW zebras off:`
with the reason. Under a calibrated narrowband preset the probe shoots under the scan light, and the
metered shutter replaces the preset's until you pick the preset again.

**Calibrate (narrowband light).** Set the ISO and the aperture you scan with. Press **+** beside the
preset dropdown, pick **Calibrate on the Film Base…**, place the small rectangle on the clear film base (the rebate strip between
frames is a good target), name the preset and run it. Calibration solves one shared
shutter plus a per-channel LED level, so each channel lands just under clipping, and
records the ISO and the aperture. The clear base becomes the *black point* after inversion,
so a clip guard checks the raw Bayer photosites and backs the exposure off if any channel
saturates. Save the preset once per film stock and reuse it.

The dropdown beside the name sets the capture mode. **Trichrome** solves each channel under
its own LED. **Single Exposure** solves the three levels together for one exposure with all
three LEDs lit, measuring how much each sensor channel reads the neighboring LEDs. If no
levels balance the channels, the run stops and says so; calibrate a Trichrome preset instead.

With **Create Sensor Crosstalk Profile** on, a Single Exposure run also saves a
[sensor calibration](#sensor-calibration) profile under the preset's name, from the same
exposures. A roll scanned with the preset takes that profile and turns Linear RAW on. Turn
the toggle off to keep a profile you made yourself.

If the target is out of reach, the run stops at the probe and a pop-up says what to do:
over-exposed, close the aperture or lower the ISO; under-exposed, open up or raise the ISO.
No preset is saved. Adjust and calibrate again in the window, which stays open.

**Presets.** A selected preset shows read-only, with its RGB levels, ISO, shutter and
aperture. The scan sets that exposure on the body before every frame. The dropdown holds your
own presets and the built-ins you have added from **+** › **Built-in Presets**; a built-in stays
in the list until the trash removes it. To build a preset by hand, press **+** beside
PRESET and pick **Set by Hand** to unlock the sliders and the exposure steppers. Set them and **Capture
mode** (Trichrome or Single Exposure); Scan uses them as they stand, and the save (floppy)
button keeps them as a preset. Only the
white-light presets use the White channel, because the Scanlight cannot light it together with the red, green and blue LEDs. The white-light presets differ only in film type: AUTO classifies each capture,
**B&W Negative**, **Color Negative** and **Slide** fix it, for the import and for Meter. **White
Light + IR, Color Negative**, **White Light + IR, Slide**, **Trichrome, Color Negative** and **Trichrome + IR, Color Negative**
need no calibration either: full LED power at the camera's shutter, which Meter sets, and
a **+ IR** preset ends each frame with an infrared exposure at **IR Shutter**. The
**Infrared** toggle beside IR Shutter does the same for any preset.

**Scan.** Pick an output folder and a preset, then press **Scan** for each frame. Files go
to an auto-numbered per-roll subfolder and are imported and merged automatically. A Single Exposure preset writes one file per frame. **Retake** shoots the current frame again
without advancing the counter. **Capture Delay** holds both for up to 10 s before the first
exposure, so a table that shakes when you let go of the camera can settle; **Stop** ends the
wait. **Delay between exposures** adds a pause between the red, green and blue captures, for
older bodies that lock up the USB/PTP link when a new capture command arrives too soon. Scan
shoots the whole triplet into the next frame and never touches an existing file. In Live
View, **R**, **G** and **B** pick which exposures Retake shoots again over the current
frame's, so one channel of a triplet can be re-shot; Retake replaces those files. **IR**, on a
light with infrared LEDs, is the Infrared setting and rides with Scan and Retake. **IR
Shutter** on the card sets the infrared exposure's shutter, which an IR-cut filter leaves
far slower than the visible channels; a step the camera does not offer falls back to the
camera's shutter. Infrared needs a light with infrared LEDs and a camera that passes 850 nm,
a full-spectrum or IR-converted body: a stock camera's IR-cut filter leaves the infrared
frame at noise. Set the lens to manual focus: autofocus hunts on the dark infrared frame and
times out. An `_IR` file stays its own frame in the Film Strip. Each scanned frame
takes the decode its light implies: white light keeps the camera's white balance, Trichrome
and Single Exposure decode Linear RAW; a frame already edited keeps its own.

**Narrowband light.** Narrowband LEDs separate the dye layers more cleanly than white light, so
colors render more saturated. The *Digital (Adobe RGB)* print stock prints them through ideal
dyes; a paper or print film in **Print stock** prints them through that stock's own dyes.

### Sensor calibration

A single-exposure scan under narrowband light can have hues that no slider fixes, such
as yellows that drift orange. The cause is **sensor crosstalk**: the camera's color-filter
passbands overlap the source's bands, so each channel reads some of the other LEDs. It is a
fixed property of your sensor and light pair, independent of the film.

Three ways to build a profile give near-identical corrections (see
[Sensor Crosstalk Profile workflows](#sensor-crosstalk-profile-workflows)):

- **With a preset.** A Single Exposure preset calibrated with **Create Sensor Crosstalk Profile** on
  saves a profile measured through the film base (workflow A).
- **Capture from Camera…** With the camera tethered, a Scanlight connected and no film in
  the holder, NegPy lights each LED in turn, sets the shutter and saves the profile. It
  uses the camera's ISO and aperture and needs no live view. One profile serves every film
  stock and every preset (workflow B, step 1).
- **From files.** Three RAW captures of the bare light (red only, green only, blue only),
  for a light NegPy cannot control (workflow D, step 1).

Film-base and bare-light profiles differ slightly, and neither is more accurate. One
profile per sensor and light pair is enough.

The profile unmixes every scan with a 3×3 matrix in the linear domain, before inversion.
Profiles are TOML files in the `NegPy/sensor` folder. Re-run **Roll Analysis** after you
change the profile.

### Sensor Crosstalk Profile workflows

You build a profile in the preset calibration that **+** opens in the **Preset**
panel, or in the **Calibration** panel on the Roll tab. A preset that made its own profile
assigns it on every scan. You assign a Calibration-panel profile yourself, once per roll.

| Your setup | Workflow | Profile built in | Assigned by |
|---|---|---|---|
| Tethered camera and Scanlight, calibrated preset | [A](#a-the-preset-makes-the-profile) | Preset calibration | The preset |
| Tethered camera and Scanlight, calibrated preset, one profile for every preset | [B](#b-a-calibrated-preset-and-a-separate-profile) | Calibration panel | You |
| Tethered camera and Scanlight, manual preset | [C](#c-a-manual-preset) | Calibration panel | You |
| Files shot without NegPy | [D](#d-files-not-scanned-with-negpy) | Calibration panel | You |

All four apply to Single Exposure frames only. A Trichrome preset needs no profile.

#### A. The preset makes the profile

The default, for calibrated Single Exposure presets.

Once per film stock:

1. Load the film. In the **Preset** panel, press **+** beside the preset
   dropdown and pick **Calibrate on the Film Base…**. The calibration window opens with its own live view.
2. Set the ISO and the aperture you scan with, click the clear film base and name the
   preset after the stock.
3. Pick **Single Exposure**, leave **Create Sensor Crosstalk Profile** on and press **Calibrate & Save**.

For every roll of that stock:

1. Pick the preset in the dropdown.
2. Scan the roll.

Notes:

- Calibrate again only when the camera, the lens, the light or the film stock changes.
- The preset holds the profile's name; the profile is a file in the `NegPy/sensor` folder.
  If that file is deleted, the preset still scans and the roll gets no correction.
- Every scan makes the preset's profile the roll's own and turns Linear RAW on, replacing a
  profile you picked by hand for that roll.
- A new calibration under the preset's name writes the profile again, and so does any
  other profile saved under that name.
- Each preset has its own profile, measured through that film's base. Frames opened or
  scanned another way do not get it.

#### B. A calibrated preset and a separate profile

Use this when one profile is to serve every preset and film stock on the rig, when you
already have a profile for this sensor and light, or when a preset calibration reports that
its exposures were too dim to measure one.

1. Build the profile once for the rig. Take the film out of the holder. In the
   **Calibration** panel, turn **Linear RAW** on, press the calibrate button under
   *Sensor Crosstalk Profile*, name the profile, press **Capture from Camera…**
   and confirm. To use three bare-light files instead, pick them in the same dialog as in
   workflow D, step 1.
2. Load the film. Calibrate the preset as in workflow A, with **Create Sensor Crosstalk Profile** off.
3. Scan the first frame. In the **Calibration** panel, check that **Linear RAW** is on,
   pick the profile under **Profile**, and press **Roll** on the panel header.
4. Scan the rest of the roll. New frames take the roll's profile.
5. Re-run **Roll Analysis** if frames were scanned before step 3.

Notes:

- The preset sets nothing on the roll. A roll with no profile picked gets no correction.
- A bare-light profile has no film base, so its correction is close to a preset's, not
  identical.
- Build the profile again when the camera or the light changes.
- The profile is a Calibration setting, so it carries to the next frames you open. Set
  **Profile** to *None* on a roll scanned under another light.

#### C. A manual preset

A manual preset meters nothing, so it cannot make a profile. A single-exposure roll scanned
with it has no correction until you pick one.

1. Build the profile as in workflow B, step 1, or use one the rig already has.
2. In the **Preset** panel, press **+** beside PRESET, pick **Set by Hand**,
   set the levels and the exposure, set **Capture mode** to **Single Exposure**
   and save the preset.
3. Scan the first frame, then assign the profile to the roll as in workflow B, step 3.

The notes for workflow B apply. The profile does not depend on the LED levels or the
shutter, so one profile serves a manual preset at any levels.

#### D. Files not scanned with NegPy

For single-exposure RAW files shot to the card, with other tethering software, or under a narrowband light NegPy cannot control.

1. Build the profile for the camera and light that shot the files. If that rig is a
   camera NegPy can tether and a Scanlight, use **Capture from Camera…** as in workflow B.
   If not, photograph the bare light three times with no film in the holder: red only,
   green only, blue only, as RAW, exposed just below clipping. In the **Calibration**
   panel, turn **Linear RAW** on, press the calibrate button, pick the three files, name
   the profile and press **Compute and Save**.
2. Open the roll. In the **Calibration** panel, turn **Linear RAW** on.
3. Pick the profile under **Profile** and press **Roll** on the panel header.
4. Re-run **Roll Analysis**.

Notes:

- The files must be camera RAW. The profile needs Linear RAW and is grayed out without it.
- Files from another camera or light need their own profile.
- A light that cannot show one color at a time cannot measure a profile.
- The profile carries to the next frames you open, as in workflow B.

### When no profile applies

Trichrome scans have no sensor crosstalk, and NegPy skips the correction for
triplet assets. A dedicated film scanner has none either (a Coolscan's mono CCD reads one
LED at a time). Its remaining color error is film-dye crosstalk, which the density-domain
**Crosstalk** matrix corrects (see [CROSSTALK.md](CROSSTALK.md)).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| The camera dot shows **"in use"** (amber hint) | Another program holds the body. On macOS it is the system camera daemon, woken by any app that watches for cameras, often a background sync client with no window (Google Drive as a login item, Dropbox). | Quit the background app, or remove it from **System Settings → General → Login Items**. Reconnecting the cable frees the camera until that app asks again. An open Preview, Photos or Image Capture does the same. NegPy reconnects when the camera is free. |
| No camera found, and nothing else is running | The body is not in PC Remote mode, or it is a mass-storage/MTP connection. | Set the camera's USB connection mode to **PC Remote**. |
| `[-10] Timeout reading from or writing to the port`, and no other program holds it | A program crashed while connected, and the *camera* still holds the old session. NegPy retries timeouts that clear by themselves, so this one did not. | Power-cycle the camera, or unplug and replug the cable. Nothing on the computer fixes it. |
| Live view is black | The body dropped out of PC Remote, or the lens cap is on. | Power-cycle the camera. |
| The scan window opens with **"no live view"** instead of a preview | libgphoto2's entry for this body has no preview capability (Sony a6000), or it is connected in MTP mode, where no body has it. | If the message names MTP, set the USB mode to **PC Remote** and reconnect. If not, scanning works, but you set framing and focus on the camera, and calibration is not available. |
| Capture says the camera returned JPEG instead of RAW | The image-quality setting is JPEG, or RAW+JPEG selected the processed file. | Set image quality to **RAW only**, then retry. |
| Scans show motion blur that Live View did not show, and **Capture Delay** does not help | The shutter shakes the stand. The mechanical and first-curtain electronic shutter modes both move a mechanical curtain. The blur shows most at slow shutter speeds on a light stand. | Set the shutter mode to **electronic** in the camera menu. NegPy cannot set it, because libgphoto2 does not expose the shutter mode. |
| The aperture stepper is grayed out | The lens has no electronic diaphragm. | Set the aperture on the lens. |
| A setting snaps back to its old value | Property writes are asynchronous. | NegPy polls until the value lands and logs a warning if it never does. Then the setting is not writable in the body's current mode. Try **M**. |
| The Scanlight is not detected | Wrong USB-C port. | Only one of the two ports carries data; the other is power only. Use the data port. |
| The Scanlight is not detected on Linux, and the data port is right | Your user cannot open the serial port (most distributions give it to the `dialout` group). | `sudo usermod -aG dialout $USER`, then log out and back in. |

## Notes and limitations

- **USB only.** libgphoto2 can reach some cameras over the network, but not Sony bodies.
- **Only Sony bodies are tested.** NegPy looks up every control and assumes no vendor.
  `iso` and `shutterspeed` have the same names everywhere. The aperture is `f-number` on
  Sony and Panasonic, and `aperture` on Canon, Nikon, Fujifilm, Olympus and Sigma. The RAW
  suffix comes from the camera, and the still goes into memory, not onto a card. Canon and
  Nikon default to the card and will not shoot without one. Reports from other brands are
  welcome.
- **The focus magnifier depends on the vendor.** Sony packs the zoom ratio and the target
  point into one property, so a click magnifies *and* aims. Canon (`eoszoom`) and Nikon
  (`liveviewimagezoomratio`) split them, with an unknown coordinate space, so a click
  magnifies where the body already looks. Other bodies have no magnifier, and the feature
  disables itself. A body that stops streaming while magnified (the Nikon D3300) returns to
  full frame, and its magnifier stays off for the session. On these, focus with the Focus
  meter.
- **Tested on macOS.** Linux should work at least as well, but this is not verified.
- **Speed.** Most of a triplet's time is the body's per-image transfer latency. Stills are
  taken inside the running live-view session, with no reconnect per frame.
- **Credit.** The R/G/B sequencing and the exposure-calibration approach come from
  [rohanpandula/TriRGB](https://github.com/rohanpandula/TriRGB). The light is
  [jackw01/scanlight](https://github.com/jackw01/scanlight). The camera is driven through
  [python-gphoto2](https://github.com/jim-easterbrook/python-gphoto2). The narrowband
  approach follows Flückiger et al.'s work on trichromatic film scanning.
