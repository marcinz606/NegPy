# Minolta DiMAGE Scan Elite 5400 and 5400 II

NegPy drives the DiMAGE Scan Elite 5400 (USB id `0686:400e`) and the Konica Minolta DiMAGE Scan
Elite 5400 II (USB id `132b:0012`) through the `dimage5400` driver. The driver talks to the scanner
directly over USB. It needs no vendor software and no firmware file. The two models share one
backend: select **Minolta DiMAGE Scan Elite 5400 / 5400 II** in the Scan panel, and the panel
lists whichever model is connected.

## Install

```bash
uv sync --group dimage5400
```

On Linux, give your user access to the scanner with a udev rule, then reconnect the scanner:

```bash
sudo tee /etc/udev/rules.d/60-dimage5400.rules <<'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="132b", ATTR{idProduct}=="0012", MODE="0666", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="0686", ATTR{idProduct}=="400e", MODE="0666", TAG+="uaccess"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

On Windows, assign the WinUSB driver to the scanner once, with [Zadig](https://zadig.akeo.ie):

1. Connect the scanner and switch it on.
2. In Zadig, select **Options > List All Devices**, then select the scanner (USB ID `0686 400E`
   for the 5400, `132B 0012` for the 5400 II).
3. Set the target driver to **WinUSB** and select **Replace Driver** (or **Install Driver**).

The Windows build includes libusb. While WinUSB is assigned, the Minolta software does not see
the scanner. To go back, uninstall the device in Device Manager with **Delete the driver
software** selected, then reconnect it.

## Start-up order

The scanner does not accept a film holder until the host has calibrated it. The driver runs this
calibration when it opens the scanner with the slot empty.

1. Switch the scanner on with no holder inside. Wait until the lamp stops blinking.
2. Open the Scan panel (or refresh the device list). This calibrates the scanner.
3. Start a preview or a scan, then insert the holder when the panel asks for it. The scanner
   pulls it in.

The 5400 pulls a holder in only while the host asks for the holder state, so insert it while a
preview or scan waits for it. The 5400 II takes it at any time after the calibration.

The calibration stays valid while the scanner is on. If the scanner rejects the holder, eject it,
refresh the device list, and insert it again.

A fast-blinking lamp is an error state. Switch the scanner off, remove the holder, and switch it on
again.

## Grain Dissolver

The 5400 has a diffuser that it can move into the light path. The **Grain Dissolver** toggle in
the Scan panel turns it on for scans; previews scan without it. It softens grain and evens out
the light, and scans take longer. The driver keeps a separate calibration and film-path profile
for each position, and uses the ones that match the scan.

## What the driver corrects

- **Shading.** Each scan is divided by the scanner's own white and dark reference, read at the
  scan's exposure, so lamp fall-off and pixel-to-pixel sensitivity do not reach the image.
- **Exposure.** Auto exposure meters the frame and sets each channel separately, up to the
  scanner's limit. The result is film transmission, so it does not change with exposure.
- **Film-path profile.** The light source gives an uneven pattern across the sensor that shows
  only when film is in the light path. The driver removes it with a profile of your scanner (see
  below).
- **Infrared.** The scanner reads the infrared line through all three color rows. The driver
  combines them into one IR channel for dust and scratch removal, weighted by the signal each row
  gets.
- **Orientation.** The 5400 reads across the film in the opposite direction to the 5400 II. The
  driver flips its images, so both models give the same orientation.

## Measure the film-path profile

Do this once per scanner, and on the 5400 once more with `--grain-dissolver`. Put a strip in the
holder that has a clear frame on it (an unexposed frame or the leader) and leave one slot without
film. A slot that the end of a short strip covers only in part also works.

```bash
uv run dimage5400 profile
```

The driver scans every slot at low resolution, finds the clear frame and the open slot, and asks
before it goes on. Give them with `--frame` and `--empty` to skip the search, and `--yes` to skip
the question. Then eject the holder and turn the strip half a turn flat on the table, so that the
other end goes in first. Keep the same side down. Insert it and run the same command again.

The film's own density changes direction between the two scans and cancels; the part that
belongs to the scanner stays. Nothing about the light is assumed: each color keeps the detail
that repeats between the two scans and drops what does not, so the narrow streaks of a bare LED
strip stay in the profile and a smooth, noisier lamp gives a smooth profile. The command prints
the finest detail it kept in each color. `--smooth` sets a fixed smoothing instead.

The first pass without the Grain Dissolver also measures the scanner's exposure response: it
scans the clear frame at nine exposures and finds where the signal stops rising. Lamps differ
from unit to unit, and auto exposure uses the measured figures. `--keep-exposure` skips it.

Everything is saved in `~/.local/share/dimage5400/` (`%LOCALAPPDATA%\dimage5400\` on Windows).
`--rebuild` computes the profile again from the two stored passes without a new scan.

## Frame position

The holder positions six frames at fixed places. A strip that sits short of the frame marks moves
every frame along the feed. Use the frame offset in the strip preview to move the scan window.
