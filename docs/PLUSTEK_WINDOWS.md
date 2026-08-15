# Windows USB setup (Plustek OpticFilm)

NegPy's Plustek USB backend uses the external [pyopticfilm](https://github.com/jboneng/pyopticfilm) driver (libusb through PyUSB). The stock Plustek Windows driver must not own the device.

## Requirements

- Windows 10 or 11
- OpticFilm **8200i SE** (`07B3:1825`, GL128). This is the only model validated for scan
- WinUSB (or libusbK) bound with [Zadig](https://zadig.akeo.ie/)
- NegPy with the `plustek` optional dependency (`uv sync --group plustek` or `pip install negpy[plustek]`). On Windows, pyopticfilm also pulls in `libusb-package`

## 1. Confirm the device

Power the scanner and plug it in, then:

1. Open **Device Manager**
2. Look under imaging or USB devices for Plustek
3. Properties → Details → Hardware Ids must include `VID_07B3&PID_1825`

## 2. Bind WinUSB (Zadig)

1. Download [Zadig](https://zadig.akeo.ie/)
2. Select **Options → List All Devices**
3. Select the Plustek Film Scanner (`07B3:1825`)
4. Replace the driver with **WinUSB**. libusbK also works
5. Keep the stock Plustek driver installer. You need it to restore VueScan or the vendor software later

While WinUSB is bound, Plustek's stock Windows scanning apps cannot see the device.

## 3. Run NegPy and scan

From a source checkout:

```powershell
cd path\to\NegPy
uv sync --group plustek
make run
```

A Windows release build works too. It bundles pyopticfilm, PyUSB and libusb. In the Scan tab, set Backend to **pyOpticfilm (Plustek)**. Refresh the device list. The SE appears when WinUSB is bound.

## 4. Restore the vendor driver

1. Unplug the scanner
2. Device Manager → uninstall the WinUSB device. Tick "delete driver software" if it is offered
3. Reinstall the Plustek or VueScan driver package
4. Plug the scanner in again

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Empty device list | Wrong PID, unplugged, or still on the vendor driver |
| `DriverBindingError` or access denied | WinUSB is not bound, or another app holds the handle |
| `UsbError` or link failures | Cable or hub. Try a direct motherboard port |
| UI hints at missing USB / PyUSB | Install the plustek group: `uv sync --group plustek` |
| The first scan at a DPI takes a few seconds | Normal. AFE plus one dark and one white shading strip, the same choreography as SilverFast. It is then cached per resolution |
| ASIC shading ready (preferred) | The log shows a colour white mean of about 50–57k, `median_gain` about 1.1–1.7x, then `ASIC shading ready` and image `shading=True` / DVDSET on. Delete `plustek_calib` and rescan after a driver change so the calibration is measured again |
| Colour white mean about 50–57k | Normal. At unity gain, DVDSET returns `raw − dark`, so a bright strip reads near full scale. The 11–13k figures in the captures are the *gains* SilverFast computed from it (`0x2000` = 1.0), not white levels |
| Median gain outside 1.02–3.0x | Below the band the strip is already at target, so there is nothing to flatten. Above it the light path is too dim. Check the lamp, the AFE gains, and that the head sits on clear home chrome |
| Carriage not at home after shading or a scan abort | The white strip arms `AGOHOME` during the measure, and clearing SCAN parks it. If the park timed out, use SilverFast or power-cycle, then retry |
| Corners darker than SilverFast on the Full window | The Full window includes about 0.8 mm of holder chrome. Shading is per column only, so residual Y falloff at those edges is expected against a tighter SilverFast frame |
| The positive is very dark until you crop (white edges on the negative) | Holder chrome is brighter than film base, so the NegPy auto bounds latch onto it. Both paths clamp border highlights to the film inset (`border highlight clamp…`). On the ASIC path this is essential, because DVDSET maps that same chrome to full scale by construction. Also raise Process → **Analysis Buffer**, or crop before auto |
| Rainbow vertical "barcode" stripes | The uploaded table was indexed differently from the image. Either the measurement went into the gain slot (gain must be `0xFFFF × 0x2000 / white`, which is proportional to `1/white`), or the blob was packed without its block padding. The AHB table is 512-byte blocks of 126 `(dark, gain)` pairs plus two `gain = 0` pad pairs, so contiguous records slide 8 bytes per block. Delete `plustek_calib` and rescan |
| Diamond or sheared scene (objects lean) | (1) Image X shrank while USB still paced the full line. The shading table must cover every acquired column (the Full window at 1800 stays about 2592 px). (2) An odd crop USB width at 1800, for example 2455. The output width must stay even (`optical_span_alignment` plus an even pixel count). Delete `plustek_calib` (cache v9+) and rescan. The log must show an even `pixels=` |
| Negative very dark, or positive washed bright, against SilverFast | Both paths reference *home* chrome, which is brighter than the light at the scan position. At 1800 dpi the film base lands near 42% of full scale, about 1.2 stops down, and NegPy meters a thin negative. `expose_film_base` lifts it with one scalar gain keyed to the brightest channel (`… exposure makeup gain=…`). The gain must stay scalar, or it neutralizes the orange mask that inversion needs |
| Strong orange or pink cast on the positive | Check `AFE search done gains=…` for a channel at or near `AFE_GAIN_MAX` (511). At the rail that channel's dark term clips to 0 (`dark0=(0, …)`), so it loses its blacks and tints the whole frame. The search substitutes SF's session-04 code for any pegged channel. A persistent peg means the AFE gain target is unreachable. The target is an *AFE-strip* level, not a shading white, so do not raise it to match SF's ~50k probe mean |
| Strong green cast on the positive | The border clamp ceiling must be **per channel**. A single joint percentile flattens the margin to neutral grey at a level the dimmest channel never reaches in the film (session 004: joint 27432 against green's own 19306), so auto Dmin meters green off chrome and lifts it 1.4x. Check `border highlight clamp peak_p99.7=(r,g,b)`. Each figure must sit just above that channel's own film peak |
| `white clipped at the rail` | The white strip is pinned near `0xFFFF`, so it carries no shape to flatten. Lower the AFE gains. DVDSET stays off and the host stretch runs |
| Scan fails: white mean below 20000 | The post-unity strip is too dim to reach target within the 4x gain clamp. Often a stale AHB strip: dark and AFE wait until the buffer has data **at home**, and a motor-busy `0xa5` is not ready there. Also check that the lamp is off for dark and that the head is on clear home chrome. Cache v6+ ignores collapsed calibration |
| Scan fails: colour ASIC shading / clear home field | The carriage is not on the clear home sensor. Park or power-cycle, then retry. The film may stay loaded |
