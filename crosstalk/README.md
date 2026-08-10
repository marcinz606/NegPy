# Crosstalk matrix gallery

Community-contributed channel-unmix matrices for NegPy's **Crosstalk** controls
(Calibration panel: *Matrix* + *Strength*).

Every `.toml` here is bundled with the app and copied into a user's
`<Documents>/NegPy/crosstalk/` folder on first run, so they show up in the sidebar
dropdown out of the box.

A matrix describes a **scanning setup**, not only a film. The dyes' unwanted absorptions
are one cause of channel mixing; the light source's spectrum and the sensor's colour
filters are others, and in the density domain all three look the same. So a datasheet-derived
matrix is a starting point, and one tuned on real scans is usually better — name yours after
the whole combination (film + light + camera) rather than just the stock.

The matrices currently in this folder are all **(approx)**: read off published spectral-dye-density
spec sheets rather than measured, and describing the dyes alone. That makes them a reasonable
starting point for a **true RGB scan** (Coolscan-style, one band at a time) or a calibrated
trichrome rig, and an incomplete story for a broadband light plus a Bayer sensor, where the
capture adds mixing of its own. **Measured profiles are especially welcome** — say so in the
PR if yours came from real scans rather than a datasheet, and drop the `(approx)` from its name.

## Film process

Each profile declares the film it describes; the dropdown and the render both gate on it.

```toml
process = "E-6"    # or "C41" — the default when absent
```

Every matrix here is currently `C41`; none ships for E-6, so on slides the dropdown starts
empty and disabled until a matching profile exists (the in-app matrix editor can create one,
and stamps it with the process in use). If you add one, note that it means something different
there: a transparency is the finished image, so unmixing its dyes moves the render away from
the slide's own appearance rather than toward the scene — a separation/punch control, not a
correction. See [`../docs/CROSSTALK.md`](../docs/CROSSTALK.md).

## Contributing

1. Add one `<film_or_scanner>.toml` file to this folder.
2. Use the format below (full reference in [`../docs/CROSSTALK.md`](../docs/CROSSTALK.md)):

   ```toml
   name = "Kodak Portra 400 (Noritsu)"   # optional display name; falls back to filename
   type = "measured"                     # measured | tuned | specsheet-based
   process = "C41"                       # C41 | E-6 — which film's dyes; C41 when absent
   matrix = [                            # 3x3, row-major (out R/G/B × in R/G/B)
     [ 1.00, -0.05, -0.02],
     [-0.04,  1.00, -0.08],
     [-0.01, -0.10,  1.00],
   ]
   ```

3. Name the file after the setup it was calibrated for — film stock plus light source and
   camera/scanner where you know them (`portra_400_scanlight_a7c2.toml`).
4. Note in your PR how the matrix was derived (datasheet, test chart, or tuned by eye on
   real scans — all three are welcome, just say which) and what rig it came from.

Keep the diagonal near `1.0` and off-diagonal terms small — NegPy row-normalizes
the matrix, so it only redistributes color *differences* between channels.
