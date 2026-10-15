# Crosstalk matrix gallery

Community-contributed channel-unmix matrices for NegPy's **Crosstalk** controls
(Calibration panel: *Matrix* + *Strength*).

NegPy bundles every `.toml` here and reads it in place, beside the user's own matrices in
`<Documents>/NegPy/crosstalk/`. They show in the sidebar dropdown from the first launch and
update with each release.

A matrix describes a **scanning setup**, not only a film: the dyes, the light source's
spectrum and the sensor's color filters all mix the channels, and in the density domain
they look the same. Name yours after the whole combination (film + light + camera).

All matrices in this folder are **(approx)**: read off published spectral-dye-density spec
sheets, not measured, and they describe the dyes alone. They are a good start for a
**Narrowband Scanner** (a Coolscan's mono sensor reads one LED at a time, fully clean; a
Pakon's trilinear array comes close, with slight residual bleed), a **Trichrome** capture,
or a Single-Shot Narrowband rig with Single-Shot Narrowband Calibration applied. With a
broadband light and a Bayer sensor they are incomplete. Measured profiles are especially
welcome: say so in the PR, and drop the `(approx)` from the name.

## Film process

Each profile declares the film it describes. The dropdown and the render both gate on it.

```toml
process = "Transparency"    # or "Color Negative" (the default when absent)
```

The old names `C41` and `E-6` still load. Every matrix here is Color Negative. None ships
for Transparency, so on slides the dropdown starts empty and disabled until a matching
profile exists (the in-app matrix editor can create one, stamped with the process in use).
On a transparency, unmixing moves the render away from the slide's own appearance: a
separation control, not a correction. See [`../docs/CROSSTALK.md`](../docs/CROSSTALK.md).

## Contributing

1. Add one `<film_or_scanner>.toml` file to this folder.
2. Use the format below (full reference in [`../docs/CROSSTALK.md`](../docs/CROSSTALK.md)):

   ```toml
   name = "Kodak Ektacolor Pro 400 (Noritsu)"   # optional display name; falls back to filename
   type = "measured"                     # measured | tuned | specsheet-based
   process = "Color Negative"            # Color Negative | Transparency; Color Negative when absent
   matrix = [                            # 3x3, row-major (out R/G/B × in R/G/B)
     [ 1.00, -0.05, -0.02],
     [-0.04,  1.00, -0.08],
     [-0.01, -0.10,  1.00],
   ]
   ```

3. Name the file after the setup it was calibrated for: film stock plus light source and
   camera/scanner where you know them (`ektacolor_pro_400_scanlight_a7c2.toml`).
4. In your PR, say how you derived the matrix (datasheet, test chart, or tuned by eye on
   real scans) and what rig it came from.

Keep the diagonal near `1.0` and the off-diagonal terms small. NegPy row-normalizes the
matrix, so it only redistributes color *differences* between channels.
