# Crosstalk matrix gallery

Community-contributed spectral-crosstalk matrices for NegPy's **Separation** control.

Every `.toml` here is bundled with the app and copied into a user's
`<Documents>/NegPy/crosstalk/` folder on first run, so they show up in the Lab
sidebar dropdown out of the box.

## Contributing

1. Add one `<film_or_scanner>.toml` file to this folder.
2. Use the format below (full reference in [`../docs/CROSSTALK.md`](../docs/CROSSTALK.md)):

   ```toml
   name = "Kodak Portra 400 (Noritsu)"   # optional display name; falls back to filename
   matrix = [                            # 3x3, row-major (out R/G/B × in R/G/B)
     [ 1.00, -0.05, -0.02],
     [-0.04,  1.00, -0.08],
     [-0.01, -0.10,  1.00],
   ]
   ```

3. Name the file after the film stock / scanner it was calibrated for.
4. Note in your PR how the matrix was derived (test chart, scanner, software).

Keep the diagonal near `1.0` and off-diagonal terms small — NegPy row-normalizes
the matrix, so it only redistributes color *differences* between channels.
