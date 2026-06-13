# Custom Crosstalk Matrices

The **Separation** control in the Lab sidebar runs *spectral crosstalk* correction.
NegPy ships with one built-in matrix (**Default**), but you can drop in your own —
calibrated per film stock or scanner — without touching any code.

---

## What it does

A color negative's three dye layers are not spectrally pure: the cyan, magenta and
yellow dyes each leak a little density into the channels they shouldn't affect. The
result is muddy, low-separation color. Crosstalk correction *unmixes* the channels by
multiplying the per-pixel **density** vector by a 3×3 matrix.

The math, per pixel:

```
density      = -log10(rgb)
density_out  = M · density
rgb_out      = 10^(-density_out)
```

`M` is your 3×3 matrix. The **Separation** slider blends it with the identity matrix
and row-normalizes the result, so `Separation = 1.0` is a no-op and higher values push
toward the full matrix:

```
M_applied = I · (1 - strength) + M · strength      # strength = Separation - 1.0
M_applied = M_applied / row_sums(M_applied)        # each row normalized to sum 1
```

Because every row is renormalized to sum to 1, a uniform gray stays gray — the matrix
only redistributes color *differences* between channels.

---

## Reading the matrix

The matrix is row-major. **Rows are output channels**, **columns are input channels**:

|            | in R   | in G   | in B   |
| :--------- | :----- | :----- | :----- |
| **out R**  | 1.00   | -0.05  | -0.02  |
| **out G**  | -0.04  | 1.00   | -0.08  |
| **out B**  | -0.01  | -0.10  | 1.00   |

- The **diagonal** stays near `1.0` (each channel keeps its own density).
- **Off-diagonal** terms are usually small and negative — they subtract the
  contamination one layer leaks into another.
- Keep rows roughly summing near `1.0`; large deviations are fine (they get
  normalized) but make the effect harder to reason about.

---

## File format (TOML)

Put `.toml` files in your user folder:

```
<Documents>/NegPy/crosstalk/
```

On first run NegPy copies the bundled gallery (`crosstalk/` in the repo) here, so
you start with some ready-made profiles. Each file is one matrix:

```toml
# my_film.toml
name = "Kodak Gold 200"        # optional display name; falls back to the filename
matrix = [                     # 3x3, row-major (out R/G/B × in R/G/B)
  [ 1.00, -0.05, -0.02],
  [-0.04,  1.00, -0.08],
  [-0.01, -0.10,  1.00],
]
```

- `matrix` is **required**: exactly 3 rows of 3 numbers.
- `name` is **optional**. If omitted, the dropdown shows the filename (without `.toml`).
- Malformed files (wrong shape, non-numbers, bad TOML) are silently skipped.
- The name `Default` is reserved for the built-in matrix and ignored if reused.

The chosen matrix is **baked into the edit** when you select it, so saved edits and
presets stay reproducible even if you later move or delete the file.

---

## Using it

1. Drop your `.toml` into `<Documents>/NegPy/crosstalk/`.
2. Open the **Lab** sidebar → the **crosstalk dropdown** under COLOR. New files appear
   the next time the panel syncs (e.g. switching photos); restart if you don't see it.
3. Pick your profile and raise **Separation** above 1.0 to apply it.
4. Pick **Default** to revert to the built-in matrix.

> Crosstalk is a color operation and is hidden in B&W mode.

---

## Contributing a matrix

Calibrated a film stock or scanner? Add your `.toml` to the repo's
[`crosstalk/`](../crosstalk/) folder and open a PR — bundled matrices ship to all
users on the next release. See [`crosstalk/README.md`](../crosstalk/README.md).
