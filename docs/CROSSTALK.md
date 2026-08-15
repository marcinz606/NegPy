# Custom Crosstalk Matrices

The **Crosstalk** control in the Calibration sidebar runs a channel-unmix correction. NegPy
ships one built-in matrix (**Generic C41**). You can drop in your own, calibrated for your
film *and your scanning setup*, without touching any code.

---

## Which film a matrix is for

A matrix describes one dye set, and Color Negative and Transparency do not share one. Each
profile therefore declares its film process:

```toml
process = "Transparency"    # or "Color Negative" (the default when the key is absent)
```

The dropdown offers only matrices for the film you are processing, and the render gates on
the same value. A mismatched profile resolves to identity rather than mixing in the wrong
correction. Every profile written before this key existed is a color negative stock, which
is why the default is `Color Negative`. Files written before the modes were renamed (`C41`,
`B&W`, `E-6`) still load.

No Transparency matrix ships with NegPy today. On slides the Matrix dropdown starts empty,
and it and the Strength slider are disabled. The editor button stays live: open it, press
**+** for a new matrix, and it is created for the process you are working in. The editor's
**Process** control sets that value on any profile you save, so you can then select a
matrix you built for a slide rig. Crosstalk is hidden outright only in B&W Negative, which
has one emulsion and so nothing to unmix.

> **Reversal film reads differently.** On a negative the dyes' unwanted absorptions are an
> error to remove before inversion, so unmixing moves the render *toward* the scene. A
> transparency **is** the finished image. What you see on a lightbox already includes those
> absorptions, so unmixing moves it *away* from the slide's own appearance and toward the
> dye-amount image behind it. In Transparency, treat Crosstalk as a color-separation
> control, not a fidelity correction, and leave Strength at 0 when you want to match the
> slide.

## What it does

A color negative's three dye layers are not spectrally pure. The cyan, magenta and yellow
dyes each leak a little density into the channels they should not affect, which gives muddy,
low-separation color. Crosstalk correction unmixes the channels: it multiplies the
per-pixel **negative density** vector by a 3×3 matrix, *before* normalization and the print
curve. That is the domain the matrices were derived in, because secondary dye absorptions
are linear in negative dye density.

**The dyes are not the only cause.** Your light source's spectrum and your sensor's
color-filter passbands mix the channels as well. A broad or oddly-shaped light samples each
dye away from its peak and picks up its neighbours, and a camera's color filters overlap.
In the density domain all three causes arrive as the same kind of error, a linear mix, which
is why one 3×3 can absorb them together, and why the same film can need different matrices
on different rigs.

So think of a matrix as a property of a **whole scanning setup**, not of a film stock. A
profile derived from a datasheet describes the dyes alone. The one that makes *your* scans
sing is the one you tuned on your own light and camera.

Two related controls sit in the same panel and are *not* substitutes. The **sensor** matrix
corrects the camera in the linear capture. **Hue Trim** rotates hue for an odd light
spectrum. See [USER_GUIDE.md](USER_GUIDE.md) §4.2.

> ### What the bundled matrices actually are
>
> Every bundled profile is **read off a published spectral-dye-density spec sheet, not
> measured on a scan**, hence the *(approx)* in each name. Two consequences are worth
> knowing before you trust one:
>
> - They model the **dyes only**. That is the complete correction only when your capture
>   reads each dye band cleanly: a **true RGB scan** (a Nikon Coolscan and friends read a
>   mono sensor under one LED at a time) or a **calibrated trichrome setup**. With a
>   broadband light and a Bayer sensor, your light and CFA add mixing the datasheet knows
>   nothing about, and a dyes-only matrix is the wrong shape for it.
> - Consumer datasheets usually publish one *aggregate* midscale-neutral curve rather than
>   separated C/M/Y curves, so even the dye part is an estimate read off a plot by eye.
>
> They are still far better starting points than identity for a trichrome rig. Just do not
> expect a bundled stock name to be "the" correction for your scanner. A matrix you tune
> yourself is the more trustworthy object. Bundled matrices are read-only; use **Make
> Editable Copy** to start from one.

The math, per pixel on the raw decoded negative:

```
density      = -log10(rgb_negative)
density_out  = M · density
```

`M` is your 3×3 matrix. The **Crosstalk** slider (0–1) blends it with the identity matrix
and row-normalizes the result, so `0` is off and `1` is the full matrix:

```
M_applied = I · (1 - strength) + M · strength
M_applied = M_applied / row_sums(M_applied)        # each row normalized to sum 1
```

Because every row is renormalized to sum to 1, a uniform grey stays grey. The matrix only
redistributes color *differences* between channels.

---

## Reading the matrix

The matrix is row-major. **Rows are output channels**, **columns are input channels**:

|            | in R   | in G   | in B   |
| :--------- | :----- | :----- | :----- |
| **out R**  | 1.00   | -0.05  | -0.02  |
| **out G**  | -0.04  | 1.00   | -0.08  |
| **out B**  | -0.01  | -0.10  | 1.00   |

- The **diagonal** stays near `1.0`, so each channel keeps its own density.
- **Off-diagonal** terms are usually small and negative. They subtract the contamination one
  layer leaks into another.
- Keep the rows summing near `1.0`. Large deviations work, because rows are normalized, but
  they make the effect harder to reason about.

---

## File format (TOML)

Put `.toml` files in your user folder:

```
<Documents>/NegPy/crosstalk/
```

The bundled gallery (`crosstalk/` in the repo) is read straight from the app and not copied
here, so those profiles stay in the dropdown beside yours and update with each release. A
bundled name wins over a user file with the same name. Each file is one matrix:

```toml
# my_film.toml
name = "Gold 200 + Spectracolor"   # optional display name; falls back to the filename
type = "tuned"                     # optional provenance; groups it in the dropdown
matrix = [                         # 3x3, row-major (out R/G/B × in R/G/B)
  [ 1.00, -0.05, -0.02],
  [-0.04,  1.00, -0.08],
  [-0.01, -0.10,  1.00],
]
```

- `matrix` is **required**: exactly 3 rows of 3 numbers.
- `name` is **optional**. Without it, the dropdown shows the filename (without `.toml`).
- `type` is **optional**. It says *where the numbers came from*, which is what decides how
  far to trust one. The dropdown groups profiles under a heading per type:

  | `type` | Dropdown group | Meaning |
  | :--- | :--- | :--- |
  | `measured` | Measured | Fitted against real scans of a known reference. |
  | `tuned` | Tuned on a rig | Dialled in by eye on real frames. What the editor saves. |
  | `specsheet-based` | From spec sheets (approx) | Read off published dye-density curves. Every bundled profile. |
  | anything else or absent | Other | Still loads and works. It just does not claim a provenance. |

- NegPy silently skips malformed files (wrong shape, non-numbers, bad TOML).
- The name `Generic C41` is reserved for the built-in matrix and ignored if reused.

NegPy **bakes the chosen matrix into the edit** when you select it, so saved edits and
presets stay reproducible even if you later move or delete the file.

---

## Using it

1. Drop your `.toml` into `<Documents>/NegPy/crosstalk/`.
2. Open the **Calibration** sidebar and find the **crosstalk dropdown** under CROSSTALK. New
   files appear the next time the panel syncs, for example when you switch photos. Restart
   if you do not see it.
3. Pick your profile and raise **Strength** above 0 to apply it.
4. Pick **Generic C41** to go back to the built-in matrix.

### Rolling your own, the short version

You do not need spectral data or a spectrophotometer to get a better matrix than *Generic
C41*, and you are the only person who can measure your own rig. Recommended loop:

1. Pick a frame whose real colors you know: foliage, sky, a grey card, skin.
2. Start on **Generic C41** and raise **Strength** until the colors separate but before they
   go garish. For many rigs this alone is the whole win.
3. Still wrong in a specific way? Open the matrix editor, press **Make Editable Copy**, and
   nudge the off-diagonal term for the pair that is off. A green leaking into red reads as
   `out R / in G`. Work one term at a time, in small steps. `0.02` is visible.
4. Save it named after the **combination**, for example `Gold 200 + Spectracolor`, not just
   the film. Re-run **Batch Analysis**, then check it on a second frame before you trust it.

If the hues are turned rather than muddied, with every color rotated the same way and
neutrals fine, that is a light-spectrum problem. **Hue Trim** is the cheaper fix, and a
matrix will fight it. Muddy, low-separation color is the crosstalk signature.

### Editing matrices in the app

The **sliders icon** beside the dropdown opens the **matrix editor**, so you do not have to
hand-edit TOML:

- Browse every profile. Bundled matrices and **Generic C41** show a lock and are
  **read-only**. Your own profiles are editable.
- Drag the off-diagonal grid sliders (`out R/G/B` × `in R/G/B`) and the preview updates live.
  The diagonal is fixed, because the matrix is row-normalized on apply, which makes the
  diagonal redundant. Only the six mixing terms are editable. The **Preview strength** slider
  controls the preview here only. Use the sidebar **Strength** slider to apply the matrix.
- **Make Editable Copy** clones the selected locked matrix into an editable profile.
- **Save** writes the profile as a `.toml` into `<Documents>/NegPy/crosstalk/`, the same
  folder profiles are read from, so it shows up in the dropdown.
- **Apply & Close** keeps what you were previewing. **Cancel** reverts.

> Crosstalk is a color operation and is hidden in B&W Negative mode. It changes what the
> normalization meters read, so re-run **Batch Analysis** (and re-save locked bounds) after
> you change the profile or the strength.

---

## Contributing a matrix

Tuned one for your setup? Please send it. Add your `.toml` to the repo's
[`crosstalk/`](../crosstalk/) folder and open a PR. Bundled matrices ship to all users on
the next release. See [`crosstalk/README.md`](../crosstalk/README.md).

Say what rig it came from, in the profile name and in the PR: the film stock, the light
source and the camera or scanner. Nobody can derive your light and sensor from a datasheet,
so a measured-in-anger profile for a common combination is more useful to other people than
any amount of theory. Light panels also get discontinued, so profiles for the ones people
own are worth keeping.
