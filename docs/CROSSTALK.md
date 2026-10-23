# Custom Crosstalk Matrices

The **Crosstalk** control in the Calibration card (Roll tab) unmixes the color channels. NegPy ships
one built-in matrix (**Generic C41**). You can add your own, calibrated for your film *and
your scanning setup*, without code.

## Which film a matrix is for

Color Negative and Transparency use different dyes, so each profile declares its film
process:

```toml
process = "Transparency"    # or "Color Negative" (the default when the key is absent)
```

The dropdown offers only matrices for the current film, and the render gates on the same
value: a mismatched profile resolves to identity. The old mode names (`C41`, `B&W`, `E-6`)
still load.

No Transparency matrix ships with NegPy. On slides the Matrix dropdown starts empty, and it
and the Strength slider are disabled. The editor button stays live: **+** in the editor
makes a matrix for the current process, and the editor's **Process** control sets the
process on any profile you save. Crosstalk is hidden only in B&W Negative, which has one
emulsion and nothing to unmix.

> **Reversal film reads differently.** On a negative, unmixing moves the render *toward*
> the scene. A transparency is the finished image, so unmixing moves the render *away* from
> the slide. In Transparency, Crosstalk is a color-separation control, not a fidelity
> correction. Leave Strength at 0 to match the slide.

## What it does

The cyan, magenta and yellow dyes of a color negative each leak some density into the other
channels, which makes color muddy and low in separation. Crosstalk correction multiplies
the per-pixel **negative density** vector by a 3×3 matrix, *before* normalization and the
print curve. Secondary dye absorptions are linear in negative dye density, so the matrices
are derived in this domain.

The light source and the camera's overlapping color filters add the same kind of linear
mix, so one 3×3 corrects all three together. A matrix is a property of a **whole scanning
setup**, not of a film stock, and the same film can need different matrices on different
rigs.

Two related controls in the same panel do a different job. The **sensor** matrix corrects
the camera in the linear capture. **Hue Trim** rotates hue for an odd light spectrum. See
[USER_GUIDE.md](USER_GUIDE.md) §10.3.

> ### What the bundled matrices actually are
>
> Every bundled profile is **read off a published spectral-dye-density spec sheet, not
> measured on a scan**, hence the *(approx)* in each name.
>
> - They model the **dyes only**. That is the full correction only when the capture reads
>   each dye band cleanly: a **Narrowband Scanner** (a Coolscan's mono sensor reads one LED
>   at a time, fully clean; a Pakon's trilinear array comes close, with slight residual
>   bleed), a **Trichrome** capture, or a Single Exposure rig with
>   **Sensor Crosstalk Profile** applied. A broadband light and a Bayer sensor
>   add mixing that a dyes-only matrix cannot model.
> - Consumer datasheets usually publish one *aggregate* midscale-neutral curve, not
>   separate C/M/Y curves. For the Kodak films, Kodak's Vision3 dye curves are fitted to
>   that curve, which assumes the film's masking couplers match Vision3's. The other
>   consumer matrices are read off the plot by eye.
>
> On a narrowband rig they still beat identity, but a matrix you tune yourself is more
> trustworthy. Bundled matrices are read-only. Use **Make Editable Copy** to start from one.

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

Every row sums to 1, so a uniform gray stays gray. The matrix only redistributes color
*differences* between channels.

## Reading the matrix

The matrix is row-major. **Rows are output channels**, **columns are input channels**:

|            | in R   | in G   | in B   |
| :--------- | :----- | :----- | :----- |
| **out R**  | 1.00   | -0.05  | -0.02  |
| **out G**  | -0.04  | 1.00   | -0.08  |
| **out B**  | -0.01  | -0.10  | 1.00   |

- The **diagonal** stays near `1.0`, so each channel keeps its own density.
- **Off-diagonal** terms are usually small and negative. They subtract what one layer leaks
  into another.
- Keep the rows summing near `1.0`. Rows are normalized, so larger deviations work, but are
  harder to reason about.

## File format (TOML)

Put `.toml` files in your user folder:

```
<Documents>/NegPy/crosstalk/
```

The app reads the bundled gallery (`crosstalk/` in the repo) in place, so those profiles
show beside yours and update with each release. A bundled name wins over a user file with
the same name. Each file is one matrix:

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
- `type` is **optional**. It says *where the numbers came from*, and so how far to trust
  them. The dropdown groups profiles by type:

  | `type` | Dropdown group | Meaning |
  | :--- | :--- | :--- |
  | `measured` | Measured | Fitted against real scans of a known reference. |
  | `tuned` | Tuned on a rig | Dialed in by eye on real frames. The editor saves this type. |
  | `specsheet-based` | From spec sheets (approx) | Read off published dye-density curves. Every bundled profile. |
  | anything else or absent | Other | Loads and works, with no provenance. |

- NegPy silently skips malformed files (wrong shape, non-numbers, bad TOML).
- The name `Generic C41` is reserved for the built-in matrix and ignored if reused.

NegPy **bakes the chosen matrix into the edit** when you select it, so saved edits and
presets stay reproducible if you later move or delete the file.

## Using it

1. Put your `.toml` into `<Documents>/NegPy/crosstalk/`.
2. Open the **Calibration** card on the Roll tab and find the **crosstalk dropdown** under CROSSTALK. New
   files show the next time the panel syncs, for example when you switch photos. If yours
   does not show, restart.
3. Pick your profile and raise **Strength** above 0.
4. Pick **Generic C41** to go back to the built-in matrix.

Crosstalk changes what the normalization meters read, so re-run **Roll Analysis** (and
re-save locked bounds) after you change the profile or the strength.

### Rolling your own, the short version

You need no spectral data or spectrophotometer to beat *Generic C41*:

1. Pick a frame whose real colors you know: foliage, sky, a gray card, skin.
2. Start on **Generic C41** and raise **Strength** until the colors separate, before they
   go garish. On many rigs this is enough.
3. Still wrong in a specific way? Open the matrix editor, press **Make Editable Copy**, and
   change the mixing term for the pair that is off. Green leaking into red is **Red ›
   From Green**. Change one term at a time, in small steps. `0.02` is visible.
4. Name it after the **combination**, for example `Gold 200 + Spectracolor`. Re-run **Roll
   Analysis** and check it on a second frame.

If every hue turns the same way and neutrals are fine, the light spectrum is the problem.
**Hue Trim** is the cheaper fix, and a matrix will fight it. Muddy, low-separation color is
the crosstalk signature.

### Editing matrices in the app

The **sliders icon** beside the dropdown opens the **matrix editor**:

- Browse every profile. Bundled matrices and **Generic C41** show a lock and are
  **read-only**. Your own profiles are editable.
- Each channel (**Red**, **Green**, **Blue**) has two sliders, **From** each other channel,
  from −1 to 1, and the preview updates live. Row normalization makes the diagonal
  redundant, so only the six mixing terms are editable. The **Preview Strength** slider
  controls only this preview. The card's **Strength** slider applies the matrix.
- **Make Editable Copy** clones the selected locked matrix into an editable profile.
- **Save** writes the profile as a `.toml` into `<Documents>/NegPy/crosstalk/`, so it shows
  in the dropdown.
- **Apply & Close** keeps what you were previewing. **Cancel** reverts.

### Measuring from a chart

A chart shot on the film is the one reference that settles a matrix. Shoot a SpyderCheckr 24 or
a ColorChecker 24 in daylight on a frame of the roll, scan it on your rig, and open it. The
**swatch button** beside the dropdown opens **Measure from Chart** (Color Negative only).

1. Drag the four dots onto the centers of the chart's four corner patches. The dot order and the
   chart's way up do not matter: NegPy reads the grid every way, sideways and mirrored included,
   and keeps the one whose colors match the chart.
2. Name the profile after the combination and press **Measure and Save**.

The fit starts from identity and runs the frame's own render for every trial matrix, so
normalization and Cast Removal re-meter each time. It prints on the Digital stock, so a paper's dyes
stay out of the rig's matrix. It minimizes CIEDE2000 against the chart's
published values (D50, adapted to the render's D65), with lightness at half weight because the
print curve owns tone, and a small pull toward identity. One exposure gain matches the mid grays
first. The result is saved with `type = "measured"` and applied at Strength 1. Re-run **Roll
Analysis** on the other frames.

After the fit, each sample box shows its patch's color ΔE00 to the chart's value: the hue and
chroma difference, with lightness left out because the print curve sets it. Green is within 2,
amber within 5, red beyond. Moving a dot clears them.

A matrix measured on one stock under one light describes that pair only. The sensor unmix runs
before Crosstalk, so the fit measures only what the frame's sensor crosstalk profile leaves; measure again
if you change that profile.

## Contributing a matrix

Add your `.toml` to the repo's [`crosstalk/`](../crosstalk/) folder and open a PR.
Bundled matrices ship to all users in the next release. See
[`crosstalk/README.md`](../crosstalk/README.md). Name the rig in the profile name and in the
PR: the film stock, the light source and the camera or scanner.
