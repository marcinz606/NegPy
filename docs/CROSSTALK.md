# Custom Crosstalk Matrices

The **Crosstalk** control in the Calibration sidebar runs channel-unmix
correction. NegPy ships with one built-in matrix (**Generic C41**), but you can drop in your
own — calibrated for your film *and your scanning setup* — without touching any code.

---

## What it does

A color negative's three dye layers are not spectrally pure: the cyan, magenta and
yellow dyes each leak a little density into the channels they shouldn't affect. The
result is muddy, low-separation color. Crosstalk correction *unmixes* the channels by
multiplying the per-pixel **negative density** vector by a 3×3 matrix, *before*
normalization and the print curve — the domain the matrices were derived in
(secondary dye absorptions are linear in negative dye density).

**The dyes are not the only cause.** Your light source's spectrum and your sensor's
colour-filter passbands mix the channels as well: a broad or oddly-shaped light samples
each dye away from its peak and picks up its neighbours, and a camera's colour filters
overlap. In the density domain all three causes arrive as the same kind of error — a
linear mix — which is why one 3×3 can absorb them together, and why the same film can
need different matrices on different rigs.

So think of a matrix as belonging to a **whole scanning setup**, not to a film stock.
A profile someone derived from a datasheet describes the dyes alone; the one that
actually makes *your* scans sing is the one you tuned on your own light and camera.

Two related controls sit in the same panel and are *not* substitutes: the **sensor**
matrix corrects the camera in the linear capture, and **Hue Trim** rotates hue for an
odd light spectrum. See [USER_GUIDE.md](USER_GUIDE.md) §4.2.

> ### What the bundled matrices actually are
>
> Every bundled profile is **read off a published spectral-dye-density spec sheet, not
> measured on a scan** — hence the *(approx)* in each name. Two consequences worth knowing
> before you trust one:
>
> - They model the **dyes only**. That is the complete correction only when your capture
>   reads each dye band cleanly — a **true RGB scan** (a Nikon Coolscan and friends read a
>   mono sensor under one LED at a time) or a **calibrated trichrome setup**. With a
>   broadband light and a Bayer sensor, your light and CFA add mixing the datasheet knows
>   nothing about, and the dyes-only matrix is not the right shape for it.
> - Consumer datasheets usually publish one *aggregate* midscale-neutral curve rather than
>   separated C/M/Y curves, so even the dye part is an estimate read off a plot by eye.
>
> None of that makes them useless — they are a far better starting point than identity for
> a trichrome rig. It does mean you should not expect a bundled stock name to be "the"
> correction for your scanner, and that a matrix you tune yourself is the more trustworthy
> object. Bundled matrices are read-only; **Make Editable Copy** to start from one.

The math, per pixel on the raw decoded negative:

```
density      = -log10(rgb_negative)
density_out  = M · density
```

`M` is your 3×3 matrix. The **Crosstalk** slider (0–1) blends it with the identity
matrix and row-normalizes the result, so `0` is off and `1` is the full matrix:

```
M_applied = I · (1 - strength) + M · strength
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

The bundled gallery (`crosstalk/` in the repo) is read straight from the app, not copied
here, so those profiles stay in the dropdown alongside yours and update with each release.
A bundled name wins over a user file with the same name. Each file is one matrix:

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
- `name` is **optional**. If omitted, the dropdown shows the filename (without `.toml`).
- `type` is **optional** and says *where the numbers came from*, which is what decides how
  far to trust one. The dropdown groups profiles under a heading per type:

  | `type` | Dropdown group | Meaning |
  | :--- | :--- | :--- |
  | `measured` | Measured | Fitted against real scans of a known reference. |
  | `tuned` | Tuned on a rig | Dialled in by eye on real frames. What the editor saves. |
  | `specsheet-based` | From spec sheets (approx) | Read off published dye-density curves. Every bundled profile. |
  | anything else / absent | Other | Still loads and works — it just isn't claiming a provenance. |

- Malformed files (wrong shape, non-numbers, bad TOML) are silently skipped.
- The name `Generic C41` is reserved for the built-in matrix and ignored if reused.

The chosen matrix is **baked into the edit** when you select it, so saved edits and
presets stay reproducible even if you later move or delete the file.

---

## Using it

1. Drop your `.toml` into `<Documents>/NegPy/crosstalk/`.
2. Open the **Calibration** sidebar → the **crosstalk dropdown** under CROSSTALK. New files
   appear the next time the panel syncs (e.g. switching photos); restart if you don't see it.
3. Pick your profile and raise **Strength** above 0 to apply it.
4. Pick **Generic C41** to revert to the built-in matrix.

### Rolling your own — the short version

You do not need spectral data or a spectrophotometer to get a better matrix than
*Generic C41*, and you are the only person who can measure your own rig. Recommended loop:

1. Pick a frame you know the real colours of — foliage, sky, a grey card, skin.
2. Start on **Generic C41** and raise **Strength** until colours separate but before they go
   garish. For many rigs this alone is the whole win.
3. Still wrong in a specific way? Open the matrix editor, **Make Editable Copy**, and nudge
   the off-diagonal term for the pair that is off (a green leaking into red reads as
   `out R / in G`). Work one term at a time and in small steps — `0.02` is visible.
4. Save it named after the **combination**, e.g. `Gold 200 + Spectracolor`, not just the
   film. Re-run **Batch Analysis**, then check it on a second frame before trusting it.

If hues are turned rather than muddied — every colour rotated the same way, neutrals
fine — that is a light-spectrum problem and **Hue Trim** is the cheaper fix; a matrix
will fight it. Muddy, low-separation colour is the crosstalk signature.

### Editing matrices in the app

The **sliders icon** next to the dropdown opens the **matrix editor**, so you don't
have to hand-edit TOML:

- Browse every profile — bundled matrices (and **Generic C41**) show a lock and are
  **read-only**; your own profiles are editable.
- Drag the off-diagonal grid sliders (`out R/G/B` × `in R/G/B`) and the preview updates
  live. The diagonal is fixed — the matrix is row-normalized on apply, which makes the
  diagonal redundant, so only the six mixing terms are editable. The **Preview strength**
  slider only controls how strongly the matrix previews here — it's view-only; use the
  sidebar **Strength** slider to actually apply it.
- **Make Editable Copy** clones the selected (locked) matrix into an editable profile.
- **Save** writes the profile as a `.toml` into `<Documents>/NegPy/crosstalk/` — the same
  folder profiles are read from — so it shows up in the dropdown.
- **Apply & Close** keeps what you were previewing; **Cancel** reverts.

> Crosstalk is a color operation and is hidden in B&W mode. Because it changes what
> the normalization meters read, re-run **Batch Analysis** (and re-save locked bounds)
> after changing the profile or strength.

---

## Contributing a matrix

Tuned one for your setup? Please send it. Add your `.toml` to the repo's
[`crosstalk/`](../crosstalk/) folder and open a PR — bundled matrices ship to all users on
the next release. See [`crosstalk/README.md`](../crosstalk/README.md).

Say what rig it came from in the profile name and the PR: the film stock, the light source
and the camera or scanner. Nobody can derive your light and sensor from a datasheet, so a
measured-in-anger profile for a common combination is more useful to other people than any
amount of theory — and light panels get discontinued, so profiles for the ones people
actually own are worth keeping.
