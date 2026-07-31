# NegPy User Guide

NegPy turns film scans into finished positives with a non-destructive, darkroom-style pipeline. Nothing is ever written back to your source files. Every edit lives in a local database, so you can experiment freely.

This guide is for new users. It explains what each control does, when you'd reach for it, and roughly what it does to your image. If you just want to know *why* the pipeline is ordered the way it is, read [PIPELINE.md](PIPELINE.md).

---

## 1. The Big Picture

### Screen layout

*   **Left, the film strip**: your loaded frames as a contact sheet, plus import, sorting, and triage tools.
*   **Centre, the canvas**: the live preview of the current frame. Most tools (crop, white-balance picker, heal brush, dodge/burn masks) are used by clicking directly on it.
*   **Right, the controls**: a pinned **Analysis** readout at the top, and below it an icon tab bar. Each icon opens a *workflow page* holding one or more collapsible panels.

### The workflow (and the order things happen)

The right-hand tabs are arranged in the order you actually work, which mirrors the processing pipeline:

| Tab | Icon | Panels | What it's for |
|-----|------|--------|---------------|
| **Setup** | cogs | Presets · Sensor Calibration · Process · Roll Analysis | Film type, negative→positive normalization, roll-wide baselines |
| **Geometry** | crop | Geometry · Flat Field | Crop, straighten, lens/falloff correction |
| **Exposure** | sun | Filtration · Tone · Dodge & Burn | White balance, print density/contrast/curve/saturation, local burns |
| **Colour** | palette | Lab · Toning | Chroma, sharpening, effects, split/chemical toning |
| **Finish** | brush | Retouch · Finishing | Dust removal, vignette, border, carrier |
| **Favourites** | star | Your chosen sliders | Quick access to the controls you use most |
| **History** | clock | Edit history | Step back through every change |
| **Export** | file | Export settings | Format, size, colour, batch output |
| **Metadata** | tags | Archival metadata | Original camera/lens/film details |
| **Scan** | camera | Scanner · Camera Scanning | Capture film directly (Linux/macOS) |

You don't have to touch every panel. NegPy's defaults are tuned to produce a good print straight away, and most frames need only a crop, maybe a white-balance nudge, and export.

A small **dot** on a panel header (and on a tab icon) means you've changed something from its default. Every panel header has a **reset** action to return that panel to defaults, and an **ⓘ** that opens this guide at that panel's section.

Both side panels can be narrowed to give the canvas more room. As the controls panel shrinks, tab icons that no longer fit move into a **»** menu at the right of the tab bar; the tab you are on always stays visible.

---

## 2. Film strip (left panel)

The header shows the NegPy logo and version (and an update link when a new release is out). Below it is the file browser.

### Importing & managing files

Toolbar buttons, left to right:

*   **Add files** / **Add folder**: load individual images or every image in a folder.
*   **Clear all**: unload everything (or, when several frames are selected, unload just those).
*   **Hot Folder**: watches the current folder and auto-loads new files as they appear, handy when a scanner drops files into a directory.
*   **RGB Scan**: treats the folder as red/green/blue exposure triplets and assembles each frame from three shots (for narrowband trichrome scanning). Right-click a frame → **Edit RGB Triplet…** to assign the three files by hand.
*   **Half Frame**: splits each scan into two frames (for half-frame cameras), edited and metered separately.
*   **Apply (clone)**: copy the current frame's settings to selected frames or the whole roll. You choose which aspects in a dialog (crop and rotation are always per-image).
*   **Sheet filter** (funnel): show *All frames*, *Keepers only*, or *Hide rejected*.
*   **Sort**: by Name or Date, ascending or descending.

Below the toolbar: a **filter box** (substring match; toggle **`.*`** for regex) and a **tally**, e.g. "36 frames · 12 keepers · 3 rejected".

Right-clicking **empty space** in the film strip offers **Add files**, **Add folder** and **Clear all**, so those tools stay in reach part-way down a long roll instead of only at the top of the panel. Here **Clear all** always means the whole session, never just the selection.

Narrow the panel and the toolbar buttons that no longer fit move into a **»** menu at its right edge, so the panel can be squeezed down to give the image more room without losing any tool.

### Triage (culling the roll)

Right-click a thumbnail (or use keyboard shortcuts) to mark frames while you review the sheet:

*   **Keep**: a small check badge marks a keeper.
*   **Reject**: a cross badge dims the frame. Rejected frames stay on the sheet but are skipped by batch exports and sidecar writes. **The file on disk is never touched.**

Marks apply to a multi-selection and persist across sessions. A badge in the top-right corner instead flags a frame that failed to decode.

The right-click menu also offers **Copy/Paste Settings** (with or without normalization bounds), **Reset Settings**, **Apply settings…**, and per-frame export.

---

<!-- panel:analysis -->
## 3. Analysis readout (always visible)

Pinned above the tabs, this is your feedback while printing. Drag the divider to resize it, or collapse it entirely. Everything in it describes the frame you're on and updates as you edit; nothing in it is a control. Top to bottom:

#### Photometric curve

The chart is the paper characteristic (H&D) curve NegPy is printing through right now. It models how a sheet of photographic paper responds, and it is not a curves editor. Left to right is **negative density**, the exposure the paper receives: dense parts of the negative (the scene's highlights) sit to the right. Bottom to top is the **print tone** that comes out. A steeper curve means more contrast, which is what Grade moves. The flattening at each end is the toe (shadows) and shoulder (highlights), where the paper runs out of range.

The crosshair marks the **pivot**: the density the curve rotates around when you change contrast, so the midtone stays put. While you drag a slider a faint **ghost** of the previous curve stays behind for comparison. If cast removal pulls the channels apart you get three separate R/G/B traces instead of one grey curve, and that spread *is* the colour correction.

#### The two histograms

Two different histograms share the chart. Behind the curve, rising from the bottom, is the **output histogram**: the tones of the print you're looking at, in R, G, B and luminance. Along the bottom axis is the **negative density histogram**, which is what the scan actually contains, before the curve.

Read them against each other: the density histogram tells you which part of the horizontal axis your negative occupies, and the curve tells you what happens to it. If the negative's data sits entirely on the flat toe, no amount of contrast will pull those shadows apart. Move the exposure so the data lands on the steep middle instead.

#### LIN / LOG toggle

Bottom-right of the chart. It switches the histogram's *height* axis (how many pixels), not the tone axis. **LIN** is literal, so a big flat sky dwarfs everything else. **LOG** compresses the tall peaks so the thin tails become visible, which is where the few hundred pixels of deep shadow or specular highlight live. Use LOG when hunting for clipping, LIN when judging where the bulk of the frame sits. The choice is remembered between sessions.

#### Clipping triangles

Small R, G and B triangles in the top corners of the chart: **top-left** = shadows crushed to pure black, **top-right** = highlights blown to pure white. They only appear once a channel passes 0.5% of the frame. A little is normal, since a real print has a black. Watch for a single channel clipping alone, which is a colour cast pushing one dye off the end rather than an exposure problem.

#### Zone shading and zone ticks

The amber wash on the left and the blue wash on the right mark the curve's toe and shoulder, the compressed ends where tonal separation is being lost. The ticks along the bottom are Adams zones I to IX, so you can read straight off the axis which zone a given negative density prints as.

#### Step wedge

A 21-step Stouffer-style grey wedge printed through your current curve, in even density increments labelled in the scan's own density units. It's a ruler for the curve: where neighbouring patches are clearly different, you have tonal separation; where they merge into one flat black or white block, those tones are gone. The brackets mark the usable span. It hides while peeking the flat scan, since there's no print curve to wedge.

#### Zone strip

Ten cells on the Adams scale, where **0 is paper black and V is 18% mid-grey**, and the last cell (IX) also absorbs paper white. The brightness of each cell is the zone's tone; how solid it looks is how much of the frame lands there. This is the fastest read of whether a frame is low-key, high-key or sitting sensibly in the middle. The end cells tint **red** when shadows are blocked up or highlights are blown. Hover a cell for its exact percentage.

#### Probe

A spot densitometer. Hover the image to read the pixel: per-channel density above film base (ΔD, relative to this scan's normalization, not absolute), the displayed tone's reflection print density, and its print zone (0 = paper black, V = 18% mid-grey, X = paper white). In B&W mode the ΔD channels read the pre-conversion colour record.

#### Negative stats

The four numeric rows at the bottom. Each one has the same explanation on hover, and each is a measurement of the negative rather than of your edit:

*   **Negative**: the negative itself, as a relative density range (luminance) plus its development character against a nominal frame: flat (≈N−1), normal, contrasty (≈N+1). It is a relative scale, comparable across a roll, and a heuristic from this scan's normalized bounds rather than a calibrated densitometer reading.
*   **Exposure**: where the frame's midtone sits, in stops from neutral, positive = brighter (high-key), negative = darker (low-key). Approximate, read off the metered midtone rather than a precise meter.
*   **Clipping**: share of pixels crushed to black (shadows) or blown to white (highlights), worst channel. Turns red above 1%.
*   **Scan clip**: share of source-scan pixels at/above sensor white, per channel. In a negative scan the film base and scene shadows sit near sensor white, so clipping there destroys base/shadow separation and no edit can undo it. Fix at capture: expose the scan lower. Turns red above 1%.

---

## 4. Setup tab

<!-- panel:presets -->
### 4.1 Presets

Save and recall a complete edit (the full workspace) by name.

*   **Preset dropdown** + **Load**: apply a saved preset to the current image.
*   **Name field** + **Save**: store the current settings as a new preset.
*   **Trash**: delete the selected preset.

<!-- panel:sensor -->
### 4.2 Sensor Calibration: un-mix your camera's channels

Only relevant for **single-shot narrowband** (RGB-LED trichrome) camera scans. The camera's colour filters overlap the light source's bands, so a pure red exposure leaks a little into green and blue. That leak is a fixed property of your sensor and light together and has nothing to do with the film, so it is corrected on the linear capture before inversion.

*   **Profile**: the sensor matrix to apply. Custom `.toml` matrices live in `<Documents>/NegPy/sensor/`.
*   **Calibrate** (vials icon): build a profile from three bare-light R/G/B exposures.

The panel greys out unless **Linear RAW** is on, since profiles are calibrated against neutral white balance and the as-shot gains would misapply the matrix. Your selection is remembered either way. It is also skipped for RGB-triplet assets, which never had the leak. Because it changes what the analysis reads, **re-run Batch Analysis** after changing it.

<!-- panel:process -->
### 4.3 Process: negative → positive

The foundation of every edit: film type, how the scan is decoded, and how the negative is normalized into a positive.

*   **Mode**: `C41` (colour negative), `B&W`, or `E-6` (slide/reversal). Changes the core conversion math and re-runs the pipeline from scratch. The wand button beside it **auto-detects** the mode when a file loads.
*   **Linear RAW**: (default on) decodes with neutral multipliers for completely raw data. When toggled off decodes RAW with the camera's as-shot white balance - **pick what works better for your scanning setup!** Toggling reloads the file.
*   **Narrowband**: corrects the oversaturation typical of narrowband (RGB-LED trichrome) scans using a bundled input profile. Leave off for ordinary broadband scans. An explicit Input ICC in Export overrides it.
*   **Lock Bounds**: freezes the analyzed normalization bounds for this frame, so cropping or moving sliders no longer re-analyzes it. Lock in once you're happy with the bounds.

**Analysis window**, where NegPy measures the black/white points:

*   **Analysis Buffer** (0.0 to 0.25): insets the measurement window from the frame edge so film rebate, sprocket holes, and scanner borders don't skew detection. Raise on scans with wide borders.
*   **Analysis Region** (square-draw tool): draw a freehand region on the canvas to meter *exactly* that area (overrides the buffer). Double-click inside to confirm; the ✕ button clears it.

**Normalization tuning:**

*   **Luma Range Clip** (-100 to 100): how aggressively the tonal range (black/white-point span) is set. Neutral already applies a small robust clip. Positive tightens it, which is good for dense or fogged negatives where a few stray pixels would push the bounds to extremes. Negative pushes the bounds *outward* for lifted blacks / unclipped highlights.
*   **Colour Clip** (-100 to 100): the per-channel colour-balance clip (orange-mask removal), independent of the tonal range. Positive tightens channel balance; negative samples nearer the extremes.
*   **Global / R / G / B** selector → **White Point** / **Black Point** (-0.25 to 0.25): manual offsets on top of the auto-detected bounds. Positive white point brightens; positive black point lifts blacks. In R/G/B mode these become per-layer trims: per-dye-layer film-base (Dmin) and Dmax corrections, i.e. scanner-style per-channel levels. Hidden in B&W.

**Crosstalk** (hidden in B&W), spectral dye unmixing applied to the raw negative before inversion:

*   **Matrix**: the crosstalk profile for your film/scanner. *Default* is built-in; drop custom `.toml` matrices in `<Documents>/NegPy/crosstalk/` (see [CROSSTALK.md](CROSSTALK.md)). The slider button opens a matrix editor.
*   **Strength** (0.0 to 1.0): how much of the unmix to apply, for richer and cleaner colour separation. Because it changes what the analysis reads, **re-run Batch Analysis** after changing it.

**Normalize** (E-6 only): auto-stretches a slide's histogram to fill the dynamic range. Useful for faded/expired slides.

<!-- panel:roll -->
### 4.4 Roll Analysis: a consistent look across the roll

Meter the whole roll once and share the baseline, so frames from the same film match.

*   **Batch Analysis**: scans every loaded file and computes a roll-average density and colour balance (outliers discarded). Run it once after importing. *(Tip: if you use Batch Autocrop, run it first, in **Image only** mode, so metering sees consistent crops.)*
*   **Use Luma Average**: this frame takes the roll-wide tonal range; colour still re-derives per frame.
*   **Use Colour Average**: this frame takes the roll-wide colour balance; tonal range still re-derives per frame. Enable both for a fully consistent roll; leave both off for per-image auto-exposure.

**ROLL**, to reuse a baseline across sessions:

*   **Roll dropdown** + **Load**: apply a saved roll's bounds and balance.
*   **Save**: store the current Batch Analysis as a named roll (useful when you shoot the same stock repeatedly).
*   **Delete**: remove the selected roll.

---

## 5. Geometry tab

<!-- panel:geometry -->
### 5.1 Geometry: crop & straighten

Where the frame gets its final shape: what's inside the print, and whether it sits level. Most scans need a pass here even when nothing else is touched.

**Crop:**

*   **Ratio**: target aspect ratio: `Free`, `1:1`, `3:2`, `4:3`, `5:4`, `6:7`, `7:5`, `65:24`, `16:9`, `16:10`, `11:8.5`. One entry per shape, since the crop tool auto-orients to portrait or landscape as you drag, so there's no separate portrait entry.
*   **Detect** (crosshairs): snap the ratio to the closest standard.
*   **Crop** tool: draw a crop rectangle on the canvas. **Reset** clears it and turns auto-crop off.
*   **Guide**: overlay a composition guide while cropping: *Thirds*, *Phi Grid*, *Diagonals*, *Golden Triangles*, *Golden Spiral*, *Armature*, *Diagonal Method*, *Grid* or *Off*. The redo button rotates guides that have orientations (the spiral has 8, the triangles 2).

**Auto Crop**, to detect the frame edge automatically:

*   **Mode**: *Image only* (exposed area) or *Film edge* (full film incl. rebate/sprockets).
*   **Crop Offset** (-5 to 100 px): inset the detected edge inward. Positive trims more; negative bleeds slightly outside (when detection clips too tightly).
*   **Rebate Trim** (0 to 150%): how far into the detected rebate to cut. 0% stops at the film edge, 100% lands on the detected image edge, above 100% bites into the picture to clear a stubborn white border. *Image only* mode; applies to both **Auto** and **Batch Autocrop**.
*   **Auto**: detect and crop this frame. Best on clean rebate.
*   **Batch Autocrop**: analyze all visible landscape frames as a roll, using confident detections to calibrate weaker ones. Runs in the background with progress and cancellation. Manual, Film-edge, portrait, and ambiguous frames are left alone. Only available in *Image only* mode.

**Alignment:**

*   **Fine Rotation** (±45°): free rotation for tilted scans, in sub-degree steps (positive = clockwise). Applied after auto-crop so the frame stays axis-aligned.
*   **Straighten** tool (ruler): draw a line along a horizon or vertical edge and NegPy rotates to make it level or plumb.

<!-- panel:flatfield -->
### 5.2 Flat Field: even out the light

Corrects uneven illumination (vignetting/falloff) from your copy-stand or scanner light, using a reference shot of the bare light source.

*   **Flatfield Correction**: apply the active reference to this image (enabled once a profile exists).
*   **Reference Profile** dropdown + **Add…** / **Delete**: pick a reference image and save it as a named profile.
*   **Distortion** (-0.25 to 0.25): radial lens-distortion correction for the rig, saved with the profile. Use the film rebate as a straight-edge reference.

---

## 6. Exposure tab

This is the heart of the print. Three panels shape light, colour, and contrast, and everything here happens in the "print" stage of the pipeline.

<!-- panel:colour -->
### 6.1 Filtration: white balance

Colour timing, like the dichroic filters on an enlarger head. A **Global / Shadows / Highlights** selector scopes the controls to the whole image or biases them toward low- or high-density tones.

*   **Pick WB** (eyedropper): click a pixel that should be neutral grey; NegPy solves the CMY filtration to make it neutral in the selected region.
*   **Roll Lock**: re-aims each newly opened frame's temperature to the current target (its own tint preserved), a per-region lock for consistent warmth across a roll.
*   **Reset** (undo-arrow icon): return the selected region's temperature and CMY to neutral.
*   **Temperature**: a warm↔cool lever driving the region's magenta/yellow pair (cyan stays put, as in a real darkroom).
*   **Cyan / Magenta / Yellow** (-1 to 1): the three filtration axes, Cyan↔Red, Magenta↔Green and Yellow↔Blue.
*   **Cast Removal** (0.0 to 1.0): neutralizes the residual colour cast a negative leaves in the print, balancing each layer so greys stay neutral from deep shadows through highlights (C-41). Applied strength scales with how many clean near-neutrals the frame has. Default ~0.5; 0 turns it off.
*   **Ring-around** (target icon, or `Shift+F`): prints the frame as a 5×5 mosaic stepping 2cc at a time out to ±4cc on the magenta and yellow axes, so the direction of a colour cast is visible instead of guessed. Each patch is a real render of the part of the frame it covers; click one to keep its filtration. The ladder is absolute and centred on neutral, so a ring printed off one frame compares to the next. `Escape` or a second press clears it, and any edit drops it. See **Rotating a proof** below.

<!-- panel:tone -->
### 6.2 Tone: density, contrast, and the print curve

The paper's response. A **Global / R / G / B** selector at the top scopes most controls to the shared curve (Global) or to per-dye-layer trims for **crossover correction**, meaning casts that differ between shadows and highlights, which filtration alone can't fix.

**Automatic helpers** (on by default; they do per-frame work so you don't have to, and turning them off lets the negative print honestly):

*   **Auto Density**: meters each frame's midtone and anchors print brightness there, so dense and flat negatives land consistently.
*   **Auto Grade**: aims each frame at a contrast target instead of printing the negative's own range, so dense negatives stop printing over-contrasty and flat ones stop printing muddy.
*   **Set Targets** (sliders icon): tune the exact brightness/contrast the two helpers aim for. Applies to every frame and is remembered between sessions.

**Test strip** (grid icon, or `Shift+T`): prints the frame as a 5×5 grid, Print Density rising left to right and ISO-R Grade softening top to bottom, so the diagonals read light-to-dark and soft-to-hard like a split-filter test strip. Both ladders are absolute and centred on their defaults, so the settings you already have are one of the patches. Each patch is a real render of the part of the frame it covers; click one to keep it. `Escape` or a second press clears it, and any edit drops it.

**Rotating a proof**: a patch only shows the slice of the frame at its own grid slot, so the part you want to judge is stuck at whichever rung sits over it. While either proof is up, the 90° **rotate** buttons and `[` / `]` turn the *ladder* instead of the image: each press moves the dense/hard end onto a different edge, and the axis labels follow. The image's own rotation is untouched, and turning is instant, because printing a proof assembles all four orientations at once. The orientation you land on is kept for the rest of the session.

**Exposure:**

*   **Print Density** (0.0 to 2.0): overall brightness, simulating enlarger exposure time. Lower = brighter, higher = denser.
*   **ISO-R Grade** (50 to 180): contrast, as a paper ISO-R value. R110 ≈ classic grade 2; **lower R = harder** (more contrast), higher = softer. In R/G/B mode a **Grade** trim rotates one layer's slope about the midtone.
*   **Shadows Density** (±0.9 ΔD) / **Highlights Density** (±0.5 ΔD): brighten or darken just the shadow or highlight zone, without reshaping the curve. Bounded by paper black/white so a burn can't exceed the print's limits. The ranges differ because density is logarithmic: the same ΔD reads far smaller near paper black than near paper white.
*   **Shadows Grade** / **Highlights Grade** (split grade, ±50 ISO-R): rotate contrast locally in the deep shadows or highlights, the digital equivalent of split-grade printing.
*   **Dye Separation** (0.5 to 1.5, hidden in B&W): saturation in density space. It pushes the print's three dye densities apart *before* the positive is decoded, in the same matrix the paper's own dye crosstalk uses. So it responds to the paper profile you picked, and it eases off automatically where the curve is already compressed at toe and shoulder, instead of forcing colour into tones that have none left to give. Below 1.0 pulls the dyes together instead, toward neutral. 1.0 = off. (Contrast **Chroma** in the Colour tab, which scales colour evenly after decode.)
*   **Separation Damping** (0 to 1, hidden in B&W): decides *where* the Dye Separation push lands, rather than adding a push of its own. At 0 every colour gets the same treatment. Turn it up and muted colour keeps the full push while colour that is already saturated gets the opposite, so a hard push puts colour into the tones that had none instead of driving the strongest colours until they flatten into a slab. Below 1.0 separation it mirrors: pastels go grey while the vivid colours survive. **Dead at Dye Separation 1.0**, where the slider greys out, because it has no look of its own. This is not the same as backing Dye Separation off: a lower value takes colour from *everything*, including the tones that had little to start with, where turning damping up takes it only from the colours that already have plenty.

**Paper Response**, the characteristic-curve shape:

*   **Paper profile**: a bundled darkroom-paper profile (RA4 colour papers in C-41, tonal B&W papers in B&W). Re-shapes the curve as a baseline; Grade/Density/toe/shoulder still trim on top. *Neutral* reproduces the defaults.
*   **Paper White**: simulate paper base density, so whites print at ~0.93 instead of pure white, like a real print.
*   **Paper Black**: show the paper's true (slightly milky) Dmax instead of compensating it to pure display black. Off (default) applies black-point compensation so the adapted eye reads black as black.
*   **Snap** (-0.5 to 0.5): midtone gamma, steepening or flattening the S-curve around the reference tone while paper white/black stay put.
*   **Toe** (-1 to 1) + **Toe Width** (0.1 to 5): the shadow roll-off into paper black. Positive toe lifts shadows for a gentle film toe; negative deepens (and, with Paper Black off, makes exact black reachable). Width sets how far the knee reaches into the midtones.
*   **Shoulder** (-1 to 1) + **Shoulder Width** (0.1 to 5): the highlight roll-off into paper white. Positive compresses highlights (film-like); negative extends them and risks clipping.

In R/G/B mode the sliders become per-layer trims on top of the global value, for that dye emulsion: **Grade** (±30 ISO-R), **Toe** / **Shoulder** (±1), **Toe Width** / **Shoulder Width** (±2), **Snap** (±0.5) and **Dye Separation** (±0.4).

<!-- panel:local -->
### 6.3 Dodge & Burn: local exposure

Paint polygon masks and lighten or darken just those areas.

*   **Draw Mask**: click to place vertices; double-click / Enter / a click near the start closes the mask; Esc cancels. To edit an existing mask, select it in the list, then drag a vertex, click an edge "+" to add a point, or right-click a vertex to delete.
*   **Mask list**: each mask shows Dodge (lighten) or Burn (darken) and its strength. The eye toggles its outline; the trash deletes it.
*   **Strength** (-1 to 1 EV): dodge (+) or burn (−) for the selected mask.
*   **Feather** (0.0 to 0.15): edge softness for the selected mask, as a fraction of the frame's short side.

---

## 7. Colour tab

<!-- panel:lab -->
### 7.1 Lab: polish and detail

Mimics what a lab scanner (Frontier/Noritsu) does automatically. Colour controls hide in B&W mode.

**Colour** (hidden in B&W):

*   **Chroma** (0.0 to 2.0): a colour scale applied after the print is decoded, even across every tone, so it is a retouching move rather than a density-space one. 1.0 = unchanged, 0 = greyscale, 2.0 = double. For saturation that behaves like a print instead, reach for **Dye Separation** in the Exposure tab. Below 1.0 is a flat scale; above 1.0, pixels that would clip the display gamut get a soft per-pixel knee toward their own in-gamut headroom instead of a hard per-channel clamp, since clamping only the overshooting channel(s) shifts the hue the flat scale itself preserves.
*   **Skin Protection** (0.0 to 1.0, default 0.5): holds skin-hued colour under a chroma ceiling so faces don't go sunburnt. Hue and lightness are untouched and chroma is only ever pulled down, never added, so asking Chroma for 0 still gives you greyscale. It is independent of Chroma and works with it at 1.0 — skin that arrived over-saturated from the print curve or the filtration gets reined in just the same. Higher values lower the ceiling: the 0.5 default only catches genuinely excessive chroma, 1.0 leaves skin matte, 0 is off. The mask is warm hue *and* skin's own chroma *and* mid lightness together, which is what keeps a red coat, a saturated sunset, brick or autumn colour out of it. What it cannot separate is warm objects sitting at the same chroma as skin — bare wood, tan leather, sand — which soften along with it. The same bound cuts the other way: skin that arrives really excessive (a sunburn) is only partly caught, so reach for Chroma or the Filtration panel for that.
*   **Chroma Denoise** (0.0 to 5.0): smooths colour noise, especially in shadows, while leaving luminance grain intact.

**Sharpen:**

*   **Method**: *Unsharp Mask* (boosts edge contrast) or *Deconvolution* (Richardson–Lucy, which reverses the scanner's optical blur; set Radius to the scan's blur width).
*   **Sharpening** (0.0 to 1.0): amount, on the L (lightness) channel so there are no colour halos.
*   **Radius** (0.5 to 3.0 px): blur width, small for fine grain and larger for soft scans. Scaled to render size so preview matches export.
*   **Masking** (0.0 to 1.0): restrict sharpening to edges, which protects flat areas like sky, skin and grain.

**Detail:**

*   **CLAHE** (0.0 to 1.0): local contrast without blowing global highlights or crushing shadows. Use sparingly, since near 1.0 can look cartoonish. (Runs before dust removal so healing operates on the final rendition.)

**Effects:**

*   **Glow** (0.0 to 1.0): lens bloom, where bright highlights scatter across all channels for a dreamy softness.
*   **Halation** (0.0 to 1.0): the red glow of light scattering back through the film base. Highlights only, strongly red-dominant.

<!-- panel:toning -->
### 7.2 Toning

Colour the print itself rather than the scene: chemical toners that convert the silver (B&W only), and a split tint that works in any mode.

**Chemical Toning** (B&W only), simulated as sequential toner baths, in the order shown, each strength 0.0 to 2.0:

*   **Selenium**: deeper blacks, cool eggplant shadows.
*   **Sepia**: warm highlights first (partial strength gives split-sepia).
*   **Gold**: cool blue-black on untoned silver; over sepia, shifts highlights orange-red.
*   **Iron Blue**: Prussian-blue shadows deepening to navy blacks.
*   **Copper**: pink to brick-red shift, with the classic Dmax loss.
*   **Vanadium**: greens the mids/highlights while deep shadows keep their black.

**Split Toning** (all modes), an additive tint in Lab space, so grain and detail are preserved:

*   **Shadow Hue** (0 to 360°) + **Shadow Strength** (0.0 to 1.0).
*   **Highlight Hue** (0 to 360°) + **Highlight Strength** (0.0 to 1.0).

---

## 8. Finish tab

<!-- panel:retouch -->
### 8.1 Retouch: dust, hairs, scratches

Spotting, the way it was done with a brush on the finished print. There are three ways to find the marks, by local contrast, by the scanner's IR channel, or by hand, and they stack.

An **Overlay** button cycles the detection overlay (Off → Marked → IR) so you can see what's being caught.

**Optical Removal** finds specks on the visible scan by local contrast, with no IR needed:

*   Toggle **Optical Removal** on, then set **Threshold** (0.01 to 1.0; lower catches more, at the risk of false positives) and **Size** (3 to 8 px; max spot radius).

**IR Removal** uses the scanner's infrared channel to remove dust invisible to the colour dyes (only enabled when the scan carries an IR plane):

*   Toggle **IR Removal** and set **IR Threshold** (0.05 to 0.95; lower catches more).
*   The IR plane is read from 4-channel TIFFs and DNGs (VueScan, NegPy's own scanner output), SilverFast's iSRD TIFFs and 64-bit **HDRi RAW DNGs**, and `_IR.tif` sidecars. Scan to HDRi (not plain HDR) if you want IR data in the file; B&W and Kodachrome block infrared like dust does, so those frames are skipped automatically.

**Manual Heal** (header shows the current spot count):

*   **Heal Tool**: click dust spots in the preview to paint them out one at a time.
*   **Scratch Tool**: click points along a scratch or hair, double-click/Enter to finish; Esc cancels, Backspace removes the last point. Right-click an overlay to delete it.
*   **Brush Size** (2 to 16 px): radius of the manual brush (shown while a manual tool is active).
*   **Undo Last** / **Clear All**: remove the most recent or all manual heals (auto-detected dust is unaffected).

<!-- panel:finish -->
### 8.2 Finishing: vignette, carrier, border

How the print is presented: edge burn, a filed-out carrier's black rebate, and the paper margin around it. Applied at the very end of the pipeline, after everything else is settled.

**Vignette** (printer's edge burn, in stops):

*   **Burn** (-2.0 to 2.0 stops): positive darkens the edges, negative holds them back (lightens). 0 = off.
*   **Size** (0.0 to 1.0): falloff radius. Small keeps it tight in the corners, large spreads it into the frame.
*   **Roundness** (0.0 to 1.0): 0 = radial (lens-like), 1 = rectangular card burn following the print edges.

**Filed Carrier**, a filed-out negative carrier: the clear rebate prints max black, framed by a margin of unexposed paper:

*   **Width** (0.0 to 5.0 mm): black rebate frame thickness. 0 = off.
*   **Roughness** (0.0 to 1.0): how raggedly the aperture was filed, on the paper-side edge of the black frame. The picture-side edge is the camera's film gate and only ever wobbles slightly.
*   **Flare** (0.0 to 1.0): light reflected off the bared metal of the filed bevel, a glow that lifts the black just inside the filed edge and stains the paper just outside it. Coloured on colour film (the hue drifts along the edge, as the stray light never passes the orange mask), neutral in B&W. 0 = off.
*   **Corners** (0.0 to 1.0): how far the aperture's corners round off, since no file cuts a sharp inside corner.

The paper margin takes the mat colour, so it runs into the border with no seam.

**Border:**

*   **Width** (0.0 to 2.5): border thickness as a fraction of the image. 0 = no border.
*   **Bottom weight** (1.0 to 2.0): thickens the bottom border (window-mat proportions).
*   **Colour swatch**: click to pick any border colour.
*   **Paper white**: tint the border with the toned paper-white instead of the picked colour.

---

## 9. Favourites tab

The sliders you reach for most, gathered in one place so a routine edit no longer costs a tab
switch and a scroll. Empty until you fill it.

*   **Edit Favourites**: opens a picker. Tick sliders on the left, order them on the right with
    the arrow buttons, then **Apply**.
*   The panel then shows those sliders in your chosen order. They are the *same* controls as in
    their home panels — moving one here moves it there, and vice versa. Nothing is duplicated or
    moved out of its own tab.
*   A favourite hides itself when its original does. Favourite a Filtration slider and it will
    disappear while you are in black & white, where it has nothing to act on.
*   Your selection is remembered between sessions.

---

## 10. History tab

A scrollable list of every edit step (last 100 kept), newest on top; the current step is bold.

*   **Click** a step to jump to that state.
*   **Right-click** → **Export this version…** to export a past state directly.

---

## 11. Export tab

### Output intent

*   **Print** (default): the full creative look you see on screen.
*   **Flat**: a flat, neutral, low-contrast master that keeps maximum tonal/colour information for editing elsewhere (Lightroom, Darktable, Photoshop). Skips the print look, effects, toning, and vignette, and writes a wide-gamut 16-bit TIFF. Your in-app preview is unaffected.
    *   **Preview Flat**: temporarily show the flat master on the canvas without changing your edit.
    *   **Roll Baseline**: measure every visible frame and share one exposure baseline, so flat masters are consistent across a roll (recommended before a flat batch).

### Export button

The primary **Export** action. Its chevron menu picks the scope: current frame (Ctrl+E), selected frames, all visible with current settings, or all visible with each frame's saved settings.

### Format / Size / Colour / Destination

*   **Format**: `JPEG`, `TIFF`, `PNG`, `JPEG XL`, or `WebP` (with quality/effort options per format).
*   **Colour Space**: `Same as Source`, `sRGB`, `Adobe RGB`, `ProPhoto RGB`, `P3 D65`, `Rec 2020`, or `Greyscale` (true B&W output).
*   **Input / Output ICC**: soft-proof against, and optionally embed, an ICC profile. Output is the destination profile (default); Input treats the profile as the source (when a scan's profile is known but untagged).
*   **Paper Aspect Ratio**: final print ratio, or *Original* (no resize).
*   **Resolution**: *Original* (full RAW resolution), *Print* (long-edge **Size** in cm + **DPI**), or *Pixels* (long-edge **px**; short side follows the paper ratio).
*   **Destination**: **Filename Pattern** (a Jinja2 template, see [TEMPLATING.md](TEMPLATING.md)), **Overwrite** toggle, and output location (subfolder of source / same as source / an absolute **Export Path** with a browse button).

### Collapsible sections

*   **Presets**: a checklist of export presets (each a saved Format/Size/Colour recipe). **Manage** edits them; **Export Presets** renders the frame(s) with every enabled preset at once.
*   **Sidecars**: **Save on export** writes a `.negpy` edit sidecar next to each source on every export; **Export sidecars** writes them for all visible frames now. (Edits always stay in the database too; sidecars are optional archival copies.)
*   **Contact Sheet**: render all visible frames into a single sheet. Choose a **Template** or set **Cell / Gap / Margin / Max tiles** by hand, pick an output **Path**, and **Export contact sheet**.
*   **Preview** (affects the on-screen preview only, never the file):
    *   **Soft proof** (on by default): simulate the export colour space and Output profile so what you see matches what you'll get. Turn off only to preview at full gamut.
    *   **Display**: the monitor profile the preview is shown through, auto-detected, or pick one manually if detection fails.

---

## 12. Metadata tab

Archival metadata for the **original analog capture** (camera, lens, film, process), written into exported files as EXIF and embedded XMP so DAMs like Lightroom show your film gear rather than the scanner.

*   **Protect original metadata**: copy the source file's EXIF/XMP to exports unchanged, adding nothing. When on, the fields below are ignored.

**Analog Gear** (searchable; type in any field to filter the library):

*   **Preset**: a reusable camera + lens + film combination. **Clear** empties gear selections.
*   **Camera / Lens / Film stock**: pick from your library. Empty = not set.
*   **Manage…**: edit cameras, lenses, film stocks, and presets. Starter data seeds into `~/NegPy/gear/` on first launch.

**Process:**

*   **Format**: `35mm`, `120`, `4×5`, `8×10`, `110`, or `Other` (with a free-text field).
*   **Developer**: e.g. `D-76 1+1`.
*   **Push / Pull**: `Push +3` … `Normal` … `Pull -3`.

**Scanning:**

*   **Scanning**: scan method/notes (EXIF `Software` is always `NegPy`).
*   **Sync custom metadata to all files in batch export**: apply this tab's values to every file in a batch.

**Exposure**: optional original shutter/aperture/ISO. Click the lock to edit a free-text string (e.g. `1/125s f/2.8 ISO 400`).

**Metadata preview**: a live view of exactly what will be embedded, grouped by capture / scan / process / file.

When you set capture gear, it's written to standard EXIF and the digitizing rig is preserved separately in `negpy:Scan*` XMP tags. Leave gear unset and your scanner/DSLR stays visible in EXIF instead.

---

## 13. Scan tab

Capture film directly into NegPy (Linux and macOS; unavailable on Windows). Two collapsible sections:

*   **Scanner (SANE)**: drive a supported flatbed/film scanner over SANE.
*   **Camera Scanning**: DSLR/mirrorless copy-stand capture. Auto-connects the camera over USB (PC-Remote mode). With a NegPy **Scanlight** connected it captures narrowband R/G/B triplets from saved film-stock presets; without one it does a single white-light exposure. A **Live View** window helps you frame and focus; captured frames land in the hot folder and flow straight into RGB-Scan mode.

Camera scanning needs the optional `python-gphoto2` dependency (`pip install gphoto2`; no Windows build). See [CAMERA_SCANNING.md](CAMERA_SCANNING.md).

---

## 14. Startup Override (`override.toml`)

If NegPy crashes on launch or has rendering glitches, you can force backend settings without touching code. On first run NegPy creates `Documents/NegPy/override.toml` with defaults for your OS. Edit it and restart.

| Setting | Values | Effect |
|---------|--------|--------|
| `rendering.backend` | `"auto"`, `"vulkan"`, `"dx12"`, `"metal"`, `"cpu"` | GPU backend for image processing. `"cpu"` disables GPU entirely. |
| `display.qt_rhi_backend` | `"auto"`, `"vulkan"`, `"d3d12"`, `"metal"`, `"opengl"`, `"software"` | Qt UI rendering backend. |
| `display.qt_platform` | `"auto"`, `"xcb"`, `"wayland"` | Window system plugin (Linux only). |
| `performance.max_texture_size` | `"auto"` or a number, e.g. `4096` | Caps GPU texture size; reduce on low-VRAM cards. |
| `performance.force_hq_preview` | `true` / `false` (or absent) | Overrides the saved HQ preview toggle. |
| `performance.preview_cache_max_bytes` | a number, e.g. `1200000000` | Preview cache memory budget (default ~1.2 GB). |
| `performance.preview_cache_max_entries` | a number, e.g. `8` | Max recently-viewed photos kept in memory. |
| `performance.preview_cache_max_full_res_entries` | a number, e.g. `2` | Full-resolution HQ preview buffers kept in memory (a 60 MP scan is ~700 MB each). |
| `performance.cpu_parallel` | `true` / `false` (or absent) | Multi-core CPU rendering kernels. Defaults on, except macOS. |
| `logging.level` | `"debug"`, `"info"`, `"warning"`, `"error"` | Log verbosity. Use `"debug"` when reporting issues. |

**Common fixes:**

*   **Crashes immediately on Linux** → `backend = "cpu"` or `qt_rhi_backend = "opengl"`.
*   **Black/blank preview on Windows** → `backend = "dx12"` or `qt_rhi_backend = "software"`.
*   **Wayland rendering issues** → `qt_platform = "xcb"` to force X11.
*   **GPU out-of-memory during export** → `max_texture_size = 4096`.

---

## Additional Info

*   **GPU acceleration**: NegPy uses your GPU for near-instant previews and responsive sliders. The Process panel's analysis (bounds, white/black point, normalize) runs on the CPU. There is no global GPU switch in the UI, so force the CPU pipeline via `override.toml` if you suspect a driver issue.
*   **Database**: all edits live in a local SQLite database keyed by file hash, so you can move or rename files without losing your work. Optional `.negpy` sidecars mirror edits next to your sources.
*   **Saving edits**: edits are written to the database on export, when you switch frames, or when you save explicitly. Closing the app mid-edit without any of those loses unsaved changes.
*   **Keyboard shortcuts**: [KEYBOARD.md](KEYBOARD.md)
*   **Filename templating**: [TEMPLATING.md](TEMPLATING.md)
*   **The pipeline in depth**: [PIPELINE.md](PIPELINE.md)
