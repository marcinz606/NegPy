# NegPy User Guide

NegPy turns film scans into finished positives with a non-destructive, darkroom-style pipeline. Nothing is ever written back to your source files — every edit lives in a local database, so you can experiment freely.

This guide is for new users. It explains what each control does, when you'd reach for it, and roughly what it does to your image. If you just want to know *why* the pipeline is ordered the way it is, read [PIPELINE.md](PIPELINE.md).

---

## 1. The Big Picture

### Screen layout

*   **Left — Film strip**: your loaded frames as a contact sheet, plus import, sorting, and triage tools.
*   **Centre — Canvas**: the live preview of the current frame. Most tools (crop, white-balance picker, heal brush, dodge/burn masks) are used by clicking directly on it.
*   **Right — Controls**: a pinned **Analysis** readout at the top, and below it an icon tab bar. Each icon opens a *workflow page* holding one or more collapsible panels.

### The workflow (and the order things happen)

The right-hand tabs are arranged in the order you actually work, which mirrors the processing pipeline:

| Tab | Icon | Panels | What it's for |
|-----|------|--------|---------------|
| **Setup** | cogs | Presets · Process · Roll Analysis | Film type, negative→positive normalization, roll-wide baselines |
| **Geometry** | crop | Geometry · Flat Field | Crop, straighten, lens/falloff correction |
| **Exposure** | sun | Filtration · Tone · Dodge & Burn | White balance, print density/contrast/curve, local burns |
| **Colour** | palette | Lab · Toning | Saturation, sharpening, effects, split/chemical toning |
| **Finish** | brush | Retouch · Finishing | Dust removal, vignette, border, carrier |
| **History** | clock | Edit history | Step back through every change |
| **Export** | file | Export settings | Format, size, colour, batch output |
| **Metadata** | tags | Archival metadata | Original camera/lens/film details |
| **Scan** | camera | Scanner · Camera Scanning | Capture film directly (Linux/macOS) |

You don't have to touch every panel. NegPy's defaults are tuned to produce a good print straight away — most frames need only a crop, maybe a white-balance nudge, and export.

A small **dot** on a panel header (and on a tab icon) means you've changed something from its default. Every panel header has a **reset** action to return that panel to defaults.

---

## 2. Film strip (left panel)

The header shows the NegPy logo and version (and an update link when a new release is out). Below it is the file browser.

### Importing & managing files

Toolbar buttons, left to right:

*   **Add files** / **Add folder**: load individual images or every image in a folder.
*   **Clear all**: unload everything (or, when several frames are selected, unload just those).
*   **Hot Folder**: watches the current folder and auto-loads new files as they appear — handy when a scanner drops files into a directory.
*   **RGB Scan**: treats the folder as red/green/blue exposure triplets and assembles each frame from three shots (for narrowband trichrome scanning). Right-click a frame → **Edit RGB Triplet…** to assign the three files by hand.
*   **Half Frame**: splits each scan into two frames (for half-frame cameras), edited and metered separately.
*   **Apply (clone)**: copy the current frame's settings to selected frames or the whole roll — you choose which aspects in a dialog (crop and rotation are always per-image).
*   **Sheet filter** (funnel): show *All frames*, *Keepers only*, or *Hide rejected*.
*   **Sort**: by Name or Date, ascending or descending.

Below the toolbar: a **filter box** (substring match; toggle **`.*`** for regex) and a **tally** — e.g. "36 frames · 12 keepers · 3 rejected".

### Triage (culling the roll)

Right-click a thumbnail (or use keyboard shortcuts) to mark frames while you review the sheet:

*   **Keep** — a small check badge marks a keeper.
*   **Reject** — a cross badge dims the frame. Rejected frames stay on the sheet but are skipped by batch exports and sidecar writes. **The file on disk is never touched.**

Marks apply to a multi-selection and persist across sessions. A badge in the top-right corner instead flags a frame that failed to decode.

The right-click menu also offers **Copy/Paste Settings** (with or without normalization bounds), **Reset Settings**, **Apply settings…**, and per-frame export.

---

## 3. Analysis readout (always visible)

Pinned above the tabs, this is your feedback while editing. Drag the divider to resize it, or collapse it entirely.

*   **Photometric curve**: the paper characteristic (H&D) curve NegPy is currently applying, drawn over two histograms — the output tones and the negative density. Grab the toggle to switch the density axis between linear and log. Hover the canvas and the curve marks where that pixel lands.
*   **Zone strip**: how your tones are distributed across print zones (shadow → highlight), with warnings when zones are clipping.
*   **Densitometer**: click-hold on the canvas to probe a pixel's values.
*   **Negative stats**: density range, metered exposure, and a **scan-clip warning** if the scanner clipped highlights or the film base (which permanently loses information — a capture problem NegPy can't undo).

---

## 4. Setup tab

### 4.1 Presets

Save and recall a complete edit (the full workspace) by name.

*   **Preset dropdown** + **Load**: apply a saved preset to the current image.
*   **Name field** + **Save**: store the current settings as a new preset.
*   **Trash**: delete the selected preset.

### 4.2 Process — negative → positive

The foundation of every edit: film type, how the scan is decoded, and how the negative is normalized into a positive.

*   **Mode**: `C41` (colour negative), `B&W`, or `E-6` (slide/reversal). Changes the core conversion math and re-runs the pipeline from scratch. The wand button beside it **auto-detects** the mode when a file loads.
*   **Linear RAW**: off (default) decodes RAW with the camera's as-shot white balance for a balanced starting point; on decodes with neutral multipliers for completely raw data. Toggling reloads the file.
*   **Narrowband**: corrects the oversaturation typical of narrowband (RGB-LED trichrome) scans using a bundled input profile. Leave off for ordinary broadband scans. An explicit Input ICC in Export overrides it.
*   **Lock Bounds**: freezes the analyzed normalization bounds for this frame, so cropping or moving sliders no longer re-analyzes it. Lock in once you're happy with the bounds.

**Analysis window** — where NegPy measures the black/white points:

*   **Analysis Buffer** (0.0–0.25): insets the measurement window from the frame edge so film rebate, sprocket holes, and scanner borders don't skew detection. Raise on scans with wide borders.
*   **Analysis Region** (square-draw tool): draw a freehand region on the canvas to meter *exactly* that area (overrides the buffer). Double-click inside to confirm; the ✕ button clears it.

**Normalization tuning:**

*   **Luma Range Clip** (-100–100): how aggressively the tonal range (black/white-point span) is set. Neutral already applies a small robust clip. Positive tightens it — good for dense or fogged negatives where a few stray pixels would push the bounds to extremes. Negative pushes the bounds *outward* for lifted blacks / unclipped highlights.
*   **Colour Clip** (-100–100): the per-channel colour-balance clip (orange-mask removal), independent of the tonal range. Positive tightens channel balance; negative samples nearer the extremes.
*   **Global / R / G / B** selector → **White Point** / **Black Point** (-0.25–0.25): manual offsets on top of the auto-detected bounds. Positive white point brightens; positive black point lifts blacks. In R/G/B mode these become per-layer trims — per-dye-layer film-base (Dmin) and Dmax corrections, i.e. scanner-style per-channel levels. Hidden in B&W.

**Crosstalk** (hidden in B&W) — spectral dye unmixing applied to the raw negative before inversion:

*   **Matrix**: the crosstalk profile for your film/scanner. *Default* is built-in; drop custom `.toml` matrices in `<Documents>/NegPy/crosstalk/` (see [CROSSTALK.md](CROSSTALK.md)). The slider button opens a matrix editor.
*   **Separation** (0.0–1.0): strength of the unmix — richer, cleaner colour separation. Because it changes what the analysis reads, **re-run Batch Analysis** after changing it.

**Normalize** (E-6 only): auto-stretches a slide's histogram to fill the dynamic range. Useful for faded/expired slides.

### 4.3 Roll Analysis — a consistent look across the roll

Meter the whole roll once and share the baseline, so frames from the same film match.

*   **Batch Analysis**: scans every loaded file and computes a roll-average density and colour balance (outliers discarded). Run it once after importing. *(Tip: if you use Batch Autocrop, run it first, in **Image only** mode, so metering sees consistent crops.)*
*   **Use Luma Average**: this frame takes the roll-wide tonal range; colour still re-derives per frame.
*   **Use Colour Average**: this frame takes the roll-wide colour balance; tonal range still re-derives per frame. Enable both for a fully consistent roll; leave both off for per-image auto-exposure.

**ROLL** — reuse a baseline across sessions:

*   **Roll dropdown** + **Load**: apply a saved roll's bounds and balance.
*   **Save**: store the current Batch Analysis as a named roll (useful when you shoot the same stock repeatedly).
*   **Delete**: remove the selected roll.

---

## 5. Geometry tab

### 5.1 Geometry — crop & straighten

**Crop:**

*   **Ratio**: target aspect ratio (`Free`, `3:2`, `4:3`, `5:4`, `1:1`, `6:7`, `65:24`, …). The crop tool auto-orients to portrait or landscape as you drag.
*   **Detect** (crosshairs): snap the ratio to the closest standard.
*   **Crop** tool: draw a crop rectangle on the canvas. **Reset** clears it and turns auto-crop off.
*   **Guide**: overlay a composition guide (thirds, golden spiral, …) while cropping; the redo button rotates guides that have orientations.

**Auto Crop** — detect the frame edge automatically:

*   **Mode**: *Image only* (exposed area) or *Film edge* (full film incl. rebate/sprockets).
*   **Crop Offset** (-5–100 px): inset the detected edge inward. Positive trims more; negative bleeds slightly outside (when detection clips too tightly).
*   **Auto**: detect and crop this frame. Best on clean rebate.
*   **Batch Autocrop**: analyze all visible landscape frames as a roll, using confident detections to calibrate weaker ones. Runs in the background with progress and cancellation. Manual, Film-edge, portrait, and ambiguous frames are left alone. Only available in *Image only* mode.

**Alignment:**

*   **Fine Rotation** (±5°): sub-degree levelling for tilted scans (positive = clockwise). Applied after auto-crop so the frame stays axis-aligned.
*   **Straighten** tool (ruler): draw a line along a horizon or vertical edge and NegPy rotates to make it level or plumb.

### 5.2 Flat Field — even out the light

Corrects uneven illumination (vignetting/falloff) from your copy-stand or scanner light, using a reference shot of the bare light source.

*   **Flatfield Correction**: apply the active reference to this image (enabled once a profile exists).
*   **Reference Profile** dropdown + **Add…** / **Delete**: pick a reference image and save it as a named profile.
*   **Distortion** (-0.25–0.25): radial lens-distortion correction for the rig, saved with the profile. Use the film rebate as a straight-edge reference.

---

## 6. Exposure tab

This is the heart of the print. Three panels shape light, colour, and contrast — everything here happens in the "print" stage of the pipeline.

### 6.1 Filtration — white balance

Colour timing, like the dichroic filters on an enlarger head. A **Global / Shadows / Highlights** selector scopes the controls to the whole image or biases them toward low- or high-density tones.

*   **Pick WB** (eyedropper): click a pixel that should be neutral grey; NegPy solves the CMY filtration to make it neutral in the selected region.
*   **Roll Lock**: re-aims each newly opened frame's temperature to the current target (its own tint preserved) — a per-region lock for consistent warmth across a roll.
*   **Reset**: return the selected region's temperature and CMY to neutral.
*   **Temperature**: a warm↔cool lever driving the region's magenta/yellow pair (cyan stays put, as in a real darkroom).
*   **Cyan / Magenta / Yellow** (-1–1): the three filtration axes — Cyan↔Red, Magenta↔Green, Yellow↔Blue.
*   **Cast Removal** (0.0–1.0): neutralizes the residual colour cast a negative leaves in the print, balancing each layer so greys stay neutral from deep shadows through highlights (C-41). Applied strength scales with how many clean near-neutrals the frame has. Default ~0.5; 0 turns it off.

### 6.2 Tone — density, contrast, and the print curve

The paper's response. A **Global / R / G / B** selector at the top scopes most controls to the shared curve (Global) or to per-dye-layer trims for **crossover correction** (fixing casts that differ between shadows and highlights — something filtration alone can't do).

**Automatic helpers** (on by default — they do per-frame work so you don't have to; turn off to let the negative print honestly):

*   **Auto Density**: meters each frame's midtone and anchors print brightness there, so dense and flat negatives land consistently.
*   **Auto Grade**: aims each frame at a contrast target instead of printing the negative's own range, so dense negatives stop printing over-contrasty and flat ones stop printing muddy.
*   **Set Targets** (sliders icon): tune the exact brightness/contrast the two helpers aim for. Applies to every frame and is remembered between sessions.

**Exposure:**

*   **Print Density** (0.0–2.0): overall brightness — simulates enlarger exposure time. Lower = brighter, higher = denser.
*   **ISO-R Grade** (50–180): contrast, as a paper ISO-R value. R110 ≈ classic grade 2; **lower R = harder** (more contrast), higher = softer. In R/G/B mode a **Grade** trim rotates one layer's slope about the midtone.
*   **Shadows Density** / **Highlights Density** (zone density): brighten or darken just the shadow or highlight zone, without reshaping the curve. Bounded by paper black/white so a burn can't exceed the print's limits.
*   **Shadows Grade** / **Highlights Grade** (split grade, ±50 ISO-R): rotate contrast locally in the deep shadows or highlights — the digital equivalent of split-grade printing.

**Paper Response** — the characteristic-curve shape:

*   **Paper profile**: a bundled darkroom-paper profile (RA4 colour papers in C-41, tonal B&W papers in B&W). Re-shapes the curve as a baseline; Grade/Density/toe/shoulder still trim on top. *Neutral* reproduces the defaults.
*   **Paper White**: simulate paper base density — whites print at ~0.93 instead of pure white, like a real print.
*   **Paper Black**: show the paper's true (slightly milky) Dmax instead of compensating it to pure display black. Off (default) applies black-point compensation so the adapted eye reads black as black.
*   **Snap** (-0.5–0.5): midtone gamma — steepens or flattens the S-curve around the reference tone while paper white/black stay put.
*   **Toe** (-1–1) + **Toe Width** (0.1–5): the shadow roll-off into paper black. Positive toe lifts shadows for a gentle film toe; negative deepens (and, with Paper Black off, makes exact black reachable). Width sets how far the knee reaches into the midtones.
*   **Shoulder** (-1–1) + **Shoulder Width** (0.1–5): the highlight roll-off into paper white. Positive compresses highlights (film-like); negative extends them and risks clipping.

In R/G/B mode, Toe/Shoulder/Snap and their Widths become per-layer trims for that dye emulsion.

### 6.3 Dodge & Burn — local exposure

Paint polygon masks and lighten or darken just those areas.

*   **Draw Mask**: click to place vertices; double-click / Enter / a click near the start closes the mask; Esc cancels. To edit an existing mask, select it in the list — drag a vertex, click an edge "+" to add a point, right-click a vertex to delete.
*   **Mask list**: each mask shows Dodge (lighten) or Burn (darken) and its strength. The eye toggles its outline; the trash deletes it.
*   **Strength** (-1–1 EV): dodge (+) or burn (−) for the selected mask.
*   **Feather**: edge softness for the selected mask.

---

## 7. Colour tab

### 7.1 Lab — polish and detail

Mimics what a lab scanner (Frontier/Noritsu) does automatically. Colour controls hide in B&W mode.

**Colour** (hidden in B&W):

*   **Saturation** (0.0–2.0): linear saturation. 1.0 = unchanged, 0 = greyscale, 2.0 = double.
*   **Dye Mute** (0.0–1.0): counters the extra saturation that harder grades create, mimicking real paper dyes' unwanted absorptions. 0 disables.
*   **Vibrance** (0.0–2.0): smart saturation that boosts muted colours more than already-saturated ones — gentler on skin tones.

**Sharpen:**

*   **Method**: *Unsharp Mask* (boosts edge contrast) or *Deconvolution* (Richardson–Lucy — reverses the scanner's optical blur; set Radius to the scan's blur width).
*   **Sharpening** (0.0–1.0): amount, on the L (lightness) channel so there are no colour halos.
*   **Radius** (0.5–3.0 px): blur width — small for fine grain, larger for soft scans. Scaled to render size so preview matches export.
*   **Masking** (0.0–1.0): restrict sharpening to edges, protecting flat areas (sky, skin, grain).

**Detail:**

*   **CLAHE** (0.0–1.0): local contrast without blowing global highlights or crushing shadows. Use sparingly — near 1.0 can look cartoonish. (Runs before dust removal so healing operates on the final rendition.)
*   **Denoise** (0.0–5.0, hidden in B&W): chroma denoise — smooths colour noise, especially in shadows, while leaving luminance grain intact.

**Effects:**

*   **Glow** (0.0–1.0): lens bloom — bright highlights scatter across all channels for a dreamy softness.
*   **Halation** (0.0–1.0): the red glow of light scattering back through the film base. Highlights only, strongly red-dominant.

### 7.2 Toning

**Chemical Toning** (B&W only) — simulated as sequential toner baths, in the order shown; each strength 0.0–2.0:

*   **Selenium**: deeper blacks, cool eggplant shadows.
*   **Sepia**: warm highlights first (partial strength gives split-sepia).
*   **Gold**: cool blue-black on untoned silver; over sepia, shifts highlights orange-red.
*   **Iron Blue**: Prussian-blue shadows deepening to navy blacks.
*   **Copper**: pink to brick-red shift, with the classic Dmax loss.
*   **Vanadium**: greens the mids/highlights while deep shadows keep their black.

**Split Toning** (all modes) — additive tint in Lab space, so grain and detail are preserved:

*   **Shadow Hue** + **Shadow Strength** (0.0–1.0).
*   **Highlight Hue** + **Highlight Strength** (0.0–1.0).

---

## 8. Finish tab

### 8.1 Retouch — dust, hairs, scratches

An **Overlay** button cycles the detection overlay (Off → Marked → IR) so you can see what's being caught.

**Optical Removal** — find specks on the visible scan by local contrast (no IR needed):

*   Toggle **Optical Removal** on, then set **Threshold** (0.01–1.0; lower catches more, risking false positives) and **Size** (3–8 px; max spot radius).

**IR Removal** — uses the scanner's infrared channel to remove dust invisible to the colour dyes (only enabled when the scan carries an IR plane):

*   Toggle **IR Removal** and set **IR Threshold** (0.05–0.95; lower catches more).

**Manual Heal** (header shows the current spot count):

*   **Heal Tool**: click dust spots in the preview to paint them out one at a time.
*   **Scratch Tool**: click points along a scratch or hair, double-click/Enter to finish; Esc cancels, Backspace removes the last point. Right-click an overlay to delete it.
*   **Brush Size** (2–16 px): radius of the manual brush (shown while a manual tool is active).
*   **Undo Last** / **Clear All**: remove the most recent or all manual heals (auto-detected dust is unaffected).

### 8.2 Finishing — vignette, carrier, border

Applied at the very end of the pipeline.

**Vignette** (printer's edge burn, in stops):

*   **Burn** (-2.0–2.0 stops): positive darkens the edges, negative holds them back (lightens). 0 = off.
*   **Size** (0.0–1.0): falloff radius — small keeps it tight in the corners, large spreads it into the frame.
*   **Roundness** (0.0–1.0): 0 = radial (lens-like), 1 = rectangular card burn following the print edges.

**Filed Carrier** — a filed-out negative carrier: the clear rebate prints max black with a rough inner edge:

*   **Width** (0.0–5.0 mm): black rebate frame thickness. 0 = off.
*   **Roughness** (0.0–1.0): how ragged the inner edge is.

**Border:**

*   **Width** (0.0–2.5): border thickness as a fraction of the image. 0 = no border.
*   **Bottom weight** (1.0–2.0): thickens the bottom border (window-mat proportions).
*   **Colour swatch**: click to pick any border colour.
*   **Paper white**: tint the border with the toned paper-white instead of the picked colour.

---

## 9. History tab

A scrollable list of every edit step (last 100 kept), newest on top; the current step is bold.

*   **Click** a step to jump to that state.
*   **Right-click** → **Export this version…** to export a past state directly.

---

## 10. Export tab

### Output intent

*   **Print** (default): the full creative look you see on screen.
*   **Flat**: a flat, neutral, low-contrast master that keeps maximum tonal/colour information for editing elsewhere (Lightroom, Darktable, Photoshop). Skips the print look, effects, toning, and vignette, and writes a wide-gamut, high-bit-depth file. Your in-app preview is unaffected.
    *   **Format**: *16-bit TIFF* (widely compatible) or *Linear DNG*.
    *   **Preview Flat**: temporarily show the flat master on the canvas without changing your edit.
    *   **Roll Baseline**: measure every visible frame and share one exposure baseline, so flat masters are consistent across a roll (recommended before a flat batch).

### Export button

The primary **Export** action; its chevron menu picks the scope — current frame (Ctrl+E), selected frames, all visible with current settings, or all visible with each frame's saved settings.

### Format / Size / Colour / Destination

*   **Format**: `JPEG`, `TIFF`, `PNG`, `JPEG XL`, or `WebP` (with quality/effort options per format).
*   **Colour Space**: `Same as Source`, `sRGB`, `Adobe RGB`, `ProPhoto RGB`, `ACES`, `P3 D65`, `Rec 2020`, `XYZ`, or `Greyscale` (true B&W output).
*   **Input / Output ICC**: soft-proof against, and optionally embed, an ICC profile. Output is the destination profile (default); Input treats the profile as the source (when a scan's profile is known but untagged).
*   **Paper Aspect Ratio**: final print ratio, or *Original* (no resize).
*   **Resolution**: *Original* (full RAW resolution), *Print* (long-edge **Size** in cm + **DPI**), or *Pixels* (long-edge **px**; short side follows the paper ratio).
*   **Destination**: **Filename Pattern** (a Jinja2 template — see [TEMPLATING.md](TEMPLATING.md)), **Overwrite** toggle, and output location (subfolder of source / same as source / an absolute **Export Path** with a browse button).

### Collapsible sections

*   **Presets**: a checklist of export presets (each a saved Format/Size/Colour recipe). **Manage** edits them; **Export Presets** renders the frame(s) with every enabled preset at once.
*   **Sidecars**: **Save on export** writes a `.negpy` edit sidecar next to each source on every export; **Export sidecars** writes them for all visible frames now. (Edits always stay in the database too — sidecars are optional archival copies.)
*   **Contact Sheet**: render all visible frames into a single sheet. Choose a **Template** or set **Cell / Gap / Margin / Max tiles** by hand, pick an output **Path**, and **Export contact sheet**.
*   **Preview** (affects the on-screen preview only, never the file):
    *   **Soft proof** (on by default): simulate the export colour space and Output profile so what you see matches what you'll get. Turn off only to preview at full gamut.
    *   **Display**: the monitor profile the preview is shown through — auto-detected, or pick one manually if detection fails.

---

## 11. Metadata tab

Archival metadata for the **original analog capture** — camera, lens, film, process — written into exported files as EXIF and embedded XMP so DAMs like Lightroom show your film gear rather than the scanner.

*   **Protect original metadata**: copy the source file's EXIF/XMP to exports unchanged, adding nothing. When on, the fields below are ignored.

**Analog Gear** (searchable — type in any field to filter the library):

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

**Exposure**: optional original shutter/aperture/ISO — click the lock to edit a free-text string (e.g. `1/125s f/2.8 ISO 400`).

**Metadata preview**: a live view of exactly what will be embedded, grouped by capture / scan / process / file.

When you set capture gear, it's written to standard EXIF and the digitizing rig is preserved separately in `negpy:Scan*` XMP tags. Leave gear unset and your scanner/DSLR stays visible in EXIF instead.

---

## 12. Scan tab

Capture film directly into NegPy (Linux and macOS; unavailable on Windows). Two collapsible sections:

*   **Scanner (SANE)**: drive a supported flatbed/film scanner over SANE.
*   **Camera Scanning**: DSLR/mirrorless copy-stand capture. Auto-connects the camera over USB (PC-Remote mode). With a NegPy **Scanlight** connected it captures narrowband R/G/B triplets from saved film-stock presets; without one it does a single white-light exposure. A **Live View** window helps you frame and focus; captured frames land in the hot folder and flow straight into RGB-Scan mode.

Camera scanning needs the optional `python-gphoto2` dependency (`pip install gphoto2`; no Windows build). See [CAMERA_SCANNING.md](CAMERA_SCANNING.md).

---

## 13. Startup Override (`override.toml`)

If NegPy crashes on launch or has rendering glitches, you can force backend settings without touching code. On first run NegPy creates `Documents/NegPy/override.toml` with defaults for your OS. Edit it and restart.

| Setting | Values | Effect |
|---------|--------|--------|
| `rendering.backend` | `"auto"`, `"vulkan"`, `"dx12"`, `"metal"`, `"cpu"` | GPU backend for image processing. `"cpu"` disables GPU entirely. |
| `display.qt_rhi_backend` | `"auto"`, `"vulkan"`, `"d3d12"`, `"metal"`, `"opengl"`, `"software"` | Qt UI rendering backend. |
| `display.qt_platform` | `"auto"`, `"xcb"`, `"wayland"` | Window system plugin (Linux only). |
| `performance.max_texture_size` | `"auto"` or a number, e.g. `4096` | Caps GPU texture size — reduce on low-VRAM cards. |
| `performance.force_hq_preview` | `true` / `false` (or absent) | Overrides the saved HQ preview toggle. |
| `performance.preview_cache_max_bytes` | a number, e.g. `1200000000` | Preview cache memory budget (default ~1.2 GB). |
| `performance.preview_cache_max_entries` | a number, e.g. `8` | Max recently-viewed photos kept in memory. |
| `logging.level` | `"debug"`, `"info"`, `"warning"`, `"error"` | Log verbosity. Use `"debug"` when reporting issues. |

**Common fixes:**

*   **Crashes immediately on Linux** → `backend = "cpu"` or `qt_rhi_backend = "opengl"`.
*   **Black/blank preview on Windows** → `backend = "dx12"` or `qt_rhi_backend = "software"`.
*   **Wayland rendering issues** → `qt_platform = "xcb"` to force X11.
*   **GPU out-of-memory during export** → `max_texture_size = 4096`.

---

## Additional Info

*   **GPU acceleration**: NegPy uses your GPU for near-instant previews and responsive sliders. The Process panel's analysis (bounds, white/black point, normalize) runs on the CPU. There is no global GPU switch in the UI — force the CPU pipeline via `override.toml` if you suspect a driver issue.
*   **Database**: all edits live in a local SQLite database keyed by file hash, so you can move or rename files without losing your work. Optional `.negpy` sidecars mirror edits next to your sources.
*   **Saving edits**: edits are written to the database on export, when you switch frames, or when you save explicitly. Closing the app mid-edit without any of those loses unsaved changes.
*   **Keyboard shortcuts**: [KEYBOARD.md](KEYBOARD.md)
*   **Filename templating**: [TEMPLATING.md](TEMPLATING.md)
*   **The pipeline in depth**: [PIPELINE.md](PIPELINE.md)
