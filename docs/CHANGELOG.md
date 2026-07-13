# Change Log


## 0.37.0

- New: **Crosstalk matrix editor** — a Manage button beside the Process → Crosstalk dropdown opens an editor: browse the bundled matrices (read-only), make an editable copy, adjust the channel-mixing terms with live preview, and save your own profiles as `.toml` files in the NegPy/crosstalk folder.
- New: **Editable Dodge & Burn masks** — the panel lists every mask you've drawn; pick one to select it, then reshape it right on the canvas (drag a point to move it, click an edge to add a point, right-click a point to remove it) and tune its sliders — no need to redraw from scratch.
- New: **Straighten tool** — draw a reference line on the image (ruler button under Geometry → Alignment, key `L`) and the frame rotates to match: lines near horizontal level the horizon, lines near vertical plumb an edge. Composes with Fine Rotation and the crop rotation handles. @linkmodo
- New: **Iron Blue, Copper and Vanadium Green toners** on the Toning page (B&W only) — navy shadows, pink-to-brick red with weakened blacks, and green mids over black shadows.
- New: **Spot densitometer** — hover the image to read the pixel under the cursor: per-channel density above film base (ΔD), the print's reflection density, and its print zone (Zone V = 18% grey); a dot tracks the pixel along the H&D curve.
- New: **Zone strip** — a Zone System bar under the chart shows how much of the print falls in each zone (0–IX) and flags blocked shadows or blown highlights in red.
- Change: **Dust & healing overhaul** — Auto Dust, IR removal and the manual Heal tool now share one texture-preserving clone instead of the old smoothed fill: sources are matched to their surroundings, edges feather with brush size, the halo around auto-fixed spots is gone, and hairs and scratches are traced along their length. Detection runs once on the source scan, so the fixed set no longer shifts as you drag sliders and preview heals exactly the spots export does. Auto Dust now works on E-6 slides too.
- Change: **Analysis chart redesign** — the histogram and the H&D curve merge into a single graph: the print's RGB histogram sits behind the curve, and a second grey histogram along the exposure axis shows where the negative's densities land on the paper curve — how much of the image rides the toe, the straight line or the shoulder, live as you drag Grade or Density (the LIN/LOG toggle scales both). The stats below reduce to the essentials: the negative's density range with a development read (flat / normal / contrasty), exposure in EV, and clipping — the scan-clip warning appears only when the scan actually clips.
- Change: **Heal & Scratch tools work fully on the canvas** — right-click opens a tool menu (Undo/Clear, Confirm Scratch, and Delete This Heal on a placed patch), the Scratch tool shows a pen cursor, `Ctrl+Z` undoes the last heal and Backspace steps back a scratch point; the two tools are now mutually exclusive and suspend/restore when you leave and return to their tab. The placed-heal outline no longer sits slightly offset from where you clicked. @linkmodo
- Change: **Toner combinations simulate sequential baths** — silver claimed by one toner is locked to the ones after it: selenium before sepia protects the shadows, partial sepia plus iron blue splits green. Single-toner looks are unchanged.
- Change: **Smoother masks and scratch heals** — Dodge & Burn mask outlines and Scratch heal strokes now follow smooth curves through their points instead of straight segments, and the line previews curved as you draw it.
- Change: **Fine Rotation now reads clockwise-positive** — dragging the slider (or the crop handles) to the right turns the image clockwise on screen, the photographer's convention, under every flip. Saved rotations render exactly as before. @linkmodo
- Change: **Customize Shortcuts redesign** — shortcuts are grouped into collapsible sections with one shortcut per row, slider rows merged with per-control keyboard step sizes, and a search box to jump to an action; the read-only shortcut overview gets the same press-to-search lookup. @jboneng
- Change: **Camera Scanning presets lock the whole exposure** — a preset now stores the ISO and aperture the film base was metered at alongside the RGB levels and shutter, and re-asserts all of them before every shot, so a bumped dial between scans can't skew a scan. @light-sntchr
- Change: **Snappier camera live view** — stepping ISO / shutter / aperture waits for you to pause before writing to the body instead of firing a slow write per click, so rapid stepping settles far quicker and no longer flickers back mid-write. @light-sntchr
- Change: **More reliable auto-calibration** — per-channel ETTR calibration now copes with dense film bases where one channel is much brighter than another: it tolerates a negligible base clip, pulls the LED down until the base truly stops clipping, and settles on the best shutter instead of oscillating. @light-sntchr
- Change: **Smaller TIFF exports** — TIFFs now use ZIP (Deflate) compression with a horizontal predictor instead of LZW, typically 15–25% smaller for 16-bit images with no change to the pixels. @RP2
- Fix: **camera reconnects cleanly after scanning** — closing the Live View or new-preset calibration window now releases the tethered camera session, so bodies (Fuji especially) that stick in tethered-capture mode no longer hang on the next connection. @bbatha
- Fix: **export no longer crashes on non-ASCII metadata** — accented or non-Latin characters in a source file's EXIF (e.g. Image Description) previously aborted a TIFF export; they're now safely substituted. @RP2
- Fix: a B&W print toned only with gold or split tints rendered grey in preview and export.
- Fix: presets no longer embed the source frame's heal strokes, and Apply settings no longer overwrites other frames' heals.
- Fix: **numpad keys bind separately** — the shortcut editor no longer collapses `Num+9` onto plain `9`, so numpad and number-row keys can hold different shortcuts. @jboneng
- Fix: **no more 'surround' warning on load** — edits saved before the old surround control was removed no longer log an unknown-key warning every time you open a file. @linkmodo
- Fix: **edits survive EXIF changes** — editing a file's EXIF metadata (e.g. tags) changes its content hash, which previously orphaned its saved edits; NegPy now falls back to matching by file path and re-homes the edits under the new hash. @RP2
- Fix: **B&W renders match between CPU and GPU** — black-and-white previews and exports no longer come out darker or subtly tinted on the CPU path, and the Analysis histogram is smooth instead of combed.
- Fix: **batch export honours the current destination** — exporting saved or selected frames now writes them to the destination you've set for the session (e.g. same-as-source) instead of the stale path each frame was last saved with. @RP2

## 0.36.0

- New: **Camera Scanning** — a new tab captures negatives with a tethered camera and feeds the RGB Scan merge. Two auto-selected modes: **Narrowband RGB** (jackw01's open-source Scanlight flashes red, then green, then blue while the camera captures each exposure) and **normal** (a single exposure under any light, imported as an ordinary RAW). Includes per-channel ETTR auto-calibration metered on the clear film base, film-stock presets, and a live view with the camera's hardware focus magnifier and live ISO/shutter controls. Cameras are detected on the USB bus automatically — no address, login or pairing. macOS and Linux only (verified on Sony bodies, other brands need testing). @light-sntchr
- New: **Split grade** — **Shadows Grade / Highlights Grade** sliders on the Tone page trim each zone's contrast in ISO-R points, like a split-grade darkroom exposure: harder shadows without blowing the highlights, or softer highlights without flattening the shadows. Mid-sparing and bounded by the paper's black and white, like the zone density sliders, and scoped per colour layer through the Global/R/G/B selector like the main Grade.
- New: **Per-layer trims (crossover correction)** — a **Global / Red / Green / Blue** selector on the Tone and Process pages scopes the curve controls (Grade, Toe, Shoulder, Widths, Snap) and the White/Black Point sliders to a single colour layer. Filtration can only *shift* a layer; these trims re-shape it, fixing casts that differ between shadows, mids and highlights. The H&D chart tracks the per-layer curves live.
- New: **Deeper control over the print curve**, grouped under a **Paper Response** header on the Tone page: **Snap** adjusts the paper's midtone punch, **Shadows / Highlights Density** darken or brighten each zone while rolling into the paper's black and white limits instead of clipping, and **True Black** (off by default) anchors the deepest print black to true display black instead of ~5% grey. The **Toe** slider is recalibrated so its full throw is felt as clearly as the Shoulder's — frames without a manual toe render identically.
- New: **Gold toner** — a third chemical toner on the Toning page (B&W only). Alone it works like the archival gold bath: a cool blue-black shift in the highlights and mids with a slight density boost, dense shadows hold. Over sepia it recreates the classic gold-over-sepia combination, pushing the toned highlights from yellow-brown toward orange-red.
- New: **Crop composition guides** — the crop tool's fixed thirds grid becomes a **Guide** dropdown: Thirds, Phi Grid, Diagonals, Golden Triangles, Golden Spiral, Armature, Diagonal Method or Grid. `O` cycles guides, `Shift+O` rotates the spiral/triangles.
- New: **Crop rotation handles** — four handles at the edges of the crop box spin the frame freehand, composing with the Fine Rotation slider for fine-tuning; both now range ±45°. @linkmodo
- New: **Reverse scroll-to-zoom** — an optional toggle in Customize Shortcuts for users who expect scroll-up to zoom out. @linkmodo
- Change: **Auto Cast is built in** — the toggle is gone: Cast Removal always adapts its strength to how confidently the frame's neutral greys read, and the slider (default 0.5) trims on top; 0 = off. Frames that had Auto Cast off will shift slightly.
- Change: **One visual language across the app** — all panels share the same section cards, button styles and sizes; every active tool and toggle uses the same red armed state; edited sliders, selectors and tabs are marked with a small red dot instead of coloured text; tooltips word-wrap; the Metadata page uses collapsible sections like the rest; the Analysis stats read-out shows plain values without the qualitative labels.
- Change: **UI polish pass** — tooltip shortcut chips render as bordered keycaps; hovered items in menus, dropdowns and combo boxes highlight clearly; active/checked tools (Linear RAW, Lock Bounds) now read as engaged; Undo/Redo move to the toolbar (Save Edits into the More Actions menu); the crop tool shows contextual hover cursors (rotate / resize / move / draw); and confirmation dialogs default to the affirmative button on Enter. @linkmodo
- Change: **Confirm before unloading** — Unload / Unload Selected / Clear All now prompt first, and the Delete key unloads the selected frame(s). @linkmodo
- Removed: **Flare** and **Contrast Lift** — Snap and the zone density sliders cover the same ground with real control. Old edits load cleanly and render without them.
- Fix: **flip under Fine Rotation now mirrors correctly** — flipping a straightened frame produced a doubled tilt instead of a true mirror of the current render. @linkmodo
- Fix: **manual crop and freehand analysis region rotate with 90/180 turns** — they stayed put before, so a quarter-turn left the crop framing the wrong area. @linkmodo
- Fix: **viewer clears when the session empties** — Clear All or Unloading the last frame left the previous image on screen with no way to dismiss it. @linkmodo
- Fix: **startup no longer crashes when the Documents folder is missing (windows)** — a OneDrive-backed Documents path that doesn't exist on disk (OneDrive unlinked or signed out) now falls back to `~/Documents` and then the home directory instead of failing to launch. @linkmodo
- Fix: **Apply Settings counts only the frames you can see** — with a filename filter active, "Whole roll" counted (and would apply to) every loaded file instead of just the visible ones. It now matches the filtered list.

## 0.35.0

- New: **Gear Library** — a searchable library of cameras, lenses and film stocks; picking gear for a frame writes scan-rig and film metadata into the exported XMP. There are some bundled items but library is easily user-extendable. @jboneng
- New: **Freehand analysis region** — draw the exposure-metering area directly on the canvas instead of relying on the centered Analysis Buffer inset; a draw/clear pair sits on the Process page. @linkmodo
- Change: **Manual healing now clones real texture** — heals copy a matching patch from elsewhere in the frame and blend the seam, keeping grain intact instead of a smoothed synthetic fill; existing heal spots convert automatically.
- New: **Scratch Tool** — heal hairs and scratches by clicking a polyline along the defect (double-click or Enter to commit), instead of forcing them through the round brush.
- New: Enter now also commits an in-progress Dodge & Burn stroke, same as double-click.
- Change: **Canvas redesign** — the status bar and info rows around the preview are gone; the image fills the whole central area. File info moved to overlay pills in the canvas corners, the toolbar floats over the image, status messages show as a short toast.
- Change: **Export now warns before overwriting files** — a batch that would clobber existing files stops for an Overwrite / Rename / Cancel prompt instead of silently overwriting; tick "Always overwrite without asking" to bring back the old silent behavior. @linkmodo
- Change: **Bundled gear and crosstalk profiles now load live from the app** instead of being copied into your docs folder. Docs folder is reserved for your personal profiles & gear  
- Change: **Selecting a thumbnail now opens it** — a single click loads the frame into the viewer (Ctrl/Shift-click still builds a multi-selection for batch actions); double-clicking inside an open crop box now confirms and closes the tool. @linkmodo
- Change: **Cyan** now has an assignable keyboard shortcut alongside the other white-balance sliders, unbound by default (you should not be touching cyan!) — set it in the shortcut editor.
- Change: **Export UX pass** — one red **Export** button replaces the toolbar Export, "Export All" and the "Sync export settings" toggle; its chevron menu *selects* what the button does (current frame, selected, all visible) and remembers the choice, with **Export Presets** working the same way. Default sizing is now **Original** resolution, multi-frame exports ask for confirmation, and failed frames are reported instead of a false "complete".
- Fix: **crop no longer inflates under Fine Rotation** — the manual crop rectangle is now measured in the same already-rotated view you draw it on, instead of being mapped through the tilt and re-bounded, which grew the cropped area as the tilt increased. @linkmodo
- Fix: **locked/disabled controls now are greyed-out** — sliders, buttons and fields under an active lock (e.g. Analysis Buffer under Roll Average, Flat-Field k1 with no reference profile) now show a grey disabled state instead of looking identical to normal ones.
- Fix: **Next/Prev buttons now follow the sorted/filtered order** shown in the filmstrip instead of raw load order — fixes them greying out mid-roll or staying stuck (but unresponsive) at the end of a roll not loaded alphabetically.
- Fix: **large (tiled) exports no longer drift from the preview** when Separation is on or a freehand analysis region is drawn — the tiled export path was metering exposure off the raw image instead of the same unmixed, region-restricted one the preview uses.
- New: **Protect original metadata** toggle in the Metadata tab — exports keep the source file's EXIF/XMP untouched instead of NegPy rewriting it. @jboneng
- Perf: **CPU rendering is now multithreaded**, ~3.5x faster on a full-res render. @linkmodo
- Fix: HQ preview toggle no longer shows a stale render; zoom % now reflects true pixel zoom (with Fit/1:1 buttons); tooltip shortcut chips and the Crosstalk tooltip no longer overflow; Batch Analysis warning now leads with the crop count. @linkmodo

## 0.34.0

- Change: **Dodge & Burn moved into the print exposure** — masks now adjust exposure before the paper curve instead of scaling the finished image, which is how dodging and burning physically work. Strong burns and dodges roll off through the paper's toe and shoulder instead of clipping flat. The section moved to the Exposure tab; existing dodge/burn edits will render slightly differently.
- Change: **Selenium and Sepia work on print density** — both toners now convert silver density instead of tinting by brightness, matching the real baths. Selenium acts on the densest areas (deeper blacks, cooler shadows); sepia acts on the thinnest (warmer highlights, shadows hold). Existing toned edits will render differently.
- Change: **Export Presets** main button now runs only the selected frames (previews the single frame when just one is selected) instead of always exporting everything. @jboneng
- New: **Export Presets** gains an export-all-visible option, with a confirmation dialog before it runs. @jboneng
- Fix: **dodge/burn mask overlay now shows the real feathered falloff** instead of a hard-edged polygon — the on-canvas mask previously didn't match the soft edge the pipeline actually renders.
- Fix: **exporting all RGB-scan triplets no longer fails** with "Input/output error" on most frames. Batch export was reusing stale saved paths for each frame's green/blue exposures instead of the ones the triplet was actually built from, so it tried to read files that weren't there; it now uses each frame's own exposures, the same as exporting one at a time.
- Fix: exported **JPEG EXIF** no longer blocks ExifTool from rewriting tags — stale RAW preview SubIFD pointers are now stripped on export. @jboneng


## 0.33.0

- New: **Temperature slider** — a Kelvin lever above the CMY white balance: drag it and Magenta/Yellow move together along the warm–cool axis in the right ratio, like re-dialing a dichroic filter pack, while your green–magenta tint stays put. Move M/Y (or Pick WB) yourself and the slider reads back the print's temperature instead. Warm sits on the right, travel is mired-linear (equal drag = equal perceived shift), and `T`/`G` nudge it from the keyboard. The thermometer button locks the temperature for the roll: every frame you open gets re-aimed to it — keeping its own tint — and the lock survives restarts.
- Change: **White balance sliders are real CC filtration** — ±1.0 = ±20cc of density on any frame. Before, the same slider position did more on a contrasty negative and less on a flat one, so a WB that worked on one frame drifted on the next. Frames with saved WB will shift slightly.
- Change: **Paper base tint sits in the paper white** — Fujicolor Crystal Archive's cool tint now lives in the base density, so it shows in the highlights and disappears into the blacks, like actual paper. Turning off Paper White turns the tint off with it.
- New: **RA-4 dye coupling** — Kodak Endura and Fujicolor Crystal Archive model the unwanted absorptions of their print dyes. Greys stay neutral; strong colours shift the way they do on real paper. Values estimated based on manufacturers technical sheets.
- New: **Histogram pixel marker** — hover the image and the Analysis histogram marks the pixel's R, G and B values with dashed lines in channel colours.
- New: **Tone-curve drag feedback** — while dragging a Tone slider, the H&D chart keeps the pre-drag curve as a faded ghost so you see exactly what moved; Toe/Shoulder sliders glow their zone of the curve, Grade/Density highlight the pivot crosshair.
- Change: **Lock Bounds** is now a labeled button beside Linear RAW in the Process panel, instead of a small icon squeezed next to the process dropdown that many likely missed.
- New: **Expanded tutorial** — the walkthrough now explains the physics behind the pipeline (log density, the H&D print, orange-mask metering) and why a tight crop or the Analysis Buffer matters for clean black/white points. New steps for the Analysis Buffer, Crosstalk dye unmixing, roll consistency, Cast Removal and the Finish panel.
- Fix: the **GPU histogram** was binning scene-linear values since the scene-linear pipeline rework (0.30.0), reading darker than the CPU one — both now bin the display-encoded image.
- Fix: the bundled **RGBScan input ICC profile** used an sRGB TRC instead of the gamma 2.2 curve NegPy's pipeline expects at that stage, lifting the toe and giving a washed-out look — rebuilt to match, same colour transform. @thetalkingdrum
- Fix: a **roll's Batch Analysis baseline no longer leaks onto other rolls** — the roll-wide bounds and colour balance were being remembered globally and quietly applied to every freshly opened file, so a different film stock (e.g. mask-less Phoenix II) came up with a heavy cast until you hit Reset. The baseline now stays with the roll it was measured on; open a new roll and it meters per frame again.

## 0.32.1

- Perf: fewer disk syncs on every edit (settings writes batched into one transaction), smoother slider dragging, faster contact-sheet/batch/tiled export, and lighter preview rendering — no change in output.

## 0.32.0

- Change: **Crosstalk moved to the Process panel** — the unmix now applies to the raw negative densities **before** analysis and inversion, making it more physically correct. Old edits migrate automatically, but expect a subtle shift on frames that used Separation. Re-run Batch Analysis after changing it.
- Change: **Process panel split** into *Process* and *Roll Analysis* collapsibles.
- New: **Print stats row** — exposure in stops and CMY white balance as dichroic CC filtration (±1.0 = ±20cc).
- New: **Scan clip warning** — per-channel share of source pixels at sensor white (red above 1%); in a negative scan that clipping destroys base/shadow separation and can only be fixed at capture.
- Change: **halation** is masked in linear light (its footprint no longer moves with Grade/Density) and, like glow, composited additively.
- Fix: the **H&D chart now matches the render at hard grades** — the grade-coupled toe/shoulder was applied by the engine but not shown by the chart.

## 0.31.2

- Fix: in the **Apply** dialog, ticking **Tonal span** or **Colour balance** no longer clears the other — each now toggles only its own axis (Use Luma Average / Use Colour Average) on the target frames and leaves the other as it was. (#375)
- Fix: the **filmstrip thumbnail** now tracks the current edit — resetting settings or adjusting sliders updates the grid preview in place, instead of leaving it stale until you switch, save or export. (#376)

## 0.31.1

- Fix: new bundled Lab Crosstalk matrices weren't copied to the user's profile folder on startup once any profile already existed there — each bundled matrix is now seeded independently, so later releases' additions show up without waiting on a fresh install.

## 0.31.0

- **Default tone curve retuned** — Auto Grade targets slightly lower contrast, and the midtone contrast boost eases in more gradually, for a softer out-of-the-box look.
- **Cast Removal is now a slider** — the toggle becomes a 0–1 **strength** slider to dial colour-cast neutralization back partway (default 0.5). A small **auto** button (like Density/Grade) sets the strength from how confidently the frame's neutral references read — clean greys full, few-neutral scenes gentler — with the slider trimming on top.
- **Lens distortion correction** — the Flat-Field profile gains a **k1** slider for radial (barrel/pincushion) correction, alongside illumination correction. Folded into the geometry transform (no RAW re-decode on drag), scale-to-fill, and kept in sync with crop/retouch/dodge-burn.
- **Apply settings dialog** — the Sync Edits / Sync Crop buttons become one **Apply** button opening a dialog: pick Selected frames or the whole roll, tick any of Process, Crop, Rotation, Exposure, Color, Finish, Tonal span and Colour balance. The bounds options broadcast the source frame's normalization as a locked roll baseline (single-frame Batch Normalization).
- **Optional edit sidecars** — mirror edits to plain `.negpy` files next to the source for archival (SQLite stays primary). Off by default; enable in the Export panel or write on demand. Loading falls back to a beside-source sidecar when there's no DB entry, and promotes it into the DB.
- **Exposure panel split into Colour + Tone** — two independent collapsible sections, each with its own edited badge and reset: **Colour** (white balance, Cast Removal) and **Tone** (density, grade, toe/shoulder, contrast lift, paper). The region selector is now an icon column beside the CMY sliders (yellow when adjusted), the heading names the active region, and Pick WB is an eyedropper. Colour shows an RGB mini-histogram, Tone the luminance one.
- **Geometry gets its own tab** — the Setup tab (Presets, Geometry, Process) is split: Geometry and Flat Field move to a dedicated tab, leaving Setup as Presets + Process.
- **Kodak Aerocolor IV 2460 crosstalk matrix** added to the bundled Lab Crosstalk profiles (also sold as SantaColor 100 / 1Hundred). @whepper
- **Flat master as a per-preset export option** — export presets gain a **render intent** (Print or Flat master), so **Export Presets** can produce a mix in one click — e.g. JPEG + PNG + Flat Master — independently of the main panel's Print/Flat toggle. Manage Presets' **+** button now offers a Print or a Flat master preset (flat presets limit format to 16-bit TIFF / Linear DNG). @jboneng
- Fix: toggling **Linear RAW** no longer leaves a stale magenta cast — the auto-meter cache invalidates when the RAW decode changes.
- Fix: the **Flatfield Correction** toggle is no longer reset to off when switching files or applying edits — it was colliding with RGB Scan's toggle in saved settings.
- Fix: sidebar labels no longer show an opaque black background patch against lighter section panels, and field labels next to combos/entries now share consistent styling across panels.
- Fix: right-panel sections now stack from the top at their natural height instead of stretching to fill the panel and splitting the leftover space between them.
- Fix: the tutorial overlay's body text no longer pans slightly wider than its popup — long unbreakable strings (e.g. file paths) now wrap instead of pushing the content past the visible width. @seanharding

## 0.30.2

- **Cast Removal — cleaner highlights** — the per-channel gray balance now anchors a third (highlight) reference, fitting a curve through highlight/midtone/shadow instead of a line. Fixes highlights occasionally overcorrecting past neutral (toward magenta) under 0.30.1.
- **More bundled crosstalk profiles** — additional Lab Crosstalk matrices for common film stocks, derived from official datasheets. @jboneng

## 0.30.1

- **Improved Cast Removal** — neutral greys no longer drift slightly green. Cast Removal now balances each colour layer at the **midtone** as well as the shadows (a true two-point per-channel gray balance), measured only on near-neutral pixels so green-heavy scenes (foliage, skin) can't pull the balance. Previously the midtone leaned on a single luminance reading that is mostly green, leaving a faint green cast on many C-41 conversions. The default look shifts slightly toward neutral.

## 0.30.0

- **Scene-linear pipeline** — the whole conversion now runs in scene-linear light internally: the creative stages (Retouch, Lab, Local, Toning, Finishing) operate on linear light instead of gamma-encoded data, so their math is physically correct, and the "print" colour space is now the wide-gamut **ProPhoto RGB**. The output/display transform is applied **only at the very end** with the correct working-space curve, fixing a latent mismatch where the internal buffer was sRGB-encoded but tagged as a wider space. Dust retouching keeps the same (perceptual) detection but now heals in linear light, with the CPU and GPU paths unified. In practice: more headroom before saturated colours clip, and a more accurate, slightly more saturated default look — existing edits will look a touch different, so re-tune Saturation/Toning to taste.
- **Independent roll average for luma and colour** — the single **Use Roll Average** toggle is now two buttons, **Use Luma Average** and **Use Colour Average**. You can take the roll-wide tonal-range (black/white-point) baseline while letting each frame find its own colour balance, or vice-versa. With both on it behaves exactly like the old Use Roll Average; with both off, like per-image local.
-  **Linear RAW on by default** — RAW files now decode with neutral (1,1,1,1) multipliers, bypassing the camera's as-shot white balance. You can still re-enable camera WB with the **Linear RAW** toggle in the Exposure sidebar (off = camera WB applied).
- **Faster auto-exposure analysis** — the block-median prefilter behind Auto Density/Grade and normalization is now multi-threaded with bit-for-bit identical results, roughly 2.5× faster on large frames, so opening files and batch analysis feel snappier.
- **Snappier live preview (GPU)** — the GPU preview no longer re-meters the negative every frame (auto-exposure analysis is cached per image and reused while you drag creative sliders), and the engine caches bind groups, uses lighter preview decodes and a source cache. Dragging sliders is dramatically smoother and repeat exports are faster, with identical results.
- **Contact-sheet output location & templates** — set an explicit output folder for the contact sheet, and save/recall named layout templates. @jboneng
- **Flat output tidies the Export panel** — the Flat intent hides controls that don't apply, and honours your Print/Pixels sizing. @jboneng
- **Export panel reorganised** — laid out in export order (output intent, settings, then the Export buttons), with presets, contact sheet and preview tucked into collapsible sections below.
- **Rule-of-thirds grid on crop**, plus a denser 10×10 leveling grid while fine-rotating.
- **Edited controls turn yellow** — changed sliders and the tabs holding them tint yellow, so you can see what you've touched.
- **VISION3 500T crosstalk matrix** added to the bundled Lab Crosstalk profiles.
- Fix: main window now fits small (1368×768) screens, and remembers its size/position.
- Fix: long monitor ICC profile names no longer force a horizontal scrollbar in the Export panel.
- Fix: Pakon `.raw` files no longer show a thin strip of garbage pixels along the left edge (and process a touch darker) — the loader now skips the file's small header.

## 0.29.1

- Fix: Previously RGB Scan mode only merged exposures if it was enabled BEFORE loading the files. Now toggling the mode after files are loaded forces the process. (#319)
- Fix: Flat export having heavy colorcasts for some files. It also follows the selected colorspace instead of hardcoding ProPhoto (#321)
- Fix: New crosstalk matrices not being correctly bundled with build in 0.29.0.

## 0.29.0

- **RGB Scan (new)** — for negatives shot as three separate frames under red, green and blue light, combine them into one clean, low-noise colour scan. Turn on the **RGB Scan** button in the Files sidebar; folders of shots are grouped into triplets automatically, and an **Edit RGB Triplet…** menu lets you fix the grouping. The frames are aligned to each other to avoid colour fringing, then run through the normal conversion as usual.
- **Asymmetric H&D print curve** — reworked exposure/normalization around a film-style characteristic curve with independent toe and shoulder, plus an absolute-percentile colour clip and a luma-decoupled.
- **Darkroom paper profiles (reintroduced)** — a paper dropdown in the Exposure panel applies per-paper curve shaping rather than post-processing (tonal, per-channel gamma, base tint) mapped from Ilford/Kodak/Foma/Fuji datasheets. Gated by process mode and sticky roll-wide.
- **JPEG XL (.jxl) export (new)** — support in python libs is somewhat restricted yet so the UI restricts it to representable spaces (sRGB, Display P3, Rec 2020, Greyscale) and doesn't support embedding custom .icc.
- **WebP (.webp) export (new)** — 8-bit lossy or lossless, with a **Lossless** toggle plus Quality and Method (encoder effort) sliders. Works with any export colour space (ICC embedded). The quality/effort knobs in JPEG and JPEG XL are now sliders too.
- **Flat master output (new)** — a new "Flat — for editing elsewhere" output intent that exports a **true log master** (S-Log/LogC-style): a flat, low-contrast, milky digital intermediate for grading in Lightroom/Darktable/Photoshop/Resolve. Instead of the print curve it emits the normalized log signal directly as the code value (no `10^-D` decode, no display gamma — that decode is what makes a normal positive), `code = clip(lift + gain·(1 − val))`, so the master holds maximal latitude and is fully invertible. Exported as a wide-gamut **16-bit TIFF** (or optional **Linear DNG**); skips the creative print look (auto density/grade, cast removal, lab, toning, finish) and maps camera RAWs to ProPhoto-linear via the camera's own matrix. New **Output Intent** section in Export with a Print/Flat toggle, **Preview Flat** peek and a **Roll Baseline** button for roll-wide consistency. Standard "Print" mode is unaffected. @jboneng
- **More film scanners supported** — dedicated film scanners on the SANE *pieusb* backend (Reflecta, Pacific Image) now work instead of being skipped, including their infrared dust-removal channel, plus several fixes to scanner mode and option handling. @hullrich
- **Histogram linear/log toggle** — a **LIN / LOG** pill in the histogram switches the count axis, lifting sparse shadow/highlight detail. Instant repaint, persisted across restarts. @jboneng
- **Unified crop tool** — the separate Manual-draw and Move Crop tools are merged into one **Crop** tool: drag corners to resize, drag inside to move, click outside to draw a fresh rect. While the tool is open it previews the full uncropped frame instead of the already-cropped image. Corner handles now track the cursor exactly (no snapping), aspect-ratio locking and min-size are computed in screen-pixel space (correct on non-square sources), and the overlay draws the actual axis-aligned crop box under fine rotation instead of a tilted quad. Dragging the rect no longer forces a full CPU recompute of exposure/retouch/lab/local on every step. On top of this. Auto-exposure bounds recompute is deferred until the crop tool closes (no per-drag re-normalization lag), enabling Auto crop exits the Manual crop tool, and a new **Reset** button beside Crop clears the manual crop and disables auto crop. @reederphill
- **Custom crosstalk profiles** — bundled Kodak Gold 200, Portra 400 and VISION3 250D matrices for the Lab Crosstalk control, plus bundled a Claude skill to derive a `.toml` matrix from a film datasheet.
- **History panel (new)** — a **History** tab in the right panel lists every edit step for the current photo. Click any step to jump back to that intermediate state (the preview updates instantly), carry on editing from there, or right-click to **Export this version**. Surfaces the existing per-file undo history (last 100 steps) as a scrollable list.
- **Expanded interactive tutorial** — the first-run walkthrough now covers the newer features too: RGB Scan, Flat-Field correction, the crop/straighten tools, paper profiles, toning, Dodge & Burn, the History panel and the Flat master export, alongside refreshed Exposure (ISO-R grade, H&D curve) and Process (Luma/Colour clip) steps.
- File pickers now reopen at your last folder instead of the system root.
- Fix: **thumbnails now update when you switch away** — editing a photo and moving to the next one refreshes the edited frame's thumbnail in the grid, instead of leaving it stale until you re-open or explicitly save it.
- Fix: **X-Trans full-res demosaic** uses DHT instead of VNG, removing dot/maze artifacts on Fujifilm sensors. @reederphill

## 0.28.0

- **Dodge & Burn (new)** — local lighten/darken with freehand **polygon masks**, darkroom-style. Click to drop vertices, double-click (or snap-to-start) to close, and click inside a mask to reselect it; each mask carries its own EV strength and feather radius. Masks are stored in raw-image space and re-projected through the geometry transform, so they survive rotation, flip and crop. The stage runs on the **GPU with bit-for-bit CPU parity**, with a **Show Masks** toggle in the new Local sidebar section. (#274, #275) @reederphill
- **Session restore** — NegPy now remembers the files you had open and offers to reopen them on the next launch. Moved or deleted files drop out automatically and your per-file edits come back. (#276)
- **Rule-of-thirds grid while straightening** — a 3×3 alignment grid appears on the canvas while you drag the Fine Rotation slider, giving horizontal/vertical references to level horizons and buildings; it fades shortly after you release. (#258, #271)
- **Cross-platform monitor profile detection** — the preview's display-profile auto-detection now works beyond Windows: **colord** on Linux (Wayland + X11), **ColorSync** on macOS, and PIL on Windows. macOS falls back to **Display P3** (not sRGB) when the OS reports no profile, matching modern Mac panels. When no profile can be detected (e.g. wlroots compositors like Hyprland/Sway, which don't register a display with colord), the **Display** selector in Export turns red, prompting you to pick your monitor's colour space manually. (#277)
- Fix: **all export presets are visible again** — the Presets list no longer clips its last entry on short screens; the nested scroll area that collapsed and swallowed wheel scrolling has been removed. (#270, #273)

## 0.27.1

- Fix: the default export colour space is now sRGB instead of Adobe RGB, so exports get a real ICC transform rather than just a wide-gamut tag that many viewers misread as sRGB. **Soft proof** now defaults on so the preview reflects the export colour-space clamp instead of silently diverging from it, and the toggle is surfaced more clearly. Monitor ICC detection failures now log a warning instead of a silent debug message. @reederphil
- Fix: **crop UX** activating Manual Crop now resets to the full image instead of keeping a stale rect or autocrop from a prior session. The Manual and Auto buttons show a red corner-dot indicator when a crop is active, so it's visible without inspecting the button state. @reederphil

## 0.27.0

- **Flat-field correction** (new, off by default) — corrects illumination falloff/vignetting from your light source or scanner using a reference scan of the bare light. New **Flat Field** section in the Setup tab: save named reference profiles, pick the active one, and toggle correction per image.
- **Contact-sheet export** — render all visible frames into a single darkroom-style contact sheet. New **Contact Sheet** section in the Export panel with cell size, gap, margin and max-tiles controls.
- **Batch progress popup** — batch analysis, batch export and thumbnail generation now show a floating popup with an animated spinner, current file and progress, alongside the existing status-bar indicator. Batch export and batch analysis can be **aborted** from the popup; aborting export keeps the files already written and skips the rest.
- **Thumbnail right-click menu** — right-click a frame in the filmstrip for quick access to Export, Copy / Paste / Reset Settings and Unload, without leaving the contact sheet. Right-clicking an unselected frame selects it first; with multiple frames selected the menu acts on the whole selection and adds **Export Selected** and **Sync Edits to Selection**.
- **Export presets** — save named export configurations (format, sizing, output destination, color space) and run them in one click. A new **Manage** dialog lets you add, duplicate, reorder and delete presets; each preset has an enable toggle. Presets are stored globally and pinned to the top of the Export panel in a collapsible section. The **Export Presets** button exports the current file through every enabled preset at once, while the normal Export / Export All buttons keep exporting with the settings shown in the panel. New installs ship with **JPEG** (enabled), **TIFF** and **PNG** (disabled) starter presets. @reederphill
- Fix: **Sync Edits** no longer overwrites the target's crop, rotation and flips — it now carries over only the manual-crop rectangle and fine rotation, leaving the target's geometry otherwise untouched. @reederphill
- Fix: the **Unload** button now clears just the selected files when more than one is selected (falling back to clearing all), with a tooltip that reflects the action. @reederphill
- Fix: next/previous navigation now follows the displayed (sorted/filtered) order. @reederphill
- Fix: **trackpad scrolling** in the filmstrip now uses pixel-precise deltas with native momentum instead of feeling much faster than the rest of the OS. @reederphill

## 0.26.0

**Reworked interface** — the panels have been reorganised around the editing workflow.

- **Editing tools split into workflow tabs** — the single long Controls list is now four icon tabs that follow the pipeline: **Setup** (Presets, Geometry, Process), **Exposure**, **Color** (Lab, Toning) and **Finish** (Retouch, Finishing), alongside **Export**, **Metadata** and **Scan**. The tab bar is icon-only with hover tooltips, and each tab scrolls on its own. Jump straight to a tab with `Ctrl+1`–`Ctrl+7` (rebindable, shown in the `?` overlay).
- **Analysis pinned to the top** — the histogram, photometric curve and stats now sit in a sticky, collapsible **Analysis** section at the top of the right panel (instead of a left-panel tab), and a draggable divider lets you resize Analysis vs. the controls below it. Its size and open/closed state persist across restarts.
- **Left panel is now just the filmstrip** — Export, Metadata, Scan and Analysis moved to the right panel, leaving the left side a clean contact sheet.
- **Contact-sheet thumbnails** — the filmstrip is now a justified grid: thumbnails scale to fill the panel width and add/remove a column as you resize, with a subtle border hugging each image (no boxy cell). The current image is shown full-brightness with a gold frame while the rest are dimmed. Labels are dropped for a denser sheet (filename still on hover).
- **Compact filmstrip toolbar** — file actions (Add files / folder, Clear, Hot Folder, Sync Edits, Sync Crop) are now a single row of icon buttons, with Sort folded into a dropdown.
- **GPU acceleration moved to the bottom toolbar** — a single ⚡ toggle (with a tooltip showing on/off and the active backend) replaces the checkbox + badge in the side panel.
- **`Esc` cancels the active tool** — deselect WB Pick / Manual Crop / Move Crop / Heal without clicking its button again.
- **`Shift+A` triggers Autocrop** — new keyboard shortcut for the Auto crop button.
- **UI scaling** — a **UI Scale** entry in the toolbar ⋯ menu scales the whole interface from 80% to 120% (applied on next launch).
- Fix: a freshly selected thumbnail could briefly show a wrong (blue) colour cast until you re-selected the file — rendered thumbnails are now colour-managed to match the canvas and captured only once the render has settled.
- Fix: **preview is now colour-managed to your monitor** — the preview previously assumed an sRGB display, so on a wide-gamut screen (e.g. Display P3) it looked over-saturated or washed out and shifted when you changed the Output ICC. The final preview is now converted to the monitor's ICC profile, auto-detected from the screen the window is on (and re-detected when you move it to another display). Falls back to sRGB when the OS reports no profile. (#243)
- **Display profile selector** (Export → ICC): a new **Display** dropdown shows the auto-detected monitor profile ("As detected — …") and lets you override it with a common space (sRGB, Display P3, Adobe RGB, Rec 2020, ProPhoto). Use the override on wide-gamut monitors whose profile the OS doesn't report (common on Linux). Affects the preview only, not the export.
- **Soft proof toggle** (Export → ICC, off by default): like Photoshop's *Proof Colors*. Off, the preview is simply your edit shown correctly on your monitor and the Output/Input ICC affect the exported file only. On, the preview simulates the Output profile. Defaulting off keeps the preview stable when you change export color space.
- **Paper/printer soft proofing**: with Soft proof on, selecting a paper/printer ICC (RGB or CMYK) as Output now simulates the print on screen — paper white, reduced black, and gamut compression (absolute-colorimetric proofing). Previously paper profiles changed nothing in the preview (and CMYK profiles silently did nothing).

## 0.25.0

- **Reworked negative conversion** — the default conversion now behaves much more like a real darkroom print: paper-like highlight roll-off, true deep blacks, and a paper-white base. Out-of-the-box results should look more natural and respond more predictably as you adjust Density and Grade.
- **Auto Density** (new, on by default) — meters each frame individually and sets a sensible brightness, so an over- or under-exposed negative comes out usable without manual tweaking. It corrects *partially*, so a deliberately dark (low-key) or bright (high-key) shot keeps its mood instead of being flattened to a neutral grey.
- **Auto Grade** (new, on by default) — chooses contrast to suit each scene, gently: a flat scene gets a little lift, a punchy scene stays punchy, but nothing is pushed to a harsh extreme. Together with Auto Density this aims for results that look right straight away while staying easy to fine-tune.
- **Grade now uses the ISO R scale** — contrast is set on the photographic ISO R paper-grade scale (R180 very soft … R110 ≈ classic grade 2 … R50 very hard) instead of the old 0–5 number, so the curve's 10–90% span maps to a real paper exposure range. Old saved 0–5 grades migrate automatically (R = 150 − 20·G), keeping the look of existing edits.
- **With the auto helpers off, the conversion follows your negative honestly** — a dense / over-exposed negative comes out dense and contrasty, a thin / flat negative comes out flat. Nothing is normalised away, so the print reflects exactly how the frame was exposed and developed. Flip the toggles off when you want the conversion to *show you your photography* rather than smooth it out — a useful way to read your own exposure and development.
- **Cast Removal** (new, on by default) — automatically neutralizes the colour cast a negative leaves in the print. It balances each colour layer so greys read neutral all the way from the deep shadows through the highlights, not just in the midtones — the usual cause of shadows or highlights drifting off-colour after a midtone white balance (C-41).
- **Contrast Lift** (new, off by default) — a gentle contrast lift about paper white based on preferred tone reproduction (Bartleson-Breneman): prints viewed in a normal room want slightly more midtone contrast than a 1:1 reproduction, so this darkens midtones a touch and adds snap.
- **Flare** (new, off by default) — a darkroom-style veiling-glare floor that gently lifts the deepest blacks and softens the toe for a more film-like look, while leaving paper white fixed.
- **Autocrop rework**: more robust film-edge detection, with a new **Autocrop Mode** selector in the Geometry sidebar — *Image only* crops to the exposed image area (default), *Film edge* crops to the full film extent, keeping the rebate/sprockets. Detection should be more reliable across stocks and border types.
- **Custom crosstalk matrices**: the Lab **Crosstalk** control (renamed from *Separation*) now has a profile dropdown. Drop your own `.toml` calibration matrices in the `NegPy/crosstalk` folder (a starter set is seeded on first run) and pick them per film stock or scanner; the built-in matrix stays available as *Default*. See `docs/CROSSTALK.md` for the format and how to contribute matrices to the bundled gallery.
- Fix: **ICC color management now apply correctly on export** — The on-screen preview is now color-managed through a working-space → sRGB display LUT, so what you see matches the exported file.
- **ICC moved into Export panel**: a dedicated **ICC** section holds explicit **Input** and **Output** selectors. The Output selector lists target color spaces and custom ICC profiles (bundled profiles that duplicate a color-space entry are hidden). The Output selection drives both the preview and the exported file (WYSIWYG — no separate apply toggle): the preview shows the output space directly, matching the file as seen in a non-color-managed viewer, and the export converts to and embeds the same profile. Input ICC corrects the source. The standalone ICC section in the right Controls panel has been removed.
- **Hideable side panels**: the left Session panel and right Controls panel can now be collapsed for a focused, canvas-only view. New toggle buttons sit at the outer edges of the bottom toolbar (and `Ctrl+[` / `Ctrl+]` shortcuts) — the button reflects each panel's current state and hidden/shown state is remembered across restarts.
- Fix: **colour adjustments now use the correct working colour space** — Saturation, Vibrance, Chroma Denoise and Split Toning compute their CIELAB without converting to sRGB, so colours — greens and cyans especially — shift more accurately and predictably. Neutral greys are unaffected.
- Fix: export no longer drops metadata when the source EXIF carries an out-of-range tag.
- Fix: **Batch Analysis** now respects the file browser filter — with a filter active, only the visible files are analyzed, matching Batch Export's behaviour (instead of always running on every file in the folder). (#237)

## 0.24.0

- Added **Before/After**: toggle button ◑ on toolbar (and `\` shortcut) to flash the un-graded auto conversion of the current frame, so you can see what your edits changed at a glance.
- **Faster preview loading**: opening and navigating between photos is now faster. Previews use a quick decode for display (full quality is still used on export), recently-viewed images are kept in memory, and an embedded thumbnail shows immediately while the full preview loads. @reederphill
- **Process** sliders: White/Black Point offsets stay editable while Roll Average is on (only Analysis Buffer and D-Range Clip are locked by roll average). Disabled sliders can no longer be scrubbed by dragging their label. @reederphill
- **Selection vs. open**: clicking a thumbnail now only selects it (for batch operations); double-click — or the arrow keys — opens it for editing. Importing files or a folder auto-opens the first one. @reederphill
- **Batch Analysis** now applies the current image's **Analysis Buffer** and **D-Range Clip** to every file in the roll before averaging — set them once on the open frame and the whole batch shares that setting (instead of each file using its own saved value). The confirmation dialog now opens every time you run it, explaining what the analysis does, how those two sliders are applied, and the crop status of the roll (so you know if any files will be analyzed on the full frame).
- Switching photos no longer blanks the canvas — the previous frame stays on screen, dimmed under a loading spinner, until the new one is ready.
- Fix: exporting a file whose source has a large embedded thumbnail or maker-note no longer silently drops all metadata — oversized EXIF is now trimmed to fit the JPEG limit instead of failing.

## 0.23.2

- Fix: a manually cropped photo no longer fails to load with "asdict() should be called on dataclass instances" after switching to another photo and returning. The crop rectangle was reloaded as a list instead of a tuple, making the config unhashable and crashing the render cache. (#228)

## 0.23.1

- Fix: adjust thresholds for film type detection to reduce negatives falsely recognized as positives.
- Made **process mode autodetect** toggleable using button next to the process dropdown.


## 0.23.0

- **D-Range Clip** now extends into negative values for outward headroom. The slider previously bottomed out at the true min/max (no clipping); pulling it below 0 pushes the normalization bounds *beyond* the histogram extremes, leaving lifted blacks and unclipped highlights for a gentler-than-default stretch. Positive values behave exactly as before (clipping the tails for more aggressive recovery).
- Added **Process mode autodetect**: new files are now analyzed on load and the process mode (C41 / B&W / E-6) is set automatically — orange-mask detection for C41, channel-correlation for B&W, balanced channels for E-6. Detection only applies to files without saved settings, so it never overrides a mode you set yourself.
- Fix: JPEG export now uses **4:4:4** chroma subsampling instead of libjpeg's default 4:2:0, preserving full color resolution at quality 95. Fine film grain and color detail no longer pick up chroma artifacts on export. (#224)

## 0.22.3

- Fix: exported images now honor the source file's EXIF orientation and match the preview — rotation and crop no longer drift on export for files carrying an orientation tag. (#218)

## 0.22.2

- Fix: tiled GPU export now correctly applies IR dust removal — it was silently skipped when the image was split into tiles during high-resolution export. (#216)
- Fix: tiled GPU export no longer applies vignette independently per tile — vignette is now computed over the full frame so seams don't appear on large exports. (#217)

## 0.22.1

- Fix: **Batch Analysis** now respects each file's crop and orientation when computing the roll-wide baseline. Previously, files with large borders (e.g. 6×6 negatives in a 3:2 scan) skewed the average because analysis ran on the full frame. (#213)
- **Sync Crop**: split the Sync Edits button in two — `Sync Edits` (exposure / lab / toning / process settings, preserves per-file crop) and `Sync Crop` (manual crop + rotation only). Useful when every frame on a roll shares the same scanner mask.
- Added **Analysis Buffer overlay**: while moving the Analysis Buffer slider, the canvas shows a dim border around the excluded region with a dashed accent-colored boundary, so you can see exactly what's being analyzed. Disappears shortly after the last slider movement.
- Pre-batch warning: if none of the selected files have a crop set, Batch Analysis prompts before running and points to either cropping or raising the Analysis Buffer.
- Status bar during Batch Analysis now indicates per-file crop state (`[cropped]` vs `[full frame]`).
- Lowered default **Analysis Buffer** to 0.05 and clamped slider max to 0.25 to match the underlying clamp in analysis.

## 0.22.0

- **Tool-aware cursor**: mouse pointer changes shape to reflect the active tool — pointing hand for WB Pick, crosshair for Manual Crop, open/closed hand for Move Crop, hidden cursor (brush circle) for Heal.
- Added **IR Dust Removal**: uses the infrared channel from IR-capable scanners (Nikon Coolscan, Epson flatbeds with SilverFast iSRD, VueScan 4-channel output) to detect and inpaint dust and scratches with near-zero false positives. Toggle and threshold slider in the Retouch panel — controls are disabled automatically when no IR channel is present in the loaded file.
- IR channel is read from: 4-channel TIFFs with ExtraSamples (VueScan, NegPy's own scanner output), multi-page TIFFs with a grayscale IR page (SilverFast iSRD), and `_IR.tif` sidecar files.
- **Tooltips**: added detailed tooltips to every sidebar control — sliders, buttons, dropdowns, and checkboxes. Controls with keyboard shortcuts show dynamic shortcut chips that update when bindings are customised.
- **Optimizations**: optimizations to preview loading speed. @reederphill

  ## 0.21.0

  - Added initial **Scanner support** on Linux and macOS: new Scan tab — select a SANE-compatible scanner, choose resolution, bit depth, output format, and filename template; scanned files auto-load into the session. This is initial implementation, tested with Plustek 8100 on Arch Linux and latest macOS. As it often is with (old in most cases of film scanners) hardware support i cannot guarantee that it will support your scanner. **IMPORTANT:** check [README.md](../README.md) for information about SANE dependencies. **IMPORTANT**
  - Added **Lock Bounds** button in the Process section — freeze normalization bounds so cropping and re-analysis don't overwrite them; useful for locking in exposure after initial analysis.
  - Added **Copy/Paste Bounds** between files — transfer normalization bounds from one file to another (**CTRL+Shift+C** -> CTRL+V).
  - **Canvas zoom**: pinch-to-zoom gesture support; smooth trackpad scrolling; increased min/max zoom bounds. @reederphill
  - Fix: TIFF export with ICC profile applied produced 8-bit tiffs.
  - Fix: Custom metadata not written correctly in tiffs.
  - Fix: Hot folder mode caused session UI debouncing lag.
  - Fix: **Saturation** slider above 1.0 darkened already-saturated reds/blues. Saturation now scales chroma in CIELAB, preserving perceived lightness (matches Vibrance behavior). (#193)
  - Docs: More detailed [USER_GUIDE.md](USER_GUIDE.md)


## 0.20.0

- Added **filtering** options to film strip - details in [FILTERING.md](FILTERING.md)
- Fix: HQ preview scaling up crop offset setting.

## 0.19.0

- Added **Metadata panel**: new "Metadata" tab in the session panel — set Film stock, Format, Developer, Push/Pull, and Scanner info written as EXIF tags into exported files. Shows read-only inherited EXIF from the source file (camera make/model, lens, exposure settings). Optionally sync custom metadata across all files in a batch export.
- Added **Detect Aspect Ratio** button in the Geometry sidebar (crosshairs icon) — finds the film frame in the image and sets the crop ratio to the closest standard aspect ratio.
- Added **Same folder as source** export option — exports files to their source directory instead of a fixed export path.
- Added **Overwrite existing files** toggle — when disabled, exports get incrementing suffixes (`_2`, `_3`) to avoid overwriting.
- **Sticky flip**: Flip Horizontal/Vertical buttons now show a pressed state when active and their state persists across files and app restarts (applied to new files only — files with saved edits keep their own flip).
- Tutorial overlay now shows keyboard navigation hints at the bottom.
- Fix: mouse input in tutorial (Windows) @alessandrv
- Fix: exports are now written atomically — no partial files left on crash or interrupt.
- Fix: Rotate CCW/CW buttons (and `[` / `]` shortcuts) reversed direction when the image was flipped horizontally or vertically (but not both).
- Performance & stability improvements.

## 0.18.2

- Fix: GPU export crop now recomputed at full resolution instead of scaling up the preview ROI — fixes misaligned crop on export when using crop offset with autocrop or manual crop.
- Fix: applying autocrop no longer silently resets the crop ratio to "Free" — the user's chosen ratio is preserved.

## 0.18.1

- Fix: startup crash on Windows systems with non-UTF-8 locale (e.g. Traditional Chinese cp950) caused by reading the stylesheet and other files without explicit encoding.
- Fix: tutorial popup body text cut off on long steps (e.g. Lab panel) — body now scrolls when content exceeds available height.

## 0.18.0

- Added **Interactive Tutorial**: step-by-step overlay walkable from the toolbar, covering the full pipeline from loading files to export.
- Added **Move Crop** tool: translate an existing manual crop rectangle without resizing it — new button in the Geometry sidebar (disabled until a crop rect is set).
- Added **Target Pixel Size** export mode: alongside the existing Print DPI and Original Resolution options, you can now export to a specific long-edge pixel count. Old `use_original_res` workspace files are automatically migrated to the new `export_resolution_mode` field.
- Improved **Autocrop**: more robust edge detection; autocrop is now off by default and resets when the button is deselected. @alessandrv
- Fix: HQ preview no longer inflates bounds analysis — analysis is always run on the downsampled image, eliminating noise from single dead pixels or sharp dust spots that could throw off normalization. (#162)

## 0.17.2

- Fix: batch normalization and batch export progress bars now properly updates on status bar.
- Fix: export filename templating now preserves `original_name` verbatim — dashes, spaces, and multiple/leading/trailing underscores in the original filename are no longer mangled by the cleanup pass.
- Changed default export filename pattern from `positive_{{ original_name }}` to `{{ original_name }}`.
- Changed default export colorspace to be `Same as source` (detected from input file).
- UX: Status bar - replaced zoom % and resolution (shown elsewhere) with a file position counter (`3 / 12`) for batch workflows.

## 0.17.1

- Fix: new vignette shaders not being bundled with appimage on linux
- Fix: queue render cleanup on worker thread to prevent use-after-free (GPU)

## 0.17.0

- Improved default conversion/normalization process, now out-of-the-box results should be better & more natural.
- Camera white balance is now applied by default — images open with the camera's as-shot WB for a balanced starting point.
- Replaced the "Camera WB" toggle with a "Linear RAW" button. Enable it to decode RAW files with neutral (1,1,1,1) multipliers, bypassing the camera WB.
- Added **Vignette effect**: new Finishing sidebar section with Strength and Size sliders. Applies a radial cosine-falloff vignette post-crop in both CPU and GPU pipelines. Negative strength darkens edges, positive brightens them.
- Moved **border controls** from the Export section to the new Finishing section. Border Width is now a slider. Old workspace files with border settings are automatically migrated.
- Increased range of **Analysis buffer** slider.
- Better error recovery in edge cases (swapchain resizing, invalid crop ratios).
- Config deserialization now warns on unknown keys — old workspace files with renamed settings won't silently lose data.
- More tests for GPU-CPU feature parity.
- Removed some dependencies that were not used.
- Added test coverage reporting to CI.
- UI refinements.

## 0.16.0

- Fix: rendering crash on some DNG and TIFF files at extreme D-Range Clip values (`kth out of bounds` error from `np.percentile` on float32 arrays.
- Fix: NaN/inf pixels in loaded images no longer propagate into normalization bounds.
- Fix: color cast on some files (cr3, raf) files.
- Fix: unsupported RAW files now show a clear error in the status bar instead of crashing.
- Updated rawpy/libraw: fixes loading of Panasonic Lumix .RW2 files from newer camera models.
- Updated many other dependencies.

**Dropped intel mac support due to lack of support for newer rawpy/libraw!**

## 0.15.0

- Add customizable slider shortcuts for all controls @alessandrv
- Fix: preview display issues (Windows) @alessandrv
- Fix: JPEG and TIFF scans now correctly linearized from sRGB before processing — density curves and color response now physically accurate for scanned negatives.
- Fix: GPU pipeline failures now log a full stack trace instead of a single-line message, making hardware acceleration issues diagnosable.
- Fix: GPU readback buffer correctly unmapped on error, preventing render failures after a hardware exception.
- UX: Make UI more subdued, replace intensive reds with subtle greys.

## 0.14.2

- Fix regression: loading monochrome DNG files no longer crashes the app (segfault in display conversion and pixel readout).

## 0.14.1

- Added **`override.toml`**: a config file in `Documents/NegPy/` (created on first run) for forcing GPU backend, Qt rendering, texture size cap, and more — without touching the app. See [Troubleshooting](../README.md#-troubleshooting) for details.
- Fix: **D-Range Clip** slider now uses a logarithmic scale for much finer control at low values.
- Fix: **Autocrop Offset** maximum raised to 100, allowing larger offsets on wide-border scans.
- Fix: 16-bit colour TIFF exports now use a proper 16-bit ICC colour transform.
- Fix: copy-pasted edits not persisting without additional manual change or explicit "Save".
- UX: Added empty-state overlay on the canvas when no image is loaded.
- Performance & stability improvements.

## 0.14.0

- Added **Split Toning**: independent shadow and highlight color tinting with hue (0–360°) and strength controls. Works on both color and B&W scans. Applied in Lab space — luminance is preserved exactly.
- **Per-section reset buttons**: each collapsible sidebar section header now has a reset button to restore that section's defaults without touching other settings.
- **Canvas background color**: three swatch buttons in the toolbar (Black / Dark Grey / Mid Grey) to set the viewport background.
- **Keyboard shortcut overlay**: press `?` to open a modal reference of all shortcuts.
- **Persistent UI preferences**: HQ preview toggle, canvas background color, auto dust removal default, and ICC profile settings (path, direction, apply-to-export) now survive app restarts.
- **Slider improvements**: sliders now show a tick mark at the default position; numeric units (px, °) displayed on relevant sliders; drag the slider label horizontally to scrub its value (hold Shift for fine control, Ctrl for coarse).
- **Cursor pixel readout**: hovering over the canvas shows the pixel's RGB and Lab values in the status bar.
- **Right click menu**: added as quicker way to access some tools/functions when right-clickign on image preview.
- Improved normalization: simplified per-channel floor analysis for more consistent results across different film stocks.
- Increased CMY white balance range for stronger correction capability.
- Toning: Selenium & Sepia sliders now hidden in color mode; Paper Profile section moved below toning controls with a clear label.
- Lab: color-only sliders (Separation, Saturation, Vibrance, Denoise) now hidden in B&W mode instead of greyed out.
- Exposure: Global / Shadows / Highlights region selector replaced with three toggle buttons; CMY label colors shift subtly per region for visual context.
- Collapsible sidebar sections now show a red dot indicator when they contain non-default values.
- Added info tooltips to many sliders.
- Many small UI refinements.
- GPU acceleration failure now shown as a status message instead of silently falling back to CPU.
- Stability: fixed GPU staging buffer leak on device loss/OOM; fixed dangling thread on app quit; thumbnail failures now surface as status bar errors.

## 0.13.2

- Fix regression: Changing camera wb, crop ratio, crop offset, manual crop, or resetting crop now forces normalization bounds re-analysis.
- Fix: Process mode change now correctly clears both floor and ceiling bounds (previously only ceiling was cleared).
- Fix: Corrupted RAW files (partial data errors) no longer crash the loader — app falls back gracefully.
- Updated dependencies: PyQt6 6.11.0, wgpu 0.31.0. (might help crashes with newer nvidia drivers)

## 0.13.1

- Fix: Pick WB not updating magenta & yellow sliders

## 0.13.0

- Added *D-Range Clip* slider for controlling the bounds of normalization process.
- Added *Sort Order* settings for contact sheet UI section (sort by name or date, ascending/descending).
- Added *HQ* toggle for viewport preview (renders preview in full resolution).
- Improvements to histogram & photometric curve rendering.
- Overall performance & stability improvements.
- Make "Pick WB" tool more predictable by averaging 8x8 area instead of sampling simple pixel.
- Fix: Linux appimage crashing on some distros.
- Fix: Border added via export menu being picked up by histogram calculation.
- Fix: mouse wheel accidentally changing sliders and dropdowns when scrolling the sidebar.
- Fix: Crash when using the Clear button to remove all files from the session.

## 0.12.0

- Added functionality to reset slider on double click (slider iteself, not only label).
- Added ability to "unzoom" for easier cropping.

## 0.11.0

- Improved normalization/autoexposure.
  - More dynamic range.
  - More neutral defaults.
  - Improved batch analysis (more aggressive outlier detection).
- Streamlined controls
  - Combined shadows+toe & highlights+shoulder sliders
- Added glow & halation effects sliders to Lab section.


## 0.10.1

- Optimized database writes to prevent stuttering during active slider movement.
- Fixed manual crop tool not being restricted to image border.
- Apply/sync export settings to all loaded files by default when using "Export All" button.

## 0.10.0

- Added **Zoom & Pan** for preview:
    - Useful for cleaning dust :)
    - Mouse wheel to zoom (up to 400%).
    - left-click (or middle click when using tools like spot healing brush) drag to pan.
    - Discrete zoom slider in the toolbar.
- Added **Persistent Undo/Redo**:
    - Standard shortcuts (Ctrl+Z / Ctrl+Y).
    - Stores up to 100 steps per file in local SQLite database.
    - History survives app restarts and file switching.
    - Track number of edits on image overlay (lower left corner)
      - Also track number of heal spots in retouch toolbar section.
- Packaged some additional requirements in Linux appimage for easier (I hope) running on debian-derived distros.
- **Fixed(?) UI rendering issues on Windows**

NOTE: due to some backend changes in storing the edits you might get weird colors on your previously edited photos. Reset should get rid of that. Nuclear option is deleting `edits.db` and `settings.db` from NegPy folder in your Documents.


## 0.9.16

- Stability improvements when using Numba-compiled functions.
- Parallelized batch normalization.

## 0.9.15

- Stability improvements when loading files and generating thumbnails (specfically for tiff & pakon files).
- Fix: white & black point offset sliders working in wrong direction when in E6 mode.
- More UI refinements.


## 0.9.14

- Fix: export folder not being correctly set on some configs on first run.
- Fix: Camera WB setting not forcing bounds re-analysis which lead on color cast stacking instead of color cast removal.
- Added [USER_GUIDE.md](USER_GUIDE.md)


## 0.9.13

- Added **Shadow Color Cast Removal**: aggressively target and neutralize color casts in the deepest shadows.
- Added **Regional Color Timing**: independent CMY adjustment for Global, Shadows, and Highlights tonal regions.
- Added **Vibrance Slider**: selectively enhance muted colors while protecting already vibrant tones.
- Added **Chroma Denoise Slider**: reduce digital color noise in LAB space while preserving natural film grain in the L-channel.
- Added **White & Black Point Offsets**: manual sliders to adjust normalization boundaries for precise highlight recovery or shadow recovery on top of auto exposure.
- Added classic Shadows & Highlights slider using dynamic Gaussian-weighted offsets.
- Reordered LAB processing pipeline for maximum signal integrity.
- Many **UI refinements**
- Added popup to ensure that export folder is properly set.

## 0.9.12

- Added macOS Intel build

## 0.9.11

- Fix color casts on exported files when heavy white balance correction is applied 

## 0.9.10

- Initial release of "E-6" mode for positives/slides
    - Optional "Normalize" step that tries to save expired slides
- Fix regression from 0.9.9 that caused colorcasts on some files.

## 0.9.9

- Added button to sync edits to selected files
    - Multiselect files in film strip using ctrl/cmd + click or drag with shift + click
- Fix "tiling" on exports that sometimes appeared on high-res exports when using CLAHE
- Fix image alignment issues when using fine rotation + manual crop

## 0.9.8

- Fix white image/thumbnail on file change when not using batch analysis.

## 0.9.7

- Improve batch normalization by discarding outliers before calculating of averages.
- Bugfixes:
    - Another small fix to Pick WB after recent changes

## 0.9.6

- UI Improvements:
    - Moved process/analysis to separate section.
    - Added options to "save rolls" 
    - Added simple switch between roll average and individual analysis.
- Bugfixes:
    - Fixed regression in Pick WB tool behaviour introduced in 0.9.5

## 0.9.5

- Features:
    - Added "Batch Normalization" button that performs bounds analysis for all loaded files and applies averaged settings to all. 
    - Added button to sync apply export settings for all files.
    - Added support for JPEG scans/files.
- Bugfixes:
    - Improved folder loading & thumbnail generation error handling & stability.
    - Fix fine rotation when manual crop is applied (credit: https://github.com/rodg)
    - Fix occasional wrong autoexposure calculation when file is rotated 90 degrees

## 0.9.4

- Brand new, native desktop UI (pyqt6) instead of electron packaged streamlit app
    - better performance.
    - more responsive.
    - more stable.
    - instant preview when moving sliders.
    - double click on slider label to reset to defaults.
    - native manual crop tool.
    - native file picker.
    - thumbnail re-rendering on inversion.
- Implemented `Analysis Buffer` to ensure that analysis is not thrown off by film border or lightsource outside of it.
- Added `Camera WB` button to use vendor-specific white balance corrections (helps green/nuclear color casts on some files)
- GPU acceleration (Vulkan/Metal)
- [keyboard](KEYBOARD.md) shortcuts
- Bugfixes: improved handling of some raw files that previously resulted in heavy colorcasts and compresssion artifacts.

## 0.9.3

- Added white balance color picker for fine-tuning white balance (click neutral grey)
- Added manual crop options (click top left and bottom right corners to set it)
- Added basic saturation slider
- Added more border options
- Added original resolution export option
- Added Input/Output .icc profile support
- Added input icc profile for narrowband RGB (should mitigate common oversaturation issues)
- Added horizontal & vertical flip options
- UI redesign: main actions moved under the preview, film strip moved to the right.
- Add new version check on startup (Displays tooltip near the logo if new version is available)

## 0.9.2

- Make export consistent with preview (same demosaic + log bounds analysis)

## 0.9.1

- Explicit support for more raw extensions for file picker.
