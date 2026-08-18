# NegPy User Guide

NegPy turns film scans into finished positives with a non-destructive, darkroom-style pipeline. It never writes back to your source files. Every edit lives in a local database, so you can experiment freely.

This guide is for new users. It explains what each control does and when to reach for it. For *why* the pipeline is ordered the way it is, read [PIPELINE.md](PIPELINE.md).

---

## 1. The Big Picture

### Screen layout

*   **Left, the film strip**: your loaded frames as a contact sheet, plus import, sorting and triage tools.
*   **Centre, the canvas**: the live preview. Most tools (crop, white-balance picker, heal brush, dodge/burn masks) are used by clicking directly on it. Scroll or pinch to zoom, drag to pan. A floating toolbar along the bottom holds Fit/1:1 zoom, undo/redo, rotate/flip and more, moving overflow items into an **⋯** menu as the window narrows. That menu also holds **Immersive Canvas** (the image fills the canvas and the toolbar overlaps it; turn it off to reserve space) **Show Slider Values** (every slider's value box stays open instead of appearing under the pointer) and **Reverse Scroll Zoom** (scroll up zooms out instead of in). Right-click the image for **Reset View** and **Sticky Zoom** (keeps the current zoom when you switch frames), plus the picker tools and copy/paste settings. With nothing loaded the canvas shows **Load some scans to get started**; click it for **Add files** or **Add folder**.
*   **Left, the film strip**: your loaded frames as a contact sheet, plus import, sorting, and triage tools.
*   **Centre, the canvas**: the live preview of the current frame. Most tools (crop, white-balance picker, heal brush, dodge/burn masks) are used by clicking directly on it. Scroll/pinch to zoom and drag to pan; a floating toolbar along the bottom holds Fit/1:1 zoom plus undo/redo, rotate/flip and more, moving overflow items into an **⋯** menu when the window narrows. That menu also has **Immersive Canvas** (image fills the canvas and the toolbar overlaps it; turn off to reserve space so it never occludes the image) **Show Slider Values** (keeps every slider's value box open instead of revealing it under the pointer; turn it on if you work by the numbers) and **Reverse Scroll Zoom** (scroll up zooms out instead of in). Right-click the image for **Reset View** and **Sticky Zoom** (keeps the current zoom level when you switch to another frame, instead of resetting to fit), alongside the picker tools, copy/paste settings, and **Unload** (removes the frame from the session; its saved edit is kept). With nothing loaded it shows **Load some scans to get started**; click it for **Add files** / **Add folder**.
*   **Right, the controls**: a pinned **Analysis** readout at the top, and below it an icon tab bar. Each icon opens a *workflow page* holding one or more collapsible panels.

### Before / After

The canvas toolbar's **◑** button (or `\`) splits the canvas in two. Left of the divider is the auto baseline: the same frame with the same film process, crop and rotation, but with every creative control (exposure, tone, Lab, dodge/burn, toning, retouch and finishing) back at its default. Right of it is your edit. Drag the divider to move the split, or grab its knob in the middle.

The split stays up while you work, so a slider moves the after side against a fixed reference. Press `\` again, or move to another frame, to close it. Peek Negative, Peek Flat Scan and the test strip take the canvas over, so they close it too.

### Peek Negative

The canvas toolbar's film button (or `N`) shows the scan as it was loaded: the negative, un-inverted, with no metering, no film-base normalization and none of your edits. Use it to judge the scan rather than the print — whether the frame is thin or dense, what colour the mask really is, whether the scanner clipped. It changes nothing and closes as soon as you touch a control. Your crop, rotation and flip still apply, so the frame stays where you put it. Because it is the file's own numbers, it gets no colour management or soft proof, so expect the raw orange cast rather than an accurate one.

### The workflow (and the order things happen)

The right-hand tabs follow the order you work in, which mirrors the processing pipeline:

| Tab | Icon | Panels | What it is for |
|-----|------|--------|---------------|
| **Setup** | cogs | Presets · Calibration · Process · Roll Analysis | Film type, capture-side color corrections, negative→positive normalization, roll-wide baselines |
| **Geometry** | crop | Geometry · Flat Field | Crop, straighten, lens and falloff correction |
| **Exposure** | sun | Filtration · Tone · Dodge & Burn | White balance, print density/contrast/curve/saturation, local burns |
| **Color** | palette | Lab · Alternative Processes · Toning | Chroma, sharpening, effects, lith and cyanotype printing, split and chemical toning |
| **Finish** | brush | Retouch · Finishing | Dust removal, vignette, border, carrier |
| **Favourites** | star | Your chosen sliders | Quick access to the controls you use most |
| **History** | clock | Work prints · Edit history | Keep named versions, step back through every change |
| **Export** | file | Export settings | Format, size, color, batch output |
| **Metadata** | tags | Archival metadata | Original camera, lens and film details |
| **Scan** | camera | Scanner · Camera Scanning | Capture film directly (Linux/macOS) |

You do not have to touch every panel. The defaults are tuned to produce a good print straight away, and most frames need only a crop, perhaps a white-balance nudge, and export.

A small **dot** on a panel header, and on a tab icon, means you changed something from its default. Every panel header has a **reset** action and an **ⓘ** that opens this guide at that panel's section.

Both side panels can be narrowed to give the canvas more room. As the controls panel shrinks, tab icons that no longer fit move into a **»** menu at the right of the tab bar. The tab you are on always stays visible.

---

## 2. Film strip (left panel)

The header shows the NegPy logo and version. The **↻** button beside the version number asks GitHub for a newer release on demand; it becomes a green **⬇** when one is out. When a newer release is out, a green **⬇ Update Available** line also appears under the version; click either to read what changed and let NegPy install it ([§15](#15-updating-negpy)). The chevron at the header's top-right folds the branding away to give the frames more room.

Below the header: the toolbar, the search box, and two collapsible sections. **Library** holds the folders your scans live in; **Film Strip** holds the frames you have open. Click either heading to fold it away; the one still open takes the whole panel. NegPy remembers which were open.

### Your library

The **Library** section is a folder tree of the places your scans live. Press **+** to add a folder, and point it at the one big `Scans` directory you keep everything under, subfolders and all. **↻** re-reads it from disk. Each row shows what is inside it ("36 photos", "2 folders"), and subfolders are read when you expand them.

**Browsing costs nothing.** NegPy opens, decodes and hashes nothing when you add a folder or click through the tree. It only lists what is there.

#### The Library button

The **Library** button (book icon, first in the toolbar, or **Ctrl+L**) opens the folder your scans live in. The first time you press it, NegPy asks you to pick that folder and remembers it. The panel also goes there on its own: on launch when you do not restore a session, and whenever you unload the last frame. Your rolls are a more useful resting state than an empty sheet.

To point it somewhere else, add another folder with **+**. To forget them all, use **Clear Library** in *Manage Database*. That clears the list of folders only, leaving your images, folders and edits untouched.

#### Walking around

*   **Click** a folder to select it, **double-click** (or **Enter**) to open it.
*   **Ctrl+click** several folders and open them together to load more than one roll at once. NegPy asks once, for the total.
*   **Alt+Up** moves the selection to the folder above.
*   The tree sorts the way the sheet does. Change **Sort** to Date or Descending and the folders follow.

When you open a folder that contains images, NegPy asks whether to **load the roll**. Only then does it hash and thumbnail them, which is the part that takes a moment on a big roll. Say no and your open frames stay as they were. Tick **Always load without asking** in that prompt if you would rather it just get on with it.

Loading a roll replaces what is in the film strip; right-click → **Add to session** appends instead. Nothing is lost either way, because your edits live in NegPy's database, not in the list of open files.

#### Folders are your folders

NegPy reads the tree straight from disk and never creates, renames, moves or deletes anything in it. Reorganize in Finder or Explorer and the tree shows the new arrangement at the next refresh. Every edit is stored against the image's content, so moving a file between folders keeps its edit, its history and its keep/reject mark.

### Importing and managing files

**A note on Nikon High Efficiency raw.** The Z 8 and Z 9 can record NEFs in **High Efficiency (HE)** or **HE\***, which use a licensed codec NegPy cannot decode. Such a file is still called `.NEF` and still carries the same TIFF compression tag as an ordinary lossless NEF, so nothing looks unusual until it fails to open. NegPy names the reason rather than reporting a generic unsupported-file error. Re-shoot in **Lossless Compressed** NEF, or convert with Adobe DNG Converter. Lossless NEFs from the same cameras open normally.

Toolbar buttons, left to right:

*   **Add files** / **Add folder**: load individual images or every image in a folder. Pick a folder that holds only *other* folders and NegPy reveals it in the Library section instead of reporting that it found nothing. Dropping a folder on the window does the same.
*   **Clear all**: unload everything, or just the selected frames.
*   **Hot Folder**: watches the current folder and auto-loads new files as they appear, which is handy when a scanner or tethering app drops files into a directory. While it is on, the "Working…" import popup stays hidden so each new frame does not raise a window; the status line over the canvas still reports the import.
*   **RGB Scan**: treats the folder as red/green/blue exposure triplets and assembles each frame from three shots, for narrowband trichrome scanning. Right-click a frame → **Edit RGB Triplet…** to assign the three files by hand. An assembled frame carries the three-dot badge described under [Triage](#triage-culling-the-roll).
*   **Half Frame**: splits each scan into two frames, for half-frame cameras. Each half is edited and metered separately and badged with which half it is. Enabling it opens a rectangle editor on the current scan: drag the green box to crop (everything outside is discarded), drag the orange line to set the split, and use **Cut thickness** to discard a band centred on the split, which is the physical black separator between the two exposures. The setting is saved and applied to every half-frame split from then on, whatever the scans were acquired with (SANE scanner, camera copy-stand, or folder import). **Adjust Half Frame**, beside Half Frame, re-opens the editor. Auto-detection of the gutter still seeds the initial split position.

    Turning Half Frame off does not lose the work: each half you edited keeps its own edit. A half you only looked at is not work, so turning Half Frame on and back off again leaves the scan as it was. A scan you did split comes back as one frame carrying both halves, a **diptych**: half 1 rendered with its own edit, half 2 with its own, joined side by side at the original spacing, with the cut band as a black gap. A scan whose halves hold edits from another folder or an earlier session is not a diptych; it stays one plain frame until you split it again. A diptych carries the both-sides-filled split badge and exports as one file named `<name>-DIPTYCH`. Its controls panel is disabled, because the edits belong to the halves: turn Half Frame back on to change either one. A scan where only one half was worked on uses that half's edit for both sides. The filmstrip thumbnail and contact-sheet tile still show the plain whole scan, so a diptych's thumbnail does not match what it exports. An export size set as a long edge applies to each half, so a diptych comes out about twice that wide. To get a plain frame back, right-click the diptych and select **Unsplit diptych**: it renders and exports whole again, and both halves' edits are deleted, so a later split starts from defaults.

    Half Frame does not apply to a frame assembled from more than one file: an RGB Scan triplet, a stitch, or an HDR merge. Those are never split, and they never come back as a diptych, even when the file they are built around was worked on as two halves earlier.
*   **Apply (clone)**: copy the current frame's settings to selected frames or the whole roll. You choose which aspects in a dialog; crop and rotation are always per-image.
*   **Sheet filter** (funnel): show *All frames*, *Keepers only*, or *Hide rejected*.
*   **Sort**: by Name or Date, ascending or descending.

Above both sections sit a **filter box**, a **`.*`** regex toggle and a **search-library** button. Inside the Film Strip section is a **tally**, for example "36 frames · 12 keepers · 3 rejected".

#### Filtering the sheet

Type a plain word and it matches the filename. Beyond that the box takes `field:value` terms, which is how you find a frame by what it *is* rather than what it was called:

| Term | Finds |
|---|---|
| `film:portra` | frames whose film stock contains "portra" |
| `camera:"Nikon F3"` | quote anything with a space |
| `iso:>=400` | numeric fields also take `>`, `>=`, `<`, `<=` (`iso`, `frame`, `push`) |
| `date:2024-03` · `date:>=2024` | by file date; a partial date is a prefix |
| `roll:` `developer:` `lens:` `format:` `scanning:` | the rest of the Metadata panel |
| `name:` `path:` `ext:tif` | file identity |
| `keeper:` `rejected:` `edited:` | frames carrying that mark, or with a saved edit |
| `-rejected:` `-film:velvia` | a leading `-` negates any term |

Terms combine with AND, so `film:portra iso:>=400 -rejected:` applies all three at once. Metadata comes from each frame's own **Metadata** panel, so it is searchable once you have filled it in; a frame you have never edited is findable by name, extension, date and mark. The **`.*`** toggle switches the box back to a plain regex over filenames, ignoring the field syntax.

#### Searching the whole library

The filter box narrows what is already open. The **magnifier-over-folder** button beside it (or **Enter** in the box) runs the same search across every library folder and loads what it finds, so `film:portra` finds your Portra frames in folders you have not opened this month. The status bar counts files as it goes.

This works without opening anything, because NegPy already knows which edit belongs to which file. Film stock, camera and the rest come from frames you have filled in. The folders are only read, never indexed in the background and never modified.

Right-click **empty space** in the film strip for **Add files**, **Add folder** and **Clear all**, so those tools stay in reach part-way down a long roll. Here **Clear all** always means the whole session, never just the selection.

#### Stitching a frame from several shots

If one negative was captured in overlapping pieces, from a copy stand at higher magnification than the frame, select the pieces and right-click → **Stitch selected frames**. NegPy finds the overlap, matches brightness across the seam and replaces the parts with a single wide composite named *a+b (Stitch)*, badged on the sheet. The parts keep their own edits on file, so right-click → **Unstitch** puts them back untouched. The registration is saved with the session and replayed on the next launch, so re-opening a composite costs nothing.

This works on RGB-scan frames too. Turn on **RGB Scan** first so each piece is already assembled from its own R/G/B triplet, then stitch the assembled frames. Each part keeps its own three exposures, and nothing is shared between parts.

#### Merging bracketed exposures (HDR)

A slide's density runs deeper than one camera exposure can record. Expose for the highlights and the darkest parts sit in sensor noise; expose for those and the bright parts blow. Bracket the capture instead, several shots of the same slide a stop apart, then select them and right-click → **Merge exposures (HDR)**. They collapse into one frame named *a +4 (HDR)* that carries the whole range, badged on the sheet so a merge is never mistaken for a single capture. Right-click → **Unmerge exposures** puts the originals back, with their own edits untouched.

Nothing needs to be set up beforehand. NegPy measures the exposures from the images themselves rather than trusting shutter tags, since several supported scanner formats have none, works out how many stops apart they are, and registers them to each other in case the camera shifted. The result is saved with the session, so re-opening a merged frame costs nothing.

**How it lands.** Two separate choices, worth keeping apart. The merge is *computed* in the units of the longest exposure that does not clip: the best reference radiometrically, since every other frame converts into it and nothing can exceed white. Which exposure the picture then *opens at* is a different question, and not one the software can answer. A slide's brightest point is denser than clear film, so the longest unclipped capture is brighter than the shot you metered, and rendering there pushes the highlights into the top of the transfer curve where it has least gradient left. That looks like lost highlight detail, because it is.

**So nominate the frame yourself.** Right-click a merged frame → **Render exposure** and pick the shot that looks the way you intended. The merge opens exactly there, and the other frames contribute only range and cleanliness. The list shows each frame in stops from the reference, with the reference itself marked *(as captured)*.

Only the reference and any **shorter** exposures are listed. The merge can never open brighter than the reference, the frame defining white, so a longer frame would render identically and is not offered. If a bracket has nothing below its reference, the menu does not appear.

**Or set it as a value.** With a merged frame selected, the Process panel gains a **Render Exposure** slider: 0 EV is the reference, the brightest the merge can open at, running down to −4 EV. The menu below snaps to exposures you actually shot; the slider goes anywhere between them, which is usually where the one you want sits. Setting a value clears any frame you had nominated, and picking a frame clears the value, so only one is ever in effect.

Left on **Bracket middle (auto)** it falls back to the middle exposure of the bracket. That is a reasonable guess only when you bracketed evenly either side of the metered shot. Bracket *upward* and every frame sits at or above the reference, so the middle lands on the reference and the setting does nothing.

**Include the shot that already looks right**, and at least one darker than it. It is tempting to bracket only upward, since frames *longer* than the reference are what buy the shadows. Do not: the menu can only offer frames you actually shot, so the darkest render you can ask for is the darkest exposure in the bracket. The reference typically lands **one to three stops above the metered frame** and is never the metered frame itself, so *(as captured)* is not the exposure you took, and the one you want is usually a stop or two below it.

What the merge buys is signal-to-noise in the deepest shadows, worth roughly two stops, with the midtones unchanged. The gain is entirely where a transparency is hardest to scan.

**A merge opens with its shadows already lifted**, by an amount derived from the range the bracket recovered, so **Shadows Density** sits off zero. That is deliberate. If your metered frame did not clip, the merge did not add *range*, it added *precision*: the tones were all recorded, just with very few levels on top of noise. Precision is invisible at the same tone, so a merge left neutral renders indistinguishable from the frame it was metered on, which is not what you merged for. The lift is measured, not a look: it goes only as far as the recovered precision affords, so the opened shadows are still quieter than the single frame's were. Drag **Shadows Density** to zero for the render faithful to the metered frame, and that choice is saved like any other edit. **Reset Settings** brings the seeded starting point back, along with the merge itself and the inherited film process.

Bracket **both ways**, for two different reasons.

**Upward, for range.** Longer exposures reach the deep shadows, and they set the reference. Metered, +1, +2, and +3 if the shadows are deep.

**Downward, for choice.** Shorter exposures add almost nothing to the *pixels*, because the reference already holds the highlights by construction. Their job is to appear in the **Render exposure** menu, and a merge is only as adjustable as the frames you gave it. Stop at the metered shot and the menu offers two entries, with no room left to go darker; go two stops below it and there are four or five, reaching −3 and −4 EV.

Metered −2 through +3 costs six frames and leaves the decision to you at the end. If you must economise, economise upward: an extra long frame buys shadow noise you may not notice, an extra short frame buys a choice you cannot make later.

Shorter frames do carry one thing outright: a **blown specular**, the sliver of water highlight or sun disc above the reference's white.

The merged frame inherits the **film process** of the exposures it came from, so a bracket of slides opens in Transparency rather than reverting to whatever mode you last set by hand. Stitched composites do the same.

It is **named after the first frame in filename order**, with an `-HDR` suffix, so a bracket of `_DSC1715`…`_DSC1719` exports as `_DSC1715-HDR.jpg`. Not the reference frame, whose identity depends on picture content; and the suffix means a merge never writes over the export of the single frame it is named after.

**Merging is for transparencies**, so the action appears on Transparency frames only. A color negative holds about 5-6 stops between its base and its densest highlight, and an ordinary black-and-white negative nearer 4, both comfortably inside one capture. A transparency runs to 10-12, which is what the merge exists for. On black-and-white the entry is shown but disabled, because reversal-processed monochrome (Scala, dr5, Fomapan R) really is a transparency and does have the range. It is simply not wired up yet.

Merging is also refused on frames that are already merged, stitched, or RGB-scan triplets. Each is its own way of building one frame from several files, and combining them is not supported.

Narrow the panel and the toolbar buttons that no longer fit move into a **»** menu at its right edge, so you can squeeze the panel down without losing any tool.

### Triage (culling the roll)

Thumbnails are positives from the start. A frame you have not opened yet is inverted straight from its preview, a quick per-channel job rather than the full pipeline, so the sheet reads as photographs while you cull. Open a frame and its thumbnail is replaced by the real render, matching the canvas exactly. Transparencies are left alone, being positives already: a frame whose film process you have already set, or that you have opened once, is taken at its word, and only a frame nothing has decided yet is guessed at from its preview.

Right-click a thumbnail, or use keyboard shortcuts, to mark frames while you review the sheet:

*   **Keep**: a small check badge marks a keeper.
*   **Reject**: a cross badge dims the frame. Rejected frames stay on the sheet but are skipped by batch exports and sidecar writes. **The file on disk is never touched.**

Marks apply to a multi-selection and persist across sessions. A badge in the top-right corner instead flags a frame that failed to decode.

#### Reading the badges

Each corner of a thumbnail means one thing, so the marks never compete:

| Corner | Badge | Means |
|---|---|---|
| Bottom-right | check | keeper |
| Bottom-right | cross, frame heavily dimmed | rejected |
| Top-right | exclamation | the file failed to decode; click to retry |
| Bottom-left | *see below* | the frame was built from more than one file |

The bottom-left badge is grey, not red, because it reports what the frame *is* rather than something you marked. Its glyph says which kind:

| Glyph | Frame |
|---|---|
| Two overlapping panes | a stitched composite ([§Stitching](#stitching-a-frame-from-several-shots)) |
| Three stacked bars | a merged bracket ([§Merging](#merging-bracketed-exposures-hdr)) |
| Three red/green/blue dots | an RGB-scan triplet |
| A split rectangle, one side filled | one half of a half-frame scan; the filled side is which half |
| A split rectangle, both sides filled | a diptych: the whole scan, each half with its own edit |

Hover any thumbnail and the tooltip says the same thing in words, with the frame count: *HDR merge of 5 exposures*, *Stitched composite of 3 frames*.

The right-click menu also offers **Copy/Paste Settings** (with or without normalization bounds), **Reset Settings**, **Apply settings…**, and per-frame export.

---

<!-- panel:analysis -->
## 3. Analysis readout (always visible)

Pinned above the tabs, this is your feedback while printing. Drag the divider to resize it, or collapse it entirely. Everything in it describes the frame you are on and updates as you edit. The zone strip is the one part you can also act through. Top to bottom:

#### Photometric curve

The chart is the paper characteristic (H&D) curve NegPy is printing through right now. It models how a sheet of photographic paper responds, and it is not a curves editor. Left to right is **negative density**, the exposure the paper receives, so dense parts of the negative (the scene's highlights) sit to the right. Bottom to top is the **print tone** that comes out. A steeper curve means more contrast, which is what Grade moves. The flattening at each end is the toe (shadows) and shoulder (highlights), where the paper runs out of range.

With a **Contrast Mask** dialled in, a violet band opens between the curve and a dashed edge. The mask shifts each pixel by how far its own value sits from its blurred surroundings, so there is no single curve for it: a large flat area prints on the dashed edge, fine detail prints on the solid curve, and everything else falls between. The band is the mask's reach. Dodge/burn, local grade and CLAHE are spatial in the same way and are deliberately absent from the chart, which plots the global curve.

The crosshair marks the **pivot**, the density the curve rotates around when you change contrast, so the midtone stays put. While you drag a slider, a faint **ghost** of the previous curve stays behind for comparison. If cast removal pulls the channels apart you get three separate R/G/B traces instead of one grey curve, and that spread *is* the color correction.

#### The two histograms

Two different histograms share the chart. Behind the curve, rising from the bottom, is the **output histogram**: the tones of the print you are looking at, in R, G, B and luminance. Along the bottom axis is the **negative density histogram**, which is what the scan contains, before the curve.

Read them against each other. The density histogram tells you which part of the horizontal axis your negative occupies, and the curve tells you what happens to it. If the negative's data sits entirely on the flat toe, no amount of contrast pulls those shadows apart. Move the exposure so the data lands on the steep middle instead.

#### LIN / LOG toggle

Bottom-right of the chart. It switches the histogram's *height* axis (how many pixels), not the tone axis. **LIN** is literal, so a big flat sky dwarfs everything else. **LOG** compresses the tall peaks so the thin tails become visible, which is where the few hundred pixels of deep shadow or specular highlight live. Use LOG to hunt for clipping, LIN to judge where the bulk of the frame sits. NegPy remembers the choice between sessions.

#### Clipping triangles

Small R, G and B triangles in the top corners of the chart. **Top-left** is shadows crushed to pure black, **top-right** is highlights blown to pure white. They appear only once a channel passes 0.5% of the frame. A little is normal, since a real print has a black. Watch for a single channel clipping alone, which is a color cast pushing one dye off the end rather than an exposure problem.

#### Zone shading and zone ticks

The amber wash on the left and the blue wash on the right mark the curve's toe and shoulder, the compressed ends where tonal separation is being lost. The ticks along the bottom are Adams zones I to IX, so you can read straight off the axis which zone a given negative density prints as.

#### Step wedge

A 21-step Stouffer-style grey wedge printed through your current curve, in even density increments labelled in the scan's own density units. It is a ruler for the curve. Where neighbouring patches are clearly different, you have tonal separation. Where they merge into one flat black or white block, those tones are gone. The brackets mark the usable span. It hides while you peek the flat scan, since there is no print curve to wedge.

#### Zone strip

Ten cells on the Adams scale, where **0 is paper black and V is 18% mid-grey**. The last cell (IX) also absorbs paper white. The brightness of each cell is the zone's tone, and how solid it looks is how much of the frame lands there. This is the fastest read of whether a frame is low-key, high-key or sitting sensibly in the middle. The end cells tint **red** when shadows are blocked up or highlights are blown. Hover a cell for its exact percentage.

It is also where you place a tone. **Click a cell, then click that spot on the photo** and the print is solved so the spot lands on that zone. See Zone placement below. The armed cell is outlined until you spend it. Click it again, or press Esc, to cancel.

#### Probe

A spot densitometer. Hover the image to read the pixel: per-channel density above film base (ΔD, relative to this scan's normalization, not absolute), the displayed tone's reflection print density, and its print zone (0 = paper black, V = 18% mid-grey, X = paper white). In B&W Negative mode the ΔD channels read the pre-conversion color record.

#### Zone placement

The probe made actionable, and what a darkroom enlarging analyser does. **Click a zone on the strip above, then click that spot on the photo.** The print is solved so the spot prints on the zone you asked for, and you see it straight away. Ask for a second and a third zone the same way. A fourth spot replaces whichever pin is nearer. Each pin appears here as a row: what zone it reads now, and the **target** it was given, which the − and + buttons trim in thirds of a zone.

With one pin, Print Density is solved so that tone prints on its target. With two, typically a shadow and a highlight, Print Density *and* ISO-R Grade are solved so both land. A third pin adds one more control for the tone between them: **Shadows Grade**, **Highlights Grade** or **Snap**, whichever can actually move that tone, which depends on where it prints rather than on which zone you call it. A line under the button says which, so nothing moves silently; if the third tone already prints where you asked, no third control is touched.

What you see until you accept is a preview. **Place zones**, or **Enter** over the photo, commits it as one undoable edit, turns off Auto Density (and Auto Grade from two pins on) since a meter left on would re-move the placed tones, and closes the tool. One pin is enough to accept. **Esc** discards instead: first the armed zone, then the pins and the preview with them. The **✕** on a row removes just that pin; remove the last one and you are back to the committed print.

Pins are handles: drag one to move it, and the zone beside it re-reads as it travels (the cursor turns into a hand over a pin you can grab). A pin keeps the zone it was asked for while it travels, and its caption reads `1 · IV⅓ → VI`, where it is now and where you are asking it to print, the arrow disappearing once the two agree. Clicking the photo without picking a zone first simply pins a reading and leaves the print alone.

Asking for a zone the paper cannot reach shows an amber `→ lands …` with the closest zone the print can make, the solve pegging at the slider's end like an analyser pegging at grade 5. With three pins the amber also appears when the three asks cannot all be met at once: placing the middle tone nudges the outer two, so the solve alternates between them and reports where they settle.

Pins are proofs, not edits. They go on any other edit and when you change frames. The measured zone reads through the print curve, the same model the chart and step wedge use, so after placing, the pin reads its target by construction; later stages (Lab, toning) can still shade the final pixel the hover probe reads.

#### Negative stats

The four numeric rows at the bottom. Each has the same explanation on hover, and each measures the negative rather than your edit:

*   **Negative**: the negative itself, as a relative density range (luminance) plus its development character against a nominal frame: flat (≈N−1), normal, contrasty (≈N+1). It is a relative scale, comparable across a roll, and a heuristic from this scan's normalized bounds rather than a calibrated densitometer reading.
*   **Exposure**: where the frame's midtone sits, in stops from neutral; positive is brighter (high-key), negative darker (low-key). Approximate, read off the metered midtone.
*   **Clipping**: share of pixels crushed to black (shadows) or blown to white (highlights), worst channel. Turns red above 1%.
*   **Scan clip**: share of source-scan pixels at or above sensor white, per channel. In a negative scan the film base and scene shadows sit near sensor white, so clipping there destroys base and shadow separation, and no edit can undo it. Fix it at capture: expose the scan lower. Turns red above 1%.

---

## 4. Setup tab

**Film mode** sits above the panels, because it is the first choice of every edit: **Color** (C-41 color negative), **B&W** (panchromatic negative) or **Slide** (transparency/reversal, E-6 and friends). Each swaps the core conversion math and re-runs the pipeline from scratch. The wand button beside them **auto-detects** the mode when a file loads.

<!-- panel:sensor -->
### 4.1 Calibration: what your rig does to the colors

Everything here corrects the *capture*, not the look. Three different things sit between the scene and your file: the camera's color filters, the film's dyes, and the light source. Each gets its own control. They are not interchangeable, and none substitutes for another.

**Capture**, how the file is decoded before any of that:

*   **Scanning setup** (bulb button): a two-question wizard, *how do you scan?* then *what light source?*, that sets Linear RAW and Narrowband for you. It runs once after the first-launch tour, and the button reopens it whenever your rig changes.
*   **Linear RAW** (default off): decodes with neutral multipliers for completely raw data. When off, it decodes RAW with the camera's as-shot white balance. Toggling it reloads the file. Let the **Scanning setup** wizard pick it, or try both and keep whichever gives better results.
*   **Narrowband**: corrects the oversaturation typical of narrowband (RGB-LED trichrome) scans, using a bundled input profile. Leave it off for ordinary broadband scans. An explicit Input ICC in Export overrides it. It is **greyed out on Transparency**; see *Narrowband and slides* below.

What the wizard sets, by rig:

| Capture | Light source | Linear RAW | Narrowband |
| --- | --- | --- | --- |
| Digital camera | White light (lightbox, CRI LED panel) | off | off |
| Digital camera | Narrowband RGB (Scanlight, RGB LED) | on | on |
| Film scanner | White light (Plustek, Epson, most flatbeds) | on | off |
| Film scanner | Narrowband RGB (Nikon Coolscan, Kodak Pakon) | on | on |

Applying it sets the defaults for newly loaded files, updates the open frame, and rewrites every already-edited frame in the session, undoable per frame with Ctrl+Z.

**Trichrome Calibration** is for **single-shot narrowband** (RGB-LED trichrome) camera scans. The camera's color filters overlap the light's bands, so a pure red exposure leaks a little into green and blue. That leak is a fixed property of your sensor and light together and has nothing to do with the film, so it is corrected on the linear capture before inversion.

*   **Profile**: the sensor matrix to apply. Custom `.toml` matrices live in `<Documents>/NegPy/sensor/`.
*   **Calibrate** (vials icon): build a profile from three bare-light R/G/B exposures.

This block greys out unless **Linear RAW** is on, since profiles are calibrated against neutral white balance and the as-shot gains would misapply the matrix. It also greys out on **Transparency**, for the reason below. Your selection is remembered either way. It is also skipped for RGB-triplet assets, which never had the leak. It changes what the analysis reads, so **re-run Batch Analysis** after changing it.

#### Narrowband and slides

**Narrowband and Trichrome Calibration do not apply to Transparency**, with or without Normalize. Both stay visible and greyed so you can see what your rig is set to, both keep their values, and both come back the moment the frame is a negative again.

Narrowband is a way to scan negatives. Its payoffs, defeating the orange mask and separating the dyes cleanly before a high-gain inversion, are things a slide does not need, and the bundled profile describes narrowband capture of *negative* dyes, which a slide does not have. Trichrome Calibration goes with it: the matrix un-mixes a narrowband light against your sensor's filters, so it means something only for a capture made under narrowband light, and there is no way to build one for a broadband capture.

This matters because both are sticky and follow your rig from frame to frame, so a profile set up for your negatives would otherwise arrive on a slide you never touched. If you do scan slides on a narrowband rig, reach for **Hue Trim** instead: it corrects the hue rotation an unusual light imposes, which is the part that can be corrected.

**Crosstalk** (hidden in B&W Negative) is a channel unmix applied to the raw densities before inversion. The dropdown lists only matrices for the film you are processing, because a Color Negative matrix does not describe a Transparency's dye set, and a mismatched stored profile resolves to no correction rather than the wrong one.

The film's dyes each absorb outside their own band, but they are not the only cause: your light's spectrum and your sensor's color filters mix the channels too, and in the density domain all three arrive as the same kind of error. So treat the matrix as *your whole scanning setup*, not just the film. A profile that works beautifully on one rig may be wrong on another with the same stock.

*   **Matrix**: the profile to apply, grouped in the dropdown by where its numbers came from (measured, tuned on a rig, or from spec sheets). *Generic C41* is the built-in; drop custom `.toml` matrices in `<Documents>/NegPy/crosstalk/` (see [CROSSTALK.md](CROSSTALK.md)). The slider button opens a matrix editor, where a **Type** control records that provenance and a **Process** control says which film the numbers describe. Process decides where the profile appears and whether it applies, so a matrix you build for slides needs it set to E-6. Anything created with **+** is already set to the process you are working in.
    When the current film process has no matrices at all, the dropdown and **Strength** are disabled and a hint says so. The editor button stays live, because it is the way to build the first one.
*   **Strength** (0.0 to 1.0): how much of the unmix to apply, for richer and cleaner color separation. It changes what the analysis reads, so **re-run Batch Analysis** after changing it.

> **The bundled film matrices are derived from published spec sheets, not measured**, which is why they are all marked *(approx)*. They describe the film's **dyes alone**, so they are the whole story only where your capture reads each dye cleanly: a **true RGB scan** (a Coolscan-style mono sensor lit one band at a time) or a **calibrated trichrome rig** (see *Trichrome Calibration* above). With a broadband light and a Bayer sensor, the capture adds mixing of its own that a dyes-only matrix does not describe. It may still help, but treat the number as a starting point rather than a correction for your setup.

**Worth experimenting with.** If a stock or a light gives you trouble, open the matrix editor, nudge the six off-diagonal terms and save the result as your own profile. Name it after the *combination*, "Gold 200 + Spectracolor", not just the film. A profile you tuned on your own rig beats any datasheet, and profiles that work are worth [contributing back](CROSSTALK.md#contributing-a-matrix), since nobody can guess your light and sensor for you.

**Light source:**

*   **Hue Trim** (-30° to 30°, default 0): rotates every hue by a fixed angle, to undo the rotation an unusual scanning light imposes. Narrowband LED and odd-phosphor panels sample the dyes away from where the film expects, which turns *every* color by roughly the same angle, so yellows read orange and greens go olive, while neutrals are left alone. That is why white balance cannot fix it: the error is a rotation, not a cast, so there is no grey to correct. Judge it on a subject whose color you know (foliage, a clear blue sky, skin), and leave it at 0 for an ordinary broadband light. The setting is **sticky**, because a light source is a property of your rig, so it carries to the next file until you change it. Neutrals are untouched, so it never disturbs the color-balance clip in **Normalization**.

<!-- panel:process -->
### 4.2 Normalization: negative → positive

How the negative is measured and normalized into a positive. The film mode that decides *which* conversion runs sits above the panels (§4), and how the scan is decoded lives in **Calibration** (§4.1).

*   **Multi-core CPU Rendering** (canvas toolbar → **»** menu, beside **GPU Acceleration**): spreads the CPU rendering kernels across your cores. It takes effect immediately, with no recompile and no restart.

    Be realistic about the gain. The kernels run much faster, but a merge is dominated by decoding the RAW files, which this does not touch, so the whole operation comes down by only about a tenth. Ordinary editing changes less again, because the GPU already carries the pipeline. The gain is largest wherever the CPU does the work: merges, exports, and any machine without a usable GPU.

    On Windows and Linux this is **on**. On macOS it is **off**, pending more evidence: the underlying threading layer terminates the process outright if two threads enter it at once, and while NegPy serialises every such call behind a lock, that has been proven on one Mac rather than on the range of them. If you turn it on and the app ever closes without warning, NegPy notices on the next launch and offers to turn it back off; that is the failure to expect, and it is recoverable. Setting `cpu_parallel` under `[performance]` in `override.toml` still wins over the menu, for a machine that cannot start.

**Analysis window**, where NegPy measures the black and white points. The slider takes half the row, the three buttons the other half:

*   **Analysis Buffer** (0.0 to 0.25): insets the measurement window from the frame edge so film rebate, sprocket holes and scanner borders do not skew detection. Raise it on scans with wide borders.
*   **Analysis Region** (square-draw tool): draw a freehand region on the canvas to meter *exactly* that area, overriding the buffer. Double-click inside to confirm; the ✕ button clears it.
*   **Lock Bounds** (padlock): freezes the analyzed normalization bounds for this frame, so cropping or moving sliders no longer re-analyzes it. Lock it in once you are happy with the bounds.

**Normalization tuning:**

*   **Luma Range Clip** (-100 to 100): how aggressively the tonal range, the black/white-point span, is set. Neutral already applies a small robust clip. Positive tightens it, which is good for dense or fogged negatives where a few stray pixels would push the bounds to extremes. Negative pushes the bounds *outward*, for lifted blacks and unclipped highlights.
*   **Color Clip** (-100 to 100): the per-channel color-balance clip (orange-mask removal), independent of the tonal range. Positive tightens channel balance; negative samples nearer the extremes.
*   **Global / R / G / B** selector → **White Point** / **Black Point** (-0.25 to 0.25): manual offsets on top of the auto-detected bounds. A positive white point brightens; a positive black point lifts blacks. In R/G/B mode these become per-layer trims: per-dye-layer film-base (Dmin) and Dmax corrections, which is scanner-style per-channel levels. The selector is hidden in B&W Negative, where per-layer trims are meaningless, and in Transparency with Normalize off, where the sliders it scopes are hidden with the rest of the normalization tuning.

**Crosstalk**, **Hue Trim** and the sensor unmix all live in **Calibration** (§4.1). They correct the capture rather than the negative-to-positive conversion.

> **No Transparency matrix ships with NegPy.** On slides the Matrix dropdown starts empty, and it and Strength are disabled until a matrix exists. The editor button stays live, so you can build your own: press **+**, and it is created for the process you are in. A `.toml` marked `process = "Transparency"` dropped into your crosstalk folder works too (the pre-rename `process = "E-6"` still loads). It means something different there: on a negative the dyes' unwanted absorptions are an error to remove before inversion, so unmixing moves the render *toward* the scene, but a transparency **is** the finished image, and what you see on a lightbox already includes those absorptions, so unmixing moves it *away* from the slide's own look. In Transparency, treat it as a color-separation control, not a fidelity correction. **Hue Trim** is unaffected: it corrects the light source, so it applies to slides exactly as it does to negatives.

**Normalize** (Transparency only) is the switch between two ways of rendering a slide.

*   **On**: auto-stretches the histogram to fill the dynamic range, metered per frame, and prints it through the paper model like a negative. This is a **rescue tool for faded or expired slides**, which is what it was added for. Where the dyes have lost their range, metering it back per frame is exactly right, and because the stretch is metered, two exposures of the same slide converge on a similar render.

    On a slide that was exposed as intended, expect it to look washed out and desaturated. That is not a miscalculation: a slide's density runs all the way to Dmax, but only its top ~1.5 decades carry picture, so a stretch measured across the whole range squeezes the picture into the top of the print curve, where the shoulder compresses tone and color together. Reach for it when the slide needs rescuing, not as a starting point.
*   **Off** (default): renders the slide **as captured**. The camera's own color matrix is applied, and the tonal window is fixed to the decoder's white level rather than measured from the frame, so a slide opens looking the way it does in Photoshop, Preview, Affinity or Darktable, and a bracketed set stays a bracket, with each exposure rendering at its own brightness. Use this mode when you exposed the slide the way you wanted it and only need to adjust from there.

    With Normalize off, the paper simulation has nothing to act on, so the print-specific controls are hidden (paper profile, Paper White/Black, Auto Density, Auto Grade, split grade, Dye Separation) along with the normalization tuning above, which only shapes a *measured* stretch. What stays is a plain transfer curve, neutral at its defaults: **Print Density** (exposure), **ISO-R Grade** (contrast), **Toe** / **Shoulder** and their **Width** sliders (shadow and highlight roll-off), **Shadows Density** / **Highlights Density** (§6.2), the per-layer R/G/B trims, and white balance. Lab, Toning and Finish work as usual.

    **On a merged bracket, Normalize is greyed out**, whatever the switch said before the merge. The merge already decides where the tones land, and **Render exposure** picks which exposure it prints at; a metered stretch divides that choice straight back out, since below the anchor at which the frame stops clipping, moving it changes nothing. The two are not wanted together in any case, because Normalize rescues faded film and fading *compresses* the density range a bracket exists to capture. Unmerge the frame if you need the stretch.

    Coming from Lightroom's basic panel, the map is: **Exposure** → Print Density (inverted, lower is brighter), **Contrast** → ISO-R Grade (inverted, 180 is softest), **Shadows** → Shadows Density, **Highlights** → Highlights Density, with Toe and Shoulder shaping how each end rolls off. The Density sliders take the darkroom sign: *positive adds density*, so negative Shadows Density is what opens the shadows. **Whites** and **Blacks** have no equivalent here, because the tonal window is fixed by design, which is what makes the render faithful to the capture.

    A source with no camera matrix (a scanner TIFF, a JPEG) is already in the working space and passes straight through.

    **Linear RAW** is greyed out in **Calibration** here, because it does not apply to an as-captured render: it decodes without the as-shot white balance, which the camera matrix assumes is present, and the multipliers are folded back in, so the render is identical either way. It stays visible, since it is a sticky setting and a hidden one is a setting you cannot see the state of. With **Normalize** on it is live again, that render being a metered stretch rather than a transfer. An explicit Input ICC in Export always applies.

    **Narrowband** and **Trichrome Calibration** are greyed out for *any* transparency, Normalize or not; see [Narrowband and slides](#narrowband-and-slides). Reproducing a slide's appearance is a colorimetric problem, and narrowband illumination samples the spectrum at three isolated wavelengths, so the inter-band overlap the eye integrates is never measured, which is the same reason narrowband scans render oversaturated and hue-rotated. No input profile recovers what was never sampled, and the bundled one describes negative dyes besides.

<!-- panel:roll -->
### 4.3 Roll Analysis: a consistent look across the roll

Meter the whole roll once and share the baseline, so frames from the same film match.

*   **Batch Analysis**: scans every loaded file and computes a roll-average density and color balance, discarding outliers. Run it once after importing. *(Tip: if you use Batch Autocrop, run it first, in **Image only** mode, so metering sees consistent crops.)*
*   **Use Luma Average**: this frame takes the roll-wide tonal range; color still re-derives per frame.
*   **Use Color Average**: this frame takes the roll-wide color balance; tonal range still re-derives per frame. Enable both for a fully consistent roll; leave both off for per-image auto-exposure.

**ROLL**, to reuse a baseline across sessions:

*   **Roll dropdown** + **Load**: apply a saved roll's bounds and balance.
*   **Save**: store the current Batch Analysis as a named roll, useful when you shoot the same stock repeatedly.
*   **Delete**: remove the selected roll (it asks first). The frames keep their current look; only the saved baseline goes.

<!-- panel:presets -->
### 4.4 Presets

Save and recall a complete edit, the full workspace, by name.

*   **Preset dropdown** + **Load**: apply a saved preset to the current image.
*   **Name field** + **Save**: store the current settings as a new preset.
*   **Trash**: delete the selected preset.

---

## 5. Geometry tab

<!-- panel:geometry -->
### 5.1 Geometry: crop and straighten

Where the frame gets its final shape: what is inside the print, and whether it sits level. Most scans need a pass here even when nothing else is touched.

**Crop:**

*   **Ratio** (default `Free`): target aspect ratio: `Free`, `1:1`, `3:2`, `4:3`, `5:4`, `6:7`, `7:5`, `65:24`, `16:9`, `16:10`, `11:8.5`. There is one entry per shape, because the crop tool auto-orients to portrait or landscape as you drag. On `Free` the crop tool is unconstrained, and auto-crop takes the ratio from the film format it detects, so 6x6, 645, 6x7 and 35mm each keep their own shape. Pick a ratio to force every frame to it instead.
*   **Detect** (crosshairs): snap the ratio to the closest standard.
*   **Crop** tool: draw a crop rectangle on the canvas. It opens on the crop already set, including one **Auto** found, so you can nudge an auto crop instead of redrawing it. Once you adjust it by hand, nothing re-detects over it. **Reset** clears it and turns auto-crop off.
*   **Guide**: overlay a composition guide while cropping: *Thirds*, *Phi Grid*, *Diagonals*, *Golden Triangles*, *Golden Spiral*, *Armature*, *Diagonal Method*, *Grid* or *Off*. The redo button rotates guides that have orientations; the spiral has 8, the triangles 2.

**Auto Crop**, to detect the frame edge automatically:

*   **Mode**: *Image only* (exposed area) or *Film edge* (full film, including rebate and sprockets).
*   **Crop Offset** (-5 to 100 px): inset the detected edge inward. Positive trims more; negative bleeds slightly outside, for when detection clips too tightly.
*   **Rebate Trim** (0 to 150%): how far into the detected rebate to cut. 0% stops at the film edge, 100% lands on the detected image edge, and above 100% bites into the picture to clear a stubborn white border. *Image only* mode; it applies to both **Auto** and **Batch Autocrop**.
*   **Auto**: detect and crop this frame. Best on clean rebate. The crop is detected once and stored, so the export is framed like the preview. **Mode**, **Ratio**, **Rebate Trim** and the orientation re-detect; **Crop Offset** moves the stored crop without re-detecting.
*   **Batch Autocrop**: analyze all visible landscape frames as a roll, using confident detections to calibrate weaker ones. It runs in the background with progress and cancellation. Manual, Film-edge, portrait and ambiguous frames are left alone. *Image only* mode only.

**Alignment:**

*   **Fine Rotation** (±45°): free rotation for tilted scans, in sub-degree steps (positive is clockwise). Applied after auto-crop so the frame stays axis-aligned.
*   **Straighten** tool (ruler): draw a line along a horizon or vertical edge and NegPy rotates to make it level or plumb.
*   **Tilt** (±15%): tip the easel about a horizontal axis to straighten converging verticals, the building that leans back because the camera pointed up. Positive stretches the top edge. The unit is per-cent of the frame, what you would measure on the easel, not a tilt angle: the same tilt keystones differently at every enlargement.
*   **Swing** (±15%): the same movement about a vertical axis, for converging horizontals. A wall shot from one side, or a copy stand not square to the film. Positive stretches the left edge.

    Both replicate a wedge along the squeezed edge, as Fine Rotation does; crop it off. Crop before correcting if you can, because the meters read the corrected frame: on an uncropped scan a big correction pulls rebate and surround into the metered area and the print darkens.

<!-- panel:flatfield -->
### 5.2 Flat Field: even out the light

Corrects uneven illumination (vignetting or falloff) from your copy-stand or scanner light, using a reference shot of the bare light source.

*   **Flatfield Correction**: apply the active reference to this image, enabled once a profile exists.
*   **Reference Profile** dropdown + **Add…** / **Delete**: pick a reference image and save it as a named profile. **Add…** reads the reference once and bakes its correction into the profile, so you can then move, rename or delete the original reference file without affecting your edits. The profile is self-contained, stored in NegPy's own `flatfield` folder like sensor and crosstalk profiles. **Delete** asks first: the baked gain map cannot be recovered, and every frame using the profile loses its correction.
*   **Distortion** (-0.25 to 0.25): radial lens-distortion correction for the rig, saved with the profile. Use the film rebate as a straight-edge reference.

---

## 6. Exposure tab

This is the heart of the print. Three panels shape light, color and contrast, and everything here happens in the "print" stage of the pipeline.

<!-- panel:color -->
### 6.1 Filtration: white balance

Color timing, like the dichroic filters on an enlarger head. A **Global / Shadows / Highlights** selector scopes the controls to the whole image, or biases them toward low- or high-density tones.

*   **Pick WB** (eyedropper): click a pixel that should be neutral grey, and NegPy solves the CMY filtration to make it neutral in the selected region.
*   **Roll Lock**: re-aims each newly opened frame's temperature to the current target, preserving its own tint. A per-region lock for consistent warmth across a roll.
*   **Reset** (undo-arrow icon): return the selected region's temperature and CMY to neutral.
*   **Temperature**: a warm-to-cool lever driving the region's magenta/yellow pair; cyan stays put, as in a real darkroom.
*   **Cyan / Magenta / Yellow** (-1 to 1): the three filtration axes, Cyan↔Red, Magenta↔Green and Yellow↔Blue.
*   **Cast Removal** (0.0 to 1.0, **Color Negative only**): neutralizes the residual color cast a negative leaves in the print, balancing each layer so greys stay neutral from deep shadows through highlights. The applied strength scales with how many clean near-neutrals the frame has. Default about 0.5; 0 turns it off.

    It is hidden in Transparency and B&W Negative, because the render ignores it there. What it defeats is the **orange mask**, a cast the manufacturer built into the film rather than part of the picture. A slide has no mask and its cast *is* the photograph, so solving for a neutral axis would strip out the light you shot in; a B&W negative has one emulsion and no channels to balance. For a slide's color, use **Temperature** and the CMY sliders above, or **Hue Trim** (§4.1) if an unusual scanning light has rotated the hues.
*   **Ring-around** (target icon, or `Shift+F`): prints the frame as a 5×5 mosaic stepping 2cc at a time out to ±4cc on the magenta and yellow axes, so the direction of a color cast is visible instead of guessed. Each patch is a real render of the part of the frame it covers; click one to keep its filtration. The ladder is absolute and centred on neutral, so a ring printed off one frame compares to the next. `Escape` or a second press clears it, and any edit drops it. See **Rotating a proof** below.

<!-- panel:tone -->
### 6.2 Tone: density, contrast and the print curve

The paper's response. A **Global / R / G / B** selector at the top scopes most controls to the shared curve (Global), or to per-dye-layer trims for **crossover correction**, meaning casts that differ between shadows and highlights, which filtration alone cannot fix.

**Automatic helpers**, on by default. They do per-frame work so you do not have to, and turning them off lets the negative print honestly.

*   **Auto Density**: meters each frame's midtone and anchors print brightness there, so dense and flat negatives land consistently.
*   **Auto Grade**: aims each frame at a contrast target instead of printing the negative's own range, so dense negatives stop printing over-contrasty and flat ones stop printing muddy.
*   **Set Targets** (sliders icon): tune the exact brightness and contrast the two helpers aim for. Applies to every frame and is remembered between sessions.

**Test strip** (grid icon, or `Shift+T`): prints the frame as a 5×5 grid, with Print Density rising left to right and ISO-R Grade softening top to bottom, so the diagonals read light-to-dark and soft-to-hard like a split-filter test strip. Both ladders are absolute and centred on their defaults, so the settings you already have are one of the patches. Each patch is a real render of the part of the frame it covers; click one to keep it. `Escape` or a second press clears it, and any edit drops it.

**Rotating a proof**: a patch shows only the slice of the frame at its own grid slot, so the part you want to judge is stuck at whichever rung sits over it. While either proof is up, the 90° **rotate** buttons and `[` / `]` turn the *ladder* instead of the image: each press moves the dense or hard end onto a different edge, and the axis labels follow. The image's own rotation is untouched, and turning is instant, because printing a proof assembles all four orientations at once. The orientation you land on is kept for the rest of the session.

**Exposure:**

*   **Print Density** (0.0 to 2.0): overall brightness, simulating enlarger exposure time. Lower is brighter, higher is denser.
*   **ISO-R Grade** (50 to 180): contrast, as a paper ISO-R value. R110 is about classic grade 2; **lower R is harder** (more contrast), higher is softer. In R/G/B mode a **Grade** trim rotates one layer's slope about the midtone.
*   **Shadows Density** (±0.9 ΔD) / **Highlights Density** (±0.5 ΔD): brighten or darken just the shadow or highlight zone, without reshaping the curve. Bounded by paper black and white, so a burn cannot exceed the print's limits. The ranges differ because density is logarithmic: the same ΔD reads far smaller near paper black than near paper white.

    These two also work in Transparency with **Normalize off**, on the same tones (the centres are mapped by position on each curve's own scale, not by raw density), and there they are the only mid-sparing controls: Shadows Density opens the quarter-tone with the highlights unmoved, where Grade and Toe drag the whole scale with them and cost the highlights.
*   **Shadows Grade** / **Highlights Grade** (split grade, ±50 ISO-R): rotate contrast locally in the deep shadows or highlights, the digital equivalent of split-grade printing.
*   **Contrast Mask** (±0.5, hidden in Transparency): sandwich the negative with a blurred, low-contrast mask, as the darkroom does with a mask film and a spacer. Densities add, so the mask's own polarity decides which way the range goes, and its gamma decides how far. The value is that gamma, signed.

    Positive is the ordinary masking case: a blurred positive, contact-printed straight off the negative, so it is dense where the negative is thin. That squeezes the range by (1 − gamma) and a harder grade then fits the paper, while the blur keeps fine detail out of the squeeze. Use it on a scene too contrasty for the grade you want, then bring Grade back down in R. Past about 0.4 a soft halo appears along strong edges, as it does on a masked print.

    Negative is a mask of the same polarity as the negative, dense where the negative is dense, which stretches the range by (1 + gamma) instead. The broad tones expand while grain and texture stay put, where a harder Grade steepens both together. It also works on a negative too flat for Grade, whose slope has bottomed out. Past about −0.4 highlights start to clip, and the Analysis panel's Clipping row says so.

*   **Mask Spacer** (2 to 6%, dead without a mask): the clear sheet that holds the mask off the negative, as a per-cent of the frame. It sets the scale above which tones are masked, so it reads backwards from a blur radius: thick works on the broad masses only and leaves detail alone, thin reaches down into the detail and bites harder. Thin also lifts shadows sitting next to something bright and hazes them, which is the mask line a printer would see on the print. Below the minimum the mask stops being unsharp at all and becomes a plain contrast change, which is Grade's job. 4% is a conservative default.

    Both controls read only your crop. A masking film is neutral, so there is no per-layer trim and both grey out in R/G/B mode.
*   **Dye Separation** (0.5 to 1.5, hidden in B&W Negative): saturation in density space. It pushes the print's three dye densities apart *before* the positive is decoded, in the same matrix the paper's own dye crosstalk uses, so it responds to the paper profile you picked and eases off automatically where the curve is already compressed at toe and shoulder, instead of forcing color into tones that have none left to give. Below 1.0 it pulls the dyes together toward neutral. 1.0 is off. Contrast this with **Chroma** in the Color tab, which scales color evenly after decode.
*   **Separation Damping** (0 to 1, hidden in B&W Negative): decides *where* the Dye Separation push lands, rather than adding a push of its own. At 0 every color gets the same treatment. Turn it up and muted color keeps the full push while color that is already saturated gets the opposite, so a hard push puts color into the tones that had none instead of driving the strongest colors until they flatten into a slab. Below 1.0 separation it mirrors: pastels go grey while the vivid colors survive. It is **dead at Dye Separation 1.0**, where the slider greys out, because it has no look of its own. This is not the same as backing Dye Separation off: a lower value takes color from *everything*, including tones that had little to start with, where turning damping up takes it only from the colors that already have plenty.

**Paper Response**, the characteristic-curve shape:

*   **Paper profile**: a bundled darkroom-paper profile, RA4 color papers in Color Negative and tonal B&W papers in B&W Negative. It re-shapes the curve as a baseline; Grade, Density, toe and shoulder still trim on top. *Neutral* reproduces the defaults. Each B&W paper also carries its own lith color path, which the Lith panel picks up: Fomatone liths warm and colorful, while *Neutral* and Ilford Multigrade stay nearly colorless.
*   **Paper White**: simulate paper base density, so whites print at about 0.93 instead of pure white, like a real print.
*   **Paper Black**: show the paper's true, slightly milky Dmax instead of compensating it to pure display black. Off (default) applies black-point compensation so the adapted eye reads black as black.
*   **Snap** (-0.5 to 0.5): midtone gamma, steepening or flattening the S-curve around the reference tone while paper white and black stay put.
*   **Toe** (-1 to 1) + **Toe Width** (0.1 to 5): the shadow roll-off into paper black. Positive toe lifts shadows for a gentle film toe; negative deepens them and, with Paper Black off, makes exact black reachable. Width sets how far the knee reaches into the midtones.
*   **Shoulder** (-1 to 1) + **Shoulder Width** (0.1 to 5): the highlight roll-off into paper white. Positive compresses highlights (film-like); negative extends them and risks clipping.

In R/G/B mode the sliders become per-layer trims on top of the global value, for that dye emulsion: **Grade** (±30 ISO-R), **Toe** / **Shoulder** (±1), **Toe Width** / **Shoulder Width** (±2), **Snap** (±0.5) and **Dye Separation** (±0.4).

<!-- panel:local -->
### 6.3 Dodge & Burn: local exposure

Draw masks and lighten or darken just those areas. Three shapes, one per darkroom move:

*   **Draw Mask** (the cut card): click to place vertices; double-click, press Enter, or click near the start to close the mask; Esc cancels. To edit an existing mask, select it in the list, then drag a vertex, click an edge "+" to add a point, or right-click a vertex to delete it.
*   **Oval** (the hole in the card, or a dodging wand): drag out an oval. Three handles: the centre moves it, the other two set each axis, so it can be stretched and tilted. It has a fixed three points, with no adding or deleting.
*   **Card Edge** (the graduated burn): drag from the edge that gets the full exposure (solid line) to where it fades out (dashed). This is the printer moving a card across the paper: a sky burn, a corner held back. The gap between the two handles is the softness, so **Feather does nothing on this shape**.

Mask handles can go outside the picture, and a tilted Card Edge usually needs that: its line must start past the corner it burns, or the tilt cuts that corner off the full-exposure side. Drag into the grey area around the frame.

*   **Mask list**: each mask shows its shape icon and Dodge (lighten), Burn (darken) or Grade (contrast only), with the values it carries. The eye toggles its outline; the trash deletes it.
*   Each mask is tinted on the canvas so its extent is visible. Hold down **Burn**, **Feather** or **Grade**, or drag a vertex, and the tint drops away until you let go, so the value is judged on the picture rather than through the tint. Any mask that **intersects** the one you are working on drops its tint with it, since stacked tints are what hides the area worst; masks clear of it keep theirs. A Card Edge covers its whole side of the frame and an inverted mask covers its surround, so both count as intersecting anything in that area.
*   **Burn** (-2 to 2 stops, default 0): print exposure for the selected mask, signed the way the rest of NegPy signs light on paper. **Positive burns** (longer exposure, darker paper), **negative dodges** (held back, brighter paper), the same direction as Print Density and the Finishing edge burn. A freshly drawn mask sits at 0, so it changes nothing until you give it a value.
*   **Feather** (0.0 to 0.15): edge softness for the selected mask, as a fraction of the frame's short side. Inactive on a Card Edge.
*   **Invert**: acts everywhere *except* inside the selected mask, so it is the card itself rather than the hole cut in it. Burn the surround and hold the face with one shape.
*   **Grade** (-40 to 40 R): prints the selected mask at its own contrast, in ISO-R points off the frame's Grade, negative being harder. This is the darkroom's burn-in through the hard filter: burn a sky at −20 R and it darkens without the highlights beside it flattening; dodge a face at +15 R and the shadow opens without going chalky. The rotation happens about the region's own midtone, so a mask with Burn 0 and a Grade set changes only contrast, not overall density. Overlapping masks add their grades, and the result is clamped to the ISO-R ladder (R50…R180) like every other grade in NegPy.

**Printing Notes** (Export tab, or **Shift+N**) turns the frame into the printer's marked-up work print. Each mask is outlined and labelled with its number and its value in stops; a Card Edge has no outline, so it is marked as the side of the frame that gets the full exposure. A card in the corner carries the print recipe: paper, Print Density, ISO-R Grade (with the split-grade trims when they are set), filtration, toe and shoulder, Snap, edge burn, and the dodge/burn list.

Two conventions are worth knowing, both borrowed from the darkroom rather than from the sliders:

*   **Burns are hatched, dodges are left open.** Shading marks where the paper gets *extra* exposure.
*   **The numbers are exposure, not brightness**, the same convention the Burn slider uses, so a mask at +1.00 st is written `Burn +1`. Values land on ⅓, ½ and ¼ fractions where they are close enough, and otherwise print as decimals.

A mask with a local **Grade** also carries the grade it actually prints at, not the trim: a burn of +1.00 st at −20 R on a frame graded R115 is written `Burn +1 @ R95`, and a grade-only mask reads `Grade @ R95`.

Every mask is on the map, including ones whose outline you hid with the eye: that eye is there to unclutter editing, and a printing record that quietly omits a burn would be wrong. The overlay steps aside while a test strip, either peek, the before/after baseline, or the crop and analysis tools own the canvas. Both the preview and its export live in the Export tab's **Printing Notes** section.

---

## 7. Color tab

<!-- panel:lab -->
### 7.1 Lab: polish and detail

Mimics what a lab scanner (Frontier or Noritsu) does automatically. Color controls hide in B&W Negative mode.

**Color** (hidden in B&W Negative):

*   **Chroma** (0.0 to 2.0): a color scale applied after the print is decoded, even across every tone, so it is a retouching move rather than a density-space one. 1.0 is unchanged, 0 is greyscale, 2.0 is double. For saturation that behaves like a print instead, reach for **Dye Separation** in the Exposure tab. Below 1.0 is a flat scale; above 1.0, pixels that would clip the display gamut get a soft per-pixel knee toward their own in-gamut headroom instead of a hard per-channel clamp, since clamping only the overshooting channels shifts the hue that the flat scale itself preserves.
*   **Skin Protection** (0.0 to 1.0, default 0.5): holds skin-hued color under a chroma ceiling so faces do not go sunburnt. Hue and lightness are untouched, and chroma is only ever pulled down, never added, so asking Chroma for 0 still gives you greyscale. It is independent of Chroma and works with it at 1.0: skin that arrived over-saturated from the print curve or the filtration gets reined in just the same. Higher values lower the ceiling: the 0.5 default catches only genuinely excessive chroma, 1.0 leaves skin matte, 0 is off. The mask is warm hue *and* skin's own chroma *and* mid lightness together, which is what keeps a red coat, a saturated sunset, brick or autumn color out of it. What it cannot separate is warm objects sitting at the same chroma as skin (bare wood, tan leather, sand), which soften along with it. The same bound cuts the other way: skin that arrives really excessive, a sunburn, is only partly caught, so reach for Chroma or the Filtration panel for that.
*   **Chroma Denoise** (0.0 to 5.0): smooths color noise, especially in shadows, while leaving luminance grain intact.

**Sharpen:**

*   **Method**: *Unsharp Mask* (boosts edge contrast) or *Deconvolution* (Richardson-Lucy, which reverses the scanner's optical blur; set Radius to the scan's blur width).
*   **Sharpening** (0.0 to 1.0): amount, on the L (lightness) channel so there are no color halos.
*   **Radius** (0.5 to 3.0 px): blur width in output pixels, small for fine grain and larger for soft scans. Sharpening acts on the pixels of the exported file, so a fit-to-window preview shows less of it than the export carries; judge it at 1:1 with the loupe or at 100% zoom.
*   **Masking** (0.0 to 1.0): restrict sharpening to edges, which protects flat areas like sky, skin and grain.

**Detail:**

*   **CLAHE** (0.0 to 1.0): local contrast without blowing global highlights or crushing shadows. Use it sparingly, since near 1.0 can look cartoonish. It runs before dust removal, so healing operates on the final rendition.

**Effects:**

*   **Glow** (0.0 to 1.0): lens bloom, where bright highlights scatter across all channels for a dreamy softness.
*   **Halation** (0.0 to 1.0): the red glow of light scattering back through the film base. Highlights only, strongly red-dominant.

<!-- panel:altproc -->
### 7.2 Alternative Processes

Two printing processes that are not ordinary silver-gelatin enlarging. Pick one with the **None / Lith / Cyanotype** buttons at the top, and only that process's controls are shown. Both are B&W Negative only, and both are off by default.

#### Lith

Lith printing is the darkroom process of massively over-exposing a lith-capable paper, developing it in a very dilute low-sulphite developer, then pulling it out part-way through. You get creamy warm highlights and an abrupt drop into hard, sooty blacks, with very little in between.

There is no color control here. The paper chosen in the Exposure panel sets the whole path, from peach highlights through an olive transition to neutral blacks. *Neutral* and the Ilford papers lith almost colorlessly; Fomatone is the one that gives you the peach and the olive.

While Lith is selected, Sepia, Iron Blue, Copper and Vanadium grey out in the Toning panel, since they do nothing distinctive on a lith print. Selenium and Gold stay live, and behave differently here (see 7.3).

*   **Exposure** (0 to 5 stops, default 2): print over-exposure. Real lith printing runs on two to four stops more light than a normal print. More light means warmer, more colorful highlights and softer gradation.
*   **Snatch Point** (0.0 to 1.0, default 0.55): how long the print stays in the developer before you pull it. Higher drops the point where the shadows go black further up the tonal scale, giving deeper, colder blacks and a wider band of undifferentiated shadow. Lower keeps the print high-key and warm, with weak blacks.
*   **Abruptness** (0.0 to 1.0, default 0.6): how suddenly the shadows go black. High turns the transition into a step, so the next zone down blocks up with no separation left in it. Low leaves a gentle roll into the blacks. In the darkroom this is the hydroquinone-to-alkali ratio of the developer.

#### Cyanotype

A cyanotype is contact-printed in UV onto paper brushed with iron salts. There is no development to time and no silver anywhere in it: the image substance is Prussian blue, which absorbs red light around 700nm, so the print never goes black, it goes blue. Highlights come out green, where the blue mixes with the yellow sensitiser left in the paper.

Two things make it look unlike an enlargement. It holds only a short density range, so a negative that prints normally on paper clips at both ends. And it compresses the midtones, so the middle of the scale is flatter than the ends.

While Cyanotype is selected, every chemical toner greys out in the Toning panel, because there is no silver for those baths to react with. Use Bleach and Tannin instead. Split toning still works.

*   **Sensitiser** (Classic or New, default Classic): *Classic (Herschel)* is the original ammonium ferric citrate mix. It loses a lot of its pigment in the wash, so it tops out at a fairly light blue and keeps a strong green stain in the highlights. *New (Ware)* is the ferric oxalate formula, which holds far more pigment through the wash, so it goes much deeper and cleaner.
*   **Exposure** (-2 to 4 stops, default 0): time under the UV source. More light drives more of the scale into blue; less leaves the print pale and high-key.
*   **Exposure Scale** (0.8 to 2.8 log D, default 1.4): the negative density range the sensitiser can print, which is the contrast control. Ware measures about 1.0 to 1.2 for the traditional formula against 2.4 for the new one, and his Simple Cyanotype comes in variants at 1.8, 2.3 and 2.7. A short scale gives a contrastier print that clips both ends.
*   **Bleach** (0.0 to 0.5, default 0): washing soda. It strips Prussian blue out of the print, highlights first. Take it far enough and only the deepest shadows keep any pigment.
*   **Tannin** (0.0 to 0.5, default 0): tea, coffee or tannic acid. It re-develops the bleached iron as a brown iron tannate, which covers more than the pigment it replaced, so the print goes browner and a little deeper. Bleach first for a full brown; use Tannin alone for a split blue-brown.

---

<!-- panel:toning -->
### 7.3 Toning

Color the print itself rather than the scene: chemical toners that convert the silver (B&W Negative only), and a split tint that works in any mode. Lith silver is much finer than normal print silver, so the toners bite harder and differently on a lith print. With Lith on, only Selenium and Gold stay enabled; with Cyanotype on all six grey out, because there is no silver in the print at all.

**Chemical Toning** (B&W Negative only), simulated as sequential toner baths, in the order shown, each strength 0.0 to 2.0:

*   **Selenium**: deeper blacks, cool eggplant shadows. On a lith print it reaches much further down the scale, lifts Dmax hard and turns the green-black shadows magenta.
*   **Sepia**: warm highlights first; partial strength gives split-sepia.
*   **Gold**: cool blue-black on untoned silver; over sepia it shifts highlights orange-red. On a lith print it works on every density evenly instead of the highlights first, and pushes the print towards blue-violet.
*   **Iron Blue**: Prussian-blue shadows deepening to navy blacks.
*   **Copper**: pink to brick-red shift, with the classic Dmax loss.
*   **Vanadium**: greens the mids and highlights while deep shadows keep their black.

**Split Toning** (all modes), an additive tint in Lab space, so grain and detail are preserved:

*   **Shadow Hue** (0 to 360°) + **Shadow Strength** (0.0 to 1.0).
*   **Highlight Hue** (0 to 360°) + **Highlight Strength** (0.0 to 1.0).

---

## 8. Finish tab

<!-- panel:retouch -->
### 8.1 Retouch: dust, hairs, scratches

Spotting, the way it was done with a brush on the finished print. There are three ways to find the marks (local contrast, the scanner's IR channel, or by hand) and they stack. However a mark is found, it is repaired the same way: the film under it is rebuilt from the clean film around it, with the frame's own grain transplanted back, and anything too wide for that goes to a fill that follows the structure through.

An **Overlay** button cycles the detection overlay (Off → Marked → IR) so you can see what is being caught: green for what Optical Removal found, magenta for IR and for defects sent to the structure-following fill.

**Optical Removal** finds specks on the visible scan by local contrast, with no IR needed:

*   Toggle **Optical Removal** on, then set **Threshold** (0.01 to 1.0; lower catches more, at the risk of false positives) and **Size** (3 to 8 px; max spot radius).

**IR Removal** uses the scanner's infrared channel to remove dust invisible to the color dyes, and is enabled only when the scan carries an IR plane.

*   Toggle **IR Removal** and set **IR Threshold** (0.05 to 0.95; lower catches more).
*   **Method** picks how the film under a defect is rebuilt. Both use the same IR plane and the same threshold slider.
    *   **NegPy** (default) divides semi-transparent dust back out, fills opaque cores with a weighted average of the clean film around them, and transplants grain from the nearest clean pixel.
    *   **OpenICE** works in log density and restores detail rather than averaging it away: at each scale it adds back the picture detail that beats the infrared's own contrast at that scale, so texture under a speck survives. Where a defect was solid there is no detail left to restore, so the repair gets Digital ICE's own synthetic grain instead, strongest in the midtones and fading out at both ends of the scale. It measures clear-film level and dye-to-infrared crosstalk from each frame, and leaves film it judges clean untouched bit-for-bit. Better on fine detail and gentler elsewhere in the frame, but less proven across scanners, so try both on a frame you know.
*   The IR plane is read from 4-channel TIFFs and DNGs (VueScan, NegPy's own scanner output), SilverFast's iSRD TIFFs and 64-bit **HDRi RAW DNGs**, and `_IR.tif` sidecars. Scan to HDRi, not plain HDR, if you want IR data in the file. B&W and Kodachrome block infrared like dust does, so those frames are skipped automatically.

**Manual Heal** (the header shows the current spot count):

The brush marks a *search area*, not a stamp: only the pixels that actually stand out from the film around them are rewritten, so you can paint generously over a speck and the clean grain inside the brush is left exactly as it was. Marks are caught in both directions: dust, which prints light, and scratches, which print dark. If the brush finds nothing wrong, it does nothing.

*   **Heal Tool**: click dust spots in the preview to paint them out one at a time, or drag to paint over a run of them.
*   **Scratch Tool**: click points along a scratch or hair, then double-click or press Enter to finish. Esc cancels, Backspace removes the last point. Right-click an overlay to delete it.
*   **Transport Line**: for the long straight marks film picks up running through a camera or lab, the ones that cross the whole frame, usually in the same place on every shot of the roll. **Click once anywhere on the scratch** and the whole line is traced and repaired; there is nothing to paint or drag.

    These are the marks the brush is worst at, and not for want of care: spread along its length, a transport scratch is far too faint to pick out from film grain at any single point. The line tool reads the evidence along the whole scratch at once, which is what makes it visible at all. It follows the scratch's own angle (film is rarely square to the sensor), widens the repair to match the scratch, and covers only the stretches where the scratch is actually present, so one that fades in and out is left alone where it fades. If a click finds nothing, it says so rather than touching the frame; click directly on the line.

    Hovering shows a **guide**: the line that would be traced and the band it would repair, before you commit. Committed lines stay drawn the same way, so you can see what each one covers; right-click one to delete it.

*   **Line Sensitivity** (0.05 to 0.95, shown while the Transport Line tool is active): how readily a scratch is followed. Lower catches fainter lines and repairs a wider band; raise it if a line starts picking up film either side. It applies to lines already placed as well as new ones, so you can trace first and tune after.

*   **Brush Size** (2 to 16 px): diameter of the manual brush, matching the on-screen cursor, shown while a heal or scratch tool is active.
*   **Undo Last** / **Clear All**: remove the most recent or all manual heals and traced lines; auto-detected dust is unaffected. Right-click a line to delete just that one.

<!-- panel:finish -->
### 8.2 Finishing: vignette, carrier, border

How the print is presented: edge burn, a filed-out carrier's black rebate, and the paper margin around it. Applied at the very end of the pipeline, after everything else is settled.

**Vignette** (printer's edge burn, in stops):

*   **Burn** (-2.0 to 2.0 stops): positive darkens the edges, negative holds them back and lightens. 0 is off.
*   **Size** (0.0 to 1.0): falloff radius. Small keeps it tight in the corners, large spreads it into the frame.
*   **Roundness** (0.0 to 1.0): 0 is radial (lens-like), 1 is a rectangular card burn following the print edges.

**Filed Carrier**, a filed-out negative carrier: the clear rebate prints max black, framed by a margin of unexposed paper.

*   **Width** (0.0 to 5.0 mm): black rebate frame thickness. 0 is off.
*   **Roughness** (0.0 to 1.0): how raggedly the aperture was filed, on the paper-side edge of the black frame. The picture-side edge is the camera's film gate and only ever wobbles slightly.
*   **Flare** (0.0 to 1.0): light reflected off the bared metal of the filed bevel, a glow that lifts the black just inside the filed edge and stains the paper just outside it. Colored on color film, with the hue drifting along the edge, because the stray light never passes the orange mask; neutral in B&W. 0 is off.
*   **Corners** (0.0 to 1.0): how far the aperture's corners round off, since no file cuts a sharp inside corner.

The paper margin takes the mat color, so it runs into the border with no seam.

**Border:**

*   **Width** (0.0 to 2.5): border thickness as a fraction of the image. 0 is no border.
*   **Bottom weight** (1.0 to 2.0): thickens the bottom border, for window-mat proportions.
*   **Color swatch**: click to pick any border color.
*   **Paper white**: tint the border with the toned paper-white instead of the picked color.

---

## 9. Favourites tab

The sliders you reach for most, gathered in one place, so a routine edit no longer costs a
tab switch and a scroll. Empty until you fill it.

*   **Edit Favourites**: opens a picker. Tick sliders on the left, order them on the right with
    the arrow buttons, then press **Apply**.
*   The panel then shows those sliders in your chosen order. They are the *same* controls as in
    their home panels, so moving one here moves it there and the other way round. Nothing is
    duplicated or moved out of its own tab.
*   A favourite hides itself when its original does. Favourite a Filtration slider and it
    disappears while you are in black & white, where it has nothing to act on.
*   Your selection is remembered between sessions.

---

## 10. History tab

Two lists: the versions you chose to keep, above the running record of every change.

### Work prints

A **work print** is a named version of this frame, the darkroom habit of keeping the prints you made on the way to the final one, so you can go back to the third attempt after deciding the fifth went too far.

*   **Save work print** (**Ctrl+Shift+S**) keeps the current edit under a name; NegPy offers *Work print 1*, *Work print 2* and so on. Saving over an existing name asks first.
*   **Click** one to make it live. That counts as an edit, so **Ctrl+Z** puts back what was on screen before; you cannot lose your place by looking at an old version.
*   **Right-click** for **Export this version…**, **Rename…** or **Delete**. Delete asks first, and a rename to an empty name is ignored.

Work prints differ from history steps in the way that matters: they are **never pruned and never thrown away by a later edit**. The undo history keeps the last 100 steps and drops the branch above you when you edit after stepping back; a work print survives both. The list appears only once you have saved one.

They belong to the frame, not to your presets: a preset is a look you apply to other images, a work print is one version of this print. Both live in NegPy's database; work prints are not written to `.negpy` sidecars.

### Edit history

A scrollable list of every edit step, the last 100 kept, newest on top. The current step is bold.

*   **Click** a step to jump to that state.
*   **Right-click** → **Export this version…** to export a past state directly.

---

## 11. Export tab

### Output intent

*   **Print** (default): the full creative look you see on screen.
*   **Flat**: a flat, neutral, low-contrast master that keeps maximum tonal and color information for editing elsewhere (Lightroom, Darktable, Photoshop). It skips the print look, effects, toning and vignette, and writes a 16-bit TIFF, or a lossless JPEG XL when JXL is selected and the color space is taggable (sRGB, P3, Rec 2020 or Greyscale). Your in-app preview is unaffected.
    *   **Preview Flat**: temporarily show the flat master on the canvas without changing your edit.
    *   **Roll Baseline**: measure every visible frame and share one exposure baseline, so flat masters are consistent across a roll. Recommended before a flat batch.
*   **Linear**: bypass the entire darkroom pipeline and dump the scanner's or camera's decoded buffer as a linear 16-bit file. The output format is selectable: **TIFF** (default, zlib-compressed, genuinely untagged) or **JPEG XL** (lossless). JPEG XL has no untagged state, so it comes out asserting sRGB primaries and a linear transfer regardless, which is not true for camera or scanner-native primaries; use TIFF if an unasserted file matters. An **Effort** slider (1–9, default 7) controls JPEG XL encoder speed against compression. No normalization, exposure, color management, flatfield or sensor correction, just the raw data with lossless geometry (rotation and flip) applied. Supported sources:
    *   **Pakon RAW**: 4× expansion by default (14-bit sensor range scaled into 16-bit). F335 files (16-bit sensor) default to no expansion.
    *   **LinearRaw DNG**: SilverFast HDRi (3-channel) and VueScan (4-channel RGB+IR). IR is written as a separate greyscale file with an `_ir` suffix, in the same Format you chose for the main output.
    *   **Camera RAW**: demosaiced with unity white balance (1,1,1,1). The camera's as-shot WB is written into XMP (`RAW-WB: R G B`) so downstream tools can apply it. Source device and timestamp are preserved. RGB-scan triplets (narrowband R/G/B exposures) are merged into a single combined TIFF. Stitch composites are assembled with flatfield and sensor correction applied per-part for clean seams; stitch and triplet combinations are also supported.
    *   **Coolscan NEF**: Nikon Coolscan scanner files. Despite the name, these are not raw sensor data: the content depends on the Nikon Scan settings used at scan time, so linear, unprocessed output needs the right settings before scanning. The full-res RGB SubIFD is read directly, and any extra channels beyond RGB are dropped, since Coolscan has no separate IR channel. No expansion.
    *   **Flextight FFF**: Imacon/Hasselblad Flextight scanner files, including both standard uncompressed 16-bit RGB exports and SGI LogLuv compressed raw files (`.3fr`/`.fff`). LogLuv files are decoded through a LogLuv → XYZ → linear sRGB pipeline with per-channel percentile normalization; LogLuv is HDR, so normalization is part of the decode, and without it the data would be truncated, not raw. The largest image IFD is selected by pixel count. Data is linear scanner transmittance. Embedded FlexColor metadata (film stock, film type, scan date, scanner serial) from the proprietary plist (tag 50457) and the firmware blob (tag 46279) is carried through to the output TIFF headers. No expansion.
    *   **Noritsu RAW**: headerless BGR 16-bit scanner dumps. Frame dimensions are auto-detected from file size against known Noritsu scan dimensions. 16× expansion by default (12-bit sensor data in 16-bit range).
    *   **TIFF**: generic scanner TIFFs. If the file has a 4th channel tagged as IR (ExtraSamples = UNSPECIFIED or missing), it is written as a separate `_ir` file in the same Format as the main output. Sidecar IR files (`_ir.tif` next to the source) and IR stored in secondary TIFF pages are also detected. **Input gamma** lets you select the gamma encoding of the source (linear, 1.8, 2.2 or sRGB) so the data can be linearized before export. Expansion is available, off by default.
    *   **Expansion**: scales the linear data before writing. The combo box shows source-appropriate options: Pakon F135/F235 default to 4×, Noritsu to 16×, F335 and LinearRaw DNG to off. Camera RAW, Coolscan NEF and Flextight FFF have no expansion option. Leave it at the default unless you know why you need to change it.
    *   **Apply ICE dust removal** (visible when an IR channel is available): applies IR-based dust and scratch correction to the linear output before writing. Off by default.
    *   **Corrections** (camera RAW only): three optional toggles that bake corrections into the linear output before writing. All default to off, following the raw-dump philosophy. **Apply white balance** multiplies by the as-shot WB gains. **Apply flatfield** applies the flatfield gain correction. **Apply sensor correction** applies the sensor crosstalk unmixing matrix. For stitch composites, flatfield and sensor correction are always applied per-part regardless of these toggles, because clean seams require it.

    Linear Output writes where the **Destination** section says, the same as a print or flat export: folder mode, subfolder, export path and Filename Pattern all apply. `_linear` is always appended to the rendered filename, so a dump written next to its source cannot overwrite that source. Without **Overwrite**, an existing file makes the next one `_linear_2`, `_linear_3` and so on.

    Linear Output runs in the background like any other batch: the progress popup shows which frame is being written, **Abort** stops it after the current one, and the finish message counts any frames that failed.

    The output file is always written clean: no ICC profiles, no EXIF color space tags, no XMP color metadata from the source. For **TIFF**, it carries raw pixels plus device metadata (Make, Model, DateTime) from the source file, and a description recording the source format, expansion, white balance and any corrections applied, including ICE. **JPEG XL carries none of that metadata at all**: no description, no device info, no record of whether ICE ran, only the pixels and the forced color tag noted above, which also is not from the source. The format simply cannot leave it unset.

### Export button

The primary **Export** action. Its chevron menu picks the scope: current frame (Ctrl+E), selected frames, or all visible frames. Every scope uses the settings below. To deliver the same frames in more than one format or size in a single run, use Export Presets.

### Format / Size / Color / Destination

*   **Format**: `JPEG`, `TIFF`, `PNG`, `JPEG XL`, or `WebP`, with quality or effort options per format. TIFF is always zlib-compressed. **JPEG XL supports only `sRGB`, `P3 D65`, `Rec 2020` or `Greyscale`** for Color Space: it tags color with compact enumerated values rather than an embedded ICC profile, and NegPy's JXL encoder cannot carry an arbitrary one, so `Adobe RGB`, `ProPhoto RGB` and a custom Output ICC are rejected with an error. Pick a supported space or a different format.
*   **Color Space**: `Same as Source`, `sRGB`, `Adobe RGB`, `ProPhoto RGB`, `P3 D65`, `Rec 2020`, or `Greyscale` (true B&W output).
*   **Input / Output ICC**: soft-proof against, and optionally embed, an ICC profile. Output is the destination profile (default); Input treats the profile as the source, for when a scan's profile is known but untagged. Not available for JPEG XL output; see the Format note above.
*   **Paper Aspect Ratio**: final print ratio, or *Original* (no resize).
*   **Resolution**: *Original* (full RAW resolution), *Print* (long-edge **Size** in cm plus **DPI**), or *Pixels* (long-edge **px**; the short side follows the paper ratio).
*   **Destination**: **Filename Pattern** (a Jinja2 template with export settings plus Metadata fields such as roll, camera and film; see [TEMPLATING.md](TEMPLATING.md)), an **Overwrite** toggle, and the output location (subfolder of source, same as source, or an absolute **Export Path** with a browse button). Destination applies to all three output intents: with **Linear** selected, Format, Size and Color hide (a raw dump has no use for them) and Destination stays.

### Collapsible sections

*   **Presets**: a checklist of export presets, each a saved Format/Size/Color/**Destination**/filename recipe. **Manage** edits them; **Export Presets** renders the frames with every enabled preset at once, and each preset uses **its own** destination, not the sidebar Destination above.
*   **Sidecars**: **Save on export** writes a `.negpy` edit sidecar next to each source on every export; **Export sidecars** writes them for all visible frames now, and reports how many failed if a source folder is read-only. Edits always stay in the database too; sidecars are optional archival copies.
*   **Contact Sheet**: render all visible frames into a single sheet. Choose a **Template** or set **Cell / Gap / Margin / Max tiles** by hand, pick an output **Path**, then press **Export contact sheet**.
*   **Preview** (affects the on-screen preview only, never the file):
    *   **Soft proof** (on by default): simulate the export color space and Output profile, so what you see matches what you get. Turn it off only to preview at full gamut.
    *   **Display**: the monitor profile the preview is shown through, auto-detected; pick one manually if detection fails.

---

## 12. Metadata tab

Archival metadata for the **original analog capture** (camera, lens, film, process), written into exported files as EXIF and embedded XMP, so DAMs like Lightroom show your film gear rather than the scanner.

*   **Protect original metadata**: copy the source file's EXIF/XMP to exports unchanged, adding nothing. When it is on, the fields below are ignored.

**Analog Gear** (searchable; type in any field to filter the library):

*   **Preset**: a reusable camera, lens and film combination. **Clear** empties the gear selections.
*   **Camera / Lens / Film stock**: pick from your library. Empty means not set.
*   **Manage…**: edit cameras, lenses, film stocks and presets. Starter data seeds into `~/NegPy/gear/` on first launch.

**Process:**

*   **Format**: `35mm`, `120`, `4×5`, `8×10`, `110`, or `Other` with a free-text field.
*   **Developer**: for example `D-76 1+1`.
*   **Push / Pull**: `Push +3` … `Normal` … `Pull -3`.

**Scanning:**

*   **Scanning**: scan method or notes. EXIF `Software` is always `NegPy`.
*   **Roll / Frame**: Scanlight capture roll name and frame number, stamped automatically on capture and editable here. Available in export filename templates as `{{ roll }}` and `{{ frame }}`, and written to XMP as `negpy:CaptureRoll` and `negpy:CaptureFrame` when set. Not the Roll Analysis normalization name.
*   **Sync custom metadata to all files in batch export**: apply this tab's values to every file in a batch.

**Exposure**: optional original shutter, aperture and ISO. Click the lock to edit a free-text string, for example `1/125s f/2.8 ISO 400`.

**Metadata preview**: a live view of exactly what will be embedded, grouped by capture, scan, process and file. **Description…** opens a checklist of which fields join into EXIF `ImageDescription`. The defaults are camera, lens, film stock and ISO; format, developer, push/pull and scanning are off until you enable them. Confirming **Description…** sets that frame's selection and becomes the sticky default for other frames that do not have their own, so the last confirm on the roll wins. Sync metadata and Sync settings can also copy a frame's selection with the rest of the metadata.

When you set capture gear, it is written to standard EXIF, and the digitizing rig is preserved separately in `negpy:Scan*` XMP tags. Leave gear unset and your scanner or DSLR stays visible in EXIF instead.

---

## 13. Scan tab

Capture film directly into NegPy. Two collapsible sections:

*   **Scanner**: drive a film scanner. Choose a **Backend**: **SANE** (Linux/macOS; Coolscans and other SANE devices) or **pyOpticfilm (Plustek)** (OpticFilm 8200i SE; Windows, macOS and Linux). Common controls are device selection, DPI, IR channel, frame range (roll feeders), scan window, output format, folder and filename template. **Depth** appears only when the device offers more than one bit depth, so it is hidden for the OpticFilm 8200i SE, which is 16-bit only. **Autofocus** and hardware **Auto-exposure** appear only when the connected device reports them, so typically on Coolscans and not on the OpticFilm 8200i SE. **Prescan** appears for devices that support a low-DPI full-window preview, such as the OpticFilm 8200i SE: run the preview, drag a crop rectangle, and the next Scan uses that hardware ROI. When the scanner exposes a `scan-exposure-time` option, as some genesys devices do, an **Exposure** slider appears; set it to override the scanner's default exposure time, and the value shows in µs, ms or s as appropriate. A device without the option hides the slider, so a saved value never breaks a different scanner.

    **pyOpticfilm (Plustek)** notes: only the **OpticFilm 8200i SE** (`07b3:1825`) is scan-ready. Other OpticFilm models may appear in the device list but cannot scan until validated; on Linux and macOS, switch Backend to **SANE** if that backend lists the scanner. Use **Prescan** to grab a 1200 dpi full-window preview, set a crop, then Scan at the chosen DPI (a hardware ROI, not a software crop).

    IR is a second USB pass, color then infrared; NegPy registers that IR plane to the color frame with a whole-pixel shift, so dust removal stays aligned after the carriage re-homes. Color scans apply ASIC shading measured at home before the film feed, the same order as SilverFast, so the strip may stay loaded. The table is cached per DPI, so later scans only re-upload it.

    The default Full window includes a little holder chrome top and bottom; host-path scans clamp those near-white margins to the film highlight so auto exposure is not skewed. Raise **Analysis Buffer** or crop if a frame still looks off. Autofocus and hardware Auto-exposure controls stay hidden, because the SE does not report those capabilities. On Windows, bind the device to **WinUSB** with Zadig before use, since the stock vendor or SilverFast driver conflicts. The driver is the optional **pyopticfilm** package: install it with `uv sync --group plustek` or `pip install negpy[plustek]`; Windows release builds bundle it. See [PLUSTEK_WINDOWS.md](PLUSTEK_WINDOWS.md).
*   **Camera Scanning**: DSLR or mirrorless copy-stand capture (macOS/Linux). It auto-connects the camera over USB in PC-Remote mode. With a NegPy **Scanlight** connected it captures narrowband R/G/B triplets from saved film-stock presets; without one it does a single white-light exposure. A **Live View** window helps you frame and focus. Captured frames land in the hot folder and flow straight into RGB-Scan mode.

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
| `performance.preview_cache_max_bytes` | a number, e.g. `1200000000` | Preview cache memory budget (default about 1.2 GB). |
| `performance.preview_cache_max_entries` | a number, e.g. `8` | Max recently-viewed photos kept in memory. |
| `performance.preview_cache_max_full_res_entries` | a number, e.g. `2` | Full-resolution HQ preview buffers kept in memory (a 60 MP scan is about 700 MB each). |
| `performance.cpu_parallel` | `true` / `false` (or absent) | Multi-core CPU rendering kernels. Defaults on, except on macOS. |
| `logging.level` | `"debug"`, `"info"`, `"warning"`, `"error"` | Log verbosity. Use `"debug"` when reporting issues. |

**Common fixes:**

*   **Crashes immediately on Linux** → `backend = "cpu"` or `qt_rhi_backend = "opengl"`.
*   **Black or blank preview on Windows** → `backend = "dx12"` or `qt_rhi_backend = "software"`.
*   **Wayland rendering issues** → `qt_platform = "xcb"` to force X11.
*   **GPU out-of-memory during export** → `max_texture_size = 4096`.

---

## 15. Updating NegPy

NegPy asks GitHub for the newest release once at startup. If there is one, a green **⬇ Update Available: vX.Y.Z** line appears under the logo in the left panel. Click it to open the update window with the release notes, the download size and one button.

To check yourself, press the **↻** button next to the version number in the left panel's header. If nothing is new, NegPy says so; if something is, the button turns into a green **⬇** and the update window opens.

**Install Update** downloads the build that matches how *this* copy was installed, then closes NegPy, installs it over the old version, and reopens on the new one. You do not download, uninstall or reinstall anything by hand. Nothing is replaced until NegPy has exited, so a failed download or a refused permission prompt leaves your working install exactly as it was.

What happens per platform:

| Install | What NegPy fetches | How it installs |
|---------|--------------------|-----------------|
| **Windows** | the `-Setup.exe` installer | Runs it silently over your existing install. Windows asks for administrator approval first, because the app lives in Program Files. Approve it *before* NegPy closes. |
| **macOS** | the `.dmg` for your chip (Apple silicon or Intel) | Mounts the image and replaces the `NegPy.app` bundle where it currently sits, then reopens it. |
| **Linux** | the `.AppImage` | Replaces the AppImage file you launched, keeps it executable, and relaunches it. |

Your edits, presets, settings and library are untouched: they live in `Documents/NegPy` and the local database, not in the installation folder.

**When the button says "Open Releases Page" instead**, NegPy cannot swap itself and sends you to GitHub. That happens when you run from a source checkout, when the release has no build for your platform, or when the app was moved somewhere it no longer matches its installer's layout. The same happens if the folder holding the app is not writable by you (an AppImage in a system directory, an app bundle in an `/Applications` you do not own); NegPy says so rather than failing halfway.

You can ask for the check again at any time with the **Check for updates** action, unbound by default, so give it a key in the shortcut editor.

---

## Additional Info

*   **GPU acceleration**: NegPy uses your GPU for near-instant previews and responsive sliders. The Normalization panel's analysis (bounds, white/black point, normalize) runs on the CPU. There is no global GPU switch in the UI, so force the CPU pipeline through `override.toml` if you suspect a driver issue.
*   **Database**: all edits live in a local SQLite database keyed by file hash, so you can move or rename files without losing your work. Optional `.negpy` sidecars mirror edits next to your sources.
*   **Saving edits**: edits are written to the database on export, when you switch frames, or when you save explicitly. Closing the app mid-edit without any of those loses unsaved changes.
*   **Keyboard shortcuts**: [KEYBOARD.md](KEYBOARD.md)
*   **Filename templating**: [TEMPLATING.md](TEMPLATING.md)
*   **The pipeline in depth**: [PIPELINE.md](PIPELINE.md)
