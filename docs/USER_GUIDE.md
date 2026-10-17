# NegPy User Guide

NegPy turns film scans into positives with a non-destructive, darkroom-style pipeline. It never writes to your source files. Edits live in a local database.

For the pipeline order and math, read [PIPELINE.md](PIPELINE.md).

---

## 1. The Big Picture

### Protected data folders on Windows

If Windows blocks the default data folder, NegPy suggests `%LOCALAPPDATA%\NegPy\data`. Click the path to pick another, then **Use This Folder**. Nothing is copied: a new folder starts empty, and an existing NegPy folder keeps its data. The choice is saved in `data-location.json` and stays even if Documents becomes writable. `NEGPY_USER_DIR` overrides it.

### Screen layout

*   **Left, the film strip**: your frames as a contact sheet, with import, sorting and triage tools.
*   **Center, the canvas**: the live preview. Scroll or pinch to zoom, drag to pan, double-click to switch between fit and 100% at that spot. Zoomed in, a navigator in the lower right shows the visible part and the zoom; click or drag it to move the view. The bottom toolbar holds navigation, zoom (**Fit to Window**, 25–400%, **HQ Preview**), rotate/flip (on every selected frame), **Tools** (arms any canvas tool: crop, straighten, the pickers, masks, retouching), undo/redo, the **Print** menu (Before / After and the peeks below) and **Overlays** (Zone Overlay, Clipping Overlay, Printing Notes, Grain Focuser, Probe at Cursor, Frame Info). An armed canvas tool shows its use, its slider and **Done** on the toolbar; its name opens the same menu to switch tools. The **⋯** menu holds every action, including **Preferences…** (§15). Right-click the toolbar for **Edit Toolbar…**. Right-click the image for **Reset View**, **Sticky Zoom**, the pickers, copy/paste settings, **Reset Settings**, **Reset to Roll Settings** and **Unload** (drop the frame from the session; its edit stays).
*   **Frame Info**: `I` cycles the corner pills. **Off**; **Basic**: file name, scan size, film mode; **Full**: adds the frame's place in the strip, film, camera and lens, the soft-proof profile, the export size and the zoom.
*   **Right, the controls**: tabs **Roll** / **Frame** / **Metadata** / **Gear** / **Export** / **Scan**. **Roll** and **Frame** have a pinned **Analysis** readout; **Frame** has its own row of tabs. Only Roll and Frame change the render.

With nothing open, the canvas offers **Load Scans…** and **Take the Tour**.

Drag a panel by its top edge to float it; its pin button docks it. **Shift+H** hides and shows both panels. **Reset Panel Layout** (**⋯** menu) restores the default layout.

### The tour

**Take the Tour** (**⋯** menu, or Find) has eight chapters. A circled step is a task. With nothing open, **Load Demo Negative** opens a synthetic negative. **Read More…** opens the panel's guide. Esc ends the tour; the next one offers **Resume Where You Left Off**.

### Find

**Ctrl+K** (or **Find Control or Action…** in the **⋯** menu) searches every slider, card, action and preset, by name or by another editor's word (*contrast* finds **ISO-R Grade**). Enter opens the row on its tab; on a preset it opens **Apply Preset**.

### Before / After

**Before / After** in the **Print** menu (or `\`) splits the canvas: left is the auto baseline (same film process, crop and rotation, every creative control at default), right is your edit. It stays up while you edit. `\`, `Esc`, a frame change, a peek or the test strip closes it. Hold `\` instead to see the whole frame as the baseline until you let go.

### Reference view

**Shift+R** (or **Reference View** in the **⋯** menu) pins the current frame in a pane beside the canvas, to match other frames to it. Press **Shift+R** twice to pin again. **✕** or **Shift+R** closes it.

### Peek Negative

**Negative** in the **Print** menu (or `N`) shows the scan as loaded: not inverted, not metered, no edits; crop, rotation and flip still apply. Use it to check density, mask color and scanner clipping. It is scaled to its brightest tone, so read density from the density histogram. It stays up through Export and Metadata changes and while you drag a crop or rotation handle or Fine Rotation. Any other edit or a frame change closes it.

### Peek Embedded Preview

**Embedded** in the **Print** menu (or `P`) shows the camera's own JPEG at your crop and rotation. NegPy uses nothing from it. Scanner TIFFs and most converter DNGs have none. It closes like Peek Negative.

`Esc` closes any peek, the split or a test strip. Hold a peek key (`N`, `M`, `P`) to peek only while it is down.

### The workflow

**Roll** holds what the whole roll shares:

| Tab | Panels | What it is for |
|-----|--------|---------------|
| **Roll** | Film Mode · Frame Assembly · Calibration · Crop · Roll Analysis · Metering · Raw Decode · Optics | Film type, capture color, crop shape, roll baselines, metering, the scanning rig |

**Frame** tabs follow the pipeline order:

| Tab | Panels | What it is for |
|-----|--------|---------------|
| **Geometry** | Geometry | Crop, straighten, easel movements |
| **Exposure** | Filtration · Tone · Dodge & Burn | White balance, density, contrast, curve, local burns |
| **Look** | Lab · Alternative Processes · Toning | Chroma, sharpening, lith, cyanotype, Sabattier, toning |
| **Finish** | Retouch · Finishing | Dust, vignette, border, carrier |
| **Favorites** | Your chosen sliders · Presets | Most-used controls, saved edits |
| **History** | Work prints · Edit history | Named versions, undo trail |

Tabs that do not change the render:

| Tab | Panels | What it is for |
|-----|--------|---------------|
| **Export** | Export settings | Format, size, color, batch |
| **Metadata** | Archival metadata | Camera, lens, film |
| **Gear** | Gear library | Cameras, lenses, films, processes, scan setups |
| **Scan** | Scanner · the scanner's cards · Output | Direct capture from a film scanner or camera |

On a slider, click the value to type one, drag the name to scrub (**Shift** for finer steps), and double-click or **Ctrl**+click the track to reset. On a dropdown, **‹** and **›** step to the previous or next choice; click the name for the full list. A **dot** on a panel header or tab marks a non-default value. Each panel header has a **reset** action and an **ⓘ** that opens this guide there.

### What carries to the next frame

An unedited frame gets the rig and roll settings: film process, crop ratio, flips, calibration, print stock, the Lab polish and export preferences. The look (density, filtration, tone curve, toning, dodge and burn) starts clean.

**Preferences → Session & Storage → Persistent Settings…** sets which settings carry. **Carry settings between frames** is the master switch; off, new frames get bare defaults.

An edited frame keeps its look; only export and metadata settings reach it. **Reset Settings** returns bare defaults and, in a roll, sets every Roll-tab card back to **Roll**. No reset changes the scanning setup (Linear RAW, Narrowband, demosaic); **Scanning setup** sets those.

### Frame or roll: the scope pair

Each section header has **Frame** (amber) and **Roll** (red) beside its reset arrow, which resets that card alone. The lit one shows where the card's values live; click the other to move them there. Frames that are not one roll (search results, several folders) show only **Frame** until **Save as Roll…**.

On a **Roll tab** or **Metadata** card the pair is a latch. On Roll, the card follows the roll and new frames inherit it; an edit flips it to Frame. **Roll** pushes this frame's value to the roll; **Frame** pins it to this frame. Right-click the card header for **Undo Apply to Roll**, which undoes this frame's last push.

On a **frame** card (Geometry, Filtration, Tone, Lab, Alternative Processes, Toning, Retouch, Finishing), **Roll** opens the section's clone picker, to apply your changes to the selection or the whole roll. After a whole-roll apply the card reads Roll until you change a pushed setting. Right-click its header for **Copy Card Settings**; Paste (`Ctrl+V`) then pastes that card alone.

**Reset to Roll**, beside the reset arrow, shows once a card differs from the roll. It puts the roll's values back on the card as one undo step; settings the roll never carried stay.

### The tab header

Tabs with several cards (Roll, Exposure, Look, Finish, Metadata) have a bar that reads **3 of 5 cards edited**. Its buttons act on every card: reset (asks first), **Reset to Roll**, one roll picker for the whole tab, and collapse/expand.

**Pick Better** (the red-green-blue balance scale at the left of the Exposure tab's bar) tunes Filtration, Tone and Paper Response: a separate window opens — a notch smaller than the main window — with three equal frames in a row: the current edit in the center, bracketed by the two candidates (the stronger on the left, the weaker on the right). Each frame is captioned above: "A" over the stronger left, "Skip" over the center and "B" over the weaker right. A hover draws a red ring around the frame under the cursor, in the Film Strip's selection style. The control the pair tunes shows as its name above the center "Skip" caption, prefixed by the round number, and stays there the whole time; the candidate values are not shown. Click a candidate to keep it; the next pair fills the same window. A left or right pick briefly shows the applied control's name and its delta above the frames. Click the center current photo to reject both candidates and move on to the next control. A **Difference** slider widens or narrows the spread between A and B live and keeps its setting for the next pair. The first eleven rounds each propose one control in the order set by **Custom Routing** — Grade, Density, Temperature, Dye Separation, Highlights Grade, Shadows Grade, Highlights Density, Shadows Density, Preflash, Snap and Toe by default — and two consecutive rounds never propose the same control; the later rounds cycle back through them, each visit a finer step. **Custom Routing** opens a window that reorders, by drag and drop, which control each round proposes, with a Randomize toggle that shuffles that order on each run — the order you set is the one used when Randomize is off, so turning it off restores it. The **OK** button (bottom-right) ends the mode; **Undo** re-shows the previous pair; the balance-scale toggle or `Esc` also ends it, keeping the last chosen settings. Luma Range Clip, Color Clip, Crop, Crosstalk, Cast Removal, Hue Trim and the Black and White Points stay at their current values.

### Menu bar (macOS)

`Ctrl` in this guide is `⌘` on macOS.

*   **NegPy**: Preferences… (`⌘,`), About and Quit.
*   **Window**: Minimize (`⌘M`), Zoom, Close (`⌘W`), Bring All to Front, and the open NegPy windows.
*   **Help**: Take the Tour, Keyboard Shortcuts, Customize Shortcuts, the Analysis panel guide, Report an Issue, Check for Updates.

Plain-key shortcuts such as `?` work but do not show in the menus. `⌘W` on the main window closes NegPy. Windows and Linux have no menu bar.

---

<!-- panel:frames -->
## 2. Film strip (left panel)

A green **Update Available** line tops the panel when a newer release is out ([§16](#16-updating-negpy)). Below are the toolbar, the search box and two sections: **Library** (imported rolls) and **Film Strip** (open frames).

<!-- panel:library -->
### Your library

**Library** lists every **roll** you imported: a named group of frames, not a live view of a folder. **Ctrl+L** expands it.

*   **+**: import a roll (below).
*   **↻**: finds new roll folders under each parent imported with **Import Subfolders as Rolls…** and recounts each roll's frames.
*   **Discovery Filters…**: folder names that importing and **↻** skip, with all their contents, one per line (default `export`). A line matches any part of a name, ignoring case; a line with `*` must match the whole name (`raw_*`).
*   **Sort**: Name or Date, ascending or descending.

A Trichrome triplet, stitch or HDR merge counts as one photo. A roll whose folder is gone shows **folder missing**. Importing only records the folder; nothing decodes until you open the roll.

#### Importing

**+** (or the list's right-click menu) offers:

*   **Import Folder as a Roll…**: the folder becomes one roll and opens. A folder with no images of its own imports its subfolders instead. If its name matches a camera or film stock in your Gear library ("penf" finds "Pen F"), Roll Settings opens pre-filled.
*   **Import Subfolders as Rolls…**: every folder below the chosen one that holds images becomes a roll, without opening, except subfolders of a roll folder (export output). A roll is named by its path ("20260901/kentmere_400_1") under a folder row. Right-click a folder row for **Delete…**, which forgets every roll in it.

Importing never changes the folder. Reorganize on disk, then import again (or **↻**). Edits follow image content, so a moved file keeps its edit and marks.

#### Opening, renaming, deleting

**Double-click** a roll (or **Enter**) to open it in the Film Strip. NegPy asks first, since loading reads every frame (**Always load without asking** skips the prompt).

Right-click a roll for:

*   **Close Roll…** (loaded roll only): empties the Film Strip. Edits stay saved.
*   **Rename…**: renames the roll. A folder roll also offers **Also rename the folder on disk** (off by default). A cloud-sync folder (Dropbox, iCloud, OneDrive) can treat that rename as a delete and upload again.
*   **Delete…**: forgets the roll only; folder, images and edits stay. **Clear Library** in *Manage Database* forgets all rolls.
*   **Roll Analysis** (loaded roll only): runs Roll Analysis ([§10.5](#105-roll-analysis)).

#### Rolls that are not folders

A roll can also be a hand-picked set, and one photo can be in several rolls. Search the library or gather frames, then **Save as Roll…**. The set is fixed: add frames while it is loaded, or Save as Roll again.

A frame has one edit in every roll that holds it. On a frame in more than one roll, right-click → **Edit Independently in This Roll** gives it its own edit here; **Use the Shared Edit Again…** deletes that edit.

### Importing and managing files

**Nikon High Efficiency raw**: Z 8 and Z 9 NEFs in **HE** or **HE\*** use a licensed codec that NegPy cannot decode. Shoot **Lossless Compressed**, or convert with Adobe DNG Converter.

The **⋮** menu on the Film Strip header:

*   **New Roll…**: clears the film strip, to drag in frames and keep them with **Save as Roll…**.
*   **Close Roll…** (**Unload All…** when the frames are not a roll): empties the film strip; edits stay saved.
*   **Reset Roll to Defaults…**: **Reset Settings** on every visible frame, each an undo step.

The grid button on the same header opens the **Light Table** (`Shift+G`): the roll as a full-window grid for culling, with the same selection, marks and menus. **Esc** or `Shift+G` goes back.

The Film Strip button row:

*   **Add**: **Add Files…** or **Add Folder…** (or drop a folder on the window). A folder with no images of its own imports its subfolders as rolls.
*   **Hot Folder**: loads new files as they appear in the current folder, for a scanner or tethering app.
*   **Trichrome Mode** and **Half Frame Mode** are on the Roll tab's Frame Assembly card ([§10.2](#102-frame-assembly)).
*   **Apply (clone)**: copies the chosen aspects of the current frame's settings to selected frames or the whole roll. Crop and rotation stay per image.
*   **Roll Settings**: writes gear, capture, place, process and scanning metadata to the frame, a selection or the whole roll. **Load** fills the fields from a metadata preset; tick the groups to write. An empty Analog Gear is matched from the folder name.
*   **Save as Roll…**: keeps the loaded frames as a roll ([Rolls that are not folders](#rolls-that-are-not-folders)).
*   **Unload…**: drops the active frame or selection from the Film Strip. It is back the next time the roll opens.
*   **Remove from Roll…** (right-click, with a roll open): takes the frame or selection out of the roll for good, with its scene, card locks and own edit in that roll. The file and its shared edit stay. A file in the roll's own folder is skipped when the folder is opened again, until you add it back with **Add Files…**. A half-frame scan goes as one file.
*   **Show Scenes**: edges each frame in its scene's color ([Scenes](#scenes)).
*   **Sort**: Name or Date, or **Scene** once the loaded roll has one, ascending or descending.
*   **Sheet filter**: *All Frames*, *Keepers Only*, *Hide Rejected* or *Unmarked Only*. **Advance After Marking** in the same menu moves on after a Keeper or Reject mark. `Home` and `End` jump to the first and last frame.

Above both sections are the **filter box**, a **`.*`** regex toggle, a **search-library** button and, once enabled in Preferences, a **search-by-meaning** toggle. The Film Strip has a **tally** of frames, keepers and rejects, and a **thumbnail size** slider. If a filter hides everything, **Show all frames** clears it.

#### Filtering the sheet

A plain word matches the filename. `field:value` terms match frame data:

| Term | Finds |
|---|---|
| `film:portra` | frames whose film stock contains "portra" |
| `camera:"Nikon F3"` | quote anything with a space |
| `iso:>=400` | numeric fields also take `>`, `>=`, `<`, `<=` (`iso`, `frame`, `push`, `devtime`, `temp`) |
| `date:2024-03` · `date:>=2024` | by file date; a partial date is a prefix |
| `shot:1998` · `shot:>=1998-07` | by capture date from the Metadata panel |
| `place:tokyo` | by capture city, state or country |
| `devtime:>=9` · `temp:20` | development time in minutes, temperature in °C |
| `roll:` `developer:` `dilution:` `lens:` `format:` `scanning:` | the rest of the Metadata panel |
| `name:` `path:` `ext:tif` | file identity |
| `scene:beach` | frames in a scene of the loaded roll whose name contains "beach" |
| `keeper:` `rejected:` `edited:` | frames with that mark, or with a saved edit |
| `-rejected:` `-film:velvia` | a leading `-` negates any term |

Terms combine with AND (`film:portra iso:>=400 -rejected:`). Metadata fields match only once filled in the **Metadata** panel. **`.*`** makes the box a plain filename regex.

#### Searching by meaning

**Search by meaning** (Preferences → Performance) finds photos by content; it downloads a small model the first time. Turn on its toggle beside the filter box and describe the picture ("a photo of a dog"). Without an index it ranks only frames loaded this session.

#### Searching the whole library

The **search-library** button (or **Enter** in the box) searches the whole library and loads the results; with **Search by meaning** on, every indexed file. Keyword search reads only stored edits, so metadata matches only frames you filled in.

#### Indexing the library

With Search by meaning on, a **database** button beside the Library refresh button indexes every photo in your library. It runs only when you ask; an aborted run resumes where it stopped.

#### Stitching a frame from several shots

For a negative captured in overlapping pieces, select them and right-click → **Stitch Selected Frames**. NegPy aligns, matches and blends them into one composite, *a+b (Stitch)*. **Unstitch** restores the parts with their edits. **Merge Frame to TIFF Negative…** ([§10.2](#102-frame-assembly)) replaces the parts with one file.

For Trichrome, turn on **Trichrome Mode** first, then stitch the assembled frames.

#### Merging bracketed exposures (HDR)

For a slide with more range than one exposure, bracket it a stop apart, select the shots and right-click → **Merge Exposures (HDR)**. They become one frame, *a +4 (HDR)*; **Unmerge Exposures** restores them.

The merge is relative to the **reference**: the longest unclipped exposure, usually brighter than your metered shot. Choose the render exposure:

*   **Render exposure** (right-click a merged frame): pick a shot, listed in stops from the reference. Only the reference and shorter exposures are listed.
*   **Render Exposure** slider (Metering card): 0 EV (the reference) to −4 EV. It and a picked shot replace each other.
*   **Bracket middle (auto)** (default): the bracket's middle exposure, correct for a bracket centered on the metered shot.

A merge opens with **Shadows Density** above zero, from the shadow detail the bracket recovered; set it to zero to match the metered frame.

Bracket both ways, including the shot that already looks right. Longer frames (+1 to +3) extend the shadows; shorter ones (−1, −2) add render choices and keep a blown specular. To drop frames, drop long ones.

The merge exports under the first file's name with `-HDR` (`_DSC1715-HDR.jpg`).

Merging shows on Transparency frames only. B&W reversal (Scala, dr5, Fomapan R) is not supported. Merged, stitched and Trichrome frames cannot be merged.

### Triage (culling the roll)

Thumbnails are positives. Until you open a frame, it shows a quick inversion of a reduced preview. Transparencies are not inverted.

Right-click a thumbnail, or use shortcuts, to mark frames (multi-selection works):

*   **Keep**: a check badge.
*   **Reject**: a cross badge and dimming. Batch exports and sidecar writes skip it. The file never changes.

#### Reading the badges

| Corner | Badge | Means |
|---|---|---|
| Bottom-right | check | keeper |
| Bottom-right | cross, frame heavily dimmed | rejected |
| Bottom-left | *see below* | the frame is built from more than one file |
| Top-left | exclamation | the file failed to decode; click to retry |
| Top-left | small amber dot | the thumbnail is older than the frame's settings; open the frame or use **Update Thumbnails** |

The bottom-left glyph shows the frame type; its tooltip names it:

| Glyph | Frame |
|---|---|
| Two overlapping panes | a stitched composite ([§Stitching](#stitching-a-frame-from-several-shots)) |
| Three stacked bars | a merged bracket ([§Merging](#merging-bracketed-exposures-hdr)) |
| Three red/green/blue dots | a Trichrome triplet |
| A split rectangle, one side filled | one half of a half-frame scan |
| A split rectangle, both sides filled | a diptych: the whole scan, each half with its own edit |

The right-click menu also has:

*   **Copy/Paste Settings**, with or without normalization bounds (untick **Normalization bounds** in the paste picker to keep the frame's own). Paste writes every selected frame.
*   **Reset Settings**; with several frames selected, **Reset N Frames…**.
*   **Reset to Roll Settings**: **Reset to Roll** on every card that differs from the roll, as one undo step.
*   **Apply Settings…**.
*   **Sync Bounds…**: pushes only this frame's measured bounds (**Tonal span**, **Color balance**) to the selection or roll.
*   **Update Thumbnail(s)**: re-renders the selection's thumbnails. **Update Thumbnails** on the toolbar re-renders every stale one in the roll. Sharpening, chroma denoise, dust and heal repairs, demosaic, display profile and soft proof do not make a thumbnail stale.
*   **Reset Roll to Defaults…**, and per-frame export.
*   **Edit Independently in This Roll** / **Use the Shared Edit Again** ([Rolls that are not folders](#rolls-that-are-not-folders)).
*   **Show in Folder**: opens the system file manager on the frame's folder, with the file selected.
*   **Move to Trash…**: moves the selection's files and sidecars to the system Trash and unloads them; the edits stay, so a restored file keeps its edit. A file on a drive with no Trash stays. Stitches, HDR merges, trichromes and half-frame scans must be dissolved first.

#### Scenes

A scene is a group of frames in one roll shot in the same light. It has its own baseline, so its frames match each other, not the rest of the roll. Scenes need a roll (**Save as Roll…** first).

*   **Scene** (right-click menu): **Group as Scene…** makes the selection a scene; **Add to** *name* and **Remove from Scene** move frames in and out. On a scene's frames, **Analyze Scene…** runs Scene Analysis; **Rename Scene…** and **Delete Scene…** manage it. A frame is in one scene at most. Deleting a scene keeps edits and baselines.
*   **Sort → Scene**: each scene's frames as a block, frames in no scene last.

---

<!-- panel:analysis -->
## 3. Analysis readout (always visible)

The readout sits above the Roll and Frame tabs and describes the current frame as you edit. Each tab remembers whether the readout is collapsed. Top to bottom:

#### Photometric curve

The paper characteristic (H&D) curve NegPy prints through: a model of the paper, not a curves editor. Horizontal is **negative density** (scene highlights to the right), vertical is **print tone**. Steeper means more contrast; the flat ends are the toe (shadows) and shoulder (highlights). The crosshair marks the **pivot** that contrast rotates around.

With a **Contrast Mask**, a violet band shows where large flat areas print; fine detail prints on the curve. Spatial tools (dodge/burn, CLAHE) do not show.

#### The two histograms

The gray **negative density histogram** shows the scan on the curve's horizontal axis; the colored **output histogram** shows the print, black left to white right. Contrast cannot separate data on the flat toe, so move the exposure onto the steep middle.

In Peek Negative the density histogram splits into R, G, B and luminance: a spike at an edge is that channel clipping, a trace apart from the others a strong cast.

#### Curve / Waveform / Parade / Vectorscope

The selector under the chart picks the chart. **Curve** is the paper curve and histograms above. **Waveform** plots print tone across the frame's width; a red tick marks a column that reaches paper black or white. **Parade** plots the print's R, G and B side by side, each channel's tone across the frame. A neutral sits at one height in all three, so judge a cast on neutrals; on neutral or even content, a channel that drifts across the frame is uneven light or a Flat Field mismatch. While you hover, a dashed line marks the pixel's green level in all three panels. In Peek Negative the Parade shows the scan: each channel's density in the camera's own channels (−log10 of the scan after any Crosstalk unmix, 0 at sensor full scale at the top, 3.0 at the bottom). On a negative, the solid lines are the range the stretch maps onto the curve, and the number beside each channel letter is that range. Scan exposure moves the traces (0.3 per stop) but not the range, so compare scans from one camera or scanner, light and unmix by their ranges. The dotted line is the clear film base: the Film Base set on the roll, else the base read from the rebate. A red tick marks a column with pixels at sensor clip (top) or in the axis's last step (bottom). **Vectorscope** plots the print's colors on the CIELAB a\*b\* plane, neutral at the center. On a color frame, a cluster off the center is a cast; on a toned B&W print, a toner draws an arm toward its hue.

*   **Parade Overlay** (layers icon): R, G and B on one plot, white where they agree. Grayed out under any Color vision choice but Standard (§15), and on the scan unless Base Relative is on.
*   **Base Relative** (ruler icon, Peek Negative): moves each channel so its film base, else its thin line, sits at the top, so the channels and the hover line compare at one height. The caption names the zero.

#### Lin / Log

Sets the count scale of the histograms, the waveform and the parade: **Log** shows thin tails (to find clipping), **Lin** the bulk of the frame. The Parade in Peek Negative keeps its own scale, **Log** by default, because the scan's thin and dense ends are small areas. On the Vectorscope it sets the chroma scale: **Log** (the B&W default) opens up the near-neutral center where a toner sits, **Lin** (the color default) keeps a color frame's spread in proportion.

#### Clipping triangles

R, G and B triangles in the top corners show crushed shadows (left) or blown highlights (right) once a channel passes 0.5% of the frame. One channel alone is a cast, not an exposure problem.

#### Clipping Overlay

The half-circle icon in the Analysis header marks on the photo every pixel with a channel at paper black or paper white.

#### Zone shading and zone ticks

The warm and cool washes mark the toe and shoulder, where separation is lost. The ticks on the left edge are Adams zones I to IX.

#### Step wedge

A 21-step wedge printed through the current curve, alternative process and toning. Patches that merge into black or white are lost tones; the brackets mark the usable span.

#### Zone strip

Eleven cells on the Adams scale (**0** paper black, **V** 18% mid-gray, **X** paper white), each zone rounded to the nearest as the Zone Overlay counts it; opacity shows how much of the frame lands in each. The two cells at each end turn **red** when shadows block up or highlights blow. Click a cell to start a Zone placement. The strip is disabled on a slide or a Positive frame.

Double-click the strip, or press `Shift+Z`, to toggle the canvas **Zone Overlay**. While it is up the canvas toolbar shows its own row: the look, a legend with each zone's share of the frame, and **✕** (or `Esc`) to close it. **Color** tints each zone over the photo (I–IV cool, V gray, VI–IX warm, in your Color vision palette) and shows paper black and paper white as solid black and white; **Grayscale** draws the print in eleven gray steps; both number the larger regions. **Grid** lays a coarse grid over the photo with one numeral per region, red at paper black and paper white. Hovering the frame outlines the pointer's zone and rings its legend cell. Hovering a legend cell or a strip cell shows that zone alone, the rest of the frame dimmed, with the Zone Overlay on or off; the highlighter icon in the Analysis header turns this off for the strip.

#### Probe

A spot densitometer. Hover the image to read the print density, its zone and ΔD per channel. ΔD is density above the thin end of the normalization window. On a negative that end is the detected thin bound plus the Black Point offset, or the roll's bound under Roll Analysis; the rebate reads below zero. On a slide it is decoder white plus the White Point offset. **Probe at Cursor** (**Overlays**, or `Shift+I`) shows the same reading in a chip beside the pointer.

#### Zone placement

Works like an enlarging analyzer: click a zone on the strip, then a spot on the photo, and NegPy solves the print so the spot prints on that zone. Up to three pins; − and + trim each pin's **target** in thirds.

*   One pin solves Print Density.
*   Two pins (usually shadow and highlight) solve Print Density and ISO-R Grade.
*   A third pin adds **Shadows Grade**, **Highlights Grade** or **Snap**, whichever can move that tone.

The result is a preview until **Place zones** (or **Enter**) commits it as one edit, which turns off Auto Density (and Auto Grade with two or more pins). **Esc** discards the armed zone, then the pins; **✕** removes one pin. Drag a pin to move it. An unreachable target shows the closest zone the print can make.

Pins are proofs, not edits: any other edit or a frame change removes them. Zone placement is off on an as-captured slide or a Positive frame.

#### Negative stats

Rows that measure the scan, not your edit; hover for details.

*   **Negative**: relative density range and development character: flat (≈N−1), normal, contrasty (≈N+1). Comparable across a roll; an estimate, not a densitometer reading. Hover for Dmax, Dmed and Dmin above the clear base: the one set with **Film Base** on the Metering card, else one read from the rebate, and the hover names which.
*   **Exposure**: approximate midtone in stops from neutral; positive is high-key, negative low-key.
*   **Clipping**: share of pixels crushed to black or blown to white, worst channel. Amber above 1%.
*   **Scan clip**: share of source pixels at or above sensor white. On a negative no edit recovers it: expose the scan lower. Amber above 1%.
*   **Repair**: share of the scan rewritten by IR Restore, dust detection and heals. A large value means the threshold is redrawing the picture. Amber above 5%.
*   **Gamut**: share of the frame the proof profile cannot print, while proofing. Amber above 2%.

---

## 4. Geometry tab

<!-- panel:geometry -->
### 4.1 Geometry: crop and straighten

**Crop:**

*   **Auto** (magic wand on the CROP header): the Crop card's Auto Crop ([§10.4](#104-crop)), for this frame.
*   **Ratio**: the roll's crop ratio, the same field as on the Crop card. **1:1.4142** is the ISO A paper shape (A4, A6).
*   **Crop** tool: draw a crop rectangle; with **Ratio** at **Free**, an edge midpoint resizes one axis. A manual crop turns **Auto Crop** off for this frame. **Reset** clears the crop and turns auto-crop off.
*   **Guide**: *Thirds*, *Phi Grid*, *Diagonals*, *Golden Triangles*, *Golden Spiral*, *Armature*, *Diagonal Method*, *Grid* or *Off*. The redo button rotates the guide.

**Alignment:**

*   **Crop by Default**: crops the wedge that Fine Rotation, Tilt and Swing leave, while no manual or auto crop is set.
*   **Fine Rotation** (±45°): sub-degree rotation, positive clockwise. Applied after auto-crop.
*   **Auto Skew**: squares the frame to its film and frame edges, never to the picture, so a deliberately tilted photo stays tilted. It sets **Fine Rotation**, and **Tilt** or **Swing** where both edges of that pair show.
*   **Straighten** tool: draw a line along a horizon or vertical edge to level or plumb it.
*   **Tilt and swing with reference lines** tool: drag a line along each rebate edge; with all four marked, Tilt and Swing solve to make them parallel and square.
*   **Tilt** (±15%): tips the easel about a horizontal axis to correct converging verticals. Positive stretches the top edge. The unit is percent of the frame, not an angle.
*   **Swing** (±15%): the same about a vertical axis, for converging horizontals. Positive stretches the left edge.

    Both leave a wedge along the squeezed edge: crop it or use **Crop by Default**. Crop first: on an uncropped scan a large correction pulls the rebate into the metering and darkens the print.

Lens distortion, chromatic aberration and flat field are in **Optics** on the Roll tab ([§10.8](#108-optics)).

---

## 5. Exposure tab

Four panels set light, color and contrast in the print stage of the pipeline.

<!-- panel:color -->
### 5.1 Filtration: white balance

Color timing, like enlarger dichroic filters. The **Global / Shadows / Highlights** selector applies the controls to the whole image, or biases them to low- or high-density tones.

*   **Pick WB** (eyedropper): click a pixel that should be neutral gray, and NegPy solves the CMY filtration for the selected region.
*   **Temperature**: warm-to-cool control on the magenta/yellow pair; cyan does not change.
*   **Cyan / Magenta / Yellow** (-1 to 1): Cyan↔Red, Magenta↔Green, Yellow↔Blue.
*   **Ring-around** (`Shift+F`): a 5×5 mosaic in 2cc steps to ±4cc on magenta and yellow, centered on neutral. Click a patch to keep its filtration; `Escape` clears it.

**Cast Removal** is on the Roll tab's Calibration card ([§10.3](#103-calibration-what-your-rig-does-to-the-colors)).

<!-- panel:tone -->
### 5.2 Tone: density and contrast

**Global / R / G / B** applies Grade and Split Grade to the shared curve, or as per-dye-layer trims for **crossover correction** (casts that differ between shadows and highlights).

**Automatic helpers**, in the **Auto** menu, are on by default:

*   **Auto Density**: meters each frame's midtone and anchors print brightness there.
*   **Auto Grade**: sets the grade partly from the frame's textured density range, harder when textured shadows need it to reach black (Shadow Reach). It opens the shadows of a contrasty frame (Shadow Hold) and holds textured highlights off paper white (Highlight Hold).
*   **Auto Density and Grade**: turns both helpers on or off together.
*   **Hold Black** (off, hidden on slides): a negative Shadows Density fades out toward paper black, so the shadows open and the black point barely moves.
*    **Set Targets…**: the brightness and contrast the helpers aim for, for all frames. Confirming a changed target warns that it moves every frame, including ones not yet baked, and that baking a frame first keeps it fixed; a **Don't show again** box stops the warning.
*    **Bake Auto Density/Grade**: writes the metered Auto Density and Auto Grade into the frame's settings and turns both helpers off, so the frame keeps the look the helpers chose without them and does not move when the targets change.
*   With a helper on, its sliders show what prints: **Print Density** the metered density, **ISO-R Grade** the grade the frame prints at, **Highlights Density** with the automatic burn. Moving one trims the helper; a double-click returns to the helper's choice.

**Test strip** (`Shift+T`): a 5×5 grid, Print Density rising left to right, ISO-R Grade softening top to bottom. Click a patch to keep it; `Escape` clears it. While a proof shows, the 90° **rotate** buttons and `[` / `]` turn the ladder, not the image.

**White Point** and **Black Point** are on the **Metering** card ([§10.6](#106-metering-negative--positive)).

**Exposure:**

*   **Print Density** (0.0 to 2.0): overall brightness (enlarger time). Lower is brighter.
*   **ISO-R Grade** (50 to 180): contrast as paper ISO-R. R110 is about grade 2; lower R is harder. In R/G/B mode a **Grade** trim rotates one layer's slope about the midtone.
*   **Shadows Density** / **Highlights Density** (±1.0 ΔD): brighten or darken only the shadows or highlights, within paper black and white. With Auto Grade on, each shows its automatic share. They also work on slides.
*   **Shadows Grade** / **Highlights Grade** (split grade, ±50 ISO-R): local contrast in the deep shadows or highlights.
*   **Preflash** (0 to 1): an even flash over the sheet, as a fraction of the paper's threshold exposure. It pulls highlight detail off paper white and softens the print slightly. Hidden on slides.
*   **Diffusion** (0 to 1, hidden in Transparency): a diffuser under the enlarger lens (a Softar, a stocking). Shadows bleed into highlights and edges soften; flat areas print as before. Grays out in R/G/B mode.
*   **Contrast Mask** (±0.5, hidden in Transparency): a blurred mask sandwiched with the negative. Positive compresses the range so a harder grade fits and fine detail stays; negative expands a negative too flat for Grade (past about −0.4, highlights clip). Past about ±0.2, strong edges get a halo.
*   **Mask Spacer** (2 to 6%, default 4%): the gap between mask and negative. Thicker masks only broad masses; thinner reaches into detail and hazes shadows next to bright areas. Both mask controls gray out in R/G/B mode.

<!-- panel:paper -->
### 5.3 Paper Response: the print curve

The paper's characteristic curve. Its own **Global / R / G / B** selector applies the curve controls to all layers or as per-dye-layer trims.

*   **Print stock**: a bundled paper or print film (RA4 papers and the Kodak 2383 and 2393 print films in Color Negative, B&W papers in B&W Negative) that sets the curve and, on color stock, the colors its dyes make, from the datasheet or, for the three Fujicolor Crystal Archive stocks (standard, HD and Pearl), from Fujifilm's measured Frontier profiles: less saturated than *Digital*, with reds leaning orange and cyans leaning blue, and grays that drift slightly as the real dyes do. The other controls trim on top. *Digital (Adobe RGB)*, the default, is an idealized print: NegPy's own curve and perfect dyes. *Digital (Pakon Minilab)* is that curve with a Pakon minilab's colors. Each B&W paper has its own lith color (§6.2).
*   **Dye Separation** (0.5 to 1.5, hidden in B&W Negative): saturation through the paper's dyes, easing off at toe and shoulder. 1.0 is off; lower values pull toward neutral. **Chroma** (Look tab) scales color evenly instead.
    *   **Separation Damping** (0 to 1): higher keeps the push on muted color and eases it on saturated color. Grayed out at Dye Separation 1.0.
*   **Paper White**: simulates paper base density, so whites print at about 0.93.
*   **Paper Black**: shows the paper's slightly milky Dmax. Off (default) applies black-point compensation.
*   **Snap** (-0.5 to 0.5): midtone gamma; paper white and black stay put.
*   **Toe** (-1 to 1) + **Toe Width** (0.1 to 5): shadow roll-off. Positive lifts shadows; negative deepens them and, with Paper Black off, reaches exact black. Width sets how far the knee reaches.
*   **Shoulder** (-1 to 1) + **Shoulder Width** (0.1 to 5): highlight roll-off. Positive compresses highlights; negative extends them and can clip.

In R/G/B mode these become per-layer trims: **Toe** / **Shoulder** (±1), **Toe Width** / **Shoulder Width** (±2), **Snap** (±0.5), **Dye Separation** (±0.4).

<!-- panel:local -->
### 5.4 Dodge & Burn: local exposure

Draw masks to lighten or darken only those areas. On a **Slide** the panel grays out; the frame keeps its masks.

*   **Draw** (Draw Mask): click to place vertices; double-click, Enter or a click near the start closes; Esc cancels. To edit, drag a vertex, click an edge "+" to add a point, or right-click a vertex to delete it.
*   **Oval**: drag out an oval. The handles move, stretch and tilt it.
*   **Card** (Card Edge, the graduated burn): drag from the full-exposure edge (solid line) to where it fades out (dashed). The gap is the softness, so Feather does nothing on this shape. A tilted Card Edge usually must start past the corner it burns.

*   **Mask list**: each mask's shape, Dodge, Burn, Grade or Flash, and values. Its icons enable, invert (act outside the shape), show the outline of, or delete the mask.
*   **SELECTED MASK**: the controls below act on the mask selected in the list.
*   **Burn** (-2 to 2 stops, default 0): positive burns (darker), negative dodges (brighter).
*   **Feather** (0.0 to 0.15): edge softness, as a fraction of the frame's short side.
*   **Grade** (-40 to 40 R): the mask's own contrast, in ISO-R points off the frame's Grade, negative harder (burn a sky at −20 R). Overlapping grades add, within R50…R180.
*   **Flash** (0 to 1, default 0): pre-exposes the paper under the mask, like the Tone card's Preflash: the region's highlights compress without the density a burn adds. Overlapping flashes add. Off on a tone-limited mask.
*   **Tone Limit** (*All*, *Highlights*, *Shadows*) with **Tone Zone** (0 to 10, in thirds, default 6) and **Tone Softness** (⅓ to 3 zones): limits the mask to tones lighter or darker than a print zone, so a sky burn on *Highlights* at VI stops at the skyline. Up to four tone-limited masks per frame.

**Printing Notes** (Export tab, or **Shift+N**) makes a marked-up work print: each mask outlined with its number and value in stops, and a card with the paper, exposure, grade, filtration, curve and dodge/burn list. Burns are hatched. Numbers are exposure, not brightness: +1.00 st is `Burn +1`, and values snap to ⅓, ½ and ¼ when close. A local **Grade** shows as `Burn +1 @ R95`, a tone limit as `Burn +1 on ≥VI`. Hidden outlines still show; disabled masks do not.

---

## 6. Look tab

<!-- panel:lab -->
### 6.1 Lab: polish and detail

What a lab scanner (Frontier or Noritsu) does automatically.

**Color** (hidden in B&W Negative):

*   **Chroma** (0.0 to 2.0): even color scale after decode; 1.0 is unchanged, 0 grayscale. For print-like saturation use **Dye Separation**.
*   **Skin Protection** (0.0 to 1.0, default 0.5): caps skin-hued chroma; 1.0 leaves skin matte, 0 is off. Red coats, sunsets and brick stay out; wood, tan leather and sand soften with it.
*   **Chroma Denoise** (0.0 to 5.0): removes color speckle finer than this radius (px of the full-size scan) and keeps sharp color edges; luminance grain stays.

**Detail:**

*   **Sharpening** (0.0 to 1.0): amount, on lightness only, so no color halos.
    *   **Method**: *Unsharp Mask* (edge contrast) or *Deconvolution* (Richardson-Lucy, reverses the scanner's blur).
    *   **Radius** (0.5 to 3.0 px): blur width in full-size pixels; judge it at 100% zoom. For a resized export, add the Export panel's **Sharpening**.
    *   **Masking** (0.0 to 1.0): limits sharpening to edges to protect sky, skin and grain.
*   **CLAHE** (0.0 to 1.0): local contrast without blown highlights or crushed shadows. Near 1.0 it can look cartoonish.

**Effects:**

*   **Glow** (0.0 to 1.0): lens bloom across all channels.
*   **Halation** (0.0 to 1.0): red glow from light that scatters back through the film base, highlights only.

<!-- panel:altproc -->
### 6.2 Alternative Processes

Pick **None / Lith / Cyanotype / Sabattier**; only that process's controls show. B&W Negative only, off by default.

#### Lith

Heavily over-exposed lith paper in dilute developer, snatched part-way: creamy warm highlights and an abrupt drop into sooty blacks. The Exposure tab's print stock sets the color: *Digital*, the Ilford and the Kodak papers are almost colorless, Fomatone gives peach and olive. In Toning only Selenium and Gold stay (§6.3).

*   **Exposure** (0 to 5 stops, default 2): over-exposure; real lith uses two to four stops. More gives warmer, more colorful highlights and softer gradation.
*   **Snatch Point** (0.0 to 1.0, default 0.55): time in the developer. Higher gives deeper, colder blacks; lower stays high-key and warm with weak blacks.
*   **Abruptness** (0.0 to 1.0, default 0.6): how suddenly shadows go black. High blocks up the next zone down; low rolls off gently.

#### Cyanotype

UV contact print on iron-salt paper: Prussian blue instead of black, with green highlights. Its short density range clips a normal negative at both ends. Chemical toners gray out; use Bleach and Tannin. Split toning still works.

*   **Sensitizer** (Classic or New, default Classic): *Classic (Herschel)* tops out at a light blue with a green highlight stain; *New (Ware)* goes deeper and cleaner.
*   **Exposure** (-2 to 4 stops, default 0): UV time. More moves more of the scale into blue.
*   **Exposure Scale** (0.8 to 2.8 log D, default 1.4): the printable density range and so the contrast; shorter is more contrasty. Traditional cyanotype is about 1.0 to 1.2, Ware's new about 2.4.
*   **Bleach** (0.0 to 0.5, default 0): washing soda; removes blue, highlights first.
*   **Tannin** (0.0 to 0.5, default 0): tea or tannic acid; turns bleached iron brown. Bleach first for full brown; Tannin alone for split blue-brown.

#### Sabattier

A white-light flash part-way through development: the light tones reverse and fold back toward gray, the dense ones hold, and a light Mackie line runs along every dark-light edge. Every toner works on it.

*   **Strength** (0 to 2, default 1.3): the second exposure's length. Below 1 the light tones flatten toward the fold; above 1 they reverse.
*   **Re-exposure** (0.1 to 0.9, default 0.45): where the fold sits, as a fraction of the paper's Dmax. Tones denser than it hold.
*   **Agitation** (0 to 1, default 0.7): rocking the tray after the flash. A still bath (0) draws wide Mackie lines; constant agitation (1) draws none.

---

<!-- panel:toning -->
### 6.3 Toning

Chemical toners (B&W Negative only) and a split tint (any mode). On a lith print only Selenium and Gold stay enabled, and they bite harder. With Cyanotype all six gray out.

**Chemical Toning**: sequential baths in the order shown, each 0.0 to 2.0.

*   **Selenium**: deeper blacks, cool eggplant shadows. On lith: green-black shadows to magenta, with a strong Dmax lift.
*   **Sepia**: warms highlights first, reaching further into the mids with strength; the deepest shadows stay black.
*   **Gold**: blue-black on untoned silver; over sepia, orange-red highlights. On lith: all densities toward blue-violet.
*   **Iron Blue**: Prussian-blue shadows to navy blacks.
*   **Copper**: pink to brick-red, with the classic Dmax loss.
*   **Vanadium**: greens mids and highlights; deep shadows stay black.

**Split Toning** (all modes): an additive Lab tint that keeps grain and detail.

*   **Shadow Strength** (0.0 to 1.0), with **Shadow Hue** (0 to 360°) under it.
*   **Highlight Strength** (0.0 to 1.0), with **Highlight Hue** (0 to 360°) under it.

---

## 7. Finish tab

<!-- panel:retouch -->
### 7.1 Retouch: dust, hairs, scratches

Spotting, as with a brush on a finished print. Marks are found by local contrast, by the scanner's IR channel or by hand, and the three stack. Each mark is rebuilt from the clean film around it, grain included.

**Overlay** (Off / Marked / IR): shows detections. IR needs an IR plane.

**Optical Removal** finds specks and hairs on the visible scan, without IR:

*   **Spot Threshold** (0.01 to 1.0): the bar for specks, against the film's grain. Lower catches more, with more false positives; 1.0 turns speck detection off.
*   **Hair Threshold** (0.01 to 1.0): the same for hairs and long thin marks. Lower it if a hair across busy detail is missed; 1.0 turns hair detection off.
*   **Size** (2 to 8 px): max spot radius.
*   To protect detail, right-drag on the canvas to paint an exclusion band, or right-click → **Exclude from Optical Removal** for one spot. Toggling **Optical Removal** clears every band.
*   The cursor button beside **Optical Removal** makes a plain right-click exclude the spot; the canvas menu is then out of reach while removal is on.

**IR Removal** uses the scanner's infrared channel, and is enabled only when the scan has one.

*   **IR Threshold** (0.05 to 0.95): lower catches more.
*   **Method**: how the film under a defect is rebuilt.
    *   **NegPy** (default): divides semi-transparent dust back out and fills opaque cores from the clean film around them.
    *   **OpenICE**: keeps picture detail under a speck and gives solid defects synthetic grain. Better on fine detail but less proven across scanners: compare both on a frame you know.
*   IR is read from 4-channel TIFFs and DNGs (VueScan, NegPy's scans), SilverFast iSRD TIFFs, 64-bit **HDRi RAW DNGs** and `_IR.tif` sidecars. B&W and Kodachrome frames are skipped, as they block infrared like dust.

**Manual Heal**. The brush marks a search area, not a stamp: only pixels that stand out from the film around them are rewritten, so paint generously.

*   **Heal**: click dust spots to paint them out, or drag over a run of them.
*   **Scratch**: click points along a scratch or hair, then double-click or press Enter. Esc cancels, Backspace removes the last point.
*   **Line** (Transport Line): for long straight transport scratches. Click once on the scratch to trace and repair the whole line.
*   **Line Sensitivity** (0.05 to 0.95): lower catches fainter lines with a wider band; raise it if a line picks up film on either side. Applies to placed lines too.
*   **Brush Size** (2 to 64 px): diameter of the heal, scratch and exclusion brushes, for new strokes. `Alt`+scroll on the canvas or pinch sets it.
*   **Undo Last** / **Clear All**: remove the last or all manual heals and traced lines; auto-detected dust stays.

Right-click a stroke, scratch or line on the canvas to delete it.

**Clone** copies film from another area over a defect the heal cannot rebuild.

*   **Clone**: the first click picks the area to copy from; then click or paint over the defect. The source follows the brush at a fixed offset. Uses **Brush Size**.
*   **Set Source**: the next click picks a new area to copy from (or `Alt`-click with the Clone tool).
*   **Match Tone** (default on): keeps the source's grain but takes the destination's brightness and color.
*   **Strength** (0 to 100%, default 100%): how much of the source covers the destination.
*   **Feather** (0 to 100%, default 50%): the soft edge, as a share of the brush radius; 0 is hard.
*   **Undo Last** / **Clear All** on the CLONE header remove the last or all strokes. Each stroke keeps the settings it was painted with.

<!-- panel:finish -->
### 7.2 Finishing: vignette, carrier, border

How the print is presented, at the end of the pipeline.

**Vignette** (printer's edge burn):

*   **Burn** (-2.0 to 2.0 stops): positive darkens the edges, negative lightens them. 0 is off.
*   **Size** (0.0 to 1.0): falloff radius, from tight in the corners to spread into the frame.
*   **Roundness** (0.0 to 1.0): 0 is radial (lens-like), 1 is a rectangular card burn along the print edges.

**Filed Carrier**: the clear rebate of a filed-out carrier prints black, framed by unexposed paper, through the frame's own curve and filtration.

*   **Width** (0.0 to 5.0 mm): black frame thickness. 0 is off.
*   **Roughness** (0.0 to 1.0): how raggedly the outer edge was filed.
*   **Flare** (0.0 to 1.0): light off the filed bevel fogs the paper just outside the edge. 0 is off.
*   **Corners** (0.0 to 1.0): how far the filed corners round off.

**Border:**

*   **Width** (0.0 to 2.5): thickness as a fraction of the image. 0 is no border.
*   **Bottom Weight** (1.0 to 2.0): thickens the bottom, for window-mat proportions.
*   **Color** (*Paper White* or *Custom*): *Paper White* tints the mat with the toned paper white; *Custom* uses the swatch color.

---

## 8. Favorites tab

The sliders you use most, in one place; empty until you fill it.

*   **Edit Favorites**: tick sliders on the left, drag them into order on the right, then press **Apply**.
*   They are the same controls as in their home panels; a favorite hides when its original does.

<!-- panel:presets -->
### Presets

Saves and recalls edit settings by name.

*   **Apply** (or double-click a preset): applies the selected preset to the current image.
*   **Save…**: pick which of the current settings to store as a new preset.
*   **Pen** and **Trash**: edit or delete the selected preset.

---

## 9. History tab

Two lists: the versions you chose to keep, above the record of every change.

### Work prints

A **work print** is a named version of this frame, like the test prints kept on the way to the final one.

*   **Save Work Print** (**Ctrl+Shift+S**) keeps the current edit under a name.
*   **Click** one to make it live. That is an edit, so **Ctrl+Z** restores the previous state.
*   **Right-click** for **Export This Version…**, **Rename…** or **Delete**.
*   **Load Metadata** (off by default; a global setting, the same for every frame and roll) restores a work print's saved metadata when you load it; off keeps the metadata on screen now and ignores the print's.

Work prints are never pruned. They live in the database, not in `.negpy` sidecars.

### Edit history

Every edit step, newest on top; NegPy keeps the last 100.

*   **Click** a step to jump to that state.
*   **Right-click** → **Export This Version…** to export a past state.

---

## 10. Roll tab

Every card here is shared by the roll, with the scope pair of [§1](#frame-or-roll-the-scope-pair): an edit makes the card **Frame** until it matches the roll again, and a push to the roll skips frames that locked the card. The line above the cards names every card this frame overrides.

<!-- panel:film -->
### 10.1 Film Mode

Decides which cards apply: **Color** (C-41 negative), **B&W** (negative) or **Slide** (transparency, E-6 and similar). The wand button detects the mode when a file loads.

**Positive** (default off, **Slide** only): for a source that is already a positive (a scanned print, another app's export). NegPy reads its embedded profile (sRGB if none) and skips metering and inversion, so the tone controls shape the image directly.

<!-- panel:assembly -->
### 10.2 Frame Assembly

How the files become frames. Neither toggle has a scope pair; each roll remembers its own state.

#### Trichrome

*   **Trichrome Mode** (three-exposure narrowband capture): assembles each frame from a red, green and blue exposure, grouped by capture time (else filename order), so shoot each frame's three back to back. Shots that do not group stay separate, for pairing by hand.
*   **Edit Triplet…**: the **Edit RGB Triplet…** dialog for the current frame. **Align channels (sub-pixel)** registers green and blue to red, removing color fringes.
*   **Merge Frame to TIFF Negative…** / **Merge Selected to TIFF Negative…** (right-click): replaces each triplet or stitch with one 16-bit linear TIFF negative beside its first source (`<red name>_RGB.tif`, `<first part name>_STITCH.tif`). It renders the same and keeps the edit, marks and scene. A merged stitch has flat field and sensor correction baked in and cannot be unstitched. **Move each merged frame's source files to the Trash** (on by default) trashes the sources once the file is verified. Brackets, slides and LinearRaw DNG sources are skipped.

A new roll, or a batch that is not one roll, takes the Trichrome Mode last chosen.

#### Half Frame

**Half Frame Mode** splits each scan into two frames, for half-frame cameras; each half is edited and metered on its own. Turning it on detects the gutter, its direction and the film crop on every loaded scan. A batch that is not one roll cannot use it.

*   **Adjust…**: drag the green box to crop and the orange line to set the split; pick **Split direction** (*Vertical* or *Horizontal*) and **Cut thickness** (the separator band to discard). **Auto-detect** finds all three again. **Apply** applies to the current, selected or all frames (all sets the roll default).
*   **Detect All**: reruns the batch detection.
*   **Unsplit** (on a split frame): makes it one plain frame and deletes both halves' edits.

Right-click a half for **Adjust Split for This Frame…** and **Reset Split to Roll Default**. Heals and masks stay anchored to the film through a resplit.

Turning Half Frame off keeps edited halves as a **diptych**: both halves side by side with their own edits. A diptych exports as `<name>-DIPTYCH`, a long-edge size applying to each half; turn Half Frame on to edit it.

Half Frame never splits a frame assembled from several files.

<!-- panel:sensor -->
### 10.3 Calibration: what your rig does to the colors

These controls correct the capture, not the look.

**Capture**:

*   **Scanning setup**: a wizard (*how do you scan?*, and for a camera *what light source?*) that sets Camera WB or Linear RAW. It runs once after the first-launch tour; open it again when your rig changes.
*   **Camera WB / Linear RAW** (default Camera WB): how a RAW file decodes. *Camera WB* applies the camera's as-shot white balance; *Linear RAW* decodes with neutral multipliers. Locked for a Trichrome triplet.

What the wizard sets:

| Capture | Light source | Decode |
| --- | --- | --- |
| Digital camera | White light (lightbox, CRI LED panel) | Camera WB |
| Digital camera | Narrowband RGB (Scanlight, RGB LED) | Linear RAW |
| Film scanner | any | Linear RAW |

Applying it sets the defaults for new files and rewrites every edited frame in the session (Ctrl+Z per frame).

**Single-Shot Narrowband Calibration**: for single-shot camera scans under narrowband light, where each color leaks into the others through the sensor's filters. Needs **Linear RAW**; grayed out on a Trichrome triplet. **Re-run Roll Analysis** after changing it.

*   **Profile**: the sensor matrix. Custom `.toml` matrices go in `<Documents>/NegPy/sensor/`.
*   **Method**: *Two-Scale* (default) subtracts the leak like *Linear* where the calibration can be trusted, and stops the speckle where a color is mostly leak (neon, deep blue). *Linear* subtracts the leak exactly and prints those colors speckled and fully saturated. *Density* has no speckle, but strong colors are slightly less vivid. A scan with a clipped film base uses Linear.
*   **Calibrate**: builds a profile from three bare-light R/G/B exposures. Pick them as files, or press **Capture from Camera…** to shoot them with a tethered camera and a Scanlight, no film in the holder. A Single Capture Scanlight preset can save a profile during its own calibration.

CAMERA_SCANNING.md has a workflow for each way to build and assign a profile.

**Crosstalk** (hidden in B&W Negative): a channel unmix on the densities before inversion. Dyes, light and sensor all mix the channels, so a matrix describes your whole scanning setup. **Re-run Roll Analysis** after changing it.

*   **Matrix**: matrices for the current film process, grouped by source. *Generic C41* is built in; custom `.toml` matrices go in `<Documents>/NegPy/crosstalk/` (see [CROSSTALK.md](CROSSTALK.md)). The slider button opens the editor; **+** makes one for the current process.
*   **Strength** (0.0 to 1.0): how much unmix to apply.
*   **Measure from Chart…** (the swatch button, Color Negative only): measures a matrix from a frame of a SpyderCheckr 24 or ColorChecker 24 shot on the film. Drag the four dots onto the corner patches, in any order, with the chart any way up; NegPy fits the six mixing terms through the frame's own render on the Digital print stock, saves a *Measured* profile, applies it at full Strength and shows each patch's color difference to its target (ΔE00 without lightness) on the frame.

> The bundled film matrices, marked *(approx)*, come from spec sheets and describe the dyes alone. They fully correct only a capture that reads each dye cleanly (a Narrowband Scanner, Trichrome, a calibrated single-shot narrowband rig); for broadband light and a Bayer sensor they are a starting point.

> No Transparency matrix ships. On slides, Matrix and Strength stay disabled until you make one with **+** or add a `.toml` with `process = "Transparency"`. Treat unmixing on a slide as a color-separation control.

To tune a matrix, adjust its six off-diagonal terms in the editor and save it as your own, named after the *combination* ("Gold 200 + Spectracolor"). Working profiles are welcome [upstream](CROSSTALK.md#contributing-a-matrix).

**Dye balance: Cast Removal** (0.0 to 1.0, color only): removes an overall color cast by balancing each layer against the frame's own grays, so neutrals stay neutral from shadows to highlights. On a negative, leaves, sky and water never count as gray, so a forest keeps its greens and a clear sky stays blue. It starts at 1.0 on Color Negative and at 0 on Transparency, for a faded slide's crossover.

**Light source:**

*   **Hue Trim** (-30° to 30°, default 0): rotates every hue, for narrowband or odd-phosphor lights that turn yellows orange and greens olive. Judge it on a known color; leave it at 0 for broadband light. It carries to the next file.

#### Narrowband and slides

Single-Shot Narrowband Calibration does not apply to Transparency; it stays grayed out with its values for the next negative. For slides on a narrowband rig, use **Hue Trim**.

<!-- panel:autocrop -->
### 10.4 Crop

The roll-wide crop shape, and what the frame detector looks for. Each frame's rectangle is its own, drawn or found in **Geometry** ([§4.1](#41-geometry-crop-and-straighten)).

*   **Ratio** (default `Free`): `Free`, `1:1`, `3:2`, `4:3`, `5:4`, `6:7`, `7:5`, `65:24`, `16:9`, `16:10`, `11:8.5`, each auto-oriented. On `Free` auto-crop uses the detected format; a ratio forces every frame to it.
*   **Detect**: snaps the ratio to the closest standard.

**Auto Crop**:

*   **Mode**: *Image only* (exposed area) or *Film edge* (full film, with rebate and sprockets).
*   **Crop Offset** (-5 to 100 px): insets the detected edge; negative bleeds slightly outside.
*   **Rebate Trim** (0 to 150%, *Image only*): 0% stops at the film edge, 100% at the detected image edge, above 100% cuts into the picture to clear a white border.
*   **Auto Crop** (magic wand): detects the frame edge and crops to it; off clears the crop. New frames start with the last value set. A hand-drawn crop turns it off for that frame alone. **Roll** asks before it changes the roll's Auto Crop for frames that follow the roll.
*   **Square Before Crop** (default off): Auto Crop first squares the frame (Fine Rotation, Tilt and Swing, as **Auto Skew**), then crops. It squares once when Auto Crop is set or Square Before Crop is turned on; a frame with its own Fine Rotation, Tilt or Swing keeps them. **Auto-crop all frames** straightens rotation either way; this adds Tilt and Swing.
*   **Auto-crop all frames** (*Image only*): crops every frame in the Film Strip together, calibrating weak detections from confident ones, and straightens each as **Auto Skew** does. Hand-drawn, Film-edge and ambiguous frames, and frames with their own Auto Crop off, are left alone.
*   **Mixing scans**: allowed. A frame that reads its own edge keeps its crop; a frame with no edge takes the roll's width and tilt. Frames from one camera, holder and format pool best.
*   **When auto-crop leaves a frame alone**: the scanner bed must be the brightest thing in the scan. A slide with highlights as bright as the bed, sprocket-exposed film, or a neighbor frame filling more than a tenth of one side stays uncropped. Crop by hand, or use *Film edge* and trim in.

A change re-detects every following frame cropped by **Auto**. Hand-drawn crops stay, but **Ratio** reshapes them around their center.

<!-- panel:baseline -->
### 10.5 Roll Analysis

Meters the roll once and shares the result, so frames of one film match.

*   **Use average: Luma**: takes the picked roll's tonal range; color stays per frame. Disables Luma Range Clip.
*   **Use average: Color**: takes the picked roll's color balance; tonal range stays per frame. Disables Color Clip. Luma and Color on gives a consistent roll; both off gives per-image auto-exposure.
*   **Use average: Cast** (Color Negative only): takes Cast Removal's gray balance from the roll or scene analysis, so frames under one light render alike. An analysis turns it on, except on frames far from the rest.
*   **Baseline**: names the roll, scene or frame the baseline came from, and warns when there is none or when an older version of the analysis measured it.
*   **Rolls** (picker): every roll in your library; a ticked roll has a saved baseline. Defaults to the loaded roll. A baseline from an older analysis shows an amber hint until you run **Reanalyze**.
*   **Reanalyze** (also in the Library, [§2](#2-film-strip-left-panel)): averages density and color balance over the loaded roll, without scene and locked frames, and saves the roll's baseline. Frames far from the rest keep their own. Run **Auto-crop all frames** ([§10.4](#104-crop)) first for consistent crops.
*   **Use This Frame**: saves this frame's bounds as the roll's baseline.
*   **Scenes**: the loaded roll's [scenes](#scenes). **Analyze** runs Scene Analysis (Reanalyze over the scene only), **Select** selects its frames, and **Delete** forgets it. An amber ✓ marks a scene baseline from an older analysis.

<!-- panel:process -->
### 10.6 Metering: negative → positive

How this frame is measured into a positive's tonal bounds.

**Slides** (Transparency) render as captured, through the camera's color matrix and a fixed tonal window, as in Photoshop or Darktable.

*   The paper controls and the normalization tuning hide. The tone controls, R/G/B trims and white balance form a transfer curve, neutral at defaults.
*   **Auto Density** and **Auto Grade** start off on a slide; turn them on to meter a faded or expired one. On a merged bracket **Render exposure** sets the exposure instead.
*   Lightroom mapping: **Exposure** → Print Density (lower is brighter), **Contrast** → ISO-R Grade (180 is softest), **Shadows** → Shadows Density, **Highlights** → Highlights Density. Positive adds density, so negative Shadows Density opens shadows. **Whites** and **Blacks** have no equivalent.
*   A source with no camera matrix (a scanner TIFF, a JPEG) passes straight through. An Input ICC in Export replaces the camera matrix.
*   **Camera WB / Linear RAW** is grayed out except with **Positive** on. **Single-Shot Narrowband Calibration** is grayed out ([Narrowband and slides](#narrowband-and-slides)).

**Analysis** sets where the black and white points are metered.

*   **Reanalyze Frame**: measures this frame again from its current crop, buffer and region.
*   **Draw Region**: draw a freehand region to meter *exactly* that area, in place of the buffer. Double-click inside to confirm. The **×** on the button clears it.
*   **Film Base**: click the clear rebate once to set the roll's film base. The **Negative** row's hover then reads Dmax, Dmed and Dmin above it on every frame of the roll scanned at the same exposure and light as the one you clicked. A click on edge print, a perforation or bare light is refused. Without a base, the rebate is read when the scan shows a film edge, and the hover says so. The **×** on the button clears the base.
*   **Analysis Buffer** (0.0 to 0.25): insets the measurement window so rebate, sprocket holes and scanner borders do not skew it. With no crop, the window is the picture Auto Crop would keep when the film sits in a holder or on bare light, so the holder never sets the exposure.
*   **Clipping**: the print's shadow and highlight clipping shares, as in the Analysis panel's row.
*   **Lock Bounds**: freezes this frame's bounds against crop and slider changes and against every Roll Analysis run.
*   **Luma Range Clip** (-100 to 100): how tightly the black/white-point span is set. Positive tightens it, for dense or fogged negatives; negative pushes the bounds *outward*, for lifted blacks and unclipped highlights.
*   **Color Clip** (-100 to 100): the per-channel color-balance clip (orange-mask removal). Positive tightens; negative samples nearer the extremes.

**White / Black Point** (-0.25 to 0.25), with a **Global** / **R** / **G** / **B** selector: offsets on the detected bounds; a positive white point brightens, a positive black point lifts blacks. In R/G/B they trim each layer's Dmin and Dmax. On a slide they offset its fixed window; elsewhere **Lock Bounds** disables them.

<!-- panel:demosaic -->
### 10.7 Raw Decode: turning the sensor mosaic into pixels

The demosaic fills in the two colors each photosite misses, which affects sharpness and grain. Bayer and X-Trans RAW only; only algorithms in your LibRaw build are listed.

**Demosaic**:

*   **Preview** / **Export** (default **Auto**): *Auto* is a fast half-size decode on screen and AHD for export. **AHD** is balanced, **VNG** smooth, **PPG** fast with clean edges, **DCB** and **DHT** favor fine detail, **AAHD** softens edges to suppress artifacts.

**Highlights: Recovery** (Transparency only, default **Off**): rebuilds a clipped highlight on a camera RAW. **Off** leaves it flat (magenta if one channel clipped first); **Blend** makes a plausible neutral, right for sun, sky, chrome or glass; **Reconstruct** is stronger and can misjudge a saturated highlight. Grayed out on rendered files and merged brackets.

<!-- panel:optics -->
### 10.8 Optics

The scanning optics: one lens correction and one light correction for every frame of the rig.

#### Lens Correction

*   **Embedded** (the file's own lens profile for DNG, Sony ARW and Panasonic RW2; RW2 carries distortion only):
    *   **Distortion**: straightens curved lines in place of the manual correction. Set it before cropping or retouching.
    *   **CA**: reduces color fringes.
*   **Distortion Correction** (-0.100 to 0.100, in steps of 0.001): positive corrects barrel, negative pincushion. Use the film rebate as a straight edge.

#### Flat Field Correction: even out the light

Corrects uneven light (vignetting, falloff) from a copy-stand or scanner light, using a shot of the bare light source.

*   **Profile**: **+** bakes a reference image into a named profile in NegPy's `flatfield` folder; the image can then be deleted. **Trash** asks first; every frame using the profile loses its correction.
*   **Apply Flat Field**: applies the selected profile to this roll.
*   **Check Flat Field**: shows how well the selected profile corrects its own reference. A good profile shows an even gray; a band or patch shows where it is off (full white or black is 10% or more). It turns and flips with the open frame, without fine rotation, keystone or crop. An older profile without check data shows the light it corrects instead; save it again to check it. Saving warns when the reference is clipped or does not correct itself evenly.

A newly chosen profile becomes the rig's default for the next roll.

---

## 11. Metadata tab

Metadata for the original analog capture (camera, lens, film, process), written into every export as EXIF and XMP, so a DAM such as Lightroom shows your film gear, not the scanner. **Protect Original Metadata** copies the source's own EXIF/XMP and resolution unchanged instead.

**Analog Gear**, **Capture**, **Process**, **Scanning** and **Exposure** are roll-wide by default ([§1](#frame-or-roll-the-scope-pair)). The frame number is always per frame.

<!-- panel:metadata_presets -->
### Metadata Presets

Saved sets of metadata values in `~/NegPy/presets/metadata/`, managed on the **Gear** tab (§12).

*   **Preset** + **Load**: writes the preset's fields onto this frame; other fields stay.

Gear loads as one unit (camera, lens, film stock, format). A film stock sets the format, so set a frame format such as `6×7` after the stock. Presets never store the frame number.

<!-- panel:metadata_gear -->
### Analog Gear

Searches your own gear (§12). **Other…** opens the full built-in catalog.

*   **Camera / Lens / Film stock**: empty means not set.
*   **Clear**: empties all three.
*   **Infer from folder name**: fills unset camera, film stock, ISO and capture date from the folder name; an ambiguous field stays empty.

<!-- panel:metadata_capture -->
### Capture

*   **Date**: `1998`, `1998-07`, `1998-07-14` or `1998-07-14 16:30`, with an optional offset such as `+02:00`. EXIF pads a partial date; XMP keeps the short form. The scan's own timestamp moves to `DateTimeDigitized`.
*   **Place**: the map-pin button opens a map; the field also takes a coordinate pair or an OpenStreetMap/Google Maps link. A place replaces the source's GPS; with none set, a geotagged source keeps its own. The map contacts OpenStreetMap.

<!-- panel:metadata_process -->
### Process

*   **Saved process**: a library recipe that fills Developer, Dilution, Push / Pull, Time and Temperature. Typing over one unlinks it.
*   **Format**: `—` (not set), `35mm`, `120`, `4×5`, `8×10`, `110`, or `Other` (free text).
*   **Developer** and **Dilution**: for example `D-76` and `1+1`.
*   **Push / Pull**: `Push +3` … `Normal` … `Pull -3`.
*   **Time** and **Temperature (°C)**: time as `9:30` or minutes; searchable as `devtime:` and `temp:`.
*   **Clear**: empties the saved process and its fields. Format stays.

<!-- panel:metadata_scanning -->
### Scanning

*   **Saved setup**: a library digitizing setup that fills Scanning. Typing over it unlinks it.
*   **Scanning**: scan method or notes. EXIF `Software` is always `NegPy`.
*   **Clear**: empties the saved setup and the note.
*   **Roll / Frame**: the capture roll name and frame number, stamped on capture; filename template fields `{{ roll }}` and `{{ frame }}`.
*   **Sequence…**: assign a running capture frame number to the selected frames or the whole roll, in film order. Set a start number, a step and ascending or descending order, and each frame gets its own number. Tick **Skip Rejected Frames** to leave frames in the rejected state out of the run.

<!-- panel:metadata_exposure -->
### Exposure

Optional original shutter, aperture and ISO. Click the lock to edit a free-text string such as `1/125s f/2.8 ISO 400`.

<!-- panel:metadata_preview -->
### Metadata Preview

A live view of what goes into the file. **Description…** picks the fields joined into EXIF `ImageDescription` (default: camera, lens, film stock, ISO); the choice becomes the default for frames without their own.

Capture gear goes to standard EXIF and the digitizing rig to `negpy:Scan*` XMP tags; with gear unset, your scanner or DSLR stays in EXIF. Other XMP tags: `photoshop:DateCreated` (with `negpy:CaptureDatePrecision`), `photoshop:City`/`State`/`Country`, `exif:GPS*` (the only GPS in a TIFF), `negpy:DevelopmentDilution`/`DevelopmentTime`/`DevelopmentTemperature`, `negpy:CaptureRoll`/`CaptureFrame`. A star rating set in the camera carries over as `xmp:Rating`.

---

## 12. Gear tab

A gear library for Metadata (§11), Roll Settings and every gear picker. **My Gear** holds physical gear; **Presets** holds metadata field sets.

<!-- panel:gear_items -->
### My Gear

**Category**: **Cameras**, **Lenses**, **Film Stocks**, **Process** (a development recipe) or **Scanning** (a digitizing setup). Your gear is saved to `~/NegPy/gear/`.

*   **+**: copy an item from the built-in catalog into your list, or **Add Custom** to enter one by hand.
*   **Catalog**: also shows the built-in reference models. Off by default.
*   **copy / trash**: duplicate or delete the selected item. Built-in entries cannot be deleted.

<!-- panel:gear_presets -->
### Presets

*   **+**: stores the current frame's metadata under a name.
*   **pen**: rename, or change which fields it stores.
*   **copy / trash**: duplicate or delete.
*   Edit a preset's fields in place. Per-frame fields (capture date, place, description) are read-only here. **Notes** is free text.

---

## 13. Export tab

### Output intent

*   **Print** (default): the look you see on screen.
*   **Flat**: a neutral, low-contrast master for editing elsewhere. It skips the print look, effects, toning and vignette, and writes a 16-bit TIFF, or lossless JPEG XL when JXL is selected with sRGB, P3, Rec 2020 or Grayscale.
    *   **Preview Flat**: shows the flat master on the canvas.
    *   Run **Roll Analysis** (Roll tab) before a flat batch so the masters match.
*   **Linear**: writes the decoded buffer as linear 16-bit, with rotation, flip and only the corrections turned on below. **TIFF** (default, untagged) or **JPEG XL** (lossless; tagged with sRGB primaries, which is wrong for native primaries). **Effort** (1 to 9, default 7) trades JPEG XL speed for size.
    *   **LinearRaw DNG**: SilverFast HDRi (3-channel) and VueScan (4-channel RGB+IR). IR goes to a separate grayscale `_ir` file.
    *   **Camera RAW**: demosaiced at unity white balance with the Raw Decode card's Export algorithm (§10.7); as-shot WB goes to XMP (`RAW-WB: R G B`). Trichrome triplets merge into one TIFF.
    *   **Coolscan NEF**: not raw sensor data but the Nikon Scan output; the full-resolution RGB is written.
    *   **Flextight FFF**: 16-bit RGB and LogLuv raw (`.3fr`/`.fff`); FlexColor metadata is kept.
    *   **Noritsu RAW**: headerless BGR 16-bit dumps; size detected from file size.
    *   **TIFF**: IR (a 4th channel, an `_ir.tif` sidecar or a second page) goes to a separate `_ir` file. **Input gamma** (linear, 1.8, 2.2 or sRGB) linearizes the data.
    *   **Expansion**: scales the data before writing. Defaults: Pakon RAW 4× (the 16-bit F335 off), Noritsu RAW 16×, LinearRaw DNG off. Camera RAW, Coolscan NEF and Flextight FFF have none.
    *   **IR Dust Removal** (when IR exists): IR dust and scratch correction. Off by default.
    *   **Corrections** (camera RAW only, all off): **White Balance** (as-shot gains), **Flat Field**, **Sensor Correction** (crosstalk unmixing). Stitch composites always get Flat Field and sensor correction per part.
    *   **Lens Correction** (off; shown when the frame has a lens profile or a **Distortion Correction**): applies the **Optics** correction (§10.8). The embedded profile needs a single camera RAW; Distortion Correction skips half-frame scans.

    Files take `_linear` (then `_linear_2`… without **Overwrite Existing Files**), so a source is never overwritten. The file carries no ICC profile. **Abort** stops after the current frame.

### Format / Size / Color Management / Destination

*   **Format**: `JPEG`, `TIFF`, `PNG`, `JPEG XL` or `WebP`, each with quality or effort options. JPEG XL supports only `sRGB`, `P3 D65`, `Rec 2020` or `Grayscale`.
*   **Bit Depth**: `8-bit` or `16-bit` (TIFF, PNG, JPEG XL).
*   **Compression** (TIFF): `Uncompressed`, `LZW` or `ZIP`, all lossless; ZIP is usually smallest.
*   **Compression** (PNG): `0` to `9`, lossless; higher is slower and smaller.
*   **Progressive** (JPEG): renders in passes while it downloads.
*   **Input ICC**: treats an untagged source as this profile (primaries only; a LUT profile's input curves still apply).
*   **Export profile**: the space the file is converted to and tagged with: `Same as Source` (Adobe RGB for an untagged scan), `sRGB`, `Adobe RGB`, `ProPhoto RGB`, `P3 D65`, `Rec 2020`, `Grayscale` (true B&W), or an imported printer or paper ICC, embedded in the file. A warning shows when the soft proof targets a different profile than the export.
*   **Import ICC**: copies a `.icc`/`.icm` into `~/NegPy/icc/`. A file named after a built-in space (`sRGB.icc`) replaces that space everywhere, after a confirmation.
*   **Paper Aspect Ratio**: final print ratio, or *Original* (no resize).
*   **Resolution**: *Original* (full resolution), *Print* (long-edge **Size** in cm plus **DPI**) or *Pixels* (**Long edge** in px). The DPI tag is your value (*Print*), the value the long edge implies (*Pixels*), or the source's, else the **DPI** field (*Original*).
*   **Sharpening** (0.0 to 1.0, *Print* and *Pixels*): sharpens the export after the resize, to restore the edge contrast that scaling down removes. It adds to the Lab **Sharpening** and does not show in the preview.
*   **Destination**: **Filename Pattern** (a Jinja2 template; see [TEMPLATING.md](TEMPLATING.md)), **Overwrite Existing Files**, and the location: subfolder of source (default, `export`), same as source, or an **Export Path**. A roll with no single source folder exports under its own folder in NegPy's data folder.

### Export button

**Export**'s chevron picks the scope: current frame (Ctrl+E), selected frames or all visible frames. For several formats or sizes in one run, use Export Presets. A batch with unedited frames asks first, since they may not match their thumbnails. After the batch, a warning names exports that rendered almost black with no crop set, where the bright scan border drove automatic levels.

### Collapsible sections

<!-- panel:export_presets -->
#### Presets

Saved Format/Size/Color Management/**Destination**/filename recipes. Each preset is a toggle; **Manage** edits them. **Export Presets** renders the frames with every preset turned on, each to its own destination.

<!-- panel:printing_notes -->
#### Printing Notes

The printer's record for this frame, as described under Dodge & Burn. **Preview** shows it on the canvas (**Shift+N**); **Export** writes it as an image beside the print.

<!-- panel:export_sidecars -->
#### Sidecars

**Save on Export** writes a `.negpy` sidecar next to each source on export. **Export Sidecars** writes them for all visible frames at once. Edits always stay in the database too. A frame with no edit in the database takes its sidecar when its folder opens. **Load Edit from Sidecar…** (right-click a frame) replaces the frame's edit with a chosen `.negpy`, as one undo step.

<!-- panel:export_bake -->
#### Auto Density/Grade

**Bake on Export** is an app-wide switch, the same on every frame: when on, every export freezes Auto Density and Auto Grade into fixed values for the frames it has metered, so the saved edit does not re-run the meters. **Bake to Roll** freezes them for every visible frame now, metering a frame it has not seen first. A frame with both helpers already off, or a composite or a roll fork, keeps its edit.

<!-- panel:contact_sheet -->
#### Contact Sheet

A darkroom proof of the roll: every visible frame at true size, as cut film strips on photographic paper, in capture order. The rebate, perforations and edge print follow the frames' **Film stock** and **Format** metadata.

*   **Path**: the folder for the sheets. Empty follows the export destination.
*   **Contact Sheet…**: opens the proof. Drag an edge or corner of the paper to resize it; a size near an Ilford sheet snaps to it.
*   **Format**: 35mm, 35mm Half Frame or 120. **Frame size** sets the 120 camera's frame (6×4.5 to 6×17).
*   **Paper**: an Ilford sheet size, or Custom. The default, 24 × 30.5 cm, holds a whole 36-exposure roll.
*   **Width** / **Height**: the paper size, 50–610 mm.
*   **DPI**: 150, 300 or 600, tagged so the sheet prints at true size.
*   **Order**: **Date**, or **Scene**, where each scene starts a new strip.
*   **Print**: **As Edited**, or **Straight Proof**: the whole roll at one exposure on grade 2 from its Roll Analysis baseline, so thin and dense negatives print light and dark. It needs the roll scanned at one exposure.
*   **Pick Frames**: click a frame to leave it out or put it back; rejected frames start out.
*   **Edge Print**: the maker's markings on the film edge (stock name, frame numbers, DX barcode).
*   **Roll Label**: the roll name, film, developer, camera and date above the strips.
*   **White Paper**: white paper in place of the darkroom black, to save ink on a home printer.
*   **Film Base** (on): prints the film base black, as a darkroom contact print. Off prints it as paper with the strip outlined, to save more ink.

Sheets are JPEGs named `contact_sheet.jpg` (`contact_sheet_1of2.jpg` and on when the roll needs more sheets), with the **JPEG Quality** and **Progressive** settings above.

#### Soft Proof

Simulates the print on screen. See below.

<!-- panel:soft_proof -->
### Soft Proof

Preview only; exports are never proofed. Paper is dimmer and has a smaller gamut than a screen, so a correct proof looks worse than the plain preview.

*   **Proof on Screen** (`Shift+P`, on by default): master switch.
*   **Preset**: a saved printer and paper setup; **None** proofs the export target with no paper simulation.
*   **Profile**: follows the **Export profile** until you pick a printer or paper, so you can proof a print while you export a web JPEG.
*   **Intent**: **Relative Colorimetric** keeps printable colors and clips the rest. **Perceptual** compresses everything so color relations survive; try it on saturated frames. **Saturation** is for charts, not photographs.
*   **Black Point Compensation** (off): maps the darkest tone to the paper's black in place of clipping.
*   **Simulate Paper White** (off): shows the paper's white and tint.
*   **Simulate Ink Black** (off): shows the paper's real black; shadows lift.
*   **Gamut Warning**: shows unprintable colors as gray.
*   **Display**: the monitor profile, auto-detected; set it by hand if detection fails.

---

## 14. Scan tab

Capture film directly into NegPy. The **Scanner** card picks the scanner; the tab shows its cards, the shared **Output** card and the **Scan** button.

<!-- panel:scan_source -->
### Scanner

*   **Scanner** (Camera by default): **Film Scanner** drives a dedicated film scanner (Device, Film & Quality, Framing); **Camera** drives a camera on a copy stand (Camera, Preset & Light).

<!-- panel:scan_device -->
### Device

**Backend**: **SANE** (Linux/macOS), **Nikon Coolscan (nkscan)** (Linux, Windows, macOS) or **pyOpticfilm (Plustek)** (all three OSes). Controls follow what the device reports.

*   **Device**: the scanner; the arrows refresh the list. The eject menu holds **Eject Now** and, on a strip feeder, **Eject When Done** (on by default), which returns the strip after a batch. Off keeps it loaded with its picks and previews. A strip the scanner returns by itself counts as an eject: insert it again, then **Detect frames**. After 9 minutes with no scanner activity NegPy measures the strip again.
*   **Debug log** (nkscan; Off, Debug, Trace): writes `nkscan.log` in the NegPy folder. For a bug report, set Trace, reproduce the problem and attach the file.
*   **Sleep**: while a scan runs, NegPy keeps macOS and Windows from going to sleep. On Linux, turn off automatic suspend before a long roll.

**pyOpticfilm (Plustek)**: the **OpticFilm 8200i SE** (`07b3:1825`), **8100 V2** (`07b3:1824`), **8300i SE** (`07b3:181f`) and **7600i v1** (`07b3:0c3b`) scan; for other models try **SANE** on Linux and macOS. Needs pyopticfilm **1.3.6** or later. **IR** comes in the same pass (on the 7600i v1, at 3600 and 7200 dpi only). Multi-Pass and Prescan are on the GL128 models. On Windows, bind the device to **WinUSB** with Zadig first ([PLUSTEK_WINDOWS.md](PLUSTEK_WINDOWS.md)). From source, install with `uv sync --group plustek` or `pip install negpy[plustek]`.

**Nikon Coolscan (nkscan)**: needs no SANE and ships in the release builds; from source, see [CONTRIBUTING.md](../CONTRIBUTING.md). On Linux, USB needs a udev rule for vendor `04b0`; FireWire/SCSI needs the `sg` module.

<!-- panel:scan_quality -->
### Film & Quality

*   **Film**: Color negative, B&W negative, Slide or Kodachrome. Sets frame detection, metering and whether IR and ICE are offered (not for B&W or Kodachrome).
*   **Film format**: frame length (135, 66, 645 and so on). **Auto** where the holder narrows it; set it for loose film in a masked carrier.
*   **Resolution**, **Bit depth**: the scan's dpi and bits per channel; type a resolution between the listed stops.
*   **Scan mode** (Plustek; Single-Pass by default): **Single-Pass**, **Multi-Pass** (stacks repeated exposures to cut noise; **Passes** 2 to 9), **Adaptive Multi-Exposure** (fuses a short and a long exposure for more range) or **Adaptive Multi-Pass** (both). Multi-Pass excludes IR.
*   **Samples** (nkscan): reads per line averaged (1 to 16). Less shadow noise, proportionally slower.
*   **IR**: a separate infrared plane for Retouch's dust removal. **ICE** (nkscan): infrared dust and scratch removal baked into the file, color film only. **ICE** and **IR** exclude each other.
*   **Superfine** (nkscan): one line per pass. Slower, with no host-side line registration.
*   **Autofocus**, **Auto-exposure**: run on the scanner before the scan.
*   **Exposure** (µs): shown when the scanner has `scan-exposure-time`; off while Auto-exposure is on.

<!-- panel:scan_framing -->
### Framing

*   **Frames**: `1-6`, `1,2,5`, or empty for all. The strip preview writes its picks here.
*   **Batch** / **Window**: on a feeder, **Preview strip…** previews every frame, to set windows and pick frames; with a manual holder, **Preview…** sets one crop window. **Clear** resets it. **Detect frames** reads the strip again after the film moves.
*   **Exposure lock** (**Meter Frame…** / **Unlock**, nkscan): nkscan meters each frame alone, so an end frame can scan in a different color. **Meter Frame…** meters one frame inside the strip (such as frame 2), and every later scan reuses that exposure, across strips and restarts, until **Unlock**. Meter again for each roll.
*   **Crop** (**Prescan…**, Plustek): a 1200 dpi full-window preview; drag a crop, and the next scan reads only that area. **Clear** scans the full window.

<!-- panel:scan_camera -->
### Camera

Copy-stand capture with a camera in **PC Remote** mode over USB (macOS/Linux): with a NegPy **Scanlight**, R/G/B triplets or one exposure with the three LEDs lit together, else one white-light exposure. Needs `python-gphoto2` (`pip install gphoto2`; no Windows build). Setup and troubleshooting: CAMERA_SCANNING.md.

*   **Camera** / **Light**: connection status, found automatically; the light turns amber once it runs warm. On Linux, the Scanlight's serial port needs your user in the `dialout` group (`sudo usermod -aG dialout $USER`, then log out and back in).
*   **Live View**: click the image to aim the focus magnifier. Turn the focus ring past best focus, then back until the **Focus meter** reads **at peak**. ISO, shutter and aperture are set from its toolbar, or locked by a calibrated RGB preset.
*   **Histogram** and **Zebras**: the frame's RGB histogram, and hatching on pixels at the top (red) and bottom (blue) of the preview range. The preview clips before the RAW does, so a zebra is an early warning.
*   **Meter** (white-light and camera-only scanning): one RAW probe sets the shutter so the region that must not clip (a negative's clear base, a slide's brightest highlights) lands just under clipping. The white-light preset's film type decides which. A clipped reading steps down two stops; meter again.
*   **Scan** and **Retake**: **Scan** shoots the next frame, auto-numbered, and imports it; **Retake** (the redo arrow beside **Live View**) shoots the last frame again. In the Live View window, `Space` or `S` scans and `R` retakes.
*   **Keep Light On** (the bulb beside **Retake**, off by default): the Scanlight goes dark after each scan; this lights it again with the framing light, so you can frame the next shot on the camera's screen. Live View lights it again without this toggle.
*   **Narrowband**: RGB-lit scans render more saturated than white light; a **Print stock** other than *Digital* prints them through real dyes.

<!-- panel:scan_light -->
### Preset & Light

Shown while a Scanlight is connected.

*   **Preset**: forces its RGB levels, ISO, shutter and aperture each frame. **+** calibrates: place the rectangle on clear film base, name it, run it; it solves a shutter and LED levels just under clipping. **Create a manual preset…** sets one by hand.
*   **Red**, **Green**, **Blue**, **White** (0 to 255): LED levels, editable while you build a manual preset. **Light Off** turns every channel off.
*   **Capture mode**: **Triplet** shoots one exposure per LED and merges them; **Single Capture** shoots one exposure with all three lit and imports it as an ordinary RAW.
*   **Create Sensor Profile** (Single Capture only, default on): also saves a sensor profile under the preset's name; a roll scanned with the preset takes it and turns Linear RAW on. Other rolls do not get it.
*   **ISO**, **Shutter**, **Aperture**: the preset's exposure, editable while you build a manual preset.
*   **Channel Delay** (0 to 5000 ms): pauses between R, G and B for bodies that lock up. Triplet presets only.

<!-- panel:scan_output -->
### Output

Shared by both scanners.

*   **Folder**: where frames are written.
*   **Scan as Roll** (on by default): the output folder opens as a roll in the Library, so Half Frame, the Roll tab and Roll Analysis apply. Frames load as they are written.
*   **Folder as Roll** (on by default): frames go into the output folder itself, which is the roll. Off writes into a **Roll** subfolder, and a new Roll name starts a new roll. A camera frame's file name starts with the roll name.
*   **New Roll** (**+** beside Roll): steps the Roll name to the next one that has no subfolder yet. A trailing number that is zero-padded or follows `_`, `-`, a space or `.` counts up (`Roll001` to `Roll002`, `portra400_1` to `portra400_2`); any other name gets `_2`.
*   **Format**, **Filename** (Film Scanner): `TIFF` or `TIFF (mono)` (one 16-bit gray plane, for B&W negatives), and the file name as a Jinja2 template of `{{ date }}` and `{{ seq }}`.

<!-- panel:scan_strip -->
### Strip preview

Dialogs end with **Cancel**, **Apply** (keep the framing) and **Scan**. During a preview **Cancel** reads **Stop Preview** and keeps the tiles read so far. Negatives preview inverted. On nkscan, **Offset** (±10 mm) and **Drift** cut the tiles again with no new scan.

*   **Cropping**: drag on a frame. **Clear Crops** removes all.
*   **Frame outline**: red box on the detected frame; offsets are measured from it.
*   **Offset**: moves every frame along the film to clear the gap. The shaded band is film the transport cannot deliver; a feeder goes one way only.
*   **Drift**: offset that grows (or shrinks) per frame position.
*   **Per-frame offset**: the slider under a tile, on top of Offset and Drift.
*   **Size**: tile size.
*   **Which frames**: tick tiles, or **All** / **None**. Eject or a restart clears ticks, crops and per-frame offsets.
*   **Preview frame**: scans that frame again.

---

## 15. Preferences

Application-wide settings: **⋯** menu → **Preferences…** or `Ctrl + ,`. Changes apply at once; startup rows show a restart notice.

### Interface

*   **UI scale** (80% to 120%): after a restart.
*   **Canvas background**: black, dark gray, mid gray (neutral for judging) or white (a print on a light table).
*   **Immersive canvas**: the toolbar floats over the image.
*   **Sticky zoom**: keeps the zoom when you switch frames.
*   **Reverse scroll zoom**: scroll up zooms out.
*   **Customize Shortcuts…**, **Edit Toolbar…**, **Reset Panel Layout**: shortcut editor, canvas toolbar picker, default panel layout.

### Accessibility

*   **Color vision** (default **Standard**): colors for marks told apart by color alone (dust overlay, dodge and burn masks, Zone Overlay, clipping overlay). *Protanopia / deuteranopia*, *Tritanopia* and *Achromatopsia* use colors from the Okabe–Ito palette.

### Performance

*   **GPU acceleration**: renders on the GPU. Off uses the slower CPU pipeline, same image. If the GPU fails at launch, an amber line here says so.
*   **Multi-core CPU rendering**: runs CPU rendering on all cores; it helps exports and machines without a usable GPU. On by default except on macOS, where it can end the app; after such a crash NegPy offers to turn it off. `cpu_parallel` in `override.toml` overrides it.
*   **Search by meaning**: find frames by description (§2).
*   **Preview size** (512 to 8192 px): canvas long edge; higher is sharper and uses more memory.
*   **Preview cache** and **Preview cache limit**: photos kept decoded, and their memory limit. Lower both on low RAM.
*   **HQ buffers**: full-resolution preview buffers kept in memory.
*   **Rendered frames**: frames kept for going back without a new render.
*   **GPU texture cap**: largest texture dimension; 0 lets the hardware decide. Lower it if exports run out of GPU memory.
*   **Show GPU memory warning**: a message when an HQ preview is downsampled to fit the texture cap.

Rows from **Preview size** down need a restart. A value in `override.toml` wins and grays out its row.

### Session & Storage

*   **Carry settings between frames**: applies your Persistent Settings to each newly opened file.
*   **Persistent Settings…**: which edits carry over (§1).
*   **Manage Database…**: row counts and sizes, clear saved edits, library roots.

### Startup override (`override.toml`)

For crashes on launch or rendering glitches: edit `Documents/NegPy/override.toml` (created on first run) and restart. Its values win over Preferences.

| Setting | Values | Effect |
|---------|--------|--------|
| `rendering.backend` | `"auto"`, `"vulkan"`, `"dx12"`, `"metal"`, `"cpu"` | GPU backend for image processing. `"cpu"` disables the GPU. |
| `display.qt_rhi_backend` | `"auto"`, `"vulkan"`, `"d3d12"`, `"metal"`, `"opengl"`, `"software"` | Qt UI rendering backend. |
| `display.qt_platform` | `"auto"`, `"xcb"`, `"wayland"` | Window system plugin (Linux only). |
| `performance.force_hq_preview` | `true` / `false` (or absent) | Overrides the saved HQ preview toggle. |
| `performance.cpu_parallel` | `true` / `false` (or absent) | Multi-core CPU rendering. On by default, except on macOS. |
| `logging.level` | `"debug"`, `"info"`, `"warning"`, `"error"` | Log verbosity. Use `"debug"` when you report issues. |

`max_texture_size`, `preview_render_size`, `preview_cache_max_bytes`, `preview_cache_max_entries`, `preview_cache_max_full_res_entries` and `render_memo_max_entries` take the same values as their Preferences rows.

**Common fixes:**

*   **Crashes immediately on Linux** → `backend = "cpu"` or `qt_rhi_backend = "opengl"`.
*   **Black or blank preview on Windows** → `backend = "dx12"` or `qt_rhi_backend = "software"`.
*   **Wayland rendering issues** → `qt_platform = "xcb"` to force X11.
*   **GPU out-of-memory during export** → `max_texture_size = 4096`.

---

## 16. Updating NegPy

NegPy checks GitHub once at startup. A new release shows a green **⬇ Update Available: vX.Y.Z** line at the top of the left panel; click it, or **Update to vX.Y.Z…** in the **⋯** menu, for the release notes and the install button. **Check for Updates…** checks by hand.

**Install Update** downloads the build for this install type, closes NegPy, installs and reopens. Nothing is replaced until NegPy exits, so a failure leaves your install as it was.

| Install | What NegPy fetches | How it installs |
|---------|--------------------|-----------------|
| **Windows** | the `-Setup.exe` installer | Runs it silently. Approve the administrator prompt *before* NegPy closes. |
| **macOS** | the `.dmg` for your chip (Apple silicon or Intel) | Replaces the `NegPy.app` bundle where it is, then reopens it. |
| **Linux** | the `.AppImage` | Replaces the AppImage file you launched and relaunches it. |

Edits, presets, settings and library live in `Documents/NegPy` and the database, so updates do not touch them.

The button reads **"Open Releases Page"** when NegPy cannot update itself: a source checkout, no build for your platform, an app moved out of its installer layout, or an app folder you cannot write to.

---

## Additional Info

*   **Database**: edits live in a local SQLite database keyed by file hash, so files can move or be renamed. Optional `.negpy` sidecars mirror them.
*   **Saving edits**: written on export, on frame switch, or on save; closing mid-edit before any of these loses unsaved changes.
*   **Keyboard shortcuts**: [KEYBOARD.md](KEYBOARD.md)
*   **Filename templating**: [TEMPLATING.md](TEMPLATING.md)
*   **The pipeline in depth**: [PIPELINE.md](PIPELINE.md)
