# NegPy User Guide

NegPy turns film scans into positives with a non-destructive, darkroom-style pipeline. It never writes to your source files; edits live in a local database.

For the pipeline order and math, read [PIPELINE.md](PIPELINE.md).

---

## 1. The Big Picture

### Protected data folders on Windows

If Windows blocks the default data folder, NegPy suggests `%LOCALAPPDATA%\NegPy\data`; click the path to pick another, then **Use This Folder**. Nothing is copied: a new folder starts empty, an existing NegPy folder keeps its data. The choice, saved in `data-location.json`, stays even if Documents becomes writable; `NEGPY_USER_DIR` overrides it.

### Screen layout

*   **Left, the film strip**: your frames as a contact sheet, with import, sorting and triage tools.
*   **Center, the canvas**: the live preview, where most tools act. Scroll or pinch to zoom, drag to pan. The bottom toolbar holds Fit/**1:1** zoom (one scan pixel per screen pixel; with **HQ** off, a **preview res · HQ off** pill marks a scaled-up preview), undo/redo, rotate/flip (on every selected frame) and more. The **⋯** menu holds every action, including **Preferences…** (§15), **Edit Toolbar…** (put any menu item on the toolbar) and **Persistent Settings…**. Right-click the image for **Reset View**, **Sticky Zoom**, the pickers, copy/paste settings, **Reset Settings**, **Reset to Roll Settings** and **Unload** (drop the frame from the session; its edit stays).
*   **Right, the controls**: tabs **Roll** / **Frame** / **Metadata** / **Gear** / **Export** / **Scan**. **Frame** has a pinned **Analysis** readout and its own row of tabs below it. Roll and Frame change the render; the other tabs do not.

With nothing open, the canvas offers **Load Scans…** (import a folder as a roll, add files, or open the Scan tab) and **Take the Tour**.

Drag a panel by its top edge (the strip above Session, or the margin around the **Find** box) to float it; its pin button docks it again. **Shift+H** hides and shows both panels. NegPy remembers the layout and each dialog's size and position; **Reset Panel Layout** (**⋯** menu) restores the default.

### The tour

**Take the Tour** (**⋯** menu, or Find) walks through NegPy in eight chapters; the chapter menu on its card jumps between them. A step with a circle is a task (drag Print Density, press a view key); a check marks it done. With nothing open, **Load Demo Negative** opens a synthetic negative. **Read More…** opens the panel's guide. Esc ends the tour; the next one offers **Resume Where You Left Off**.

### Find

**Ctrl+K** (or **Find Control or Action…** in the **⋯** menu) searches every slider, card and action, by name or by another editor's word: *contrast* finds **ISO-R Grade**, *white balance* **Filtration**, *exposure* **Print Density**. Enter opens the row on its tab; a slider row is the live control. Presets are listed by name; Enter opens **Apply Preset**.

### Before / After

**◑** on the toolbar (or `\`) splits the canvas: left is the auto baseline (same film process, crop and rotation, every creative control at default), right is your edit. Drag the divider. The split stays up while you edit; `\`, `Esc`, a frame change, a peek or the test strip closes it.

### Reference view

**Shift+R** (or **Reference View** in the **⋯** menu) pins the current frame to a pane beside the canvas, to match other frames to it. The pane keeps the frame as pinned; press **Shift+R** twice to pin again. **✕** or **Shift+R** closes it.

### Peek Negative

The toolbar's film button (or `N`) shows the scan as loaded: not inverted, not metered, no edits; crop, rotation and flip still apply. Use it to check density, mask color and scanner clipping. It is scaled to its brightest tone, so read density from the density histogram, not the brightness. It stays up while you drag a crop or rotation handle or the Fine Rotation slider, and through Export and Metadata settings. Releasing the handle or slider closes it, and so do a rotate, flip, straighten or keystone, any other edit and a frame change.

### Peek Embedded Preview

**Peek Embedded Preview** (**⋯** menu, or `P`) shows the camera's own JPEG at your crop and rotation. NegPy uses nothing from it. Files without one (scanner TIFFs, most converter DNGs) disable the item. It closes the same way as Peek Negative.

A peek shows a **NEGATIVE**, **EMBEDDED** or **FLAT SCAN** badge. `Esc` closes any peek, the split or a test strip.

### The workflow

**Roll** holds what the whole roll shares:

| Tab | Panels | What it is for |
|-----|--------|---------------|
| **Roll** | Film Mode · Frame Assembly · Calibration · Crop · Roll Analysis · Metering · Raw Decode · Optics | Film type, capture color, crop shape, roll baselines, negative→positive metering, the scanning rig |

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

A slider row reads name, track and value: click the value to type one, drag the name to scrub (**Shift** for finer steps), double-click or **Ctrl**+click the track to reset. A thin rail joins controls that work together, such as Toe and Toe Width. A **dot** on a panel header or tab marks a non-default value. Each panel header has a **reset** action and an **ⓘ** that opens this guide there. Tabs that do not fit move into a **»** menu.

### What carries to the next frame

An unedited frame gets the rig and roll settings: film process, crop ratio, flips, calibration, paper stock, the Lab polish and export preferences. The look (density, filtration, tone curve, toning, dodge and burn) starts clean.

**Preferences → Session & Storage → Persistent Settings…** edits that list: tick a setting or a group to make it carry. **Carry settings between frames** is the master switch; off, new frames get bare defaults.

An edited frame keeps its look; only export and metadata settings reach it. **Reset Settings** returns bare defaults and, in a roll, sets every Roll-tab card back to **Roll**. No reset, a card's arrow included, changes the scanning setup (Linear RAW, Narrowband, demosaic); **Scanning setup** sets those.

### Frame or roll: the scope pair

Each section header has **Frame** (picture, amber) and **Roll** (film roll, red) beside its reset arrow, which resets that card alone; **· 2** after the card name counts its non-default settings. The lit one shows where the card's values live; click the other to move them. A card with non-default values has a stripe in that color. Frames that are not one roll (search results, several folders) show only **Frame** until **Save as Roll…**.

On a **Roll tab** or **Metadata** card the pair is a latch: on Roll, the card follows the roll and new frames inherit it; an edit flips it to Frame. **Roll** pushes this frame's value to the roll; **Frame** pins it to this frame. Right-click a Roll tab card's header for **Undo Apply to Roll**, which gives the roll back that card's values from before the last push from this frame.

On a **frame** card (Geometry, Filtration, Tone, Lab, Alternative Processes, Toning, Retouch, Finishing), **Roll** opens the clone picker for that section, to apply your changes to the selection or the whole roll. After a whole-roll apply the card reads Roll until you touch a pushed setting. Right-click its header for **Copy Card Settings**: Paste (`Ctrl+V`) then offers that card's settings alone.

**Reset to Roll**, beside the reset arrow, appears once a card differs from the roll. It puts the roll's values back on that card as one undo step; a setting the roll never carried keeps this frame's value.

### The tab header

Tabs with several cards (Roll, Exposure, Look, Finish, Metadata) have a bar reading **3 of 5 cards edited**. Its buttons act on every card: reset arrow (asks first), **Reset to Roll**, roll button (one picker for the whole tab) and double chevron (collapse/expand).

### Menu bar (macOS)

`Ctrl` in this guide is `⌘` on macOS, as the app shows it.

*   **NegPy**: Preferences… (`⌘,`), beside About and Quit.
*   **Window**: Minimize (`⌘M`), Zoom, Close (`⌘W`), Bring All to Front, and a list of open NegPy windows (main, live view, calibration).
*   **Help**: Take the Tour, Keyboard Shortcuts, Customize Shortcuts, the Analysis panel guide, Report an Issue, Check for Updates.

Full screen is the green window button. Menus show only `⌘` keys; plain-key shortcuts such as `?` still work. `⌘W` on the main window closes NegPy. Windows and Linux have no menu bar.

---

<!-- panel:frames -->
## 2. Film strip (left panel)

A green **Update Available** line tops the panel when a newer release is out ([§16](#16-updating-negpy)). **About NegPy…** in the **⋯** menu shows the version.

Below it are the toolbar, the search box and two sections: **Library** (imported rolls) and **Film Strip** (open frames). Click a heading to fold it; drag the handle between them to resize.

<!-- panel:library -->
### Your library

**Library** lists every **roll** you have imported: a named group of frames, not a live view of a folder. **Ctrl+L** expands it.

*   **+**: import a roll (below).
*   **↻**: finds new roll folders under each parent imported with **Import Subfolders as Rolls…** and re-reads each roll's frame count.
*   **Discovery Filters…**: folder names that importing and **↻** skip, with everything inside them, one per line (default `export`). A line matches any part of a name, ignoring case; a line with `*` must match the whole name (`raw_*`). Saving runs **↻**: rolls a filter now catches leave the list, and come back when the filter goes.
*   **Sort**: Name or Date, ascending or descending.

Each row shows name and count ("36 photos"), where an assembled Trichrome triplet, stitch or HDR merge is one photo; a roll whose folder is gone shows **folder missing** in amber. Importing only records the folder; nothing is decoded until you open the roll.

#### Importing

**+** (or the list's right-click menu) offers:

*   **Import Folder as a Roll…**: the folder becomes one roll and opens; a folder with no images of its own imports its subfolders instead. If its name matches a camera or film stock in your Gear library ("penf" finds "Pen F"), Roll Settings opens pre-filled.
*   **Import Subfolders as Rolls…**: every folder below the chosen one that holds images becomes a roll, without opening; a roll folder's own subfolders (export output) do not. A roll is named by its path ("20260901/kentmere_400_1") and listed under a **20260901** folder row. Right-click a folder row for **Delete…**, which forgets every roll in it.

Importing never changes the folder. Reorganize on disk, then re-import (or **↻**); edits follow image content, so a moved file keeps its edit and marks.

#### Opening, renaming, deleting

**Double-click** a roll (or **Enter**) to open it; NegPy asks first, since loading reads every frame (**Always load without asking** skips the prompt). Opening replaces the Film Strip contents.

Right-click a roll for:

*   **Close Roll…** (**loaded** roll only): empties the Film Strip and returns to the Library. Asks first; edits stay saved.
*   **Rename…**: renames the roll. A folder roll also offers **Also rename the folder on disk** (off by default). In a cloud-sync folder (Dropbox, iCloud, OneDrive) the sync can treat that rename as a delete and re-upload.
*   **Delete…**: forgets the roll only; folder, images and edits stay. **Import Folder as a Roll…** brings it back. **Clear Library** in *Manage Database* forgets all rolls.
*   **Roll Analysis** (**loaded** roll only): runs Roll Analysis ([§10.5](#105-roll-analysis)) and stores the roll's baseline.

#### Rolls that are not folders

A roll can also be a hand-picked set, and one photo can be in several rolls (folder rolls show a **folder** icon, others a **magnifier**). Search the library or gather frames, then **Save as Roll…**. The set is fixed, not a live search: add frames while it is loaded, or Save as Roll again.

A frame has one edit in every roll that holds it. On a frame in more than one roll, right-click → **Edit Independently in This Roll** gives it its own edit here; **Use the Shared Edit Again…** deletes that edit.

### Importing and managing files

**Nikon High Efficiency raw**: Z 8 and Z 9 NEFs in **HE** or **HE\*** use a licensed codec NegPy cannot decode. Shoot **Lossless Compressed**, or convert with Adobe DNG Converter.

The **⋮** menu on the Film Strip header, beside its ⓘ guide:

*   **New Roll…**: clears the film strip so you can drag in frames and keep them with **Save as Roll…**.
*   **Close Roll…**: empties the film strip and returns to the Library; **Unload All…** when the frames are not a roll. Asks first; edits stay saved.
*   **Reset Roll to Defaults…**: **Reset Settings** on every visible frame. Asks first; each reset is an undo step.

The grid button on the same header opens the **Light Table** (`Shift+G`): the roll as a full-window grid for culling, with the same selection, marks and menus. Double-click or **Enter** opens a frame; **Esc** or `Shift+G` goes back. It does not open while the strip shows no frames.

The Film Strip button row:

*   **Add** (import icon): **Add Files…** or **Add Folder…** (or drop a folder on the window). A folder with no images of its own imports its subfolders as rolls.
*   **Hot Folder**: loads new files as they appear in the current folder, for a scanner or tethering app.
*   **Trichrome Mode** and **Half Frame Mode** are on the Roll tab's Frame Assembly card ([§10.2](#102-frame-assembly)).
*   **Apply (clone)**: copies the current frame's settings, aspects chosen in a dialog, to selected frames or the whole roll. Crop and rotation stay per-image.
*   **Roll Settings** (tag icon): writes gear, capture, place, process and scanning metadata to the frame, a selection or the whole roll. Fields start from the active frame; **Load** fills them from a metadata preset; tick the groups to write. Empty Analog Gear is matched from the folder name, as at import.
*   **Save as Roll…** (red folder icon): keeps the loaded frames as a roll. See [Rolls that are not folders](#rolls-that-are-not-folders).
*   **Unload…**: drops the active frame or selection. For the whole roll, use **Close Roll…**.
*   **Show Scenes** (layers icon): edges each frame in its scene's color. See [Scenes](#scenes).
*   **Sort** (arrows): Name or Date, or **Scene** once the loaded roll has one ([Scenes](#scenes)), ascending or descending.
*   **Sheet filter** (funnel): *All Frames*, *Keepers Only*, *Hide Rejected* or *Unmarked Only*. **Advance After Marking** in the same menu moves to the next frame after a Keeper or Reject mark. `Home` and `End` jump to the first and last frame.

Above both sections sit the **filter box**, a **`.*`** regex toggle, a **search-library** button and, once enabled in Preferences, a **search-by-meaning** toggle. The Film Strip has a **tally** ("Ektacolor Pro 400 — 36 frames · 12 keepers · 3 rejected", or **Collection** for frames that are not one roll) and a **thumbnail size** slider. With a filter active the tally names it; if it hides everything, **Show all frames** clears it.

Right-click empty space for **Add Files**, **Add Folder** and **Close Roll…**. Buttons that do not fit move into a **»** menu.

#### Filtering the sheet

A plain word matches the filename. `field:value` terms match frame data:

| Term | Finds |
|---|---|
| `film:portra` | frames whose film stock contains "portra" |
| `camera:"Nikon F3"` | quote anything with a space |
| `iso:>=400` | numeric fields also take `>`, `>=`, `<`, `<=` (`iso`, `frame`, `push`, `devtime`, `temp`) |
| `date:2024-03` · `date:>=2024` | by file date; a partial date is a prefix |
| `shot:1998` · `shot:>=1998-07` | by capture date from the Metadata panel, not the file date |
| `place:tokyo` | by capture city, state or country |
| `devtime:>=9` · `temp:20` | development time in minutes, and temperature in °C |
| `roll:` `developer:` `dilution:` `lens:` `format:` `scanning:` | the rest of the Metadata panel |
| `name:` `path:` `ext:tif` | file identity |
| `scene:beach` | frames in a scene of the loaded roll whose name contains "beach" |
| `keeper:` `rejected:` `edited:` | frames carrying that mark, or with a saved edit |
| `-rejected:` `-film:velvia` | a leading `-` negates any term |

Terms combine with AND (`film:portra iso:>=400 -rejected:`). Metadata fields match only once filled in the **Metadata** panel. **`.*`** makes the box a plain filename regex.

#### Searching by meaning

**Search by meaning** (Preferences → Performance) finds photos by content; it downloads a small model the first time. Turn on its toggle beside the filter box and describe the picture ("a photo of a dog") to narrow loaded frames to the clear matches, best first. Without an index it ranks only frames loaded this session.

#### Searching the whole library

The **magnifier-over-folder** button (or **Enter** in the box) runs the search across the whole library and loads the results; with **Search by meaning** on, across every indexed file. Keyword search reads only stored edits, so metadata matches only frames you filled in.

#### Indexing the library

With Search by meaning on, a **database** button beside the Library refresh button indexes every photo in your library for it. It runs only when you ask and can be aborted; the next run does only the rest.

#### Stitching a frame from several shots

For a negative captured in overlapping pieces, select them and right-click → **Stitch Selected Frames**. NegPy aligns the overlap, matches brightness, cuts a seam and shows one composite named *a+b (Stitch)*, kept across launches. **Unstitch** restores the parts with their edits. **Merge Frame to TIFF Negative…** ([§10.2](#102-frame-assembly)) replaces the parts with one file and ends the composite.

For Trichrome, turn on **Trichrome Mode** first, then stitch the assembled frames; each part keeps its own three exposures.

#### Merging bracketed exposures (HDR)

For a slide with more range than one exposure: bracket it a stop apart, select the shots and right-click → **Merge Exposures (HDR)**. They become one frame named *a +4 (HDR)*; **Unmerge Exposures** restores them.

The merge is relative to the longest unclipped exposure, the **reference**, which is usually brighter than your metered shot. Choose the render exposure:

*   **Render exposure** (right-click a merged frame): pick a shot, listed in stops from the reference (marked *(as captured)*). Only the reference and shorter exposures are listed.
*   **Render Exposure** slider (Metering card): 0 EV (the reference) to −4 EV. It and a picked shot replace each other.
*   **Bracket middle (auto)** (default): the bracket's middle exposure, right for a bracket centered on the metered shot.

A merge opens with **Shadows Density** above zero, from the shadow detail the bracket recovered; set it to zero to match the metered frame. **Reset Settings** restores that value.

Bracket both ways, including the shot that already looks right: longer frames (+1, +2, +3 for deep shadows) extend the shadows; shorter ones (−1, −2) add render choices and keep a blown specular. Metered −2 through +3 is six frames; to drop some, drop long ones.

The merge takes its exposures' film process and exports under the first file's name with `-HDR` (`_DSC1715-HDR.jpg`).

Merging is for transparencies and shows on Transparency frames only; B&W reversal (Scala, dr5, Fomapan R) is not supported yet. Merged, stitched and Trichrome frames cannot be merged.

### Triage (culling the roll)

Thumbnails are positives. An unopened frame shows a quick inversion of a reduced preview (a neutral square when the format has none) until you open it. Transparencies are not inverted. Applying settings to other frames re-renders their thumbnails in the background.

Right-click a thumbnail, or use shortcuts, to mark frames (multi-selection works; marks persist):

*   **Keep**: a check badge.
*   **Reject**: a cross badge and dimming; batch exports and sidecar writes skip it. The file is never changed.

#### Reading the badges

| Corner | Badge | Means |
|---|---|---|
| Bottom-right | check | keeper |
| Bottom-right | cross, frame heavily dimmed | rejected |
| Bottom-left | *see below* | the frame was built from more than one file |
| Top-left | exclamation | the file failed to decode; click to retry |
| Top-left | small amber dot | the thumbnail is older than the frame's settings; open the frame or use **Update Thumbnails** |

The gray bottom-left glyph shows the frame type:

| Glyph | Frame |
|---|---|
| Two overlapping panes | a stitched composite ([§Stitching](#stitching-a-frame-from-several-shots)) |
| Three stacked bars | a merged bracket ([§Merging](#merging-bracketed-exposures-hdr)) |
| Three red/green/blue dots | a Trichrome triplet |
| A split rectangle, one side filled | one half of a half-frame scan; the filled side is which half |
| A split rectangle, both sides filled | a diptych: the whole scan, each half with its own edit |

The tooltip names it (*HDR merge of 5 exposures*).

The right-click menu also has:

*   **Copy/Paste Settings**, with or without normalization bounds (untick **Normalization bounds** in the paste picker to keep the frame's own). With several frames selected, Paste writes every one of them.
*   **Reset Settings**; with several frames selected, **Reset N Frames…**, confirmed first.
*   **Reset to Roll Settings**: **Reset to Roll** on every card that differs from the roll, as one undo step; the rest of the edit stays.
*   **Apply Settings…**.
*   **Sync Bounds…**: pushes only this frame's measured bounds (**Tonal span**, **Color balance**) to the selection or roll.
*   **Update Thumbnail(s)**: re-renders the selection's thumbnails; **Update Thumbnails** on the toolbar re-renders every stale one in the roll, from this session or an earlier one. Changes a thumbnail cannot show (sharpening, chroma denoise, dust and heal repairs, demosaic, display profile, soft proof) do not make it stale. Both become **Cancel** while running.
*   **Reset Roll to Defaults…**, and per-frame export.
*   **Edit Independently in This Roll** / **Use the Shared Edit Again**; see [Rolls that are not folders](#rolls-that-are-not-folders).

#### Scenes

A scene is a group of frames in one roll shot in the same light. It has its own baseline, so its frames match each other, not the rest of the roll. Scenes need a roll (**Save as Roll…** first).

*   **Scene** (right-click menu): **Group as Scene…** makes the selection a scene; **Add to** *name* and **Remove from Scene** move frames in and out. On a scene's frames, **Analyze Scene…** runs Scene Analysis; **Rename Scene…** and **Delete Scene…** manage it. A frame is in one scene at most; deleting a scene keeps edits and baselines.
*   **Sort → Scene**: each scene's frames as a block on a band of its color, frames in no scene last. Grouping a roll's first scene switches to it.

---

<!-- panel:analysis -->
## 3. Analysis readout (always visible)

The readout sits above the tabs and describes the current frame as you edit; drag the divider to resize or collapse it. Top to bottom:

#### Photometric curve

The paper characteristic (H&D) curve NegPy prints through: a model of the paper, not a curves editor. Horizontal is **negative density** (scene highlights to the right), vertical is **print tone**. Steeper means more contrast; the flat ends are the toe (shadows) and shoulder (highlights).

With a **Contrast Mask**, a violet band opens to a dashed edge: large flat areas print on the dashed edge, fine detail on the curve. Spatial tools (dodge/burn, CLAHE) do not show.

The crosshair marks the **pivot** that contrast rotates around. A faint **ghost** shows the previous curve while you drag. Separate R/G/B traces show cast removal's correction.

#### The two histograms

Behind the curve is the **output histogram** (print tones); along the bottom, the **negative density histogram** (the scan). Contrast cannot separate data on the flat toe: move the exposure onto the steep middle.

In Peek Negative the density histogram splits into R, G, B and luminance: a spike at an edge is that channel clipping, a trace apart from the others a strong cast.

#### LIN / LOG toggle

Bottom-right of the chart: the histogram's height axis. **LOG** compresses tall peaks so thin tails show (to find clipping); **LIN** shows the bulk of the frame.

#### Clipping triangles

R, G and B triangles in the top corners: **top-left** for crushed shadows, **top-right** for blown highlights, once a channel passes 0.5% of the frame. One channel alone is a cast, not an exposure problem.

#### Zone shading and zone ticks

The amber wash (left) and blue wash (right) mark the toe and shoulder, where separation is lost. The bottom ticks are Adams zones I to IX.

#### Step wedge

A 21-step wedge printed through the current curve, then the alternative process and toning, so it shows the print's own colors. Patches that merge into black or white are lost tones; the brackets mark the usable span, and a Sabattier wedge's span ends where its light tones fold back.

#### Zone strip

Ten cells on the Adams scale (**0** paper black, **V** 18% mid-gray); opacity shows how much of the frame lands in each. The end cells turn **red** when shadows block up or highlights blow.

Click a cell, then a spot on the photo, to place that tone (see Zone placement); Esc cancels. On a slide or a Positive frame the strip reads **Disabled for Slides/Reversal Film**.

Double-click the strip to turn the canvas **Zone Overlay** on or off (also `Shift+Z` and the canvas toolbar's grid icon): every region of the print is outlined and labeled with its zone.

#### Probe

A spot densitometer: hover the image for per-channel density above film base (ΔD), the print density and its zone.

#### Zone placement

Works like an enlarging analyzer: click a zone on the strip, then a spot on the photo, and the print is solved so the spot prints on that zone. Up to three pins; each row shows the current zone and its **target**, which − and + trim in thirds.

*   One pin solves Print Density.
*   Two pins (usually shadow and highlight) solve Print Density and ISO-R Grade.
*   A third pin adds **Shadows Grade**, **Highlights Grade** or **Snap**, whichever can move that tone; a line under the button names it.

The result is a preview until **Place zones** (or **Enter**) commits it as one edit, turning off Auto Density (and Auto Grade with two or more pins). **Esc** discards the armed zone, then the pins; **✕** removes one pin.

Drag a pin to move it; its caption reads `1 · IV⅓ → VI` (now → target) until they agree. A click with no zone armed only pins a reading. An unreachable target shows an amber `→ lands …` with the closest zone the print can make.

Pins are proofs, not edits: any other edit or a frame change removes them. Zone placement is off on an as-captured slide or a Positive frame.

#### Negative stats

Rows that measure the scan, not your edit; hover for details. A row with nothing to measure reads —.

*   **Negative**: relative density range (luminance) and development character: flat (≈N−1), normal, contrasty (≈N+1). Comparable across a roll; estimated from the normalized bounds, not a densitometer reading.
*   **Exposure**: midtone in stops from neutral, approximate; positive is high-key, negative low-key.
*   **Clipping**: share of pixels crushed to black or blown to white, worst channel. Red above 1%.
*   **Scan clip**: share of source pixels at or above sensor white, per channel. On a negative no edit can recover it: expose the scan lower. Red above 1%.
*   **Repair**: share of the scan rewritten by IR Restore, dust detection and heals. A large value means the threshold is redrawing the picture. Red above 5%.
*   **Gamut**: share of the frame the proof profile cannot print, while proofing to one. Zero is normal. Red above 2%.

---

## 4. Geometry tab

<!-- panel:geometry -->
### 4.1 Geometry: crop and straighten

**Crop:**

*   **Auto** (magic wand on the CROP header): detect the frame edge and crop to it. Its settings and the whole-roll run are on the Roll tab's **Crop** card ([§10.4](#104-crop)).
*   **Ratio**: the roll's crop ratio, the same field as on the Crop card; the crop tool snaps to it. **1:1.4142** is the ISO A paper shape (A4, A6).
*   **Crop** tool (crop icon on the CROP header): draw a crop rectangle; with **Ratio** at **Free**, an edge midpoint resizes one axis. It opens on the current crop; a manual change stops auto-detection. Drag past the viewport edge to pan. **Reset** (undo icon on the CROP header) clears the crop and turns auto-crop off.
*   **Guide**: *Thirds*, *Phi Grid*, *Diagonals*, *Golden Triangles*, *Golden Spiral*, *Armature*, *Diagonal Method*, *Grid* or *Off*. The redo button rotates guides with orientations (spiral 8, triangles 2).

**Alignment:**

*   **Crop by Default** (crop icon, right end of the ALIGNMENT header): crops the wedge Fine Rotation, Tilt and Swing leave, while no manual or auto crop is set.
*   **Fine Rotation** (±45°): sub-degree rotation, positive clockwise. Applied after auto-crop.
*   **Auto Skew** (square ruler, on the ALIGNMENT header): squares the frame to its film and frame edges, never to the picture, so a deliberately tilted photo stays tilted. It sets **Fine Rotation**, and **Tilt** or **Swing** where both edges of that pair show. A frame it cannot read confidently is left alone.
*   **Straighten** tool (ruler, on the ALIGNMENT header): draw a line along a horizon or vertical edge to level or plumb it; drag past the viewport edge to pan.
*   **Tilt and swing with reference lines** tool: drag a line along each rebate edge (top, bottom, left, right); once all four are marked, Tilt and Swing solve to make them parallel and square. Re-drag an edge to refine.
*   **Tilt** (±15%): tip the easel about a horizontal axis to correct converging verticals. Positive stretches the top edge. The unit is percent of the frame, not an angle.
*   **Swing** (±15%): the same about a vertical axis, for converging horizontals. Positive stretches the left edge.

    Both leave a wedge along the squeezed edge: crop it or use **Crop by Default**. Crop first: on an uncropped scan a large correction pulls the rebate into the metering and darkens the print.

Scanning-lens distortion, chromatic aberration and flat field are in **Optics** on the Roll tab ([§10.8](#108-optics)).

---

## 5. Exposure tab

Four panels set light, color and contrast in the print stage of the pipeline.

<!-- panel:color -->
### 5.1 Filtration: white balance

Color timing, like enlarger dichroic filters. The **Global / Shadows / Highlights** selector applies the controls to the whole image or biases them to low- or high-density tones.

*   **Pick WB** (eyedropper, right of the region selector): click a pixel that should be neutral gray; NegPy solves the CMY filtration for the selected region.
*   **Temperature**: warm-to-cool lever on the magenta/yellow pair; cyan stays put.
*   **Cyan / Magenta / Yellow** (-1 to 1): Cyan↔Red, Magenta↔Green, Yellow↔Blue.
*   **Ring-around** (target icon, or `Shift+F`): a 5×5 mosaic in 2cc steps to ±4cc on magenta and yellow, centered on neutral. Click a patch to keep its filtration; `Escape` or a second press clears it. See **Rotating a proof** below.

**Cast Removal** is on the Roll tab's Calibration card ([§10.3](#103-calibration-what-your-rig-does-to-the-colors)).

<!-- panel:tone -->
### 5.2 Tone: density and contrast

**Global / R / G / B** applies Grade and Split Grade to the shared curve, or as per-dye-layer trims for **crossover correction** (casts that differ between shadows and highlights).

**Automatic helpers**, in the **Auto** menu (magic-wand icon) beside the channel selector, on by default:

*   **Auto Density**: meters each frame's midtone and anchors print brightness there.
*   **Auto Grade**: sets the grade partly from the frame's textured density range, harder when needed so textured shadows reach black (Shadow Reach), opens the shadows of a contrasty frame (Shadow Hold), and holds textured highlights off paper white (Highlight Hold).
*   **Auto Density and Grade**: turns both helpers on or off together.
*   **Set Targets…** (last item in the Auto menu): the brightness and contrast the helpers aim for, for all frames.
*   With a helper on, its sliders show what prints: **Print Density** the metered density, **ISO-R Grade** the grade the frame prints at, **Highlights Density** with the automatic burn. Moving one trims the helper, the tick marks the helper's own choice and a double-click returns to it. Turning a helper off drops its share, so the sliders show your own values; turning it on adds it back.

**Test strip** (grid icon, or `Shift+T`): a 5×5 grid, Print Density rising left to right, ISO-R Grade softening top to bottom; one patch is your current setting. Click a patch to keep it; `Escape` or a second press clears it.

**Rotating a proof**: while a proof shows, the 90° **rotate** buttons and `[` / `]` turn the ladder, not the image.

**White Point** and **Black Point** are on the **Metering** card ([§10.6](#106-metering-negative--positive)).

**Exposure:**

*   **Print Density** (0.0 to 2.0): overall brightness (enlarger time). Lower is brighter.
*   **ISO-R Grade** (50 to 180): contrast as paper ISO-R. R110 is about grade 2; **lower R is harder**. In R/G/B mode a **Grade** trim rotates one layer's slope about the midtone.
*   **Shadows Density** / **Highlights Density** (±1.0 ΔD): brighten or darken only the shadows or highlights, within paper black and white. With Auto Grade on, each shows its automatic share (Shadow Hold's lift, Highlight Hold's burn). They also work on slides.
*   **Shadows Grade** / **Highlights Grade** (split grade, ±50 ISO-R): local contrast in the deep shadows or highlights.
*   **Preflash** (0 to 1): an even flash over the sheet, as a fraction of the paper's threshold exposure. It pulls highlight detail off paper white and softens the print slightly; bare paper stays white. Hidden on slides.
*   **Diffusion** (0 to 1, hidden in Transparency): a diffuser under the enlarger lens, a Softar or a stocking. The value is how much light it scatters: a stronger diffuser, or more of the exposure given through it. Shadows bleed into highlights and edges soften; flat areas print as before. Grays out in R/G/B mode.
*   **Contrast Mask** (±0.5, hidden in Transparency): a blurred mask sandwiched with the negative; the value is its signed gamma. Positive compresses the range by (1 − gamma) so a harder grade fits, keeping fine detail. Negative expands the range by (1 + gamma) for a negative too flat for Grade; past about −0.4, highlights clip. Past about ±0.2, strong edges get a halo.
*   **Mask Spacer** (2 to 6%, default 4%): the gap between mask and negative, as percent of the frame. Thicker masks only broad masses; thinner reaches into detail and hazes shadows next to bright areas. Both mask controls gray out in R/G/B mode.

<!-- panel:paper -->
### 5.3 Paper Response: the print curve

The paper's characteristic curve, drawn faintly behind the card header. Its own **Global / R / G / B** selector applies the curve controls to all layers or as per-dye-layer trims.

*   **Paper**: a bundled paper (RA4 in Color Negative, B&W papers in B&W Negative) that sets the curve; the other controls trim on top. *Neutral* gives the defaults. Each B&W paper has its own lith color (§6.2).
*   **Dye Separation** (0.5 to 1.5, hidden in B&W Negative): saturation in density space, through the paper's dyes, so it eases off at toe and shoulder. 1.0 is off; below pulls toward neutral. **Chroma** (Look tab) scales color evenly instead.
    *   **Separation Damping** (0 to 1): higher keeps the push on muted color and eases it on saturated color; below 1.0 separation, pastels go gray first. Grayed out at Dye Separation 1.0.
*   **Paper White** (page icon, right of the channel selector): simulate paper base density, so whites print at about 0.93.
*   **Paper Black** (circle icon, right of the channel selector): show the paper's slightly milky Dmax. Off (default) applies black-point compensation.
*   **Snap** (-0.5 to 0.5): midtone gamma; paper white and black stay put.
*   **Toe** (-1 to 1) + **Toe Width** (0.1 to 5): shadow roll-off. Positive lifts shadows; negative deepens them and, with Paper Black off, reaches exact black. Width sets how far the knee reaches.
*   **Shoulder** (-1 to 1) + **Shoulder Width** (0.1 to 5): highlight roll-off. Positive compresses highlights; negative extends them and can clip.

In R/G/B mode these become per-layer trims: **Toe** / **Shoulder** (±1), **Toe Width** / **Shoulder Width** (±2), **Snap** (±0.5), **Dye Separation** (±0.4).

<!-- panel:local -->
### 5.4 Dodge & Burn: local exposure

Draw masks and lighten or darken only those areas. On a **Slide** the panel grays out, because the slide's transfer curve takes no masks; the frame keeps them. The **MASKS** header shows how many the frame has:

*   **Draw** (Draw Mask, the cut card): click to place vertices; double-click, Enter or click near the start to close; Esc cancels. To edit, select the mask, then drag a vertex, click an edge "+" to add a point, or right-click a vertex to delete it.
*   **Oval** (the hole in the card, or a dodging wand): drag out an oval. The center handle moves it; the other two set each axis, so you can stretch and tilt it.
*   **Card** (Card Edge, the graduated burn): drag from the full-exposure edge (solid line) to where it fades out (dashed). The gap is the softness, so **Feather does nothing on this shape**.

Handles can go into the gray area outside the frame. A tilted Card Edge usually must start past the corner it burns.

*   **Mask list**: shape icon, Dodge, Burn, Grade or Flash, and values. The shape icon enables or disables the mask; the yin-yang inverts it (it acts outside its shape); the eye toggles the outline (shown only on the Exposure tab); the trash deletes it.
*   The canvas tint of the current mask, and of masks that intersect it, hides while you drag **Burn**, **Feather**, **Grade**, **Flash** or a vertex.
*   **SELECTED MASK**: the controls below act on the mask selected in the list, and gray out with none selected.
*   **Burn** (-2 to 2 stops, default 0): **positive burns** (darker), **negative dodges** (brighter), like Print Density and the Finishing edge burn.
*   **Feather** (0.0 to 0.15): edge softness, as a fraction of the frame's short side.
*   **Grade** (-40 to 40 R): the mask's own contrast, in ISO-R points off the frame's Grade, negative harder (burn a sky at −20 R). It pivots on the region's midtone; overlapping grades add, within R50…R180.
*   **Flash** (0 to 1, default 0): pre-exposes the paper under the mask, as a fraction of the threshold exposure like the Tone card's Preflash: the region's highlights compress without the density a burn adds (flash a sky through a card). Overlapping flashes add. Off on a tone-limited mask.
*   **Tone Limit** (*All*, *Highlights*, *Shadows*) with **Tone Zone** (0 to 10, in thirds, default 6) and **Tone Softness** (⅓ to 3 zones): limits the mask to tones lighter or darker than a print zone, so a sky burn on *Highlights* at VI stops at the skyline. The tint shows the tones it selects. Up to four tone-limited masks per frame.

**Printing Notes** (Export tab, or **Shift+N**) makes a marked-up work print: each mask outlined with its number and value in stops, and a corner card with the paper, exposure, grade, filtration, curve and dodge/burn list.

*   Burns are hatched, dodges left open.
*   Numbers are exposure, not brightness: +1.00 st is `Burn +1`; values snap to ⅓, ½ and ¼ when close.
*   A mask with a local **Grade** shows the grade it prints at: `Burn +1 @ R95` (−20 R on an R115 frame), or `Grade @ R95` for a grade-only mask. A tone-limited mask adds its zone: `Burn +1 on ≥VI`.

Hidden outlines stay on the map; disabled masks do not.

---

## 6. Look tab

<!-- panel:lab -->
### 6.1 Lab: polish and detail

What a lab scanner (Frontier or Noritsu) does automatically.

**Color** (hidden in B&W Negative):

*   **Chroma** (0.0 to 2.0): even color scale after decode; 1.0 is unchanged, 0 grayscale. Out-of-gamut pixels roll off softly, keeping hue. For print-like saturation use **Dye Separation**.
*   **Skin Protection** (0.0 to 1.0, default 0.5): caps skin-hued chroma; 0.5 catches excessive chroma, 1.0 leaves skin matte, 0 is off. Red coats, sunsets and brick stay out; wood, tan leather and sand soften with it.
*   **Chroma Denoise** (0.0 to 5.0): smooths color noise, mainly in shadows; luminance grain stays.

**Detail:**

*   **Sharpening** (0.0 to 1.0): amount, on the L (lightness) channel, so no color halos.
    *   **Method**: *Unsharp Mask* (edge contrast) or *Deconvolution* (Richardson-Lucy, reverses the scanner's blur; set Radius to the blur width).
    *   **Radius** (0.5 to 3.0 px): blur width in output pixels; judge it at 100% zoom.
    *   **Masking** (0.0 to 1.0): limit sharpening to edges to protect sky, skin and grain.
*   **CLAHE** (0.0 to 1.0): local contrast without blowing highlights or crushing shadows. Near 1.0 it can look cartoonish. Runs before dust removal.

**Effects:**

*   **Glow** (0.0 to 1.0): lens bloom across all channels.
*   **Halation** (0.0 to 1.0): red glow from light scattering back through the film base, highlights only.

<!-- panel:altproc -->
### 6.2 Alternative Processes

Pick **None / Lith / Cyanotype / Sabattier**; only that process's controls show. B&W Negative only, off by default.

#### Lith

Heavily over-exposed lith paper in dilute developer, snatched part-way: creamy warm highlights and an abrupt drop into sooty blacks. The Exposure tab's paper sets the color: *Neutral* and the Ilford papers are almost colorless, Fomatone gives peach and olive. In Toning only Selenium and Gold stay, and act differently (§6.3).

*   **Exposure** (0 to 5 stops, default 2): over-exposure; real lith uses two to four stops. More gives warmer, more colorful highlights and softer gradation.
*   **Snatch Point** (0.0 to 1.0, default 0.55): time in the developer. Higher gives deeper, colder blacks and more flat shadow; lower stays high-key and warm with weak blacks.
*   **Abruptness** (0.0 to 1.0, default 0.6): how suddenly shadows go black (the developer's hydroquinone-to-alkali ratio). High blocks up the next zone down; low rolls off gently.

#### Cyanotype

UV contact print on iron-salt paper: Prussian blue instead of black, with green highlights from leftover sensitizer. Its short density range clips a normal negative at both ends. Chemical toners gray out (no silver); use Bleach and Tannin. Split toning still works.

*   **Sensitizer** (Classic or New, default Classic): *Classic (Herschel)*, ammonium ferric citrate, tops out at a light blue with a green highlight stain. *New (Ware)*, ferric oxalate, goes deeper and cleaner.
*   **Exposure** (-2 to 4 stops, default 0): UV time. More moves more of the scale into blue.
*   **Exposure Scale** (0.8 to 2.8 log D, default 1.4): the printable density range, the contrast control; shorter is more contrasty. Traditional cyanotype is about 1.0 to 1.2, Ware's new about 2.4.
*   **Bleach** (0.0 to 0.5, default 0): washing soda; removes blue, highlights first.
*   **Tannin** (0.0 to 0.5, default 0): tea, coffee or tannic acid; turns bleached iron brown and a little deeper. Bleach first for full brown; Tannin alone for split blue-brown.

#### Sabattier

The print is flashed with white light part-way through development: the light tones reverse and fold back toward gray, the dense ones hold, and a light Mackie line runs along every edge between dark and light. The print stays neutral silver, so every toner works on it.

*   **Strength** (0 to 2, default 1.3): the second exposure's length. Below 1 the light tones flatten toward the fold; above 1 they reverse, the lighter the darker.
*   **Re-exposure** (0.1 to 0.9, default 0.45): where the fold sits, as a fraction of the paper's Dmax. Tones denser than it hold.
*   **Agitation** (0 to 1, default 0.7): rocking the tray after the flash. In a still bath (0) the shadows' bromide creeps into the light tones and draws wide Mackie lines; constant agitation (1) washes it away and draws none.

---

<!-- panel:toning -->
### 6.3 Toning

Chemical toners (B&W Negative only) and a split tint (any mode). On a lith print toners bite harder: only Selenium and Gold stay enabled. With Cyanotype all six gray out.

**Chemical Toning**, sequential baths in the order shown, each 0.0 to 2.0:

*   **Selenium**: deeper blacks, cool eggplant shadows. On lith: further down the scale, strong Dmax lift, green-black shadows to magenta.
*   **Sepia**: warms highlights first, and more strength reaches further into the mids; the deepest shadows stay black at any strength.
*   **Gold**: blue-black on untoned silver; over sepia, orange-red highlights. On lith: all densities evenly, toward blue-violet.
*   **Iron Blue**: Prussian-blue shadows to navy blacks.
*   **Copper**: pink to brick-red, with the classic Dmax loss.
*   **Vanadium**: greens mids and highlights; deep shadows stay black.

**Split Toning** (all modes), an additive Lab tint that keeps grain and detail:

*   **Shadow Strength** (0.0 to 1.0), with **Shadow Hue** (0 to 360°) under it.
*   **Highlight Strength** (0.0 to 1.0), with **Highlight Hue** (0 to 360°) under it.

---

## 7. Finish tab

<!-- panel:retouch -->
### 7.1 Retouch: dust, hairs, scratches

Spotting, as with a brush on a finished print. Marks are found by local contrast, by the scanner's IR channel or by hand, and the three stack. Each mark is rebuilt from the clean film around it, grain included.

**Overlay** (Off / Marked / IR, at the top): shows detections, green for Optical Removal, magenta for IR (**Color vision** in Preferences picks other colors). IR needs an IR plane.

**Optical Removal** finds specks and hairs on the visible scan, with no IR needed:

*   **Spot Threshold** (0.01 to 1.0): the bar for specks, measured against the film's grain. Lower catches more, with more false positives; 1.0 turns speck detection off. Dust in busy detail may need a lower value, IR or Heal.
*   **Hair Threshold** (0.01 to 1.0): the same for hairs and long thin marks; it rises along tonal edges, so a bright rim is not taken for a hair. Lower it if a hair across busy detail is missed; 1.0 turns hair detection off.
*   **Size** (2 to 8 px): max spot radius. A mark covers the whole speck or hair.
*   To protect detail, right-drag on the canvas to paint an exclusion band (Brush Size wide, amber with the overlay on), or right-click → **Exclude from Optical Removal** for one spot. Toggling **Optical Removal** clears every band.
*   The cursor button beside **Optical Removal** makes a plain right-click exclude the spot, with no menu; the canvas menu is then out of reach while removal is on.

**IR Removal** uses the scanner's infrared channel; it is enabled only when the scan has one.

*   **IR Threshold** (0.05 to 0.95): lower catches more.
*   **Method** (beside **IR Removal**): how the film under a defect is rebuilt.
    *   **NegPy** (default): divides semi-transparent dust back out and fills opaque cores from the clean film around them.
    *   **OpenICE**: keeps picture detail under a speck and gives solid defects Digital ICE's synthetic grain; clean film stays untouched. Better on fine detail but less proven across scanners: compare both on a frame you know.
*   IR is read from 4-channel TIFFs and DNGs (VueScan, NegPy's scans), SilverFast iSRD TIFFs, 64-bit **HDRi RAW DNGs** (plain HDR has no IR) and `_IR.tif` sidecars. B&W and Kodachrome frames are skipped, since they block infrared like dust.

**Manual Heal** (the header shows the spot count). The brush marks a search area, not a stamp: only pixels that stand out from the film around them are rewritten, so paint generously.

*   **Heal**: click dust spots to paint them out, or drag over a run of them.
*   **Scratch**: click points along a scratch or hair, then double-click or press Enter. Esc cancels, Backspace removes the last point. Right-click an overlay to delete it.
*   **Line** (Transport Line): for long straight transport scratches across the frame. Click once anywhere on the scratch to trace and repair the whole line; hovering shows the band it would repair. Right-click a placed line to delete it.
*   **Line Sensitivity** (0.05 to 0.95, live with the Line tool): lower catches fainter lines with a wider band; raise it if a line picks up film on either side. Applies to placed lines too.
*   **Brush Size** (2 to 64 px): diameter of the heal, scratch and exclusion brushes, for new strokes. It stays the same from frame to frame; strokes already made keep their own size. Hold `Alt` and scroll on the canvas, or pinch, while a brush is live.
*   **Undo Last** / **Clear All** (undo and bin icons, on the MANUAL HEAL header): remove the last or all manual heals and traced lines; auto-detected dust is unaffected.

**Clone** (the header shows the stroke count): copies film from another area over a defect the heal cannot rebuild.

*   **Clone**: the first click picks the area to copy from; then click or paint over the defect. The source follows the brush at a fixed offset, shown as a dashed circle. Uses **Brush Size**. Right-click a stroke to delete it.
*   **Set Source**: the next click picks a new area to copy from. `Alt`-click with the Clone tool, or **Pick New Source** in the right-click menu, does the same.
*   **Match Tone** (default on): keeps the source's grain but takes the destination's brightness and color.
*   **Strength** (0 to 100%, default 100%): how much of the source covers the destination.
*   **Feather** (0 to 100%, default 50%): the soft edge, as a share of the brush radius; 0 is hard.
*   **Undo Last** / **Clear All** on the CLONE header remove the last or all strokes. Each stroke keeps the settings it was painted with.

<!-- panel:finish -->
### 7.2 Finishing: vignette, carrier, border

How the print is presented, at the end of the pipeline. The crop, analysis-region and tilt/swing tools show the frame without carrier and border.

**Vignette** (printer's edge burn):

*   **Burn** (-2.0 to 2.0 stops): positive darkens the edges, negative lightens them. 0 is off.
*   **Size** (0.0 to 1.0): falloff radius, from tight in the corners to spread into the frame.
*   **Roundness** (0.0 to 1.0): 0 is radial (lens-like), 1 is a rectangular card burn along the print edges.

**Filed Carrier**: the clear rebate of a filed-out carrier prints black, framed by unexposed paper, through the frame's own curve and filtration. The film sits off center, so the top and left rebates print wider.

*   **Width** (0.0 to 5.0 mm): black frame thickness. 0 is off.
*   **Roughness** (0.0 to 1.0): how raggedly the outer edge was filed.
*   **Flare** (0.0 to 1.0): light off the filed bevel fogs the paper just outside the edge. 0 is off.
*   **Corners** (0.0 to 1.0): how far the filed corners round off.

The paper margin takes the border color.

**Border:**

*   **Width** (0.0 to 2.5): thickness as a fraction of the image. 0 is no border.
*   **Bottom Weight** (1.0 to 2.0): thickens the bottom, for window-mat proportions.
*   **Color** (*Paper White* or *Custom*): *Paper White* tints the mat with the toned paper white; *Custom* uses the color from the swatch button beside it.

---

## 8. Favorites tab

The sliders you use most, in one place. Empty until you fill it.

*   **Edit Favorites**: tick sliders on the left, drag them into order on the right, then press **Apply**.
*   They are the same controls as in their home panels; a favorite hides when its original does.

<!-- panel:presets -->
### Presets

Saves and recalls edit settings by name.

*   **Apply** (or double-click a preset): apply the selected preset to the current image.
*   **Save…**: pick which of the current settings to store as a new preset.
*   **Pen** and **Trash**: edit or delete the selected preset.

---

## 9. History tab

Two lists: the versions you chose to keep, above the record of every change.

### Work prints

A **work print** is a named version of this frame, like the test prints kept on the way to the final one.

*   **Save Work Print** (save icon on the WORK PRINTS header, or **Ctrl+Shift+S**) keeps the current edit under a name. Saving over a name asks first.
*   **Click** one to make it live. That is an edit, so **Ctrl+Z** restores the previous state.
*   **Right-click** for **Export This Version…**, **Rename…** or **Delete**. Delete asks first.

Work prints are never pruned, unlike the undo history. They belong to the frame and live in the database, not in `.negpy` sidecars.

### Edit history

Every edit step, the last 100 kept, newest on top. The current step is bold.

*   **Click** a step to jump to that state.
*   **Right-click** → **Export This Version…** to export a past state.

---

## 10. Roll tab

Every card here is shared by the roll, with the scope pair of [§1](#frame-or-roll-the-scope-pair): an edit makes the card **Frame** until it matches the roll again, and a push to the roll leaves frames that locked the card alone. The line above the cards names every card this frame overrides.

<!-- panel:film -->
### 10.1 Film Mode

Decides which cards apply: **Color** (C-41 negative), **B&W** (negative) or **Slide** (transparency, E-6 and similar). The wand button auto-detects the mode when a file loads.

**Positive** (default off, **Slide** only): for a source that is already a positive (a scanned print, another app's export). NegPy reads its embedded profile (sRGB if none) and skips metering and inversion, so the tone controls shape the image directly. Leaving Slide turns it off.

<!-- panel:assembly -->
### 10.2 Frame Assembly

How the files become frames. Neither toggle has a scope pair.

#### Trichrome

*   **Trichrome Mode** (three-exposure narrowband capture): assembles each frame from a red, green and blue exposure, grouped by capture time (else filename order), so shoot each frame's three back to back. Shots that are not one of each color of the same frame stay separate, for pairing by hand. An assembled frame has the three-dot badge ([Triage](#triage-culling-the-roll)).
*   **Edit Triplet…** (pen icon): the **Edit RGB Triplet…** dialog for the current frame. **Align channels (sub-pixel)** registers green and blue to red, removing color fringes.
*   **Merge Frame to TIFF Negative…** / **Merge Selected to TIFF Negative…** (right-click): replaces each triplet or stitch with one 16-bit linear TIFF negative beside its first source (`<red name>_RGB.tif`, `<first part name>_STITCH.tif`). It renders the same and keeps the edit, marks and scene. A stitch also bakes in flat field and sensor correction and cannot be unstitched. **Move each merged frame's source files to the Trash** (on by default) trashes the sources once the file is verified. A frame already merged is left alone (delete its negative to merge again); brackets, slides and LinearRaw DNG sources are skipped.

The line under the buttons names the two exposures the frame is assembled from. Each roll remembers its own Trichrome Mode; a new roll, or a batch that is not one roll, takes the mode last chosen.

#### Half Frame

**Half Frame Mode** splits each scan into two frames, for half-frame cameras; each half is edited and metered on its own. Turning it on detects the gutter, its direction and the film crop on every loaded scan. Each roll remembers its own state; it is disabled for a batch that is not one roll.

*   **Adjust…**: drag the green box to crop and the orange line to set the split; pick **Split direction** (*Vertical* cuts left/right, *Horizontal* top/bottom) and **Cut thickness** (the separator band to discard). **Auto-detect** re-finds all three. **Apply**'s ▾ applies to the current, selected or all frames (all sets the roll default).
*   **Detect All**: re-runs the batch detection.
*   **Unsplit** (enabled on a split frame): reverts it, as does its right-click item.

Right-click a half for **Adjust Split for This Frame…** and, with an override, **Reset Split to Roll Default**. Heals and masks stay anchored to the film through a recrop or resplit.

Turning Half Frame off keeps edited halves: the scan becomes a **diptych**, both halves side by side with their own edits (one edited half is used for both). A diptych exports as `<name>-DIPTYCH`, a long-edge size applying to each half; turn Half Frame on to edit it. **Unsplit** makes it one plain frame and deletes both halves' edits.

Half Frame never splits a frame assembled from several files (a Trichrome triplet, a stitch, an HDR merge).

<!-- panel:sensor -->
### 10.3 Calibration: what your rig does to the colors

These controls correct the capture, not the look: sensor filters, film dyes and the light source each have their own control.

**Capture**:

*   **Scanning setup** (bulb icon on the CAPTURE header): a wizard (*how do you scan?*, *what light source?*) that sets Linear RAW and Narrowband. It runs once after the first-launch tour; reopen it when your rig changes.
*   **Linear RAW** (default off): decodes RAW with neutral multipliers; off uses the as-shot white balance. Locked on for a Trichrome triplet.
*   **Narrowband**: corrects the oversaturation of narrowband (RGB-LED) capture with a bundled input profile; leave it off for broadband light. An Input ICC in Export overrides it. Grayed out on Transparency.

What the wizard sets:

| Capture | Light source | Linear RAW | Narrowband |
| --- | --- | --- | --- |
| Digital camera | White light (lightbox, CRI LED panel) | off | off |
| Digital camera | Narrowband RGB (Scanlight, RGB LED) | on | on |
| Film scanner | White light (Plustek, Epson, most flatbeds) | on | off |
| Narrowband Scanner | Nikon Coolscan, Kodak Pakon | on | on |

Applying it sets the defaults for new files and rewrites every edited frame in the session (Ctrl+Z per frame).

**Single-Shot Narrowband Calibration**: for single-shot camera scans under narrowband light, where each color leaks into the others through the sensor's filters.

*   **Profile**: the sensor matrix. Custom `.toml` matrices go in `<Documents>/NegPy/sensor/`.
*   **Method**: how the matrix is applied. *Two-Scale* (default) is *Linear* wherever the calibration can be trusted; where a color is mostly leak, as in neon or deep blue, it stops the speckle and keeps the grain at the film's own. *Linear* subtracts the leak exactly and prints those colors as speckled, fully saturated color. *Density* applies the matrix to densities: no speckle, but strong colors come out slightly less vivid. Two-Scale and Density read the film base color from the frame; a scan clipped there uses Linear.
*   **Calibrate** (vials icon on the header): build a profile from three bare-light R/G/B exposures. Pick them as files, or press **Capture from Camera…** to shoot and measure them with a tethered camera and a Scanlight, with no film in the holder; it asks before the first exposure, and the button is grayed out until both are connected. A Single Capture Scanlight preset can save one during its own calibration.

Needs **Linear RAW**; grayed out on Transparency and on a Trichrome triplet. **Re-run Roll Analysis** after changing it. CAMERA_SCANNING.md has a workflow for each way to build and assign a profile.

**Crosstalk** (hidden in B&W Negative): a channel unmix on the densities before inversion. Dyes, light and sensor all mix the channels, so a matrix describes your whole scanning setup.

*   **Matrix**: matrices for the current film process, grouped by source. *Generic C41* is built in; custom `.toml` matrices go in `<Documents>/NegPy/crosstalk/` (see [CROSSTALK.md](CROSSTALK.md)). The slider button opens the editor, with **Type** and **Process** (where the matrix applies); **+** makes one for the current process.
*   **Strength** (0.0 to 1.0): how much unmix to apply. **Re-run Roll Analysis** after changing it.

> The bundled film matrices, marked *(approx)*, come from spec sheets and describe the dyes alone. They fully correct only a capture that reads each dye cleanly (a Narrowband Scanner, Trichrome, or a calibrated single-shot narrowband rig); for broadband light and a Bayer sensor they are a starting point.

> No Transparency matrix ships. On slides, Matrix and Strength stay disabled until you make one with **+** or add a `.toml` with `process = "Transparency"`. Unmixing moves a slide away from its own look: treat it as a color-separation control.

To tune a matrix, adjust its six off-diagonal terms in the editor and save it as your own, named after the *combination* ("Gold 200 + Spectracolor"). Working profiles are welcome [upstream](CROSSTALK.md#contributing-a-matrix).

**Dye balance: Cast Removal** (0.0 to 1.0, color only): balances each layer against the frame's own grays so neutrals stay neutral from shadows to highlights. On Color Negative it removes the orange mask and starts at 1.0; on Transparency it starts at 0, for a faded slide's crossover. Hidden for B&W Negative.

**Light source:**

*   **Hue Trim** (-30° to 30°, default 0): rotates every hue by a fixed angle, for narrowband or odd-phosphor lights that turn yellows orange and greens olive; white balance cannot fix that. Judge it on a known color; leave it at 0 for broadband light. It carries to the next file.

#### Narrowband and slides

Narrowband and Single-Shot Narrowband Calibration do not apply to Transparency; they stay grayed with their values for the next negative. For slides on a narrowband rig, use **Hue Trim**.

<!-- panel:autocrop -->
### 10.4 Crop

The shape every frame is cut to and what the frame detector looks for, roll-wide. Each frame's rectangle is its own, drawn or found in **Geometry** ([§4.1](#41-geometry-crop-and-straighten)).

*   **Ratio** (default `Free`): `Free`, `1:1`, `3:2`, `4:3`, `5:4`, `6:7`, `7:5`, `65:24`, `16:9`, `16:10`, `11:8.5`, each auto-oriented. On `Free` auto-crop uses the detected format (6x6, 645, 6x7, 35mm); a ratio forces every frame to it.
*   **Detect** (crosshairs): snap the ratio to the closest standard.

**Auto Crop**:

*   **Mode**: *Image only* (exposed area) or *Film edge* (full film, including rebate and sprockets).
*   **Crop Offset** (-5 to 100 px): inset the detected edge; negative bleeds slightly outside.
*   **Rebate Trim** (0 to 150%): 0% stops at the film edge, 100% at the detected image edge, above 100% cuts into the picture to clear a white border. *Image only*; applies to **Frame** and **Roll**.
*   **Auto-crop this frame** (magic wand on the AUTO CROP header): detect this frame's edge and crop to it, the same toggle as **Auto** in Geometry. Off clears the crop.
*   **Auto-crop the roll** (layers icon on the AUTO CROP header, *Image only*): crops all visible landscape frames together, calibrating weak detections from confident ones, and straightens each as **Auto Skew** does. Portrait frames are cropped alone. Manual, Film-edge and ambiguous frames are left alone.
*   **Mixing scans**: allowed. A frame that reads its own edge keeps its crop; a frame with no edge takes the roll's width and tilt. Frames from one camera, holder and format pool best.
*   **When auto-crop leaves a frame alone**: the scanner bed must be the brightest thing in the scan. A slide with highlights as bright as the bed, sprocket-exposed film, or a neighbor frame filling more than a tenth of one side stays uncropped. Crop by hand, or use *Film edge* and trim in.

A change re-detects every following frame cropped by **Auto**. Hand-drawn crops are kept, except that **Ratio** reshapes them around their center.

<!-- panel:baseline -->
### 10.5 Roll Analysis

Meter the roll once and share the result, so frames of one film match. The **Use average** toggles are this card's roll defaults.

*   **Use average: Luma**: take the picked roll's tonal range; color stays per frame. Disables Luma Range Clip.
*   **Use average: Color**: take the picked roll's color balance; tonal range stays per frame. Disables Color Clip. Luma and Color on gives a consistent roll; both off gives per-image auto-exposure.
*   **Use average: Cast** (Color Negative only): take Cast Removal's gray balance from the roll or scene analysis, so frames under one light render alike. Analysis turns it on, except on frames far from the rest; grayed out until one has run.
*   **Baseline** (line shown while either average is on): names the roll or scene analyzed, or the frame **Sync Bounds…** took it from, and warns when there is none.
*   **Rolls** (picker): search every roll in your library; a ticked roll has a saved baseline. Defaults to the loaded roll. Picking one loads its baseline at once.
*   **Reanalyze** (gauge, on the ROLLS header; also in the Library, [§2](#2-film-strip-left-panel)): averages density and color balance over the loaded roll, scene and locked frames excluded, and saves the roll's baseline. Frames far from the rest keep their own; the status message names them. Run **Auto-crop the roll** first for consistent crops.
*   **Use This Frame** (crosshairs, on the ROLLS header): saves this frame's bounds as the roll's baseline, for a reference frame.
*   **Scenes**: the loaded roll's [scenes](#scenes), ticked once analyzed. **Analyze** runs Scene Analysis (Reanalyze over the scene only); **Select** selects its frames; **Delete** forgets it. Roll Analysis skips scene frames.

<!-- panel:process -->
### 10.6 Metering: negative → positive

How this frame is measured into a positive's tonal bounds.

**Slides** (Transparency) render as captured, through the camera's color matrix and a fixed tonal window, as in Photoshop or Darktable.

*   The paper controls and the normalization tuning hide. **Print Density**, **ISO-R Grade**, **Toe** / **Shoulder** and their widths, **Shadows Density** / **Highlights Density**, the R/G/B trims and white balance form a transfer curve, neutral at defaults.
*   **Auto Density** and **Auto Grade** start off on a slide; turn them on to meter a faded or expired one. On a merged bracket they are grayed out (**Render exposure** sets the exposure).
*   Lightroom mapping: **Exposure** → Print Density (lower is brighter), **Contrast** → ISO-R Grade (180 is softest), **Shadows** → Shadows Density, **Highlights** → Highlights Density. Positive adds density, so negative Shadows Density opens shadows. **Whites** and **Blacks** have no equivalent.
*   A source with no camera matrix (a scanner TIFF, a JPEG) passes straight through.
*   **Linear RAW** is grayed out (it renders the same either way), except with **Positive** on. An Input ICC in Export replaces the camera matrix; the as-shot white balance still applies.
*   **Narrowband** and **Single-Shot Narrowband Calibration** are grayed out ([Narrowband and slides](#narrowband-and-slides)).

**Analysis** sets where the black and white points are metered.

*   **Analysis Buffer** (0.0 to 0.25): insets the measurement window so rebate, sprocket holes and scanner borders do not skew it. Raise it for wide borders.
*   **Reanalyze Frame** (circular arrow, on the ANALYSIS header): measures this frame again from its current crop, buffer and region. Grayed out with Lock Bounds on or with both averages on.
*   **Draw Region** / **Clear Region**: draw a freehand region to meter *exactly* that area, overriding the buffer. Double-click inside to confirm.
*   **Clipping** (line under the ANALYSIS header): the print's shadow and highlight clipping shares, worst channel, the Analysis panel's row; amber above 1%. The card's header carries the print's luma histogram.
*   **Lock Bounds** (lock icon, on the ANALYSIS header): freezes this frame's bounds against crop and slider changes and against every Roll Analysis run.

**Tonal Range**:

*   **Luma Range Clip** (-100 to 100): how tightly the black/white-point span is set. Neutral applies a small robust clip. Positive tightens it, for dense or fogged negatives; negative pushes the bounds *outward*, for lifted blacks and unclipped highlights.
*   **Color Clip** (-100 to 100): the per-channel color-balance clip (orange-mask removal). Positive tightens; negative samples nearer the extremes.

**White / Black Point** (-0.25 to 0.25), with a **Global** / **R** / **G** / **B** selector: offsets on the detected bounds; positive white point brightens, positive black point lifts blacks. In R/G/B they trim each layer's Dmin and Dmax. On a slide they offset its fixed window; elsewhere **Lock Bounds** disables them.

<!-- panel:demosaic -->
### 10.7 Raw Decode: turning the sensor mosaic into pixels

The demosaic fills in the two colors each photosite misses, which affects sharpness and grain. Bayer and X-Trans RAW only; only algorithms in your LibRaw build are listed.

**Demosaic**:

*   **Preview** / **Export** (sticky, default **Auto**): *Auto* is a fast half-size decode on screen (full-size PPG on X-Trans) and AHD for export. For the preview, Auto and Linear are fastest; the others decode at full size. **AHD** is balanced, **VNG** smooth, **PPG** fast with clean edges, **DCB** and **DHT** favor fine detail, **AAHD** softens edges to suppress artifacts.
*   A hint under them says so when the open frame is not a Bayer or X-Trans RAW.

**Highlights: Recovery** (Transparency only, default **Off**): rebuilds a clipped highlight on a camera RAW. **Off** leaves it flat (magenta if one channel clipped first); **Blend** makes a plausible neutral, right for sun, sky, chrome or glass; **Reconstruct** is stronger and can misjudge a saturated highlight. Grayed out on rendered files and merged brackets.

<!-- panel:optics -->
### 10.8 Optics

The scanning optics: one lens correction and one light correction for every frame of the rig. One scope pair covers both.

#### Lens Correction

*   **Embedded** (the file's own lens profile for DNG, Sony ARW and Panasonic RW2, enabled when the file has one; RW2 carries distortion only):
    *   **Distortion**: straightens curved lines in place of the manual correction, scaling the image to fill the frame. Set it before cropping or retouching.
    *   **CA**: reduces color fringes. Works with or without **Distortion** and manual correction.
*   **Distortion Correction** (-0.100 to 0.100, in steps of 0.001): positive corrects barrel, negative pincushion. Use the film rebate as a straight edge. Applied before Tilt and Swing. Grayed out while the embedded **Distortion** is on.

#### Flat Field Correction: even out the light

Corrects uneven illumination (vignetting, falloff) from a copy-stand or scanner light, using a shot of the bare light source.

*   **Profile**, with **+** and **trash** on the FLAT FIELD CORRECTION header: **+** bakes a reference image into a named profile in NegPy's `flatfield` folder, after which the image can be deleted. **Trash** asks first; every frame using the profile loses its correction.
*   **Apply Flat Field** (bulb toggle beside the dropdown): apply the selected profile to this roll, enabled once a profile exists.
*   **Check Flat Field** (eye toggle): shows how well the selected profile corrects its own reference, from any frame and without the reference file. A good profile shows an even gray; a band or patch shows where it is off, and the stronger it is, the larger the error: full white or black is 10% or more. The status line gives the spread. A profile saved before this check shows the light it corrects instead; save it again to check it. Saving a profile warns when its reference is clipped or does not correct itself evenly.

A newly chosen profile becomes the rig's default for the next roll.

---

## 11. Metadata tab

Metadata for the original analog capture (camera, lens, film, process), written into every export as EXIF and XMP, so a DAM such as Lightroom shows your film gear, not the scanner. **Protect Original Metadata**, at the top, copies the source's own EXIF/XMP and resolution unchanged instead, and grays out the cards below.

**Analog Gear**, **Capture**, **Process**, **Scanning** and **Exposure** are roll-wide by default, with the scope pair ([§1](#frame-or-roll-the-scope-pair)). The frame number is always per frame.

<!-- panel:metadata_presets -->
### Metadata Presets

Saved sets of metadata values in `~/NegPy/presets/metadata/`, separate from the Favorites tab's edit presets. You manage them on the **Gear** tab (§12).

*   **Preset** + **Load**: write the preset's fields onto this frame. Other fields stay. Hover to see what a preset holds.

Gear loads as one unit (camera, lens, film stock, format). Picking a film stock sets the format, so set a frame format such as `6×7` after the stock. Presets never store the frame number.

<!-- panel:metadata_gear -->
### Analog Gear

Searches your own gear (§12). **Other…** opens the full built-in catalog.

*   **Camera / Lens / Film stock**: empty means not set.
*   **Clear**: empties all three.
*   **Infer from folder name**: fills unset camera, film stock, ISO and capture date from the folder name; a field with two plausible readings stays empty.

<!-- panel:metadata_capture -->
### Capture

*   **Date**: `1998`, `1998-07`, `1998-07-14` or `1998-07-14 16:30`, with an optional offset such as `+02:00`; an impossible date turns red. EXIF pads a partial date, XMP keeps the short form. The scan's own timestamp moves to `DateTimeDigitized`.
*   **Place**: the map-pin button opens a map (search, click or paste coordinates); the field also takes a coordinate pair or an OpenStreetMap/Google Maps link, and ✕ empties it. A place replaces the source's GPS; with none set, a geotagged source keeps its own. The map contacts OpenStreetMap.

<!-- panel:metadata_process -->
### Process

*   **Saved process**: a library recipe that fills Developer, Dilution, Push / Pull, Time and Temperature. Typing over one unlinks it.
*   **Format**: `—` (not set), `35mm`, `120`, `4×5`, `8×10`, `110`, or `Other` (free text).
*   **Developer** and **Dilution**: for example `D-76` and `1+1`.
*   **Push / Pull**: `Push +3` … `Normal` … `Pull -3`.
*   **Time** and **Temperature (°C)**: time as `9:30` or minutes; searchable as `devtime:` and `temp:`.
*   **Clear**: empties the saved process and its fields. Format stays (the film stock sets it).

<!-- panel:metadata_scanning -->
### Scanning

*   **Saved setup**: a library digitizing setup that fills Scanning. Typing over it unlinks it.
*   **Scanning**: scan method or notes. EXIF `Software` is always `NegPy`.
*   **Clear**: empties the saved setup and the note. Roll and Frame stay.
*   **Roll / Frame**: the capture roll name and frame number, stamped on capture; filename template fields `{{ roll }}` and `{{ frame }}`.

<!-- panel:metadata_exposure -->
### Exposure

Optional original shutter, aperture and ISO. Click the lock to edit a free-text string such as `1/125s f/2.8 ISO 400`.

<!-- panel:metadata_preview -->
### Metadata Preview

A live view of what will be embedded. **Description…** picks the fields joined into EXIF `ImageDescription` (default: camera, lens, film stock, ISO); the choice becomes the default for frames without their own.

Capture gear goes to standard EXIF and the digitizing rig to `negpy:Scan*` XMP tags; with gear unset, your scanner or DSLR stays in EXIF. Other XMP tags: `photoshop:DateCreated` (with `negpy:CaptureDatePrecision`), `photoshop:City`/`State`/`Country`, `exif:GPS*` (the only GPS in a TIFF), `negpy:DevelopmentDilution`/`DevelopmentTime`/`DevelopmentTemperature`, `negpy:CaptureRoll`/`CaptureFrame`.

---

## 12. Gear tab

A gear library for Metadata (§11), Roll Settings and every gear picker. **My Gear** holds physical gear, **Presets** metadata field sets.

<!-- panel:gear_items -->
### My Gear

**Category**: **Cameras**, **Lenses**, **Film Stocks**, **Process** (a development recipe) or **Scanning** (a digitizing setup). Your gear is saved to `~/NegPy/gear/`.

*   **+**: copy an item from the built-in catalog into your list, or **Add Custom** to enter one by hand.
*   **Catalog**: show the built-in reference models too. Off by default.
*   **copy / trash**: duplicate or delete the selected item. Trash is disabled on built-in entries.

<!-- panel:gear_presets -->
### Presets

*   **+**: store the current frame's metadata under a name.
*   **pen**: rename, or change which fields it stores.
*   **copy / trash**: duplicate or delete.
*   Edit a preset's fields in place, with no frame open; per-frame fields (capture date, place, description) are read-only here. **Notes** is free text.

---

## 13. Export tab

### Output intent

*   **Print** (default): the look you see on screen.
*   **Flat**: a neutral, low-contrast master for editing elsewhere. It skips the print look, effects, toning and vignette, and writes a 16-bit TIFF, or lossless JPEG XL when JXL is selected with sRGB, P3, Rec 2020 or Grayscale.
    *   **Preview Flat**: show the flat master on the canvas. Rotation and flip keep it up; any other edit or a frame change closes it.
    *   **Roll Analysis** (Roll tab): share one exposure baseline across all visible frames so flat masters match. Run it before a flat batch.
*   **Linear**: write the decoded buffer as linear 16-bit, with rotation, flip and only the corrections turned on below. **TIFF** (default, untagged) or **JPEG XL** (lossless; always tagged with sRGB primaries, wrong for native primaries). **Effort** (1 to 9, default 7) trades JPEG XL speed for size.
    *   **Pakon RAW**: 4× expansion by default; F335 files (16-bit sensor) none.
    *   **LinearRaw DNG**: SilverFast HDRi (3-channel) and VueScan (4-channel RGB+IR). IR goes to a separate grayscale `_ir` file in the same Format.
    *   **Camera RAW**: demosaiced at unity white balance with the Raw Decode card's Export algorithm (§10.7); as-shot WB goes to XMP (`RAW-WB: R G B`). Trichrome triplets merge into one TIFF; stitches get Flat Field and sensor correction per part.
    *   **Coolscan NEF**: not raw sensor data but the Nikon Scan output; the full-resolution RGB is written.
    *   **Flextight FFF**: 16-bit RGB and LogLuv raw (`.3fr`/`.fff`); FlexColor metadata is kept.
    *   **Noritsu RAW**: headerless BGR 16-bit dumps; size detected from file size. 16× expansion by default.
    *   **TIFF**: IR (a 4th channel, an `_ir.tif` sidecar or a second page) goes to a separate `_ir` file. **Input gamma** (linear, 1.8, 2.2 or sRGB) linearizes the data.
    *   **Expansion**: scales the data before writing. Defaults: Pakon F135/F235 4×, Noritsu 16×, F335 and LinearRaw DNG off. Camera RAW, Coolscan NEF and Flextight FFF have none.
    *   **IR Dust Removal** (when IR exists): IR dust and scratch correction. Off by default.
    *   **Corrections** (camera RAW only, all off): **White Balance** (as-shot gains; grayed out for a Trichrome triplet or Single-Shot Narrowband capture), **Flat Field**, **Sensor Correction** (crosstalk unmixing). Stitch composites always get Flat Field and sensor correction per part.
    *   **Lens Correction** (off; shown when the frame has a lens profile or a **Distortion Correction**): applies the **Optics** correction (§10.8), resampling the pixels. The embedded profile needs a single camera RAW; Distortion Correction skips half-frame scans.

    Files take `_linear` (then `_linear_2`… without **Overwrite Existing Files**), so a source is never overwritten. **Abort** stops after the current frame.

    The file carries no ICC profile or source color metadata; it keeps Make, Model and DateTime and describes the decode and corrections.

### Format / Size / Color Management / Destination

*   **Format**: `JPEG`, `TIFF`, `PNG`, `JPEG XL` or `WebP`, each with quality or effort options. **JPEG XL supports only `sRGB`, `P3 D65`, `Rec 2020` or `Grayscale`**; `Adobe RGB`, `ProPhoto RGB` and a custom Output ICC give an error.
*   **Bit Depth**: `8-bit` or `16-bit` (TIFF, PNG, JPEG XL). Hidden for JPEG, WebP and a flat master (always 16-bit).
*   **Compression** (TIFF): `Uncompressed`, `LZW` or `ZIP`, all lossless; ZIP is usually smallest.
*   **Compression** (PNG): `0` to `9`, lossless; higher is slower and smaller.
*   **Progressive** (JPEG): renders in passes while it downloads.
*   **Input ICC**: treat an untagged source as this profile. Primaries only: a matrix profile's TRC is ignored; a LUT profile's input curves still apply.
*   **Export profile**: the space the file is converted to and tagged with: `Same as Source` (Adobe RGB for an untagged scan), `sRGB`, `Adobe RGB`, `ProPhoto RGB`, `P3 D65`, `Rec 2020`, `Grayscale` (true B&W), or an imported printer or paper ICC, which is embedded in the file.
*   **Import ICC** (folder button on the COLOR MANAGEMENT header): copies a `.icc`/`.icm` into `~/NegPy/icc/`, available at once. A file named after a built-in space (`sRGB.icc`) replaces that space everywhere, after a confirmation.
*   **Proof on Screen** is in **Soft Proof** below. A warning shows here when nothing is proofed or the proof targets a different profile than the export.
*   **Paper Aspect Ratio**: final print ratio, or *Original* (no resize).
*   **Resolution**: *Original* (full resolution), *Print* (long-edge **Size** in cm plus **DPI**) or *Pixels* (**Long edge** in px). The DPI tag is your value (*Print*), the one the long edge implies (*Pixels*), or the source's, else the **DPI** field (*Original*).
*   **Destination**: **Filename Pattern** (a Jinja2 template with export and Metadata fields; see [TEMPLATING.md](TEMPLATING.md)), **Overwrite Existing Files**, and the location: subfolder of source (default, `export`), same as source, or an **Export Path**. A roll with no single source folder exports under its own folder in NegPy's data folder. With **Linear**, only Destination shows.

### Export button

**Export**, under the form; its chevron picks the scope: current frame (Ctrl+E), selected frames or all visible frames. For several formats or sizes in one run, use Export Presets. A batch holding frames without a saved edit asks for confirmation first: they export with the current settings and may not match their thumbnails. After the batch, a warning names exports that rendered almost black with no crop set, where the bright scan border drove automatic levels.

### Collapsible sections

<!-- panel:export_presets -->
#### Presets

Saved Format/Size/Color Management/**Destination**/filename recipes. Each preset is a toggle; **Manage** edits them. **Export Presets** renders the frames with every preset turned on, each to its own destination.

<!-- panel:printing_notes -->
#### Printing Notes

The printer's record for this frame: the numbered dodge/burn masks and a card with paper, density, grade, filtration and the burn list. **Preview** shows it on the canvas (**Shift+N**); **Export** writes it as an image beside the print. Its conventions are under Dodge & Burn.

<!-- panel:export_sidecars -->
#### Sidecars

**Save on Export** writes a `.negpy` sidecar next to each source on export. **Export Sidecars** writes them for all visible frames now and reports failures in read-only folders. Edits always stay in the database too. A frame with no edit in the database takes its sidecar when its folder opens. **Load Edit from Sidecar…** (right-click a frame or the image) replaces the frame's edit with a chosen `.negpy`, as one undo step.

<!-- panel:contact_sheet -->
#### Contact Sheet

A darkroom proof of the roll: every visible frame at true size, as cut film strips on photographic paper, in capture order. The rebate, perforations and edge print follow the frames' **Film stock** and **Format** metadata. Each frame prints with its own edit, without the border.

*   **Path**: the folder for the sheets. Empty follows the export destination.
*   **Contact Sheet…**: opens the proof. Drag an edge or a corner of the paper to resize it; the strips refill to fit. A size near an Ilford sheet snaps to it.
*   **Format**: 35mm, 35mm Half Frame or 120, read from the frames' metadata and the roll's Half Frame mode. **Frame size** sets the 120 camera's frame (6×4.5 to 6×17).
*   **Paper**: an Ilford sheet size, or Custom. The turn button swaps width and height. The default, 24 × 30.5 cm, holds a whole 36-exposure roll.
*   **Width** / **Height**: the paper size, 50–610 mm.
*   **DPI**: 150, 300 or 600, tagged so the sheet prints at true size.
*   **Order**: **Date**, or **Scene** (once the roll has one), where each scene starts a new strip.
*   **Print**: **As Edited**, or **Straight Proof**: the whole roll at one exposure on grade 2 from its Roll Analysis baseline (per scene in **Scene** order), so thin and dense negatives print light and dark. It needs the roll scanned at one exposure; positives from other software cannot be proofed.
*   **Pick Frames**: click a frame to leave it out or put it back; rejected frames start out.
*   **Edge Print**: the maker's markings on the film edge (stock name, frame numbers, DX barcode). Off prints plain film.
*   **Roll Label**: the roll name, film, developer, camera and date above the strips, set in the edge print's capitals and ink.
*   **White Paper**: white paper and a dark roll label instead of the darkroom black, to save ink on a home printer. The film strips print as before.
*   **Film Base** (beside White Paper, on): prints the film base black, as a darkroom contact print. Off prints it as paper, with the strip outlined, the edge print and frame numbers in dark ink and the perforations as rings, so a home printer inks the frames, the markings, the outline and the rings alone.

Sheets are JPEGs named `contact_sheet.jpg` (`contact_sheet_1of2.jpg` and on when the roll needs more sheets), with the **JPEG Quality** and **Progressive** settings above.

#### Soft Proof

Simulate the print on screen. See below.

<!-- panel:soft_proof -->
### Soft Proof

Preview only; exports are never proofed. Paper is dimmer and has a smaller gamut than a screen, so a correct proof looks worse than the plain preview. Judge it in room light.

*   **Proof on Screen** (`Shift+P`, on by default): master switch. Off grays out the settings on the rail under it.
*   **Preset**: a saved printer and paper setup; **None** proofs the export target with no paper simulation.
*   **Profile**: follows the **Export profile** until you pick a printer or paper, so you can proof a print while you export a web JPEG. Lists imported ICC profiles only.
*   **Intent**: **Relative Colorimetric** keeps printable colors and clips the rest, so saturated areas can flatten. **Perceptual** compresses everything so color relations survive; printer profiles carry their own table, so try it on saturated frames. **Saturation** is for charts, not photographs.
*   **Black Point Compensation** (off): map the darkest tone to the paper's black instead of clipping.
*   **Simulate Paper White** (off): show the paper's white and tint.
*   **Simulate Ink Black** (off): show the paper's real black; shadows lift.
*   **Gamut Warning**: show unprintable colors as gray (the Analysis panel's **Gamut** row counts them). The edge fades, so read it as a region.
*   **Display**: the monitor profile, auto-detected; set it by hand if detection fails.

---

## 14. Scan tab

Capture film directly into NegPy. The **Scanner** card picks the scanner; the tab shows its cards, the shared **Output** card and, at the bottom, the **Scan** button, which reads **Stop** while scanning.

<!-- panel:scan_source -->
### Scanner

*   **Scanner** (Camera by default): **Film Scanner** drives a dedicated film scanner (Device, Film & Quality, Framing); **Camera** drives a camera on a copy stand (Camera, Preset & Light). Both write through the same Output card.

<!-- panel:scan_device -->
### Device

**Backend**: **SANE** (Linux/macOS), **Nikon Coolscan (nkscan)** (direct Coolscan driver, Linux, Windows, macOS) or **pyOpticfilm (Plustek)** (OpticFilm 8200i SE and 8100 V2, all three OSes). Controls follow what the device reports; a row or card with nothing for the device is hidden.

*   **Device**: the scanner; the arrows refresh the list. The eject button's menu holds **Eject Now** and, on a strip feeder, **Eject When Done** (on by default), which returns the strip after a batch. Off keeps it loaded with its picks and previews, so more frames scan without a new preview. A strip the scanner returns by itself (its idle timeout) counts as an eject: insert it again, then **Detect frames**. After 9 minutes with no scanner activity NegPy measures the strip again.
*   **Debug log** (nkscan; Off, Debug, Trace): writes `nkscan.log` in the NegPy folder; the folder button opens it. For a bug report, set Trace, reproduce the problem and attach the file.

**pyOpticfilm (Plustek)**: the **OpticFilm 8200i SE** (`07b3:1825`) and **8100 V2** (`07b3:1824`) scan; for other models try **SANE** on Linux and macOS. **IR** comes in the same pass. On Windows, bind the device to **WinUSB** with Zadig first ([PLUSTEK_WINDOWS.md](PLUSTEK_WINDOWS.md)). From source, install with `uv sync --group plustek` or `pip install negpy[plustek]`.

**Nikon Coolscan (nkscan)**: needs no SANE and ships in the release builds; from source, see [CONTRIBUTING.md](../CONTRIBUTING.md). On Linux, USB needs a udev rule for vendor `04b0`; FireWire/SCSI needs the `sg` module.

<!-- panel:scan_quality -->
### Film & Quality

*   **Film**: Color negative, B&W negative, Slide or Kodachrome. Sets frame detection, metering and whether IR and ICE are offered (not for B&W or Kodachrome).
*   **Film format**: frame length (135, 66, 645 and so on). **Auto** where the holder narrows it; set it for loose film in a masked carrier. Shown only where the transport measures the film.
*   **Resolution**, **Bit depth**: the scan's dpi and bits per channel; type a resolution between the listed stops.
*   **Scan mode** (Plustek; Single-Pass by default): **Single-Pass**, **Multi-Pass** (stacks repeated exposures to cut noise; **Passes** 2 to 9), **Adaptive Multi-Exposure** (fuses a short and a long exposure for more range) or **Adaptive Multi-Pass** (both). Each takes longer than Single-Pass; Multi-Pass excludes IR.
*   **Samples** (nkscan): reads per line averaged (1 to 16). Less shadow noise, proportionally slower.
*   **IR**: a separate infrared plane for Retouch's dust removal. **ICE** (nkscan): infrared dust and scratch removal baked into the file, color film only. **ICE** and **IR** exclude each other.
*   **Superfine** (nkscan): one line per pass. Slower, with no host-side line registration.
*   **Autofocus**, **Auto-exposure**: run on the scanner before the scan, where it offers them.
*   **Exposure** (µs): shown when the scanner has `scan-exposure-time` (some genesys devices); off while Auto-exposure is on.

<!-- panel:scan_framing -->
### Framing

*   **Frames**: `1-6`, `1,2,5`, or empty for all. The strip preview writes its picks here. The line above **Scan** states frame count, resolution, extra passes, approximate disk use and, in amber, an active exposure lock.
*   **Batch** / **Window**: on a feeder, **Preview strip…** previews every frame, to set windows and pick frames; with a manual holder, **Preview…** sets one crop window. Only the window is scanned; **Clear** resets it. On nkscan the preview reads the whole strip in one pass; **Detect frames** reads it again after the film moves.
*   **Exposure lock** (**Meter Frame…** / **Unlock**, nkscan): nkscan meters each frame alone, so an end frame can scan in a different color. **Meter Frame…** meters one frame inside the strip (such as frame 2), and every later scan reuses that exposure, across strips and restarts, until **Unlock**. Meter again for each roll.
*   **Crop** (**Prescan…**, Plustek): a 1200 dpi full-window preview; drag a crop and leave with **Apply Crop** or **Scan Frame**, and the next scan reads only that area. **Clear** scans the full window.

<!-- panel:scan_camera -->
### Camera

Copy-stand capture with a camera in **PC Remote** mode over USB (macOS/Linux): with a NegPy **Scanlight**, R/G/B triplets or one exposure with the three LEDs lit together, else one white-light exposure. Needs `python-gphoto2` (`pip install gphoto2`; no Windows build). Setup and troubleshooting: CAMERA_SCANNING.md.

*   **Camera** / **Light**: connection status, found automatically; the light shows its LED temperature, amber once it runs warm. On Linux, the Scanlight's serial port needs your user in the `dialout` group (`sudo usermod -aG dialout $USER`, then log out and back in).
*   **Live View**: click the image to aim the focus magnifier, again for the full frame. The **Focus meter** reads sharpness against the best seen: turn the focus ring past best focus, then back until it reads **at peak** (a click, or the body's own zoom change, resets it). ISO, shutter and aperture are set from its toolbar, or locked by a calibrated RGB preset.
*   **Histogram** and **Zebras** (live view toolbar): the frame's R, G and B histogram under the image, and hatching on pixels at the top (red) and bottom (blue) of the preview range. The preview clips before the RAW does, so a zebra is an early warning.
*   **Meter** (live view toolbar, white-light and camera-only scanning): one RAW probe at the current exposure sets the shutter so the region that must not clip lands just under clipping: a negative's clear base, or a slide's brightest highlights. The white-light preset's film type decides which (AUTO classifies the probe as an import would). The status line names the medium, the level read and the step's move; Scan waits for the body to confirm the write. A clipped reading steps down two stops; meter again. After a probe the red zebras hatch where the RAW will clip at the new shutter, while the exposure and the framed picture hold.
*   **Scan** and **Retake**: **Scan** shoots the next frame, auto-numbered, and imports it; **Retake** shoots the last frame again. In the Live View window, `S` scans and `R` retakes; both can be rebound under Camera Live View in Keyboard Shortcuts.
*   **Narrowband**: RGB-lit scans render more saturated; the Calibration card's **Narrowband** toggle corrects this.

<!-- panel:scan_light -->
### Preset & Light

Shown while a Scanlight is connected.

*   **Preset**: shows its RGB levels, ISO, shutter and aperture and forces them each frame. **+** calibrates: place the rectangle on clear film base, name it, run it; it solves a shutter and LED levels just under clipping, or says which way to adjust. **Create a manual preset…** sets one by hand, and the save button stores it.
*   **Red**, **Green**, **Blue**, **White** (0 to 255): LED levels, editable while building a manual preset. **Light Off** turns every channel off.
*   **Capture mode**: **Triplet** shoots one exposure per LED and merges them; **Single Capture** shoots one exposure with red, green and blue lit together and imports it as an ordinary RAW. Picked in the calibration window, or here while building a manual preset.
*   **Create Sensor Profile** (calibration window, Single Capture only, default on): also saves a sensor profile under the preset's name; a roll scanned with the preset takes it and turns Linear RAW on. A preset's profile is not carried to other rolls.
*   **ISO**, **Shutter**, **Aperture**: the preset's exposure, editable while building a manual preset.
*   **Channel Delay** (0 to 5000 ms): pauses between R, G and B for bodies that lock up. Triplet presets only.

<!-- panel:scan_output -->
### Output

Shared by both scanners.

*   **Folder**: where frames are written.
*   **Scan as Roll** (on by default): the output folder opens as a roll in the Library, so Half Frame, the Roll tab and Roll Analysis apply. Frames load as they are written.
*   **Folder as Roll** (on by default): frames go into the output folder itself, which is the roll and gives it its name. Off writes into a **Roll** subfolder, and a new Roll name starts a new roll. A camera frame's file name starts with the roll name.
*   **New Roll** (**+** beside Roll, with Folder as Roll off): steps the Roll name to the next one that has no subfolder yet. A trailing number that is zero-padded or follows `_`, `-`, a space or `.` counts up (`Roll001` to `Roll002`, `portra400_1` to `portra400_2`); any other name gains `_2` (`portra400` to `portra400_2`).
*   **Format**, **Filename** (Film Scanner): `TIFF` or `TIFF (mono)` (one 16-bit gray plane, for B&W negatives), and the file name as a Jinja2 template of `{{ date }}` and `{{ seq }}`.

<!-- panel:scan_strip -->
### Strip preview

Dialogs end with **Cancel**, **Apply** (keep the framing) and **Scan**. Apply reads **Apply Framing** on a strip, **Apply Window** on a single holder, **Apply Crop** after a Prescan.

During a preview **Cancel** reads **Stop Preview** (keeps the tiles read so far). Negatives preview inverted. On nkscan, **Offset** (±10 mm) and **Drift** re-cut the tiles with no new scan.

*   **Cropping**: drag on a frame; corners resize, inside moves. **Clear Crops** removes all.
*   **Frame outline**: red box on the detected frame; offsets are measured from it.
*   **Offset**: moves every frame along the film to clear the gap. The shaded band on the right is film the transport cannot deliver; a feeder goes one way only.
*   **Drift**: offset that grows (or shrinks) per frame position.
*   **Per-frame offset**: the slider under a tile, on top of Offset and Drift. Double-click resets.
*   **Size**: tile size, remembered. Double-click resets.
*   **Which frames**: tick tiles, or **All** / **None**. Eject or a restart clears ticks, crops and per-frame offsets.
*   **Preview frame** (eye, beside the tick): scans that frame again; hidden where the tiles come from one strip pass.

---

## 15. Preferences

Application-wide settings: canvas **⋯** menu → **Preferences…**, `Ctrl + ,`, or the macOS application menu. Changes apply at once; startup rows show a restart notice.

### Interface

*   **UI scale** (80% to 120%): after a restart.
*   **Canvas background**: black, dark gray, mid gray (neutral for judging) or white (a print on a light table).
*   **Immersive canvas**: toolbar floats over the image.
*   **Sticky zoom**: keep the zoom when you switch frames.
*   **Reverse scroll zoom**: scroll up zooms out.
*   **Customize Shortcuts…**, **Edit Toolbar…**, **Reset Panel Layout**: shortcut editor, canvas toolbar picker, default panel layout.

### Accessibility

*   **Color vision** (default **Standard**): colors for marks told apart by color alone. The Retouch dust overlay is green and magenta under *Standard*, Dodge & Burn masks and Printing Notes amber (dodge) and blue (burn). *Protanopia / deuteranopia* uses sky blue and vermilion for dust, orange and sky blue for masks; *Tritanopia* vermilion and bluish green; *Achromatopsia* white and gray. The color-blind colors come from the Okabe–Ito palette.

### Performance

*   **GPU acceleration**: render on the GPU (backend named below). Off uses the slower CPU pipeline, same image. If the GPU fails at launch, an amber line here says so.
*   **Multi-core CPU rendering**: runs CPU rendering on all cores, effective at once; it helps exports and machines without a usable GPU. On by default except on macOS, where it can end the app; after such a crash NegPy offers to turn it off. `cpu_parallel` in `override.toml` overrides it.
*   **Search by meaning**: find frames by description instead of `field:value` terms (§2). Downloads its model on first use.
*   **Preview size** (512 to 8192 px): canvas long edge; higher is sharper and costs more memory.
*   **Preview cache** and **Preview cache limit**: photos kept decoded, and their memory limit. Lower both on low RAM.
*   **HQ buffers**: full-resolution preview buffers kept in memory.
*   **Rendered frames**: frames kept for going back without a re-render.
*   **GPU texture cap**: largest texture dimension. 0 lets the hardware decide. Lower it if exports run out of GPU memory.
*   **Show GPU memory warning**: a message when an HQ preview is downsampled to fit the texture cap; off hides only the message.

Rows from **Preview size** down need a restart. A value in `override.toml` wins and grays out its row.

### Session & Storage

*   **Carry settings between frames**: apply your Persistent Settings to each newly opened file. Off, each file starts from its saved edit or defaults.
*   **Persistent Settings…**: which edits carry over (§2).
*   **Manage Database…**: row counts and sizes, clear saved edits, library roots.

### Startup override (`override.toml`)

For crashes on launch or rendering glitches: edit `Documents/NegPy/override.toml` (created on first run) and restart. Its values win over Preferences.

| Setting | Values | Effect |
|---------|--------|--------|
| `rendering.backend` | `"auto"`, `"vulkan"`, `"dx12"`, `"metal"`, `"cpu"` | GPU backend for image processing. `"cpu"` disables GPU entirely. |
| `display.qt_rhi_backend` | `"auto"`, `"vulkan"`, `"d3d12"`, `"metal"`, `"opengl"`, `"software"` | Qt UI rendering backend. |
| `display.qt_platform` | `"auto"`, `"xcb"`, `"wayland"` | Window system plugin (Linux only). |
| `performance.force_hq_preview` | `true` / `false` (or absent) | Overrides the saved HQ preview toggle. |
| `performance.cpu_parallel` | `true` / `false` (or absent) | Multi-core CPU rendering kernels. Defaults on, except on macOS. |
| `logging.level` | `"debug"`, `"info"`, `"warning"`, `"error"` | Log verbosity. Use `"debug"` when reporting issues. |

`max_texture_size`, `preview_render_size`, `preview_cache_max_bytes`, `preview_cache_max_entries`, `preview_cache_max_full_res_entries` and `render_memo_max_entries` take the same values as their Preferences rows.

**Common fixes:**

*   **Crashes immediately on Linux** → `backend = "cpu"` or `qt_rhi_backend = "opengl"`.
*   **Black or blank preview on Windows** → `backend = "dx12"` or `qt_rhi_backend = "software"`.
*   **Wayland rendering issues** → `qt_platform = "xcb"` to force X11.
*   **GPU out-of-memory during export** → `max_texture_size = 4096`.

---

## 16. Updating NegPy

NegPy checks GitHub once at startup. A new release shows a green **⬇ Update Available: vX.Y.Z** line at the top of the left panel and a green dot on the **⋯** menu; click the line, or **Update to vX.Y.Z…** in the menu, for the release notes and the install button. **Check for Updates…** in the **⋯** menu checks by hand.

**Install Update** downloads the build for this install type, closes NegPy, installs and reopens. Nothing is replaced until NegPy exits, so a failure leaves your install as it was.

| Install | What NegPy fetches | How it installs |
|---------|--------------------|-----------------|
| **Windows** | the `-Setup.exe` installer | Runs it silently over your install. Approve the administrator prompt *before* NegPy closes. |
| **macOS** | the `.dmg` for your chip (Apple silicon or Intel) | Mounts the image and replaces the `NegPy.app` bundle where it currently sits, then reopens it. |
| **Linux** | the `.AppImage` | Replaces the AppImage file you launched, keeps it executable, and relaunches it. |

Edits, presets, settings and library live in `Documents/NegPy` and the database, so updates do not touch them.

The button reads **"Open Releases Page"** when NegPy cannot update itself: a source checkout, no build for your platform, an app moved out of its installer layout, or an app folder you cannot write to.

---

## Additional Info

*   **GPU acceleration**: turn it off in **Preferences → Performance**, or force a backend in `override.toml`, if you suspect a driver issue.
*   **Database**: edits live in a local SQLite database keyed by file hash, so files can move or be renamed. Optional `.negpy` sidecars mirror them.
*   **Saving edits**: written on export, on frame switch, or on save. Closing mid-edit before any of these loses unsaved changes.
*   **Keyboard shortcuts**: [KEYBOARD.md](KEYBOARD.md)
*   **Filename templating**: [TEMPLATING.md](TEMPLATING.md)
*   **The pipeline in depth**: [PIPELINE.md](PIPELINE.md)
