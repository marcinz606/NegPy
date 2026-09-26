# NegPy User Guide

NegPy turns film scans into positives with a non-destructive, darkroom-style pipeline. It never writes to your source files; edits live in a local database.

For the pipeline order and math, read [PIPELINE.md](PIPELINE.md).

---

## 1. The Big Picture

### Protected data folders on Windows

If Windows blocks the default data folder, NegPy suggests `%LOCALAPPDATA%\NegPy\data`. Click the path to pick another folder, then **Use This Folder**. No data is copied or moved: a new folder starts empty, an existing NegPy data folder uses its own data. The dialog also explains how to allow NegPy through folder protection instead. The choice is saved in `data-location.json` and wins even if Documents becomes writable; `NEGPY_USER_DIR` wins over it. If the saved folder is not accessible, you get an error.

### Screen layout

*   **Left, the film strip**: your frames as a contact sheet, with import, sorting and triage tools.
*   **Center, the canvas**: the live preview. Most tools (crop, white-balance picker, heal brush, dodge/burn masks) work by clicking on it. Scroll or pinch to zoom, drag to pan. The bottom toolbar holds Fit/**1:1** zoom (one scan pixel per screen pixel; below **HQ** a **preview res · HQ off** pill shows the preview is scaled up), undo/redo, rotate/flip and more. Rotate 90° and flip act on every selected frame. Items that do not fit go into the **⋯** menu, which holds every action, including **Preferences…** (all app-wide settings, §15), **Edit Toolbar…** (puts any tool or action from the menu on the row, except About, updates and the tour) and **Persistent Settings…**. Right-click the image for **Reset View**, **Sticky Zoom** (keep the zoom across frames), the pickers, copy/paste settings and **Unload** (remove the frame from the session, keep its edit). An empty canvas shows **Load some scans to get started**; click it for **Import Folder as a Roll…** (the folder becomes a roll and opens, as in Library) or **Add Files…**.
*   **Right, the controls**: tabs **Roll** / **Frame** / **Metadata** / **Gear** / **Export** / **Scan**. **Frame** has a pinned **Analysis** readout and its own row of tabs below it. Roll and Frame change the render; the other tabs do not.

Drag a panel by its top edge (the thin strip above Session, the margin around the Controls panel's **Find** box) to float it; its pin button docks it again. **Shift+H** hides both panels, and brings both back. NegPy remembers the layout. **Reset Panel Layout** in the **⋯** menu restores the default layout.

### Find

**Ctrl+K**, or **Find Control or Action…** in the **⋯** menu, opens one box for every slider, card and action. Type a name, or another editor's word for the job: *contrast* finds **ISO-R Grade**, *white balance* **Filtration**, *exposure* **Print Density**. Up and Down pick a row and Enter opens it on its tab. A slider row is the live control, so you can drag it without leaving the list.

### Before / After

**◑** on the toolbar (or `\`) splits the canvas. The left side is the auto baseline: same film process, crop and rotation, with every creative control (exposure, tone, Lab, dodge/burn, toning, retouch and finishing) at default. The right side is your edit. Drag the divider or its knob. The split stays up while you edit; `\`, `Esc`, a frame change, Peek Negative, Peek Flat Scan or the test strip close it.

### Reference view

**Shift+R**, or **Reference View** in the **⋯** menu, pins the frame on the canvas to a pane beside it; open other frames to match them to it. The pane keeps the frame as it looked when pinned, so press **Shift+R** twice to pin again. Drag the divider to share the width; **✕** or **Shift+R** closes it.

### Peek Negative

The toolbar's film button (or `N`) shows the scan as loaded: not inverted, no metering, no film-base normalization, no edits. Crop, rotation and flip still apply. Use it to check density, mask color and scanner clipping. Touching a control closes it. It is color managed and scaled to its brightest tone, so thin and dense captures look equally bright: read density from the density histogram. The soft proof is off.

### Peek Embedded Preview

**Peek Embedded Preview** in the **⋯** menu (or `P`) shows the camera's own JPEG at your crop and rotation, with the camera's white balance, tone curve and clipping. NegPy uses nothing from it. Files without a preview (scanner TIFFs, most converter DNGs) disable the item.

During a peek the canvas shows a **NEGATIVE**, **EMBEDDED** or **FLAT SCAN** badge and the menu item has a checkmark. `Esc` closes any peek, the split or a test strip.

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
| **Look** | Lab · Alternative Processes · Toning | Chroma, sharpening, lith, cyanotype, toning |
| **Finish** | Retouch · Finishing | Dust, vignette, border, carrier |
| **Favorites** | Your chosen sliders · Presets | Most-used controls, saved edits |
| **History** | Work prints · Edit history | Named versions, undo trail |

Tabs that do not change the render:

| Tab | Panels | What it is for |
|-----|--------|---------------|
| **Export** | Export settings | Format, size, color, batch |
| **Metadata** | Archival metadata | Camera, lens, film |
| **Gear** | Gear library | Cameras, lenses, films, processes, scan setups |
| **Scan** | Scanner · Camera Scanning | Direct capture (Linux/macOS) |

A slider row reads name, track and value: click the value to type one, drag the name to scrub (**Shift** for finer steps), and double-click or **Ctrl**+click the track to reset. A thin rail at the left joins controls that work together, such as Toe and Toe Width. A **dot** on a panel header or tab marks a non-default value. Each panel header has a **reset** action and an **ⓘ** that opens this guide there. In a narrow panel, tabs that do not fit move into a **»** menu; the current tab stays visible.

### What carries to the next frame

An unedited frame gets the rig and roll settings: film process, crop ratio, flips, calibration, paper stock, the Lab polish and export preferences. The look (density, filtration, tone curve, toning, dodge and burn) starts clean.

**Preferences → Session & Storage → Persistent Settings…** edits that list, grouped by panel, with values from your last saved edit. Tick a setting or a group header to make it carry. The **Carry settings between frames** checkbox is the master switch; off, new frames get bare defaults and your ticks stay saved.

An edited frame keeps its look; only export and metadata settings reach it. **Reset Settings** ignores the list and returns bare defaults, except for the scanning setup: Linear RAW, Narrowband and the demosaic choices stay as a new frame gets them. In a roll, a reset also sets every Roll-tab card back to **Roll**, so the frame takes the roll's values.

### Frame or roll: the scope pair

Each section header has **Frame** (picture, amber) and **Roll** (film roll, red) beside its reset arrow. The lit one shows where the card's values live; click the other to move them. A card with non-default values has a stripe in that color down its header. Frames that are not one roll (search results, several folders) show **Frame** everywhere and **Roll** grayed out, until Save as Roll.

On a **Roll tab** or **Metadata** card the pair is a latch. On Roll, the card follows the roll's value, and new frames in the roll inherit it. Edit a slider and it flips to Frame. Click **Roll** to push this frame's value to the roll; click **Frame** to pin the current value to this frame.

On a **frame** card (Geometry, Filtration, Tone, Lab, Alternative Processes, Toning, Retouch, Finishing), **Roll** opens the film strip's clone picker for that section, ticked for what you changed, to apply to the selected frames or the whole roll. After a whole-roll apply the card reads Roll until you touch a pushed setting. A selection apply leaves it on Frame.

**Reset to Roll**, beside the reset arrow, appears once a card differs from the roll: a Roll tab or Metadata card this frame took off the roll, or a frame card that no longer matches its last whole-roll apply. It puts the roll's values back on that card as one undo step; a setting the roll apply never carried keeps this frame's value.

### The tab header

Tabs with several cards (Roll, Exposure, Color, Finish, Metadata) have a bar reading **3 of 5 cards edited** (or **No cards edited**). Its buttons act on all cards: reset arrow (appears once something is edited, asks first), **Reset to Roll** (appears once a card differs from the roll, one undo step), roll button (one picker for the whole tab, selected frames or whole roll) and double chevron (collapse/expand). Cards the film mode has retired are skipped. Geometry has one card and no bar.

### Menu bar (macOS)

`Ctrl` in this guide is `⌘` on macOS, as the app shows it.

*   **NegPy**: Preferences… (`⌘,`), beside About and Quit.
*   **Window**: Minimize (`⌘M`), Zoom, Close (`⌘W`), Bring All to Front, and a list of open NegPy windows (main, live view, calibration) with a tick on the front one. Minimize and Zoom are off in full screen.
*   **Help**: Take the Tour, Keyboard Shortcuts, Customize Shortcuts, the Analysis panel guide, Report an Issue (opens the issue tracker in your browser), Check for Updates.

For full screen use the green window button; NegPy has no Enter Full Screen item.

Menus show only `⌘` keys. Plain-key shortcuts such as `?` for Keyboard Shortcuts still work, but are not in the menu, because a menu key would fire while you type `?` in the film strip search box. A rebound shortcut updates its menu item.

`⌘W` on the main window closes NegPy. Windows and Linux have no menu bar.

---

<!-- panel:frames -->
## 2. Film strip (left panel)

When a newer release is out, a green **⬇ Update Available** line tops the panel and a green dot marks the **⋯** menu, whose **Check for Updates…** item then reads **Update to vX.Y.Z…**. Click either to see the changes and install ([§16](#16-updating-negpy)). **About NegPy…** in the **⋯** menu shows the version.

Below it are the toolbar, the search box and two collapsible sections: **Library** (imported rolls) and **Film Strip** (open frames). Click a heading to fold its section; drag the handle between them to resize. NegPy remembers both.

<!-- panel:library -->
### Your library

**Library** lists every **roll** you have imported: a named, openable group of frames, not a live view of a folder. **Ctrl+L** expands it, and offers an import when you have no roll. Buttons: **+** imports a roll, **↻** finds new roll folders under each parent imported with **Import Subfolders as Rolls…** and re-reads each roll's frame count from disk, **Discovery Filters…** lists folder names that importing subfolders and **↻** skip, with everything inside them, one per line (default `export`): a line matches any part of a name, ignoring case, and a line with `*` must match the whole name (`raw_*`). Saving runs **↻**, which also drops rolls a filter now catches; removing the filter brings them back. **Sort** orders the roll list by Name or Date, ascending or descending, apart from the Film Strip's own Sort, and **index** appears when Search by Meaning is on. Each row shows name and count ("36 photos"). A roll whose folder is gone from disk shows **folder missing** in amber.

Importing only recognizes a folder; nothing is decoded or hashed until you open the roll.

#### Importing

**+** (or the list's right-click menu) offers:

*   **Import Folder as a Roll…**: the folder becomes one roll and opens. A folder with no images of its own imports its subfolders as with **Import Subfolders as Rolls…**. If its name matches a camera or film stock in your Gear library (a word, or a run like "penf" for "Pen F"), Roll Settings opens pre-filled and ticked; Apply keeps it, Cancel skips it.
*   **Import Subfolders as Rolls…**: each folder under the chosen parent, at any depth, that holds images becomes a roll, without opening. NegPy does not look inside a roll folder, so its subfolders (for example export output) do not become rolls. Each roll is named by its path from the chosen folder, for example "20260901/kentmere_400_1", and the list shows it under a **20260901** folder row with its roll count. A folder row only groups rolls; it does not open. Right-click it for **Delete…**, which forgets every roll in it.

NegPy never creates, renames, moves or deletes anything in the folder. Reorganize on disk, then re-import (or **↻**). Edits are keyed to image content, so a moved file keeps its edit, history and keep/reject mark.

#### Opening, renaming, deleting

**Click** a roll to select it; **double-click** (or **Enter**) to open it. NegPy asks whether to **load the roll**, since loading hashes every frame. Tick **Always load without asking** to skip the prompt. Opening replaces the Film Strip contents; edits stay in the database.

Right-click a roll for:

*   **Rename…**: renames the roll, not the folder rows above it. A folder roll also offers **Also rename the folder on disk** (unticked by default, asked each time). NegPy refuses with a warning if a sibling has that name or permission is missing. Inside a cloud-sync folder (Dropbox, iCloud, OneDrive), the sync can treat a rename as delete and re-upload.
*   **Delete…**: forgets the roll record only; folder, images and edits stay. **↻** does not bring it back; **Import Folder as a Roll…** restores it. **Clear Library** in *Manage Database* forgets all rolls.
*   **Roll Analysis** (**loaded** roll only): runs Roll Analysis on every frame outside a scene ([§10.5](#105-roll-analysis)) and stores it as the roll's baseline, for any frame's **Use average** toggles, in this roll or another.

#### Rolls that are not folders

A roll can also be a hand-picked set, and one photo can be in several rolls. Folder rolls show a **folder** icon; rolls from a search or a picked set show a **magnifier**. Search the library (the magnifier-over-folder button) or gather frames, then use **Save as Roll…** in the Film Strip's button row. The roll is a fixed set, not a live search: to add frames, reopen it and Save as Roll again, or add to the session while it is loaded.

A frame has one edit in every roll that holds it. On a frame in more than one roll, right-click → **Edit Independently in This Roll** gives it its own edit here; **Use the Shared Edit Again…** deletes that edit.

### Importing and managing files

**Nikon High Efficiency raw.** Z 8 and Z 9 NEFs in **High Efficiency (HE)** or **HE\*** use a licensed codec NegPy cannot decode, and NegPy says so when one fails. Shoot **Lossless Compressed** NEF, or convert with Adobe DNG Converter.

The **⋮** menu on the Film Strip header, beside its ⓘ guide:

*   **New Roll…**: clears the film strip so you can drag in frames and keep them with **Save as Roll…**. Same as **Clear All…**.
*   **Reset Roll to Defaults…**: **Reset Settings** on every visible frame. Asks first; each reset is an undo step.

The grid button on the same header opens the **Light Table** (`Shift+G`): the roll as a grid in place of the canvas, to cull and pick frames with the same selection, marks and menus. Double-click or **Enter** opens a frame on the canvas; **Esc** or `Shift+G` goes back. The controls panel steps aside while it shows.

The Film Strip button row:

*   **Add** (import icon): **Add Files…** or **Add Folder…**. A folder with no images of its own imports its subfolders into Library as rolls. Dropping a folder on the window does the same as Add Folder.
*   **Hot Folder**: loads new files as they appear in the current folder, for a scanner or tethering app. The "Working…" popup stays hidden; the status line reports each import.
*   **Trichrome Mode** and **Half Frame Mode** are on the Roll tab's Frame Assembly card ([§10.2](#102-frame-assembly)).
*   **Apply (clone)**: copies the current frame's settings, aspects chosen in a dialog, to selected frames or the whole roll. Crop and rotation stay per-image.
*   **Roll Settings** (tag icon): tags gear, capture, place, process and scanning metadata for the frame, a selection or the whole roll (default when a roll is loaded). Fields start from the active frame; **Load** a metadata preset to fill and tick fields, then tick groups to write. With Analog Gear empty, it matches the folder name against your Gear library, as at import, and never overwrites tagged gear.
*   **Save as Roll…** (red folder icon): keeps the loaded frames as a roll. See [Rolls that are not folders](#rolls-that-are-not-folders).
*   **Unload…**: drops the active frame or selection. For the whole roll, use *Clear All…*.
*   **Show Scenes** (layers icon): edges each frame in its scene's color, with the selection ring just outside it. See [Scenes](#scenes).
*   **Sort** (arrows): orders the frames by Name or Date, or by **Scene** once the loaded roll has one ([Scenes](#scenes)), ascending or descending. The Library's roll list has its own.
*   **Sheet filter** (funnel): *All Frames*, *Keepers Only* or *Hide Rejected*, for every roll.

Both of the last two are remembered between sessions.

Above both sections are the **filter box**, a **`.*`** regex toggle and a **search-library** button, shared by Library and Film Strip, plus a **search-by-meaning** toggle once enabled in Preferences. The Film Strip has a **tally** ("36 frames · 12 keepers · 3 rejected") and a **thumbnail size** slider. With a filter active, the tally names it ("3 of 36 frames · Keepers filter"); if it hides everything, **Show all frames** clears the filter box and funnel. The tally starts with the roll name ("Portra 400 — 36 frames"), or **Collection** for frames from a search, several folders or added by hand; their edits also show in each frame's own roll. In a narrow panel the tally is cut short with …; hover it to read it in full.

Right-click **empty space** for **Add Files**, **Add Folder** and **Clear All…** (always the whole session). Toolbar buttons that do not fit a narrow panel move into a **»** menu.

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

Terms combine with AND (`film:portra iso:>=400 -rejected:`). Metadata fields match only once filled in the **Metadata** panel; unedited frames match by name, extension, date and mark. **`.*`** makes the box a plain filename regex.

#### Searching by meaning

**Search by meaning** (Preferences → Performance) searches by photo content; it downloads a small model the first time. Check its toggle beside the filter box and type a description ("a photo of a dog") to narrow loaded frames to clear matches, best first. It excludes regex mode. The cutoff is relative to the other loaded frames, not a fixed score. Unindexed, it ranks only frames loaded this session, as their thumbnails finish. For the whole library, see **Indexing the library**.

#### Searching the whole library

The **magnifier-over-folder** button (or **Enter** in the box) runs the search across every library folder and loads the results. With **Search by meaning** on, it queries every indexed file. Keyword search opens no files: it uses the edits NegPy already stores, so metadata matches only frames you filled in. Folders are never indexed in the background or changed.

#### Indexing the library

With Search by meaning on and its model downloaded, a **database** button appears beside the Library refresh button. It decodes and embeds every photo in your library folders so search by meaning covers the whole library. It runs only when you ask, with progress and an Abort button; finished work is kept, and the next run does only the rest.

#### Stitching a frame from several shots

For a negative captured in overlapping pieces, select them and right-click → **Stitch Selected Frames**. NegPy registers the overlap, matches brightness across it, cuts the seam where the two shots agree best and shows one composite named *a+b (Stitch)*, badged on the sheet. **Unstitch** restores the parts with their edits. The registration is saved, so the composite persists across folders and launches.

For Trichrome, turn on **Trichrome Mode** first, then stitch the assembled frames; each part keeps its own three exposures.

#### Merging bracketed exposures (HDR)

A slide has more range than one exposure. Bracket it (shots a stop apart), select them and right-click → **Merge Exposures (HDR)**. They become one badged frame named *a +4 (HDR)*; **Unmerge Exposures** restores the originals. NegPy measures the exposure steps from the images (not shutter tags, which some scanner formats lack) and registers the frames. The merge is saved and persists across folders and launches.

The merge is computed relative to the longest unclipped exposure, the **reference**. It is usually brighter than your metered shot, and rendering there crowds the highlights. Choose the render exposure:

*   **Render exposure** (right-click a merged frame): pick a shot, listed in stops from the reference, which is marked *(as captured)*. Only the reference and **shorter** exposures are listed, since the merge cannot open brighter; with none shorter, the menu does not appear.
*   **Render Exposure** slider (Process panel): 0 EV (the reference) down to −4 EV, any value. Setting it clears a picked frame, and picking a frame clears it.
*   **Bracket middle (auto)**: the default, the bracket's middle exposure. Right only for an even bracket around the metered shot; an upward-only bracket makes it the reference.

A merge opens with **Shadows Density** above zero, set from the precision the bracket recovered, so the shadows stay quieter than the single frame's. Set it to zero for a render that matches the metered frame. **Reset Settings** restores the seeded value, the merge and the inherited film process.

Bracket both ways and include the shot that already looks right:

*   **Upward, for range**: longer frames reach the shadows and set the reference. Metered, +1, +2 (+3 for deep shadows).
*   **Downward, for choice**: shorter frames add little to the pixels but fill the **Render exposure** menu, and the darkest render is the darkest shot. The reference is usually **one to three stops above the metered frame**, so *(as captured)* is not your metered shot. Two stops below metered gives four or five entries, down to −3 and −4 EV.

Metered −2 through +3 is six frames; if you must drop some, drop long ones. Short frames also keep a **blown specular** above the reference's white. The gain is about two stops of shadow signal-to-noise, midtones unchanged.

The merge inherits its exposures' **film process** (stitched composites too), and is **named after the first frame in filename order** with an `-HDR` suffix: `_DSC1715`…`_DSC1719` exports as `_DSC1715-HDR.jpg`.

**Merging is for transparencies** (10-12 stops, against about 5-6 for color negative and near 4 for black-and-white), so it appears on Transparency frames only. On black-and-white it shows disabled: reversal monochrome (Scala, dr5, Fomapan R) is not supported yet. Frames already merged, stitched or Trichrome triplets cannot be merged.

### Triage (culling the roll)

Thumbnails are positives. An unopened frame is inverted in the background from a size-limited source preview (a per-channel inversion, not the full pipeline), quick previews first, then memory-bounded sources one at a time. Loading pauses while the selected frame renders. When idle, NegPy predecodes one neighboring frame if the source is bounded or small and its estimated decoder and cache memory fit system memory. A placeholder shows the frame loading. Automatic thumbnails never run a full camera RAW demosaic. Large TIFF-based scans, including LinearRaw DNGs, are read in bounded segments; a format with no bounded preview shows a neutral square. Opening a frame swaps in the real render. Applying settings or a preset to other frames re-renders their thumbnails in the background. Transparencies are not inverted: a frame with a set film process, or opened once, uses it; others are guessed from the preview.

Right-click a thumbnail, or use shortcuts, to mark frames (multi-selection works; marks persist):

*   **Keep**: a check badge.
*   **Reject**: a cross badge and dimming. Batch exports and sidecar writes skip it. The file on disk is never changed.

#### Reading the badges

| Corner | Badge | Means |
|---|---|---|
| Bottom-right | check | keeper |
| Bottom-right | cross, frame heavily dimmed | rejected |
| Bottom-left | *see below* | the frame was built from more than one file |
| Top-left | exclamation | the file failed to decode; click to retry |
| Top-left | small amber dot | the thumbnail predates a settings change (a bulk apply reached the file before a render reached its thumbnail); open the frame to refresh it |

The gray bottom-left glyph shows the frame type:

| Glyph | Frame |
|---|---|
| Two overlapping panes | a stitched composite ([§Stitching](#stitching-a-frame-from-several-shots)) |
| Three stacked bars | a merged bracket ([§Merging](#merging-bracketed-exposures-hdr)) |
| Three red/green/blue dots | a Trichrome triplet |
| A split rectangle, one side filled | one half of a half-frame scan; the filled side is which half |
| A split rectangle, both sides filled | a diptych: the whole scan, each half with its own edit |

The tooltip says it in words: *HDR merge of 5 exposures*, *Stitched composite of 3 frames*.

The right-click menu also has:

*   **Copy/Paste Settings**, with or without normalization bounds. Copied bounds show in the paste picker as **Normalization bounds**, ticked; untick to keep the frame's own.
*   **Reset Settings**; with several frames selected, **Reset N Frames**, confirmed first.
*   **Reset to Roll Settings**: **Reset to Roll** on every card of this frame that differs from the roll, as one undo step. Unlike **Reset Settings**, the rest of the frame's edit stays. Also in the canvas **⋯** menu.
*   **Apply Settings…**.
*   **Sync Bounds…**: pushes only this frame's measured bounds, as **Tonal span** and **Color balance**, to the selection or roll. Also in the canvas right-click and overflow menus.
*   **Update Thumbnail(s)**: re-renders the selection's thumbnails; **Update Thumbnails** on the toolbar does every stale one in the roll. Both become **Cancel** while running.
*   **Reset Roll to Defaults…**, and per-frame export.
*   **Edit Independently in This Roll** / **Use the Shared Edit Again**; see [Rolls that are not folders](#rolls-that-are-not-folders).

#### Scenes

A scene is a group of frames in one roll shot in the same light (the beach half of a roll that is also a night walk). It has its own baseline, so its frames match each other, not the rest of the roll. Scenes need a roll: open one, or **Save as Roll…** first.

*   **Scene** (right-click menu): **Group as Scene…** makes the selection a scene; **Add to** *name* and **Remove from Scene** move frames in and out. On a scene's frames, **Analyze Scene…** runs Scene Analysis; **Rename Scene…** and **Delete Scene…** manage it. A frame is in one scene at most; deleting a scene keeps edits and baselines.
*   **Sort → Scene** (offered once the roll has a scene): each scene's frames as their own block, from a new row, on a band in the scene's color; frames in no scene come last. Grouping a roll's first scene switches to it.

---

<!-- panel:analysis -->
## 3. Analysis readout (always visible)

The readout sits above the tabs and describes the current frame as you edit. Drag the divider to resize it, or collapse it. Top to bottom:

#### Photometric curve

The paper characteristic (H&D) curve NegPy prints through. It models the paper; it is not a curves editor. Horizontal is **negative density** (dense negative, scene highlights, to the right); vertical is **print tone**. Steeper means more contrast (Grade). The flat ends are the toe (shadows) and shoulder (highlights).

With a **Contrast Mask** set, a violet band opens between the curve and a dashed edge: large flat areas print on the dashed edge, fine detail on the solid curve. Dodge/burn, local grade and CLAHE are spatial and do not show on the chart.

The crosshair marks the **pivot**, the density the curve rotates around when contrast changes. While you drag a slider, a faint **ghost** shows the previous curve. When cast removal pulls the channels apart, you see three R/G/B traces; that spread is the color correction.

#### The two histograms

Behind the curve is the **output histogram** (print tones in R, G, B and luminance). Along the bottom axis is the **negative density histogram** (the scan, before the curve). If the negative's data sits on the flat toe, contrast cannot separate those shadows: move the exposure so it lands on the steep middle.

In Peek Negative the curve, output histogram and zone strip go, and the density histogram splits into R, G, B and luminance, each scaled to its own peak. A spike at an edge is that channel clipping; a trace apart from the others is a strong cast.

#### LIN / LOG toggle

Bottom-right of the chart. It sets the histogram's height axis (pixel count). **LIN** is literal; **LOG** compresses tall peaks so thin shadow and highlight tails show. Use LOG to find clipping, LIN to see the bulk of the frame. The choice is kept between sessions.

#### Clipping triangles

R, G and B triangles in the top corners: **top-left** is shadows crushed to black, **top-right** highlights blown to white. They show when a channel passes 0.5% of the frame. A little is normal. One channel clipping alone is a color cast, not an exposure problem.

#### Zone shading and zone ticks

The amber wash (left) and blue wash (right) mark the toe and shoulder, where separation is lost. The bottom ticks are Adams zones I to IX.

#### Step wedge

A 21-step Stouffer-style gray wedge printed through the current curve, in even density steps labelled in the scan's density units. Patches that merge into flat black or white are lost tones. The brackets mark the usable span. Hidden while you peek the flat scan.

#### Zone strip

Ten cells on the Adams scale: **0 is paper black and V is 18% mid-gray**; IX also holds paper white. Brightness is the zone's tone, opacity is how much of the frame lands there. The end cells turn **red** when shadows block up or highlights blow. Hover a cell for its percentage.

**Click a cell, then click that spot on the photo** to place that tone (see Zone placement). The armed cell stays outlined; click it again or press Esc to cancel.

#### Probe

A spot densitometer. Hover the image for per-channel density above film base (ΔD, relative to this scan's normalization), the reflection print density of the displayed tone, and its print zone (0 = paper black, V = 18% mid-gray, X = paper white). In B&W Negative mode ΔD reads the color record before conversion.

#### Zone placement

Works like an enlarging analyser. **Click a zone on the strip above, then click that spot on the photo**, and the print is solved so the spot prints on that zone. Add up to three pins; a fourth spot replaces the nearer pin. Each pin row shows its current zone and its **target**, which − and + trim in thirds of a zone.

*   One pin solves Print Density.
*   Two pins (usually shadow and highlight) solve Print Density and ISO-R Grade.
*   A third pin adds **Shadows Grade**, **Highlights Grade** or **Snap**, whichever can move that tone (set by where it prints). A line under the button names it; if the tone is already on target, nothing more changes.

The result is a preview until you accept. **Place zones**, or **Enter** over the photo, commits one undoable edit, turns off Auto Density (and Auto Grade with two or more pins) and closes the tool. **Esc** discards the armed zone first, then the pins and preview. The **✕** on a row removes that pin.

Drag a pin to move it (the cursor becomes a hand); it keeps its target and its caption reads `1 · IV⅓ → VI` (now → target) until the two agree. A click with no zone armed only pins a reading.

An unreachable target shows an amber `→ lands …` with the closest zone the print can make. With three pins, amber also shows when the targets cannot all be met, with where they settle.

Pins are proofs, not edits: any other edit or a frame change removes them. They read through the print curve, so later stages (Lab, toning) can still change the pixel the hover probe reads. Zone placement is off on an as-captured slide or a Positive frame, which print through no paper curve.

#### Negative stats

Rows that measure the scan, not your edit; hover for details. A row with nothing to measure reads —.

*   **Negative**: relative density range (luminance) and development character: flat (≈N−1), normal, contrasty (≈N+1). Comparable across a roll; estimated from the normalized bounds, not a densitometer reading.
*   **Exposure**: midtone in stops from neutral, approximate; positive is high-key, negative low-key.
*   **Clipping**: share of pixels crushed to black or blown to white, worst channel. Red above 1%.
*   **Scan clip**: share of source pixels at or above sensor white, per channel. In a negative scan this destroys base and shadow separation and no edit can undo it: expose the scan lower. Red above 1%.
*   **Repair**: share of the whole scan (border included) rewritten by IR Restore, dust detection and painted heals; none until one fires. A large value means the threshold is redrawing the picture. Red above 5%.
*   **Gamut**: share of the frame the proof profile cannot print, while proofing to one. Zero is normal. Red above 2%.

---

## 4. Geometry tab

<!-- panel:geometry -->
### 4.1 Geometry: crop and straighten

**Crop:**

*   **Auto**: detect the frame edge and crop to it. Its settings and the whole-roll run are on the Roll tab's **Crop** card ([§10.4](#104-crop)).
*   **Ratio**: the roll's crop ratio, the same field as on the Crop card; the crop tool snaps to it.
*   **Crop** tool: draw a crop rectangle; when **Ratio** is **Free**, drag an edge midpoint to resize one axis. It opens on the current crop, including one **Auto** found; after a manual change nothing re-detects over it. **Reset** clears it and turns auto-crop off.
*   **Guide**: *Thirds*, *Phi Grid*, *Diagonals*, *Golden Triangles*, *Golden Spiral*, *Armature*, *Diagonal Method*, *Grid* or *Off*. The redo button rotates guides with orientations (spiral 8, triangles 2).

**Alignment:**

*   **Crop by Default**: crop the wedge Fine Rotation, Tilt and Swing leave, so no edge shows extrapolated pixels. Live, only while no manual or auto crop is set. While you adjust a slider below, the canvas briefly darkens the margin it trims.
*   **Fine Rotation** (±45°): sub-degree rotation, positive clockwise. Applied after auto-crop.
*   **Straighten** tool (ruler): draw a line along a horizon or vertical edge to level or plumb it.
*   **Tilt** (±15%): tip the easel about a horizontal axis to correct converging verticals. Positive stretches the top edge. The unit is percent of the frame, not an angle.
*   **Swing** (±15%): the same about a vertical axis, for converging horizontals. Positive stretches the left edge.

    Both leave a wedge along the squeezed edge: crop it or use **Crop by Default**. Crop first if you can: the meters read the corrected frame, and a large correction on an uncropped scan pulls the rebate in and darkens the print.

Scanning-lens distortion, chromatic aberration and flat field are in **Optics** on the Roll tab ([§10.8](#108-optics)).

---

## 5. Exposure tab

Three panels set light, color and contrast in the print stage of the pipeline.

<!-- panel:color -->
### 5.1 Filtration: white balance

Color timing, like enlarger dichroic filters. **Global / Shadows / Highlights** applies the controls to the whole image or biases them to low- or high-density tones.

*   **Pick WB** (eyedropper): click a pixel that should be neutral gray; NegPy solves the CMY filtration for the selected region.
*   **Roll Lock**: re-aims each newly opened frame's temperature to the current target, keeping its tint. Per region.
*   **Reset** (undo-arrow icon): set the region's temperature and CMY to neutral.
*   **Temperature**: warm-to-cool lever on the magenta/yellow pair; cyan stays put.
*   **Cyan / Magenta / Yellow** (-1 to 1): Cyan↔Red, Magenta↔Green, Yellow↔Blue.
*   **Cast Removal** (0.0 to 1.0, **color only**): balances each layer against the frame's own grays so neutrals stay neutral from shadows to highlights; strength scales with how many clean near-neutrals the frame has. On Color Negative it removes the **orange mask** and starts at 1.0. On Transparency it starts at 0 and corrects a faded slide's crossover (a slide's cast can be the photograph). For other slide color use **Temperature**, the CMY sliders or **Hue Trim** (§10.3). Hidden for B&W Negative.
*   **Ring-around** (target icon, or `Shift+F`): a 5×5 mosaic in 2cc steps to ±4cc on magenta and yellow, centered on neutral, so rings compare across frames. Each patch renders the part of the frame it covers; click one to keep its filtration. `Escape` or a second press clears it; any edit drops it. See **Rotating a proof** below.

<!-- panel:tone -->
### 5.2 Tone: density, contrast and the print curve

**Global / R / G / B** applies most controls to the shared curve, or as per-dye-layer trims for **crossover correction** (casts that differ between shadows and highlights).

**Automatic helpers** (on by default; turn off to print the negative as it is):

*   **Auto Density**: meters each frame's midtone and anchors print brightness there.
*   **Auto Grade**: partially picks the grade from the frame's textural density range, goes harder if needed so the darkest textured tones reach black (Shadow Reach in Set Targets), and burns the brightest textured tones off paper white (Highlight Hold).
*   **Set Targets** (sliders icon): the brightness and contrast the helpers aim for. All frames, kept between sessions.

**Test strip** (grid icon, or `Shift+T`): a 5×5 grid, Print Density rising left to right, ISO-R Grade softening top to bottom. Both ladders are centered on their defaults, so your settings are one patch. Click a patch to keep it. `Escape` or a second press clears it; any edit drops it.

**Rotating a proof**: each patch shows only the part of the frame under it. While a proof shows, the 90° **rotate** buttons and `[` / `]` turn the ladder, not the image, and the axis labels follow. The orientation stays for the session.

**White Point** and **Black Point** are on the **Metering** card ([§10.6](#106-metering-negative--positive)).

**Exposure:**

*   **Print Density** (0.0 to 2.0): overall brightness (enlarger time). Lower is brighter.
*   **ISO-R Grade** (50 to 180): contrast as paper ISO-R. R110 is about grade 2; **lower R is harder**. In R/G/B mode a **Grade** trim rotates one layer's slope about the midtone.
*   **Shadows Density** (±0.9 ΔD) / **Highlights Density** (±0.5 ΔD): brighten or darken only the shadow or highlight zone, bounded by paper black and white. The ranges differ because the same ΔD looks smaller near paper black. They also work in Transparency, where they are the only controls that spare the midtones.
*   **Shadows Grade** / **Highlights Grade** (split grade, ±50 ISO-R): local contrast in the deep shadows or highlights.
*   **Contrast Mask** (±0.5, hidden in Transparency): a blurred mask sandwiched with the negative; the value is its signed gamma. Positive (a positive mask) compresses the range by (1 − gamma) so a harder grade fits the paper, keeping fine detail; use it on a scene too contrasty for your grade, then lower Grade in R. Past about 0.4 edges get a soft halo. Negative expands the range by (1 + gamma) without steepening grain, and works on a negative too flat for Grade; past about −0.4 highlights clip (see the Clipping row).
*   **Mask Spacer** (2 to 6%, no effect without a mask): the gap between mask and negative, as percent of the frame. Thick masks only broad masses; thin reaches into detail, bites harder, and hazes shadows next to bright areas. 4% is a conservative default. Both mask controls read only your crop and gray out in R/G/B mode.
*   **Dye Separation** (0.5 to 1.5, hidden in B&W Negative): saturation in density space, applied before decode in the paper's crosstalk matrix, so it follows the paper profile and eases off at toe and shoulder. On a slide it applies to density directly. Below 1.0 pulls toward neutral; 1.0 is off. **Chroma** (Look tab) instead scales color evenly after decode.
*   **Separation Damping** (0 to 1, hidden in B&W Negative): where the Dye Separation push lands. Higher keeps the full push on muted color and reduces it on saturated color; below 1.0 separation, pastels go gray first. Grays out **at Dye Separation 1.0**.

**Paper Response**:

*   **Paper profile**: a bundled paper profile (RA4 in Color Negative, tonal B&W papers in B&W Negative) that sets the curve baseline; the other controls trim on top. *Neutral* gives the defaults. Each B&W paper has its own lith color path: Fomatone warm and colorful, *Neutral* and Ilford Multigrade nearly colorless.
*   **Paper White**: simulate paper base density, so whites print at about 0.93.
*   **Paper Black**: show the paper's slightly milky Dmax. Off (default) applies black-point compensation.
*   **Preflash** (0 to 1): an even flash of light over the whole sheet, as a fraction of the paper's threshold exposure. It pulls highlight detail off paper white and softens the print a little overall; a harder Grade gives the contrast back. Bare paper stays white. Hidden on slides.
*   **Snap** (-0.5 to 0.5): midtone gamma; paper white and black stay put.
*   **Toe** (-1 to 1) + **Toe Width** (0.1 to 5): shadow roll-off. Positive lifts shadows; negative deepens them and, with Paper Black off, reaches exact black. Width sets how far the knee reaches.
*   **Shoulder** (-1 to 1) + **Shoulder Width** (0.1 to 5): highlight roll-off. Positive compresses highlights; negative extends them and can clip.

In R/G/B mode these become per-layer trims: **Grade** (±30 ISO-R), **Toe** / **Shoulder** (±1), **Toe Width** / **Shoulder Width** (±2), **Snap** (±0.5), **Dye Separation** (±0.4).

<!-- panel:local -->
### 5.3 Dodge & Burn: local exposure

Draw masks and lighten or darken only those areas:

*   **Draw Mask** (the cut card): click to place vertices; double-click, Enter or click near the start to close; Esc cancels. To edit, select the mask, then drag a vertex, click an edge "+" to add a point, or right-click a vertex to delete it.
*   **Oval** (the hole in the card, or a dodging wand): drag out an oval. The center handle moves it; the other two set each axis, so you can stretch and tilt it.
*   **Card Edge** (the graduated burn): drag from the full-exposure edge (solid line) to where it fades out (dashed). The gap is the softness, so **Feather does nothing on this shape**.

Handles can go into the gray area outside the frame. A tilted Card Edge usually must start past the corner it burns.

*   **Mask list**: shape icon, Dodge (lighten), Burn (darken) or Grade (contrast only), and values. Click the shape icon to enable or disable the mask (its row grays out). The yin-yang inverts the mask, so it acts everywhere except inside its shape (red while on). The eye toggles the outline; the trash deletes it.
*   The canvas tint of the current mask, and of masks that **intersect** it, goes while you hold **Burn**, **Feather** or **Grade** or drag a vertex. A Card Edge or an inverted mask intersects anything on its side.
*   **Burn** (-2 to 2 stops, default 0): **positive burns** (darker), **negative dodges** (brighter), like Print Density and the Finishing edge burn.
*   **Feather** (0.0 to 0.15): edge softness, as a fraction of the frame's short side.
*   **Grade** (-40 to 40 R): the mask's own contrast, in ISO-R points off the frame's Grade, negative harder (burn a sky at −20 R, dodge a face at +15 R). It rotates about the region's midtone, so with Burn 0 only contrast changes. Overlapping grades add, clamped to R50…R180.
*   **Tone Limit** (*All*, *Highlights*, *Shadows*) with **Tone Zone** (0 to 10, in thirds, default 6) and **Tone Softness** (⅓ to 3 zones): limit the mask to tones lighter (*Highlights*) or darker (*Shadows*) than a zone of the print before any mask, like a lith mask registered with the negative. A sky burn on *Highlights* at VI stops at the skyline instead of darkening a band of it. The tint shows the tones it selects. Up to four masks per frame.

**Printing Notes** (Export tab, or **Shift+N**) makes a marked-up work print: each mask outlined with its number and value in stops (a Card Edge marks its full-exposure side), and a corner card with paper, Print Density, ISO-R Grade (and split-grade trims), filtration, toe and shoulder, Snap, edge burn and the dodge/burn list.

*   **Burns are hatched, dodges are left open.**
*   **The numbers are exposure, not brightness**: +1.00 st is `Burn +1`. Values snap to ⅓, ½ and ¼ when close, else decimals.
*   A mask with a local **Grade** shows the grade it prints at: `Burn +1 @ R95` (−20 R on an R115 frame), or `Grade @ R95` for a grade-only mask. A tone-limited mask adds its zone: `Burn +1 on ≥VI`.

Masks with a hidden outline stay on the map; disabled masks do not. The overlay hides while a test strip, either peek, the before/after baseline, or the crop and analysis tools use the canvas. Preview and export are in the Export tab's **Printing Notes** section.

---

## 6. Look tab

<!-- panel:lab -->
### 6.1 Lab: polish and detail

What a lab scanner (Frontier or Noritsu) does automatically.

**Color** (hidden in B&W Negative):

*   **Chroma** (0.0 to 2.0): even color scale after decode. 1.0 is unchanged, 0 grayscale, 2.0 double. Above 1.0 out-of-gamut pixels get a soft per-pixel knee, so hue stays. For print-like saturation use **Dye Separation** (Exposure tab).
*   **Skin Protection** (0.0 to 1.0, default 0.5): holds skin-hued chroma under a ceiling; it only lowers chroma, independent of Chroma. 0.5 catches excessive chroma, 1.0 leaves skin matte, 0 is off. The mask needs warm hue, skin-level chroma and mid lightness, so red coats, sunsets and brick stay out; wood, tan leather and sand soften with it. A strong sunburn is only partly caught: use Chroma or the Filtration panel.
*   **Chroma Denoise** (0.0 to 5.0): smooths color noise, mainly in shadows; luminance grain stays.

**Sharpen:**

*   **Method**: *Unsharp Mask* (edge contrast) or *Deconvolution* (Richardson-Lucy, reverses the scanner's blur; set Radius to the blur width).
*   **Sharpening** (0.0 to 1.0): amount, on the L (lightness) channel, so no color halos.
*   **Radius** (0.5 to 3.0 px): blur width in output pixels. It acts on export pixels, so judge it at 1:1 with the loupe or at 100% zoom.
*   **Masking** (0.0 to 1.0): limit sharpening to edges to protect sky, skin and grain. The deepest shadows always get a third of the amount.

**Detail:**

*   **CLAHE** (0.0 to 1.0): local contrast without blowing highlights or crushing shadows. Near 1.0 it can look cartoonish. Runs before dust removal.

**Effects:**

*   **Glow** (0.0 to 1.0): lens bloom across all channels.
*   **Halation** (0.0 to 1.0): red glow from light scattering back through the film base, highlights only.

<!-- panel:altproc -->
### 6.2 Alternative Processes

Pick **None / Lith / Cyanotype**; only that process's controls show. B&W Negative only, off by default.

#### Lith

A heavily over-exposed lith paper in dilute low-sulphite developer, pulled part-way: creamy warm highlights and an abrupt drop into sooty blacks. The paper in the Exposure panel sets the color path (peach, olive, neutral blacks): *Neutral* and the Ilford papers are almost colorless, Fomatone gives the peach and olive. Sepia, Iron Blue, Copper and Vanadium gray out in Toning; Selenium and Gold act differently (see 6.3).

*   **Exposure** (0 to 5 stops, default 2): over-exposure; real lith uses two to four stops. More gives warmer, more colorful highlights and softer gradation.
*   **Snatch Point** (0.0 to 1.0, default 0.55): time in the developer. Higher gives deeper, colder blacks and more flat shadow; lower stays high-key and warm with weak blacks.
*   **Abruptness** (0.0 to 1.0, default 0.6): how suddenly shadows go black (the developer's hydroquinone-to-alkali ratio). High blocks up the next zone down; low rolls off gently.

#### Cyanotype

UV contact print on iron-salt paper. The image is Prussian blue (absorbs red around 700nm), so the print goes blue, not black, with green highlights from leftover yellow sensitizer. It holds a short density range, so a normal negative clips at both ends, and compresses the midtones. Every chemical toner grays out (no silver); use Bleach and Tannin. Split toning still works.

*   **Sensitizer** (Classic or New, default Classic): *Classic (Herschel)*, ammonium ferric citrate, tops out at a light blue with a green highlight stain. *New (Ware)*, ferric oxalate, goes deeper and cleaner.
*   **Exposure** (-2 to 4 stops, default 0): UV time. More moves more of the scale into blue.
*   **Exposure Scale** (0.8 to 2.8 log D, default 1.4): the printable density range, the contrast control. Ware: about 1.0 to 1.2 traditional, 2.4 new, Simple Cyanotype 1.8, 2.3 and 2.7. Shorter is more contrasty.
*   **Bleach** (0.0 to 0.5, default 0): washing soda; removes blue, highlights first.
*   **Tannin** (0.0 to 0.5, default 0): tea, coffee or tannic acid; turns bleached iron brown and a little deeper. Bleach first for full brown; Tannin alone for split blue-brown.

---

<!-- panel:toning -->
### 6.3 Toning

Chemical toners (B&W Negative only) and a split tint (any mode). On a lith print toners bite harder: only Selenium and Gold stay enabled. With Cyanotype all six gray out.

**Chemical Toning**, sequential baths in the order shown, each 0.0 to 2.0:

*   **Selenium**: deeper blacks, cool eggplant shadows. On lith: further down the scale, strong Dmax lift, green-black shadows to magenta.
*   **Sepia**: warms highlights first; partial strength gives split-sepia.
*   **Gold**: blue-black on untoned silver; over sepia, orange-red highlights. On lith: all densities evenly, toward blue-violet.
*   **Iron Blue**: Prussian-blue shadows to navy blacks.
*   **Copper**: pink to brick-red, with the classic Dmax loss.
*   **Vanadium**: greens mids and highlights; deep shadows stay black.

**Split Toning** (all modes), an additive Lab tint that keeps grain and detail:

*   **Shadow Hue** (0 to 360°) + **Shadow Strength** (0.0 to 1.0).
*   **Highlight Hue** (0 to 360°) + **Highlight Strength** (0.0 to 1.0).

---

## 7. Finish tab

<!-- panel:retouch -->
### 7.1 Retouch: dust, hairs, scratches

Spotting, as done with a brush on a finished print. Marks are found by local contrast, by the scanner's IR channel or by hand, and the three stack. Each mark is rebuilt from the clean film around it, with the frame's own grain put back; a mark too wide for that goes to a fill that follows the structure through.

**Overlay** cycles the detection overlay (Off → Marked → IR): green for Optical Removal finds, magenta for IR finds and for defects sent to the structure-following fill.

**Optical Removal** finds specks and hairs on the visible scan, with no IR needed:

*   **Threshold** (0.01 to 1.0): lower catches more, with more false positives. Above the default the bar rises faster, so the top end leaves sharp highlights and dense lines alone. The bar is measured against the film's own grain, so a value means the same on any scan. It rises in busy detail, where a thin dark line looks like a hair, so dust on textured film may need a lower Threshold or the IR or Heal tools.
*   **Size** (3 to 8 px): max spot radius. A mark covers the whole speck or hair.
*   To protect detail, right-drag on the canvas to paint an exclusion band, or right-click and pick **Exclude From Optical Removal** for one spot. Only what you paint is excluded. The band is Brush Size wide and shows in amber with the overlay on. Toggling **Optical Removal** clears every band.
*   The cursor button beside **Optical Removal** makes a plain right-click exclude the spot, with no menu. The canvas menu (Copy Settings, Reset View) is then out of reach while the removal is on; the Heal and Scratch tools keep their right-click. The setting is remembered between sessions.

**IR Removal** uses the infrared channel to remove dust the dyes do not show. It is enabled only when the scan has an IR plane.

*   **IR Threshold** (0.05 to 0.95): lower catches more.
*   **Method**: how the film under a defect is rebuilt, from the same IR plane and threshold.
    *   **NegPy** (default): divides semi-transparent dust back out, fills opaque cores with a weighted average of the clean film around them, and takes grain from the nearest clean pixel.
    *   **OpenICE**: works in log density and, at each scale, adds back picture detail stronger than the infrared's contrast, so texture under a speck survives. A solid defect gets Digital ICE's synthetic grain, strongest in the midtones. It measures clear-film level and dye-to-infrared crosstalk per frame and leaves clean film untouched bit-for-bit. Better on fine detail but less proven across scanners, so compare both on a frame you know.
*   The IR plane is read from 4-channel TIFFs and DNGs (VueScan, NegPy's own scanner output), SilverFast's iSRD TIFFs and 64-bit **HDRi RAW DNGs**, and `_IR.tif` sidecars. Scan to HDRi, not plain HDR, to keep IR data. B&W and Kodachrome frames are skipped automatically, since they block infrared like dust.

**Manual Heal** (the header shows the spot count). The brush marks a *search area*, not a stamp: only pixels that stand out from the film around them are rewritten, both dust (prints light) and scratches (print dark), so you can paint generously. If it finds nothing, it does nothing.

*   **Heal Tool**: click dust spots to paint them out, or drag over a run of them.
*   **Scratch Tool**: click points along a scratch or hair, then double-click or press Enter. Esc cancels, Backspace removes the last point. Right-click an overlay to delete it.
*   **Transport Line**: for long straight marks from camera or lab transport that cross the frame. **Click once anywhere on the scratch** to trace and repair the whole line. Such a scratch is too faint to see at one point, so the tool reads its full length. It follows the scratch's angle and width and repairs only where the scratch is present. If a click finds nothing, it says so; click directly on the line. Hovering shows a **guide** of the line and band it would repair, and placed lines stay drawn; right-click one to delete it.
*   **Line Sensitivity** (0.05 to 0.95, shown with the Transport Line tool): lower catches fainter lines and repairs a wider band; raise it if a line picks up film on either side. It also applies to placed lines.
*   **Brush Size** (2 to 64 px): diameter of the heal, scratch and exclusion brushes, as the cursor shows. Shown while a heal or scratch tool is active or Optical Removal is on. Hold `Alt` and scroll on the canvas, or pinch while a brush is live.
*   **Undo Last** / **Clear All**: remove the last or all manual heals and traced lines; auto-detected dust is unaffected.

<!-- panel:finish -->
### 7.2 Finishing: vignette, carrier, border

How the print is presented. Applied at the end of the pipeline.

**Vignette** (printer's edge burn):

*   **Burn** (-2.0 to 2.0 stops): positive darkens the edges, negative lightens them. 0 is off.
*   **Size** (0.0 to 1.0): falloff radius, from tight in the corners to spread into the frame.
*   **Roundness** (0.0 to 1.0): 0 is radial (lens-like), 1 is a rectangular card burn along the print edges.

**Filed Carrier**: the clear rebate of a filed-out carrier prints at the paper's black, framed by unexposed paper. It prints through the frame's own curves and filtration, so the filed edge's fringe takes the paper's toe color and, with **Paper Black** on, the rebate stops at the paper's D-max. The film sits off center, so the top and left rebates print wider than the bottom and right.

*   **Width** (0.0 to 5.0 mm): black frame thickness. 0 is off.
*   **Roughness** (0.0 to 1.0): how raggedly the paper-side edge was filed: straight file strokes with nicks, most of them near the corners. The picture-side edge (the film gate) wobbles only slightly.
*   **Flare** (0.0 to 1.0): light off the filed bevel exposes the paper just outside the edge, in the same toe color as the fringe; neutral in B&W. 0 is off.
*   **Corners** (0.0 to 1.0): how far the filed aperture's corners round off. The picture's corners are the camera gate's and stay nearly square.

The paper margin takes the mat color, so it joins the border with no seam.

**Border:**

*   **Width** (0.0 to 2.5): thickness as a fraction of the image. 0 is no border.
*   **Bottom Weight** (1.0 to 2.0): thickens the bottom, for window-mat proportions.
*   **Color swatch**: click to pick a border color.
*   **Paper White**: use the toned paper-white instead of the picked color.

---

## 8. Favorites tab

The sliders you use most, in one place. Empty until you fill it.

*   **Edit Favorites**: tick sliders on the left, drag them into order on the right, then press **Apply**.
*   They are the same controls as in their home panels, so a change here is a change there. A favorite hides when its original does (a Filtration slider in black & white).
*   Your selection is remembered between sessions.

<!-- panel:presets -->
### Presets

A collapsible card below your favorites. It saves and recalls edit settings by name.

*   **Apply** (or double-click a preset): apply the selected preset to the current image.
*   **Save…**: pick which of the current settings to store as a new preset.
*   **Pen** and **Trash**: edit or delete the selected preset.

---

## 9. History tab

Two lists: the versions you chose to keep, above the record of every change.

### Work prints

A **work print** is a named version of this frame, like the test prints kept on the way to the final one.

*   **Save Work Print** (**Ctrl+Shift+S**) keeps the current edit under a name; NegPy offers *Work print 1*, *Work print 2* and so on. Saving over a name asks first.
*   **Click** one to make it live. That is an edit, so **Ctrl+Z** restores the previous state.
*   **Right-click** for **Export This Version…**, **Rename…** or **Delete**. Delete asks first; an empty name is ignored.

Work prints are **never pruned and never thrown away by a later edit**, unlike the undo history, which keeps the last 100 steps and drops the branch above you when you edit after stepping back. The list appears once you save one. Work prints belong to the frame (a preset is a look for other images). They are stored in NegPy's database, not in `.negpy` sidecars.

### Edit history

Every edit step, the last 100 kept, newest on top. The current step is bold.

*   **Click** a step to jump to that state.
*   **Right-click** → **Export This Version…** to export a past state.

---

## 10. Roll tab

**Every card here is shared by every frame in the roll**, once applied. A control edits the current frame only; when the card stops matching the roll, its scope pair shows **Frame** (see [§1](#frame-or-roll-the-scope-pair)), and it returns to **Roll** if you set it back to the roll's value.

The card's **Roll** button pushes this frame's value to the roll, and the frame rejoins it. **Reset to Roll** does the reverse: the frame takes the roll's value. Other frames that locked the card keep their own value. The line above the cards names every card this frame overrides. A frame outside any roll has no scope pair.

<!-- panel:film -->
### 10.1 Film Mode

Always expanded and first, because it decides which cards apply: **Color** (C-41 color negative), **B&W** (panchromatic negative) or **Slide** (transparency/reversal, E-6 and similar). Each changes the conversion math and re-runs the pipeline. The wand button **auto-detects** the mode when a file loads.

**Positive** (default off, shown on **Slide** only) is for a source that is already a positive (a scanned print, another app's export, a negative the scanner positivized), not a raw capture. NegPy decodes its embedded profile (sRGB if none) and skips metering, inversion, the exposure lift and the filmic roll-off, so the Print/tone controls in Metering (§10.6) shape the image directly and its bounds and clip controls hide. Leaving Slide turns it off.

<!-- panel:assembly -->
### 10.2 Frame Assembly

How the files become frames. Neither toggle has a scope pair.

#### Trichrome

*   **Trichrome Mode** (three-exposure narrowband capture, also called trichromatic capture): assembles each frame from a red, green and blue exposure. Shots are grouped by the capture time in the files, so shoot each frame's three exposures back to back; filenames need no convention. With no capture time, filename order is used. Three shots are assembled only if they are one of each color *and* show the same frame; otherwise they stay separate for pairing by hand. An assembled frame has the three-dot badge (see [Triage](#triage-culling-the-roll)).
*   **Edit Triplet…**: the film strip's right-click **Edit RGB Triplet…** dialog, for the current frame. Its **Align channels (sub-pixel)** registers green and blue to red, which removes color fringing from capture drift.

The line under the buttons names the two exposures the frame is assembled from. The toggle is one rig-wide flag, so it has no scope pair and applies to every roll.

#### Half Frame

**Half Frame Mode** splits each scan into two frames, for half-frame cameras. Each half is edited and metered separately and badged. Turning it on auto-detects the gutter and the outer film crop on every loaded scan, cropping into the rebate where one shows; a side with no visible or confidently readable rebate stays uncropped. Each roll remembers its own state. It is disabled for a batch that is not one roll (a library-wide search, a restored session with no shared roll).

*   **Adjust…**: a rectangle editor for the current scan. Drag the green box to crop, drag the orange line to set the split, and set **Cut thickness** to discard the black separator band centered on the split. **Auto-detect** re-finds crop and gutter on this scan. **Apply** is a split button: its ▾ picks *Apply to current frame*, *Apply to selected frames* or *Apply to all frames* (the roll default for frames without an override), remembered for next time.
*   **Detect All**: re-runs the batch detection.
*   **Unsplit** (enabled on a split frame): reverts it, as does its right-click item.

Right-click a half-frame asset for **Adjust Split for This Frame…** (defaulted to *Apply to current frame*) and, with an override, **Reset Split to Roll Default**. Heal strokes, dust spots, scratch lines and dodge/burn masks move with a recrop or resplit, anchored to the film.

Turning Half Frame off keeps each edited half. A scan with edited halves becomes a **diptych**: both halves rendered with their own edits, side by side at the original spacing, with the cut band as a black gap. If only one half was edited, its edit is used for both. A scan whose halves hold edits from another folder or an earlier session stays one plain frame. A diptych has the both-sides-filled split badge, exports as `<name>-DIPTYCH`, and has its controls panel disabled; turn Half Frame on to edit a half. Its filmstrip thumbnail and contact-sheet tile show the whole scan. A long-edge export size applies to each half, so a diptych is about twice that wide. **Unsplit** makes it one plain frame and deletes both halves' edits.

Half Frame never splits a frame assembled from several files (a Trichrome triplet, a stitch, an HDR merge).

<!-- panel:sensor -->
### 10.3 Calibration: what your rig does to the colors

These controls correct the *capture*, not the look: the camera's color filters, the film's dyes and the light source each have their own control, and none replaces another.

**Capture**:

*   **Scanning setup** (bulb button): a wizard (*how do you scan?*, *what light source?*) that sets Linear RAW and Narrowband. It runs once after the first-launch tour; reopen it when your rig changes.
*   **Linear RAW** (default off): decodes RAW with neutral multipliers; off uses the camera's as-shot white balance. Toggling it reloads the file. **Locked on for a Trichrome triplet**, since a narrowband exposure has no scene white balance; it stays visible and remembered.
*   **Narrowband**: corrects the oversaturation of narrowband (RGB-LED) capture with a bundled input profile. Leave it off for broadband light. An explicit Input ICC in Export overrides it. **Grayed out on Transparency** (see *Narrowband and slides*).

What the wizard sets:

| Capture | Light source | Linear RAW | Narrowband |
| --- | --- | --- | --- |
| Digital camera | White light (lightbox, CRI LED panel) | off | off |
| Digital camera | Narrowband RGB (Scanlight, RGB LED) | on | on |
| Film scanner | White light (Plustek, Epson, most flatbeds) | on | off |
| Narrowband Scanner | Nikon Coolscan, Kodak Pakon | on | on |

Applying it sets the defaults for new files and rewrites the open frame and every edited frame in the session, undoable per frame with Ctrl+Z.

**Crosstalk** (hidden in B&W Negative): a channel unmix on the raw densities before inversion. The dropdown lists only matrices for the current film process; a mismatched stored profile gives no correction. The dyes, your light's spectrum and your sensor's filters all mix the channels the same way in density, so a matrix describes *your whole scanning setup*, and may not suit another rig with the same stock.

*   **Matrix**: grouped by source (measured, tuned on a rig, or from spec sheets). *Generic C41* is built in; custom `.toml` matrices go in `<Documents>/NegPy/crosstalk/` (see [CROSSTALK.md](CROSSTALK.md)). The slider button opens a matrix editor, with **Type** (the source) and **Process** (which film the numbers describe, and so where the matrix appears and applies; a slide matrix needs E-6). A matrix made with **+** gets the current process. With no matrix for the current process, the dropdown and **Strength** are disabled with a hint; the editor button stays live.
*   **Strength** (0.0 to 1.0): how much unmix to apply. **Re-run Roll Analysis** after changing it.

> **The bundled film matrices come from published spec sheets**, marked *(approx)*, and describe the **dyes alone**. They are a full correction only where the capture reads each dye cleanly: a **Narrowband Scanner** (a Coolscan's mono sensor; a Pakon's trilinear array, with slight bleed), a **Trichrome** capture, or a Single-Shot Narrowband rig with **Single-Shot Narrowband Calibration**. With broadband light and a Bayer sensor, treat them as a starting point.

> **No Transparency matrix ships with NegPy.** On slides the Matrix dropdown and Strength stay disabled until you press **+** in the editor or drop in a `.toml` with `process = "Transparency"` (`process = "E-6"` also loads). A slide already shows its dyes' absorptions, so unmixing moves it *away* from its own look: treat it as a color-separation control. **Hue Trim** applies to slides as to negatives.

To tune a matrix, adjust its six off-diagonal terms in the editor and save it as your own, named after the *combination* ("Gold 200 + Spectracolor"). Working profiles are welcome [upstream](CROSSTALK.md#contributing-a-matrix).

**Single-Shot Narrowband Calibration**: for single-shot camera scans under narrowband light, where the sensor's filters overlap the light's bands and each color leaks into the others. The leak belongs to sensor and light, so it is corrected on the linear capture before inversion.

*   **Profile**: the sensor matrix. Custom `.toml` matrices go in `<Documents>/NegPy/sensor/`.
*   **Calibrate** (vials icon): build a profile from three bare-light R/G/B exposures.

Grayed out unless **Linear RAW** is on (profiles assume neutral white balance) and on **Transparency**; skipped for RGB-triplet assets. The selection is remembered. **Re-run Roll Analysis** after changing it.

**Light source:**

*   **Hue Trim** (-30° to 30°, default 0): rotates every hue by a fixed angle, to undo narrowband LED and odd-phosphor lights, which turn every color by about the same angle (yellows read orange, greens go olive) and leave neutrals alone. White balance cannot fix a rotation. Judge it on a known color (foliage, blue sky, skin); leave it at 0 for broadband light. It is **sticky** and carries to the next file. Neutrals are untouched, so the color-balance clip in **Metering** is unaffected.

#### Narrowband and slides

**Narrowband and Single-Shot Narrowband Calibration do not apply to Transparency**. They stay visible and grayed, keep their values, and return on a negative. The bundled profile describes negative dyes, and a narrowband light cannot be calibrated against a slide render. For slides on a narrowband rig, use **Hue Trim** to correct the light's hue rotation.

<!-- panel:autocrop -->
### 10.4 Crop

The shape every frame is cut to and what the frame detector looks for, roll-wide. Each frame's rectangle is its own, drawn or found in **Geometry** ([§4.1](#41-geometry-crop-and-straighten)).

*   **Ratio** (default `Free`): `Free`, `1:1`, `3:2`, `4:3`, `5:4`, `6:7`, `7:5`, `65:24`, `16:9`, `16:10`, `11:8.5`, each auto-oriented as you drag. On `Free` the crop tool is unconstrained and auto-crop uses the detected format (6x6, 645, 6x7, 35mm). A ratio forces every frame to it. **Geometry** shows the same Ratio beside the crop tool.
*   **Detect** (crosshairs): snap the ratio to the closest standard.

**Auto Crop**:

*   **Mode**: *Image only* (exposed area) or *Film edge* (full film, including rebate and sprockets).
*   **Crop Offset** (-5 to 100 px): inset the detected edge; negative bleeds slightly outside.
*   **Rebate Trim** (0 to 150%): 0% stops at the film edge, 100% at the detected image edge, above 100% cuts into the picture to clear a white border. *Image only*; applies to **Frame** and **Roll**.
*   **Frame**: detect this frame's edge and crop to it, the same toggle as **Auto** in Geometry. Off clears the crop.
*   **Roll**: crops all visible landscape frames as a roll, calibrating weak detections from confident ones. Portrait frames are cropped alone, as with **Frame**. With no readable film edge anywhere (film overfills the sensor), each frame trims its own border; a roll of only a few frames, including an RGB triplet, trims by the edges brighter than the picture. Runs in the background with progress and cancel. Manual, Film-edge and ambiguous frames are left alone. *Image only* mode only.
*   **Mixing scans**: allowed. A frame that reads its own edge keeps its crop; a frame with no edge takes the roll's width and tilt. Frames from one camera, holder and format pool best.
*   **When auto-crop leaves a frame alone**: the scanner bed must be the brightest thing in the scan. A slide with highlights as bright as the bed, sprocket-exposed film, or a neighbor frame filling more than a tenth of one side stays uncropped. Crop by hand, or use *Film edge* and trim in.

A change re-detects every following frame cropped by **Auto**. Hand-drawn crops are kept, except that **Ratio** reshapes them around their center.

<!-- panel:baseline -->
### 10.5 Roll Analysis

Meter the roll once and share the result, so frames of one film match. The **Use average** toggles are this card's roll defaults.

*   **Use average: Luma**: take the picked roll's tonal range; color stays per frame. Disables Luma Range Clip.
*   **Use average: Color**: take the picked roll's color balance; tonal range stays per frame. Disables Color Clip. Luma and Color on gives a consistent roll; both off gives per-image auto-exposure.
*   **Use average: Cast** (**Color Negative** only): take Cast Removal's gray balance from the roll or scene analysis instead of this frame's own grays. The film's color curve is the same on every frame; each frame keeps only as much of its own color level as the analysis found real differences between frames, so a scene under one light renders alike. Roll and Scene Analysis turn it on, except on frames far from the rest; grayed out until an analysis has run.
*   **Baseline** (line shown while either average is on): names the roll or scene analyzed, or the frame **Sync Bounds…** took it from, and warns when there is none.
*   **Rolls** (picker): search every roll in your library; a ticked roll has a saved baseline. Defaults to the loaded roll. Picking one loads its baseline at once; a different ticked roll shows a hint that its baseline belongs to that roll.
*   **Reanalyze** (gauge, beside the picker): runs Roll Analysis on the loaded roll (also on the Library's roll list, [§2](#2-film-strip-left-panel)): averages density and color balance over every loaded frame outside a scene and saves it as the roll's baseline. Locked frames are skipped. A frame whose color is far from the rest keeps its own exposure and color (**Use average** off); the status message names these frames. *(Tip: run Auto Crop **Roll** first, in **Image only** mode, for consistent crops.)* Grayed out on a roll that is not loaded.
*   **Use This Frame** (crosshairs, beside the picker): saves this frame's bounds as the roll's baseline, for a reference frame. Frames outside a scene with **Use average: Luma** / **Color** follow it; locked frames are skipped. A frame opened later with no baseline takes its scene's, else the roll's. Grayed out on a roll that is not loaded; on an unrendered frame it says there are no bounds yet.
*   **Scenes**: lists the loaded roll's [scenes](#scenes) with number, color, frame count and a tick once analyzed. **Analyze** runs Scene Analysis (Reanalyze over the scene's frames only, saved on the scene). **Select** selects its frames in the Film Strip. **Delete** (trash) forgets it after asking. Roll Analysis and a picked roll baseline skip scene frames.

<!-- panel:process -->
### 10.6 Metering: negative → positive

How this frame is measured into a positive's tonal bounds. The film mode is in §10.1, the capture corrections in **Calibration** (§10.3), where the bounds come from in **Roll Analysis** (§10.5). The whole card follows the scope pair ([§10](#10-roll-tab)).

**Slides** (Transparency) render **as captured**, with the camera's color matrix and a tonal window fixed to the decoder's white level, as in Photoshop, Preview, Affinity or Darktable. A bracket keeps each exposure's own brightness.

*   The paper controls hide (paper profile, Paper White/Black, split grade, Preflash), as does the normalization tuning. What stays is a transfer curve, neutral at defaults: **Print Density**, **ISO-R Grade**, **Toe** / **Shoulder** and their **Width** sliders, **Shadows Density** / **Highlights Density** (§5.2), the per-layer R/G/B trims and white balance; Lab, Toning and Finish work as usual. **Dye Separation**, its R/G/B trims and **Separation Damping** apply directly to density.
*   **Auto Density** and **Auto Grade** start off on a slide, to leave a bracket alone: entering Slide turns them off if they were at the negative default and leaving restores them, unless you changed them. Turn them on to meter a faded or expired slide. On a merged bracket they are grayed out, since **Render exposure** already picks the print exposure.
*   Lightroom mapping: **Exposure** → Print Density (lower is brighter), **Contrast** → ISO-R Grade (180 is softest), **Shadows** → Shadows Density, **Highlights** → Highlights Density. *Positive adds density*, so negative Shadows Density opens shadows. **Whites** and **Blacks** have no equivalent, because the window is fixed.
*   A source with no camera matrix (a scanner TIFF, a JPEG) passes straight through.
*   **Linear RAW** is grayed out (the as-shot multipliers are folded back in, so the render is identical) but stays visible. It is live with **Positive** on. An explicit Input ICC in Export replaces the camera's primaries rotation; the as-shot white balance still applies.
*   **Narrowband** and **Single-Shot Narrowband Calibration** are grayed out for *any* transparency ([Narrowband and slides](#narrowband-and-slides)): narrowband light samples three isolated wavelengths, and no profile recovers the rest of the spectrum.

**Analysis** sets where the black and white points are metered.

*   **Analysis Buffer** (0.0 to 0.25): insets the measurement window so rebate, sprocket holes and scanner borders do not skew it. Raise it for wide borders.
*   **Reanalyze Frame** (circular arrow): measures this frame again from its current crop, buffer and region. Grayed out with Lock Bounds on or with both averages on.
*   **Draw Region** / **Clear Region**: draw a freehand region to meter *exactly* that area, overriding the buffer. Double-click inside to confirm.
*   **Lock Bounds**: freezes this frame's bounds against crop and slider changes and against every Roll Analysis run.

**Tonal Range**:

*   **Luma Range Clip** (-100 to 100): how tightly the black/white-point span is set. Neutral applies a small robust clip. Positive tightens it, for dense or fogged negatives; negative pushes the bounds *outward*, for lifted blacks and unclipped highlights.
*   **Color Clip** (-100 to 100): the per-channel color-balance clip (orange-mask removal). Positive tightens; negative samples nearer the extremes.

**White / Black Point** (-0.25 to 0.25), with a **Global** / **R** / **G** / **B** selector: offsets on the detected bounds. Positive white point brightens; positive black point lifts blacks. In R/G/B they are per-layer Dmin and Dmax trims, a fact of the film stock, so they are roll defaults like the rest of the card. On a slide they offset its fixed window; elsewhere **Lock Bounds** disables them.

<!-- panel:demosaic -->
### 10.7 Raw Decode: turning the sensor mosaic into pixels

A color sensor records one color per photosite; the demosaic algorithm fills in the other two, which affects sharpness and grain. Bayer and X-Trans RAW only (a scanner TIFF, a Pakon scan or a linear DNG is already de-mosaiced), and only algorithms in your LibRaw build are listed.

*   **Preview** / **Export** (sticky, default **Auto**): *Auto* is a fast half-size decode on screen and AHD for export. For the preview, Auto and Linear are fastest; the others decode at full size. **AHD** is balanced, **VNG** smooth, **PPG** fast with clean edges, **DCB** and **DHT** favor fine detail, **AAHD** softens edges to suppress artifacts.

**Highlight Recovery** (Transparency only, default **Off**): recovers a clipped highlight on a camera RAW. **Off** leaves it flat, or magenta if one channel clipped first. **Blend** recovers a plausible neutral from the unclipped channels, right for sun, sky, chrome or glass. **Reconstruct** is libraw's more aggressive level and can misjudge a saturated-color highlight. Greyed out on a scanner TIFF, JPEG or other rendered file and on a merged bracket (the merge recovers highlights itself), hidden outside Transparency, inert under Narrowband.

<!-- panel:optics -->
### 10.8 Optics

The scanning optics: one lens correction and one light correction for every frame of the rig. One scope pair covers both.

#### Lens Correction

*   **Distortion Correction** (-0.100 to 0.100, in steps of 0.001): positive corrects barrel, negative pincushion. Use the film rebate as a straight edge. Applied before Tilt and Swing.
*   **Embedded Profile**, from lens data in the file (enabled when the file has it):
    *   **Distortion**: straightens curved lines, replacing manual distortion correction. Set it before cropping or retouching.
    *   **CA**: reduces color fringes. Works with or without **Distortion** and manual correction.

#### Flat Field Correction: even out the light

Corrects uneven illumination (vignetting, falloff) from a copy-stand or scanner light, using a shot of the bare light source.

*   **Profile** dropdown, **+** and **trash**: **+** reads a reference image once and bakes it into a named profile in NegPy's `flatfield` folder, so the reference file can then be moved or deleted. **Trash** asks first: the gain map is lost, and every frame using it loses its correction.
*   **Apply Flat Field** (bulb toggle beside the dropdown): apply the selected profile to this roll, enabled once a profile exists.

A newly chosen profile becomes the rig's default for the next roll.

---

## 11. Metadata tab

Metadata for the original analog capture (camera, lens, film, process). NegPy writes it into every export format (JPEG, TIFF, PNG, JPEG XL, WebP) as EXIF and XMP, so a DAM such as Lightroom shows your film gear, not the scanner. A TIFF holds the capture position in XMP only. EXIF text is 7-bit, so `4×5` is written `4x5`. **Protect original metadata** (§13) writes the source's own EXIF/XMP instead.

**Analog Gear**, **Capture**, **Process**, **Scanning** and **Exposure** are roll-wide by default: each has the scope pair from [§1](#frame-or-roll-the-scope-pair) with **Roll** lit. Edit a field and that card flips to **Frame**; its **Roll** button pushes this frame's value to the roll, and **Reset to Roll** takes the roll's value back. The line under the panel title lists the cards this frame has taken off the roll. The frame number is always per frame.

<!-- panel:metadata_presets -->
### Metadata Presets

Saved sets of metadata values in `~/NegPy/presets/metadata/`, separate from the Favorites tab's edit presets. You manage them on the **Gear** tab (§12).

*   **Preset** + **Load**: write the preset's fields onto this frame. Other fields stay. Hover to see what a preset holds.

Gear loads as one unit: camera, lens, film stock, film format and the values read from them, so a 120 stock cannot leave the frame at 35mm. Picking a film stock sets the format, so set a frame format such as `6×7` after the stock. Presets never store the frame number.

<!-- panel:metadata_gear -->
### Analog Gear

Searches your own gear (§12). **Other…** opens the full built-in catalog.

*   **Camera / Lens / Film stock**: empty means not set.
*   **Clear**: empties all three.
*   **Infer from folder name**: fills unset camera, film stock, ISO and capture date from the folder name. Camera and film stock match the gear catalog, as Roll Settings does on import. A field with two plausible readings stays empty. It uses the active folder roll's folder, or the frame's directory when no roll is active.

<!-- panel:metadata_capture -->
### Capture

*   **Date**: `1998`, `1998-07`, `1998-07-14` or `1998-07-14 16:30`, with an optional offset such as `+02:00`. An impossible date turns red and is not saved. EXIF `DateTimeOriginal` pads the missing parts; XMP `photoshop:DateCreated` keeps the short form, with `negpy:CaptureDatePrecision`. The scan file's timestamp moves to `DateTimeDigitized`.
*   **Place**: the map-pin button opens a map (search, click, or paste coordinates; zoom with **+** / **−**, scroll or pinch, drag to pan). The ✕ empties the place. The field also takes a coordinate pair or an OpenStreetMap/Google Maps link. Coordinates go to EXIF GPS and XMP `exif:GPS*` (XMP only in a TIFF), names to XMP `photoshop:City`/`State`/`Country`. A place you set replaces the source's whole GPS block. With no place set, a geotagged source keeps its coordinates and the map opens on them. The map contacts OpenStreetMap; typed coordinates need no network.

<!-- panel:metadata_process -->
### Process

*   **Saved process**: a library recipe that fills Developer, Dilution, Push / Pull, Time and Temperature. Typing over one unlinks it.
*   **Format**: `—` (not set), `35mm`, `120`, `4×5`, `8×10`, `110`, or `Other` (free text).
*   **Developer** and **Dilution**: for example `D-76` and `1+1`, `1+50` or `stock`, joined in EXIF `ImageDescription` as `D-76 1+1`. Dilution also goes to XMP `negpy:DevelopmentDilution`.
*   **Push / Pull**: `Push +3` … `Normal` … `Pull -3`.
*   **Time** and **Temperature (°C)**: time as `9:30` or minutes; an unreadable time turns red and is not saved. XMP `negpy:DevelopmentTime` and `negpy:DevelopmentTemperature`; searchable as `devtime:` (minutes) and `temp:`.
*   **Clear**: empties the saved process and its fields. Format stays (the film stock sets it).

<!-- panel:metadata_scanning -->
### Scanning

*   **Saved setup**: a library digitizing setup that fills Scanning. Typing over it unlinks it.
*   **Scanning**: scan method or notes. EXIF `Software` is always `NegPy`.
*   **Clear**: empties the saved setup and the note. Roll and Frame stay.
*   **Roll / Frame**: Scanlight capture roll name and frame number, stamped on capture, editable. Filename template fields `{{ roll }}` and `{{ frame }}`; XMP `negpy:CaptureRoll` and `negpy:CaptureFrame`. Not the Roll Analysis card's roll name.

<!-- panel:metadata_exposure -->
### Exposure

Optional original shutter, aperture and ISO. Click the lock to edit a free-text string such as `1/125s f/2.8 ISO 400`.

<!-- panel:metadata_preview -->
### Metadata Preview

A live view of what will be embedded, grouped by capture, scan, process and file. The Scan group shows the source's own timestamp and coordinates. **Description…** picks the fields joined into EXIF `ImageDescription` (default: camera, lens, film stock, ISO; format, developer, push/pull and scanning off). A confirm sets this frame and becomes the default for frames without their own selection. Sync metadata and Sync settings copy the selection too.

Capture gear goes to standard EXIF and the digitizing rig to `negpy:Scan*` XMP tags. With gear unset, your scanner or DSLR stays in EXIF.

---

## 12. Gear tab

A searchable gear library used by Metadata (§11), Roll Settings and every gear picker. **My Gear** holds physical gear, **Presets** holds metadata field sets. Only My Gear has the Catalog toggle.

<!-- panel:gear_items -->
### My Gear

**Category**: **Cameras**, **Lenses**, **Film Stocks**, **Process** (a development recipe) or **Scanning** (a digitizing setup). The list shows gear you added, saved to `~/NegPy/gear/`. An empty category reads "You haven't added any…yet".

*   **+**: copy an item from the built-in catalog into your list, or **Add Custom** to enter one by hand.
*   **Catalog**: show the built-in reference models too. Off by default.
*   **copy / trash**: duplicate or delete the selected item. Trash is disabled on built-in entries.

<!-- panel:gear_presets -->
### Presets

*   **+**: store the current frame's metadata under a name.
*   **pen**: rename, or change which fields it stores.
*   **copy / trash**: duplicate or delete.
*   Edit a preset's fields in place, with no frame open: camera, lens, film stock, saved process, saved setup, developer, dilution, push, time, temperature, scanning note, roll, exposure. Pickers work as on the Metadata tab, with **Other…** for the full catalog. Capture date, place, description fields and flags show but are per-frame, so they are not editable here. **Notes** is free text.

---

## 13. Export tab

### Output intent

*   **Print** (default): the look you see on screen.
*   **Flat**: a neutral, low-contrast master for editing elsewhere. It skips the print look, effects, toning and vignette, and writes a 16-bit TIFF, or lossless JPEG XL when JXL is selected with sRGB, P3, Rec 2020 or Grayscale.
    *   **Preview Flat**: show the flat master on the canvas.
    *   **Roll Analysis** (Roll tab): share one exposure baseline across all visible frames so flat masters match. Run it before a flat batch.
*   **Linear**: skip the pipeline and write the decoded buffer as linear 16-bit, with only rotation and flip applied (no normalization, exposure, color management, flatfield or sensor correction). **TIFF** (default, zlib, untagged) or **JPEG XL** (lossless). JPEG XL always tags sRGB primaries and a linear transfer, which is wrong for native primaries; use TIFF when that matters. **Effort** (1 to 9, default 7) sets JPEG XL speed against size.
    *   **Pakon RAW**: 4× expansion by default; F335 files (16-bit sensor) none.
    *   **LinearRaw DNG**: SilverFast HDRi (3-channel) and VueScan (4-channel RGB+IR). IR goes to a separate grayscale `_ir` file in the same Format.
    *   **Camera RAW**: demosaiced at unity white balance (1,1,1,1) with the Export algorithm from the Raw Decode card (§10.7). As-shot WB goes to XMP (`RAW-WB: R G B`). Trichrome triplets merge into one TIFF. Stitch composites (and stitch plus triplet) get flatfield and sensor correction per part for clean seams.
    *   **Coolscan NEF**: not raw sensor data; the content depends on the Nikon Scan settings at scan time. Reads the full-res RGB SubIFD and drops extra channels. No expansion.
    *   **Flextight FFF**: uncompressed 16-bit RGB and SGI LogLuv raw (`.3fr`/`.fff`). LogLuv is HDR and decodes through LogLuv → XYZ → linear sRGB with per-channel percentile normalization. The largest image IFD is used. FlexColor metadata (tags 50457 and 46279) goes into the TIFF headers. No expansion.
    *   **Noritsu RAW**: headerless BGR 16-bit dumps; size detected from file size. 16× expansion by default.
    *   **TIFF**: a 4th channel tagged as IR (ExtraSamples = UNSPECIFIED or missing), a sidecar `_ir.tif` or IR in a secondary page goes to a separate `_ir` file. **Input gamma** (linear, 1.8, 2.2 or sRGB) linearizes the data. Expansion available, off by default.
    *   **Expansion**: scales the data before writing. Defaults: Pakon F135/F235 4×, Noritsu 16×, F335 and LinearRaw DNG off. Camera RAW, Coolscan NEF and Flextight FFF have none.
    *   **Apply ICE dust removal** (when IR exists): IR dust and scratch correction. Off by default.
    *   **Corrections** (camera RAW only, all off): **Apply white balance** (as-shot gains; grayed out for a Trichrome triplet or Single-Shot Narrowband capture), **Apply flatfield**, **Apply sensor correction** (crosstalk unmixing). Stitch composites always get flatfield and sensor correction per part.

    Linear Output uses **Destination** like any export and always appends `_linear`, so it cannot overwrite its source. Without **Overwrite**, an existing file gives `_linear_2`, `_linear_3` and so on. It runs as a background batch: **Abort** stops after the current frame, and the finish message counts failures.

    The file has no ICC profile and no color metadata from the source (JPEG XL excepted, as above). It carries Make, Model and DateTime from the source and a description of source format, expansion, demosaic algorithm, white balance and corrections, ICE included.

### Export button

**Export**; its chevron picks the scope: current frame (Ctrl+E), selected frames or all visible frames. For several formats or sizes in one run, use Export Presets.

*   **Protect original metadata**: copy the source's EXIF/XMP unchanged. The Metadata tab (§11) is ignored and the source's resolution is copied exactly, even when resized. A source without a resolution stays without one, except in TIFF, which states the export's own resolution.
*   **Sync custom metadata to all files in batch export**: write this frame's capture, gear and process values to every file in a batch or preset export. Disabled with Protect original metadata.

### Format / Size / Color Management / Destination

*   **Format**: `JPEG`, `TIFF`, `PNG`, `JPEG XL` or `WebP`, each with quality or effort options. **JPEG XL supports only `sRGB`, `P3 D65`, `Rec 2020` or `Grayscale`**; `Adobe RGB`, `ProPhoto RGB` and a custom Output ICC give an error, because NegPy's encoder cannot embed an ICC profile.
*   **Bit Depth**: `8-bit` or `16-bit` (TIFF, PNG, JPEG XL). Hidden for JPEG, WebP and a flat master (always 16-bit).
*   **Compression** (TIFF): `Uncompressed`, `LZW` or `ZIP`, all lossless; ZIP is usually smallest.
*   **Compression** (PNG): `0` to `9`, lossless; higher is slower and smaller.
*   **Progressive** (JPEG): renders in passes while it downloads.
*   **Input ICC**: treat an untagged source as this profile. Primaries only: a matrix profile's TRC is ignored; a LUT profile's input curves still apply.
*   **Export profile**: the space the file is converted to and tagged with: `Same as Source` (names what it resolves to; Adobe RGB for an untagged scan), `sRGB`, `Adobe RGB`, `ProPhoto RGB`, `P3 D65`, `Rec 2020`, `Grayscale` (true B&W), or an imported printer or paper ICC, which is embedded in the file.
*   **Import ICC** (folder button on the COLOR MANAGEMENT header): copies a `.icc`/`.icm` into `~/NegPy/icc/`, available at once. A file named after a built-in space (`sRGB.icc`) replaces that space everywhere, after a confirmation.
*   **Proof on screen** is in **Soft Proof** below. A warning shows here when nothing is proofed or the proof targets a different profile than the export.
*   **Paper Aspect Ratio**: final print ratio, or *Original* (no resize).
*   **Resolution**: *Original* (full resolution), *Print* (long-edge **Size** in cm plus **DPI**) or *Pixels* (long-edge **px**). Every file is tagged with a DPI: *Print* your value, *Pixels* the one its long edge implies, *Original* the source's own (EXIF or e.g. JFIF density), else the **DPI** field. Linear output follows the same rule.
*   **Destination**: **Filename Pattern** (a Jinja2 template with export and Metadata fields; see [TEMPLATING.md](TEMPLATING.md)), **Overwrite**, and the location: subfolder of source (default, `export`), same as source, or an **Export Path**. A roll with no single source folder exports under its own folder in NegPy's data folder with Subfolder of Source, and the status bar says so. With **Linear**, only Destination shows.

### Collapsible sections

<!-- panel:export_presets -->
#### Presets

Saved Format/Size/Color Management/**Destination**/filename recipes. **Manage** edits them. **Export Presets** renders the frames with every enabled preset, each to its own destination.

<!-- panel:printing_notes -->
#### Printing Notes

The printer's record for this frame: the numbered dodge/burn masks and a card with paper, density, grade, filtration and the burn list. **Preview** shows it on the canvas (**Shift+N**); **Export** writes it as an image beside the print. Its conventions are under Dodge & Burn.

<!-- panel:export_sidecars -->
#### Sidecars

**Save on export** writes a `.negpy` sidecar next to each source on export. **Export Sidecars** writes them for all visible frames now and reports failures in read-only folders. Edits always stay in the database too.

<!-- panel:contact_sheet -->
#### Contact Sheet

All visible frames on one sheet. Pick a **Template** or set **Cell / Gap / Margin / Max tiles**, choose a **Path**, press **Export Contact Sheet**. Written as JPEG with the **JPEG Quality** and **Progressive** settings above.

#### Soft Proof

Simulate the print on screen. See below.

<!-- panel:soft_proof -->
### Soft Proof

Preview only; exports are never proofed. Paper is dimmer and has a smaller gamut than a screen, so a correct proof looks worse than the plain preview. Judge it in room light.

*   **Preset**: a saved printer and paper set-up (profile, intent, toggles). **None** proofs the export target with no paper simulation. Save names the set-up, the bin removes it. The box goes blank once you change a setting. Separate from the export **Presets**.
*   **Proof on screen** (`Shift+P`, on by default): master switch. Off grays out the section.
*   **Profile**: follows the **Export profile** until you pick a printer or paper, so you can proof a print while you export a web JPEG. Lists imported ICC profiles only.
*   **Intent**: **Relative Colorimetric** keeps printable colors and clips the rest, so saturated areas can flatten. **Perceptual** compresses everything so color relations survive; printer profiles carry their own table, so try it on saturated frames. **Saturation** is for charts, not photographs.
*   **Black point compensation** (off): map the darkest tone to the paper's black instead of clipping.
*   **Simulate paper white** (off): show the paper's white and tint.
*   **Simulate ink black** (off): show the paper's real black; shadows lift.
*   **Gamut warning**: show unprintable colors as gray (the Analysis panel's **Gamut** row counts them). The edge fades, so read it as a region.
*   **Display**: the monitor profile, auto-detected; set it by hand if detection fails.

---

## 14. Scan tab

Capture film directly into NegPy. Two collapsible sections.

<!-- panel:scan_sane -->
### Film Scanner

**Backend**: **SANE** (Linux/macOS), **Nikon Coolscan (nkscan)** (direct Coolscan driver, Linux, Windows, macOS) or **pyOpticfilm (Plustek)** (OpticFilm 8200i SE and 8100 V2, all three OSes). Controls group as **Film**, **Quality**, **Framing** and **Output**; a group with nothing for the device is hidden.

*   **Format**: `TIFF` or `TIFF (mono)` (one 16-bit gray plane, for B&W negatives).
*   **Frames**: `1-6`, `1,2,5`, or empty for all. The strip preview writes its picks here. The line above **Scan** states frame count, resolution, extra passes and approximate disk use.
*   **Depth**, **Autofocus**, hardware **Auto-exposure**: shown only when the device offers them (not on the OpticFilm 8200i SE).
*   **Prescan**: a low-DPI full-window preview; drag a crop and the next Scan uses that hardware ROI.
*   **Exposure**: shown when the scanner has `scan-exposure-time` (some genesys devices); overrides the exposure time, in µs, ms or s.

**pyOpticfilm (Plustek)**: the **OpticFilm 8200i SE** (`07b3:1825`) and **8100 V2** (`07b3:1824`) scan; other listed models cannot yet (try **SANE** on Linux and macOS). **Prescan** takes a 1200 dpi preview; set a crop and leave with **Apply Crop** or **Scan Frame**. **Scan mode** (Single-Pass by default) chooses among **Single-Pass** (one exposure), **Multi-Pass** (repeats the exposure and stacks the results to reduce noise, with a **Passes** slider from 2 to 9), **Adaptive Multi-Exposure** (fuses a short and long exposure for extended dynamic range) and **Adaptive Multi-Pass** (both together); every mode past Single-Pass takes longer, and Multi-Pass cannot combine with IR. From pyopticfilm 1.1.2, orientation matches SilverFast; rescan older files if left-right matters. **IR** comes in the same pass, aligned to color. Shading is measured before the film feed, so the strip can stay loaded. The holder margins in the default Full window are clamped so they do not skew auto exposure; if a frame still looks off, raise **Analysis Buffer** or crop. On Windows, bind the device to **WinUSB** with Zadig first. Install with `uv sync --group plustek` or `pip install negpy[plustek]` (bundled in Windows builds). See [PLUSTEK_WINDOWS.md](PLUSTEK_WINDOWS.md).

**Nikon Coolscan (nkscan)**: needs no SANE. **Preview strip…** reads the strip in one pass, finds every frame and cuts the tiles from it; it starts when the dialog opens and holds until eject. **Detect frames** runs it again after the film moves. Adjust framing with **Offset** (±10 mm, both ways) and **Drift**; tiles re-cut with no new scan. **Scan** with nothing picked scans every frame. For a subset, type **Frames** or untick tiles. Eject clears the selection; **Offset** and **Drift** stay. Backend-only controls:

*   **ICE**: infrared dust and scratch removal baked into the file (Retouch's IR Restore stays editable). Color film only. **ICE** and **IR** exclude each other; **IR** keeps the plane for Retouch.
*   **Samples**: reads per line averaged (1 to 16). Less shadow noise, proportionally slower.
*   **Superfine**: one line per pass. Slower, with no host-side line registration.
*   **Film**: Color negative, B&W negative, Slide or Kodachrome. Sets how frame boundaries are read, whether IR and ICE are offered (not for B&W or Kodachrome), and metering: a color negative is metered per channel to take the orange mask off before conversion; other films keep the factory balance.
*   **Film format**: frame length (135, 66, 645 and so on). **Auto** where the holder narrows it; set it for loose film in a masked carrier. Shown only where the transport measures the film.
*   **Exposure** (**Meter Frame…** / **Unlock**): nkscan meters every frame on its own, so a strip end, which meters on the bare light past the cut, keeps a color negative's orange mask and scans with a different color. **Meter Frame…** meters one frame of the loaded strip (pick one inside the strip, such as frame 2) and every later scan on this scanner reuses its exposure, across strips and restarts, until **Unlock**. Meter again for each new roll.

Controls follow what the unit reports; an LS-50 hides Samples and Superfine. The release builds include **nkscan**; from source, see [CONTRIBUTING.md](../CONTRIBUTING.md). On Linux, USB needs a udev rule for vendor `04b0`; FireWire/SCSI needs the `sg` module.

**SANE scan window**: on a feeder, **Preview strip…** previews every frame, sets per-frame windows and picks frames. With a manual holder, **Preview…** previews one position for one crop window. The window sets the hardware scan area, so only that region is read.

During a preview, a progress bar shows, **Cancel** reads **Stop Preview** (keeps tiles read so far), and **Apply** and **Scan** stay disabled. Negative stock previews inverted, Slide and Kodachrome not.

<!-- panel:scan_rgb -->
### Camera Scanning

Copy-stand capture with a camera in **PC Remote** mode over USB (macOS/Linux). With a NegPy **Scanlight**, it captures narrowband R/G/B triplets from film-stock presets; without, one white-light exposure. Frames go to the hot folder and into Trichrome Mode.

*   **Live View & Scan**: click the image to aim the focus magnifier, click again for the full frame. ISO, shutter and aperture are set from the toolbar, or locked by a calibrated RGB preset.
*   **Preset**: shows its RGB levels, ISO, shutter and aperture and forces them each frame. **+** calibrates: place the rectangle on clear film base, name it, run it. It solves a shutter and per-channel LED levels just under clipping, or says which way to adjust and saves nothing. **Create a manual preset…** sets one by hand.
*   **Scan** and **Retake**: **Scan** shoots into a per-roll subfolder, auto-numbered, and imports; **Retake** shoots again without advancing. **Delay between exposures** pauses between R, G and B for bodies that lock up.
*   **Narrowband**: RGB-lit scans render more saturated; the Calibration card's **Narrowband** toggle corrects this.

Needs `python-gphoto2` (`pip install gphoto2`; no Windows build). See CAMERA_SCANNING.md for setup, the macOS camera-daemon note and troubleshooting.

<!-- panel:scan_strip -->
### Strip preview

Dialogs end with **Cancel**, **Apply** (keep the framing) and **Scan**. Apply reads **Apply Framing** on a strip, **Apply Window** on a single holder, **Apply Crop** after a Prescan.

*   **Cropping**: drag on a frame; corners resize, inside moves. **Clear Crops** removes all.
*   **Frame outline**: red box on the detected frame; offsets are measured from it.
*   **Offset**: moves every frame along the film to clear the gap. The shaded band on the right is film the transport cannot deliver, so too much offset cuts the frame's tail. A feeder goes one way only.
*   **Drift**: offset that grows (or shrinks) per frame position.
*   **Per-frame offset**: the slider under a tile, on top of Offset and Drift. Double-click resets.
*   **Size**: tile size, remembered. Double-click resets.
*   **Which frames**: each tile has a tick; **All** and **None** set all, with a count. Eject clears ticks, crops and per-frame offsets.

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

### Performance

*   **GPU acceleration**: render on the GPU (backend named below). Off uses the slower CPU pipeline, same image. If the GPU viewport fails at launch, a toast and an amber line here say so.
*   **Multi-core CPU rendering**: runs the CPU rendering kernels on all cores, effective immediately, no restart. It helps most where the CPU does the work (exports, machines without a usable GPU); merges gain little, since RAW decoding dominates. **On** for Windows and Linux, **off** for macOS, whose threading layer ends the process if two threads enter it at once. If the app closes without warning after you enable it, NegPy offers to turn it off at the next launch. `cpu_parallel` under `[performance]` in `override.toml` overrides Preferences.
*   **Search by meaning**: find frames by description instead of `field:value` terms (§2). Downloads its model on first use.
*   **Preview size** (512 to 8192 px): canvas long edge. Higher is sharper and costs proportionally more VRAM and CPU; lower the cache limits to match. Camera RAW decodes at half sensor size.
*   **Preview cache** and **Preview cache limit**: photos kept decoded, and their memory limit. Lower both on low RAM.
*   **HQ buffers**: full-resolution preview buffers kept (a 60 MP scan is about 700 MB each).
*   **Rendered frames**: frames kept for going back without a re-render.
*   **GPU texture cap**: largest texture dimension. 0 lets the hardware decide (integrated GPUs get a conservative default). Lower it if exports run out of GPU memory.
*   **Show GPU memory warning**: the status message when an HQ preview is downsampled for exceeding the texture cap above. Off only hides the message — the downsampling itself still happens.

Rows from **Preview size** down need a restart. A value in `override.toml` wins and grays out its row.

### Session & Storage

*   **Carry settings between frames**: apply your Persistent Settings to each newly opened file. Off, each file starts from its saved edit or defaults.
*   **Persistent Settings…**: which edits carry over (§2).
*   **Manage Database…**: row counts and sizes, clear saved edits, library roots.

### Startup override (`override.toml`)

For crashes on launch or rendering glitches. NegPy creates `Documents/NegPy/override.toml` on first run; edit it and restart. It also holds the `[performance]` values and wins over Preferences, so it works when the app does not start.

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

At startup NegPy checks GitHub once. A new release shows a green **⬇ Update Available: vX.Y.Z** line at the top of the left panel and a green dot on the **⋯** menu; click the line, or **Update to vX.Y.Z…** in the menu, for the release notes, download size and install button. To check by hand, use **Check for Updates…** in the **⋯** menu, or run the **Check for updates** action (no default key).

**Install Update** downloads the build for this install type, closes NegPy, installs and reopens. Nothing is replaced until NegPy exits, so a failure leaves your install as it was.

| Install | What NegPy fetches | How it installs |
|---------|--------------------|-----------------|
| **Windows** | the `-Setup.exe` installer | Runs it silently over your existing install. Windows asks for administrator approval first, because the app lives in Program Files. Approve it *before* NegPy closes. |
| **macOS** | the `.dmg` for your chip (Apple silicon or Intel) | Mounts the image and replaces the `NegPy.app` bundle where it currently sits, then reopens it. |
| **Linux** | the `.AppImage` | Replaces the AppImage file you launched, keeps it executable, and relaunches it. |

Edits, presets, settings and library live in `Documents/NegPy` and the database, so updates do not touch them.

The button reads **"Open Releases Page"** when NegPy cannot update itself: a source checkout, no build for your platform, an app moved out of its installer layout, or an app folder you cannot write to.

---

## Additional Info

*   **GPU acceleration**: previews render on the GPU; Normalization analysis (bounds, white/black point, normalize) runs on the CPU. Turn it off in **Preferences → Performance** or force a backend in `override.toml` if you suspect a driver issue.
*   **Database**: edits live in a local SQLite database keyed by file hash, so files can move or be renamed. Optional `.negpy` sidecars mirror them.
*   **Saving edits**: written on export, on frame switch, or on save. Closing mid-edit before any of these loses unsaved changes.
*   **Keyboard shortcuts**: [KEYBOARD.md](KEYBOARD.md)
*   **Filename templating**: [TEMPLATING.md](TEMPLATING.md)
*   **The pipeline in depth**: [PIPELINE.md](PIPELINE.md)

