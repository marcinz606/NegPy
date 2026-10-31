# NegPy User Guide

NegPy turns scans of film negatives and slides into finished pictures. It works like a darkroom: you pick a paper, then set brightness, contrast and color. Your original files never change; NegPy keeps your edits in its own database.

For the math behind each step, read [PIPELINE.md](PIPELINE.md).

---

## 1. The Big Picture

### Words in this guide

*   **Roll**: a group of frames, usually one film. What a roll shares is on the Roll tab.
*   **Folder roll**: a roll of the image files in one folder. New files in the folder join it.
*   **Virtual roll**: a roll of a fixed list of frames, from any folders. New files never join it.
*   **Density**: how dark a part of the negative or the print is. A dense negative is dark; on a print, more density is darker.
*   **Print Density**: how light or dark the print is, like the exposure time under an enlarger.
*   **Grade**: the contrast of the print, like the contrast grade of darkroom paper. NegPy uses the ISO-R scale, where a lower number gives more contrast.
*   **Stop**: a doubling of light. +1 stop is twice the light, −1 stop half.
*   **Print stock**: the paper the picture prints on. Each paper has its own contrast and colors.
*   **Filtration**: color correction, like the color filters of an enlarger.
*   **Dodge and burn**: making one area of the print lighter (dodge) or darker (burn).
*   **Shadows, midtones, highlights**: the dark, middle and bright parts of the picture.
*   **Toe and shoulder**: the dark end and the bright end of the paper's curve, where tones get closer together.
*   **Rebate**: the border of the film around the picture.

### Screen layout

*   **Canvas** (center): the picture. Scroll or pinch to zoom, drag to pan, and double-click to switch between fit and 100%. Right-click for view options, copy and paste of settings, resets and **Unload**.
*   **Film Strip** (under the canvas): the open frames in one row. Drag its top edge to resize it; `Ctrl+[` folds it away.
*   **Toolbar** (bottom): **Photo** holds navigation, zoom, rotate, **Tools** (crop, pickers, masks, light leaks, retouching), undo, the **Print** menu (Before / After and the peeks), **Roll Average**, **Export** and **Overlays**. **Export** does what the Export panel's button does, and its chevron picks the frames. **Roll** holds the Film Strip's import, filter, sort and roll actions. The **⋯** menu holds every action, **Preferences…** included.
*   **Controls** (right): the tabs **Library**, **Roll**, **Frame**, **Metadata**, **Export** and **Scan**. Only Roll and Frame change the picture. Drag the panel by its top edge to float it.
*   **Frame Info** (`I`): the strip above the canvas. It shows the file name, film and size on the left, the print recipe on the right, and messages in the middle. A running job fills the bar along its bottom edge, one segment per frame for a short batch; **Abort** stops it.

**Shift+H** hides the panels, and **Reset Panel Layout** (Preferences → Interface, or `Ctrl+Shift+L`) restores them. With nothing open, the canvas offers **Load Scans…** and **Take the Tour**.

### The workflow

1.  On the **Library** tab, import a folder as a roll and open it.
2.  On the **Roll** tab, check **Setup** (film type and scanner).
3.  On **Frame › Geometry**, crop and straighten. Then run **Roll Analysis** on the Roll tab, so the frames match.
4.  On the other **Frame** tabs, set brightness, contrast and color (**Exposure**), the look (**Look**), and remove dust (**Finish**).
5.  Export on the **Export** tab.

| Tab | Cards |
|---|---|
| **Roll** | Setup · Raw Decode · Optics · Calibration · Metering · Roll Analysis · Frame Assembly |
| **Frame › Geometry** | Crop · Straighten |
| **Frame › Exposure** | Filtration · Tone · Paper Response · Dodge & Burn |
| **Frame › Look** | Lab · Alternative Processes · Toning |
| **Frame › Finish** | Retouch · Finishing |
| **Frame › Favorites** (star) | Your chosen sliders · Presets |
| **Frame › History** (clock) | Work prints · Edit history |
| **Metadata** | Metadata Preview · Shot · Process · Presets · My Gear |
| **Export** | File · Proof · Presets · Alongside Exports |
| **Scan** | Scanner · the scanner's cards · Output |

### Using the controls

On a slider, click the value to type a number, drag the name to adjust (**Shift** for fine steps), and double-click to reset. A chevron hides fine-tune controls; a dot on a closed fold, a card or a tab marks a changed value. On a dropdown, **‹** and **›** step through the choices. Each card has a reset arrow and an **ⓘ** that opens its part of this guide.

A tab with several cards has a bar such as **3 of 5 cards edited**, with reset, **Reset to Roll**, **Apply** (to the selection or the roll) and collapse for the whole tab.

### Frame or roll

Each card header has **Frame** (amber) and **Roll** (red). The lit one shows where the card's values live.

*   On a **Roll** tab or **Metadata** card, the card follows the roll until you edit it on this frame. **Roll** pushes this frame's values to the roll; **Frame** keeps them on this frame only.
*   On a **Frame** card, **Roll** copies your changes to the selection or the whole roll.
*   **Reset to Roll** puts the roll's values back on the card.

Right-click a card header for **Copy Card Settings** and **Undo Apply to Roll**. Frames that are not one roll show only **Frame** until you save them as a roll.

### What carries to the next frame

A new frame gets the roll's film type, and the crop, print stock and Lab settings of the frame before it. The look (brightness, color, dodge and burn, toning) starts clean. **Preferences → Session & Storage → Persistent Settings…** sets what carries.

**Reset Settings** returns a frame to the defaults. It does not change the scanning choices on the **Setup** card.

### Viewing

*   **Before / After** (`\`, or the **Print** menu): splits the canvas between the default conversion and your edit. Hold `\` to see the default alone. To compare with an earlier version, right-click it in **History**.
*   **Peeks**: hold `N` for the scan as loaded (the negative), `M` for the flat master, and `P` for the camera's own JPEG. The **Print** menu has them too.
*   **Roll Average** (`U`, or the **Print** menu): prints this frame on the average and changes nothing. The bar has **Luma**, **Color** and **Cast**; right-click one to turn them all on or off. On a frame in a scene, **Scene** and **Roll** choose which average. **Apply** saves them to the frame.
*   **Roll Average** button (filmstrip icon on the toolbar): opens the **Roll Average** bar. It is lit while the frame uses an average. Right-click turns every average on or off at once, as one edit. An amber icon marks a frame far from the roll color.
*   **Export Size** (**Print** menu): shows the frame as it exports, at the size the **Export** panel sets, at 100%. Use it to judge sharpening and grain. It updates a moment after each edit.
*   **Grain Focuser** (`Shift+L`, **Overlays** menu): a magnifier at the pointer that shows the exported pixels at their true size, with a sharpness number.
*   **Reference View** (`Shift+R`): pins this frame beside the canvas, to match other frames to it.
*   **Probe at Cursor** (`Shift+I`, **Overlays** menu): shows the density and zone under the pointer.
*   **Find** (`Ctrl+K`): finds any slider, card or action by name, or by another app's word (*contrast* finds **ISO-R Grade**).

`Esc` closes any peek, split or test strip.

### Pick Better

**Pick Better** (the balance scale on the Exposure tab's bar) helps when you are not sure which slider to move. It shows your edit in the middle with two versions beside it. Click the better one, or the middle to skip. Each round tunes one control, and later rounds repeat in finer steps. **Difference** sets how far apart the two versions are, and **Custom Routing** sets the order of the controls. **OK** or `Esc` ends it and keeps your picks.

### The tour

**Take the Tour** (⋯ → **Help**) walks through NegPy in short chapters. With nothing open, **Load Demo Negative** opens a sample. `Esc` ends the tour; the next one offers **Resume Where You Left Off**.

### macOS and Windows

`Ctrl` in this guide is `⌘` on macOS, where the **NegPy**, **Window** and **Help** menus hold Preferences, window commands, the tour and update checks.

If Windows blocks the default data folder, NegPy suggests `%LOCALAPPDATA%\NegPy\data`. Click the path to pick another, then **Use This Folder**. Nothing is copied. `NEGPY_USER_DIR` overrides the choice.

---

## 2. Library and film strip

<!-- panel:library -->
### Library

The **Library** tab (`Ctrl+L`) lists your **rolls**. A roll is a named group of frames, usually one film: a **folder roll** or a **virtual roll**. Importing records the folder; NegPy never moves, renames or deletes your files.

*   **+**: **Import Folder as a Roll…** makes one roll of a folder and opens it. **Import Subfolders as Rolls…** makes a roll of each subfolder that holds images.
*   **Double-click** a roll (or press Enter) to open it in the Film Strip.
*   **⋮** or right-click a roll: **Open**, **Unload All…** (the open roll), **Roll Analysis**, **Rename…**, **Save to Folder Roll**, **Restore Removed Frames…** and **Delete…**. Restore Removed Frames brings back the frames that Remove from Roll took out of a folder roll; the line under the roll's name counts them. Delete keeps the files and their shared edits. It loses the roll's settings, baseline, film base, scenes, card locks and the edits made only in that roll, and Refresh does not add a deleted folder roll again.
*   **Filter rolls…** and **Sort** help you find a roll. **⋮ → Refresh** finds new subfolders and recounts the frames. **⋮ → Discovery Filters…** lists folder names to skip (`export` at first).

A roll whose folder is gone shows **folder missing**. A folder name that matches your gear ("penf" for "Pen F") opens Roll Settings filled in.

A folder roll holds only the files in its folder. A file you add from another folder stays for this session only.

To save the frames in the Film Strip as a roll, use the Film Strip. **Save to Folder Roll** makes frames that all come from one folder that folder's roll; its other files show as removed. **Save as Virtual Roll…** keeps exactly these frames, from any folders. Save to Folder Roll also turns a virtual roll of one folder into that folder's roll. The new roll starts with the open roll's settings, card locks and the edits made only in that roll. A frame has one edit in every roll that holds it; right-click → **Edit Independently in This Roll** gives it its own edit in one roll.

<!-- panel:frames -->
### Film Strip

The **Film Strip** under the canvas shows the open frames in one row. Drag its top edge to make the frames larger.

The toolbar's **Roll** row holds:

*   The roll's name and its count of frames, keepers and rejects.
*   The **filter box**: type part of a file name, or terms such as `film:portra iso:>=400` (below). The search button, or **Enter**, searches the whole library.
*   **Sort**: by name, date or scene.
*   **Sheet filter**: all frames, keepers only, hide rejected, or unmarked only. The button is highlighted while a filter hides frames.
*   **Roll Settings** (the tag): gear and film details for this frame, the selection or the roll.
*   **Apply**: copies chosen settings of this frame to the selection or the roll.
*   **⋮**: **Add Files…**, **Add Folder…**, **Save to Folder Roll**, **Save as Virtual Roll…**, **Update Thumbnails**, **Hot Folder** (loads new files that a scanner saves in the newest frame's folder), **Unload…**, **Unload All…** (empties the Film Strip; the roll stays in the Library), **Reset Roll to Defaults…**, **Restore Removed Frames…** and **Film Strip Guide…**.

The grid button opens the **Light Table** (`Shift+G`): the whole roll in rows, for sorting. A click selects a frame, and a double-click or `Enter` opens it. The toolbar acts on the selected frames: **Keep**, **Reject**, rotate and flip, **Auto Crop** and **Export**, and ⋯ holds the rest of the right-click menu. Its **Frames per Row** slider, `+` and `-`, or `Ctrl` with the wheel change the frame size. `Esc` goes back.

#### Filtering

A plain word matches the file name. `field:value` terms match frame data, and a leading `-` excludes:

| Term | Finds |
|---|---|
| `film:portra` | film stock containing "portra" |
| `camera:"Nikon F3"` | quotes for a value with spaces |
| `iso:>=400` | numbers also take `>`, `>=`, `<`, `<=` |
| `date:2024-03` · `shot:>=1998` | file date · capture date |
| `place:tokyo` · `lens:` · `roll:` | other Metadata fields |
| `scene:beach` | frames in a scene |
| `keeper:` · `rejected:` · `edited:` | marked or edited frames |

Terms combine: `film:portra iso:>=400 -rejected:`. Metadata terms match only frames whose Metadata you filled in. **Search by Meaning** (turn it on in Preferences → Performance) finds pictures by description, such as "a dog on a beach".

### Marking frames

Right-click a frame, or use the shortcuts, to **Keep** it (a check) or **Reject** it (a cross; the frame dims and batch exports skip it). Your files never change.

A film mark in the top-right corner means the frame prints on the roll average; it is amber when the frame's color is far from that average. A mark in the bottom-left corner means the frame is built from several files (a stitch, HDR merge, Trichrome or half frame); hover it for the name. An exclamation mark means the file did not open (click to retry); a small amber dot means the thumbnail is out of date.

The right-click menu also copies, pastes and resets settings, applies them to other frames, reanalyzes frames ([§10.5](#105-metering-negative--positive)), puts the selection on the roll average or back on its own measurement, shows the file in its folder, and moves files to the Trash (the edit stays).

### Stitch and HDR

*   **Stitch**: select overlapping shots of one large negative, then right-click → **Stitch Selected Frames**. **Unstitch** separates them again.
*   **HDR** (slides): select a bracket of exposures one stop apart, then right-click → **Merge Exposures (HDR)**. Right-click the merged frame → **Render Exposure** to pick the exposure it shows, or use the **Render Exposure** slider on the Metering card. **Unmerge Exposures** separates them. Bracket both ways around your metered shot.
*   **Merge Frame to TIFF Negative…** saves a stitch as one file ([§10.7](#107-frame-assembly)).

### Scenes

A scene is a group of frames shot in the same light. It gets its own Roll Analysis baseline, so its frames match each other. **Detect Scenes** in Roll Analysis finds these groups for you. To make one yourself, select frames, then right-click → **Scene → Group as Scene…**. The same menu adds and removes frames, runs **Analyze Scene…**, and renames or deletes the scene. **Sort → Scene** groups them in the strip. Scenes need a roll; without one, the **Scene** menu is grayed out.

---

<!-- panel:analysis -->
## 3. Analysis readout

The readout above the Roll and Frame tabs describes the current frame while you edit.

#### Chart

The buttons above the chart pick one view:

*   **Curve**: the paper's curve, from the negative (left to right) to the print (bottom to top). Steeper means more contrast. The gray histogram is the negative, the colored one the print.
*   **Waveform**: print brightness across the frame.
*   **Parade**: red, green and blue side by side; a gray subject sits at one height in all three. **Parade Overlay** puts the three on one plot.
*   **Vectorscope**: the print's colors around a center point. A cluster off center is a color cast.

**Log** shows small areas, such as clipped ends.

#### Clipping

Triangles in the top corners warn when shadows go pure black (left) or highlights pure white (right). **Clipping Overlay** (the half-circle button) marks those parts on the picture. **Zone Highlight** (the marker button) shows the zone under the cursor alone on the frame.

#### Zones and zone placement

The **step wedge** prints 21 gray steps; steps that merge into black or white are lost tones. The **zone strip** shows how much of the frame falls in each zone, from 0 (black) through V (middle gray) to X (white). Double-click it (or press `Shift+Z`) for the **Zone Overlay**, which colors the picture by zone.

Click a zone on the strip, then a spot on the picture, and NegPy sets the print so that spot prints in that zone. One pin sets Print Density; a second also sets ISO-R Grade. **Place Zones** (or Enter) applies the result; `Esc` cancels.

#### Readouts

The **Readouts** fold holds numbers about the scan. An amber value is a warning; hover a row for details.

*   **Negative**: the negative's density range (Δ). Hover for flat, normal or contrasty.
*   **Exposure**: bright (positive) or dark (negative) frame, in stops.
*   **Clipping**: share of the print that is pure black, then pure white.
*   **Scan clip**: share of each channel, red, green and blue, at the scanner's maximum. No edit fixes it; expose the scan lower.
*   **Repair**: share of the picture that dust removal rewrote.
*   **Gamut**: share the proof printer cannot print, while proofing.

---

## 4. Geometry tab

Two cards: **Crop** sets the part of the picture you keep, and **Straighten** turns and levels it. Both hold this frame's own values. Their **Roll** button copies them to the selection or the whole roll ([§1](#frame-or-roll)).

<!-- panel:autocrop -->
### 4.1 Crop

Choose the part of the picture to keep, by hand or with Auto Crop.

*   **Draw**: drag a crop on the picture. Drag a corner to resize it, the inside to move it, or outside the box to rotate. `Enter` or **Done** keeps the crop, and `Esc` puts back the crop you had before. The **×** removes a crop you drew. Drawing turns Auto Crop off.
*   **Auto Crop** (also on the Geometry tab's bar): finds the edges of the picture and crops to them.
*   **Ratio**: the crop shape, such as `3:2` or `1:1`, turned to fit the frame. **Free** uses the shape NegPy finds, and you can drag one side at a time. **Detect** picks the nearest standard shape.
*   **Guide**: composition lines, such as thirds, while you draw. The rotate button turns them.
*   **Mode** (Auto Crop Settings fold): **Image only** crops to the picture. **Film edge** keeps the whole film with its border and holes.
*   **Crop Offset** and **Rebate Trim** (fold under Mode): move the crop inside (positive) or outside (negative) the found edge, and set how far it cuts into the film border (Image only). 0% stops at the film edge and 100% at the picture edge; above 100% cuts into the picture to remove a white border.
*   **Square First**: straightens the frame to its film edges before Auto Crop crops it.
*   **Crop All Frames** (Image only): crops every frame together. Clear frames help unclear ones, and crops you drew stay.

Auto Crop needs the scanner light or holder to be the brightest part of the scan. It leaves a frame alone when the picture is as bright as the light, or when the next frame fills a large part of one side. Crop such a frame by hand, or use **Film edge** and trim in.

<!-- panel:geometry -->
### 4.2 Straighten

Turn the picture and make it level.

*   **Rotate** and **Flip**: turn the picture a quarter turn, or mirror it. The toolbar has the same buttons.
*   **Level**: draw a line along the horizon or a vertical edge, and the picture turns to make it level.
*   **Square to Film** (also on the Geometry tab's bar): lines the picture up with the edges of the film. It ignores what is in the picture, so a photo taken at a tilt stays tilted.
*   **Fine Rotation**: turns the picture by small amounts. Positive is clockwise.
*   **Tilt** and **Swing** (fold): straighten a building shot from below (Tilt) or a wall shot from one side (Swing). **Draw Lines** sets both from lines along the four film edges.
*   **Crop Empty Corners**: crops away the empty corners that rotation, Tilt and Swing leave.

Crop before a large Tilt or Swing, or the film border can come into view and darken the print. Lens distortion is fixed in **Optics** ([§10.3](#103-optics)).

---

## 5. Exposure tab

These four cards set the brightness, contrast and color of the print.

<!-- panel:color -->
### 5.1 Filtration: color balance

Corrects the color of the print, as the color filters of a darkroom enlarger do. **Global / Shadows / Highlights** applies the color controls to the whole picture, or mostly to its dark or bright parts.

*   **Pick WB**: click something in the picture that should be gray, and NegPy corrects the color to make it gray.
*   **Temperature**: makes the picture warmer or cooler.
*   **Cyan / Magenta / Yellow**: shift the color between cyan and red, magenta and green, or yellow and blue.
*   **Ring-around** (`Shift+F`): shows the picture as a grid of slightly different colors. Click the one that looks right. `Esc` closes it.
*   **Cast Removal**: removes a color cast that differs between the dark and bright parts of the picture. It starts at 0.5 (half) on color negatives and at 0 on slides. Leaves, sky and water do not count as gray, so they keep their color.
*   **Filter**: a colored filter, as on the camera lens for black-and-white film. It shows only on a color negative printed in B&W, and replaces the controls above. Each filter darkens the colors opposite to its own. Yellow #8, Deep Yellow #15, Orange #21 and Red #25 make a blue sky darker, step by step. Red also makes leaves dark and skin pale. Yellow-Green #11 and Green #58 make leaves lighter and skin darker. None gives the look of a normal black-and-white film.

<!-- panel:tone -->
### 5.2 Tone: brightness and contrast

How light or dark the print is, and how much contrast it has.

*   **Print Density**: how light or dark the print is (the darkroom's exposure time). Lower is lighter.
*   **ISO-R Grade**: contrast (the darkroom's paper grade). A lower number gives more contrast; about 110 is normal.
*   **Auto** (the wand, also on the Exposure tab's bar): **Auto Density** and **Auto Grade** set brightness and contrast for each frame, so a roll prints evenly. A slider then shows the automatic value: move it to adjust, double-click it to go back. Right-click the wand to turn both on or off. The Auto menu also has:
    *   **Set Targets…**: the brightness and contrast Auto aims for.
    *   **Bake Auto Density/Grade**: keeps this frame's automatic values and turns Auto off for it.
    *   **Hold Black**: when you lighten the shadows, the deepest blacks stay black.
*   **Test Strip** (`Shift+T`): a grid of versions at different brightness and contrast. Click the one you like. `Esc` closes it.
*   **Shadows Density** / **Highlights Density** (fold under Print Density): darken or lighten only the shadows or the highlights.
*   **Shadows Grade** / **Highlights Grade** (fold under ISO-R Grade): contrast in the shadows or the highlights only.
*   **Printing Aids** (fold):
    *   **Preflash**: brings back detail in the brightest areas.
    *   **Diffusion**: a soft glow; dark areas spread into bright ones.
    *   **Contrast Mask**: opens up large dark areas and holds back large bright ones, and keeps fine detail. In the darkroom, a printer gets this with a soft, blurred mask placed on the negative. In digital printing, Fujifilm Frontier scanners get a similar effect with HyperTone. Areas far from mid-gray change most, so a low-contrast frame changes little.
*   **Global / R / G / B**: R, G or B applies the grades to one color only, for a cast that differs between shadows and highlights.

**White Point** and **Black Point** are on the Roll tab's **Metering** card ([§10.5](#105-metering-negative--positive)).

<!-- panel:paper -->
### 5.3 Paper Response: the paper

The paper the picture prints on, and how it turns light into tones from black to white.

*   **Print stock**: the paper. A real paper sets the contrast curve and the colors from its maker's data. **Digital (Adobe RGB)**, the default, is an ideal paper. **Digital (Pakon)** has the colors of a Pakon lab scanner. **Digital (Frontier)** has the colors and the contrast curve of a Fujifilm Frontier lab scan.
*   **Paper Test Strip** (the striped button): prints the frame on every paper. Click a paper to see it large, step with the arrow keys, and press **Keep** to use it.
*   **Dye Separation**: color strength in the print. 1 is unchanged. **Separation Damping** puts more of the change on dull colors and less on strong ones. **Chroma** on the Look tab changes all colors evenly instead.
*   **Paper White**: shows the white of real paper, a little darker than pure white.
*   **Paper Black**: shows the black of real paper, a little lighter than pure black.
*   **Curve Shape** (fold):
    *   **Snap**: midtone contrast. Black and white do not move.
    *   **Toe**: shapes the shadows. Positive lifts them, negative makes them deeper. **Toe Width** sets how far into the midtones it reaches.
    *   **Shoulder**: shapes the highlights. Positive softens them, negative brightens them and can lose detail. **Shoulder Width** sets how far it reaches.
*   **Global / R / G / B**: R, G or B applies the curve controls to one color layer.

<!-- panel:local -->
### 5.4 Dodge & Burn: lighten or darken areas

Draw a shape, then make only that area lighter (dodge) or darker (burn). A slide does not use this card, but its masks stay.

*   **Draw**: click to place points. Double-click, press Enter or click near the start to close the shape. Drag a point to move it, click a "+" on an edge to add one, and right-click a point to delete it.
*   **Oval**: drag out an oval. Its handles move, stretch and turn it.
*   **Card**: a burn that fades out, like a card moved over the paper. Drag from the full-strength edge (solid line) to where it ends (dashed line).
*   **Mask list**: click a mask to select it. Its icons turn it on or off, invert it (act everywhere except inside it), show its outline or delete it. The sliders below change the selected mask.
*   **Burn**: positive darkens the area, negative lightens it. The unit is stops: +1 doubles the light.
*   **Feather**: how soft the edge is.
*   **Grade**: contrast inside the mask. Negative gives more contrast.
*   **Flash**: brings back detail in bright areas inside the mask, like Preflash.
*   **Tone Limit**: **Highlights** or **Shadows** makes the mask act only on tones lighter or darker than **Tone Zone** (0 is black, V middle gray, X white). A sky burn then stops at the skyline. **Tone Softness** sets how gradually it starts.

**Printing Notes** (`Shift+N`) marks the masks on the picture with their numbers and values, as a printer's notes do.

---

## 6. Look tab

<!-- panel:lab -->
### 6.1 Lab: color and detail

The adjustments a lab scanner makes after the print.

*   **Chroma**: color strength. 1 is unchanged, 0 is black and white.
*   **Skin Protection**: keeps skin tones from getting too strong in color.
*   **Chroma Denoise**: removes colored speckles. Film grain stays. It is on at 2.5 for color film.
*   **Sharpening**: sharpens detail without colored edges. Its fold holds **Method** (Unsharp Mask makes edges crisper; Deconvolution undoes the scanner's blur), **Radius** (the size of the detail, in exported pixels; judge it in **Export Size** or with the **Grain Focuser**) and **Masking** (sharpen only edges, so sky and skin stay smooth).
*   **Auto Masking** (the wand beside Method, also on the tab header): sets Masking from the grain of each frame. Masking shows the value in use, and a drag moves it up or down from there. An export measures the grain again at its own size.
*   **CLAHE**: local contrast, to bring out detail. Strong values can look unnatural.
*   **Glow** (Effects fold): a soft glow around bright areas, like the soft focus of a minilab scanner.
*   **Halation** (Effects fold): a red glow around bright areas, as some films give.

<!-- panel:altproc -->
### 6.2 Alternative Processes

Historic print processes for B&W negatives. Pick **None**, **Lith**, **Cyanotype** or **Sabattier**, and only that process's controls show.

#### Lith

Creamy, warm highlights and hard, deep blacks. The print stock on Paper Response sets the color. Only the Selenium and Gold toners work on a lith print.

*   **Exposure**: more gives warmer, more colorful highlights and softer tones.
*   **Snatch Point**: when the print leaves the developer. Higher gives deeper, colder blacks.
*   **Abruptness**: how suddenly the shadows turn black.

#### Cyanotype

A blue print with green highlights. Toners do not work on it; use Bleach and Tannin.

*   **Sensitizer**: **Classic** gives a light blue with green highlights. **New** gives deeper, cleaner blues.
*   **Exposure**: more makes the print bluer.
*   **Exposure Scale**: contrast. A shorter scale gives more contrast.
*   **Bleach**: removes blue, highlights first.
*   **Tannin**: turns bleached areas brown. Use Bleach first for a full brown.

#### Sabattier

A flash of light during development. The light tones reverse toward gray, and thin light lines (Mackie lines) appear along edges between dark and light.

*   **Strength**: the length of the flash. Above 1 the light tones reverse.
*   **Re-exposure**: where the reversal starts. Darker tones stay as they are.
*   **Agitation**: less agitation gives wider Mackie lines.

---

<!-- panel:toning -->
### 6.3 Toning

Toners are chemical baths that change the color of a B&W print. Split Toning adds color to any print.

**Chemical Toning** (B&W negatives) acts in the order shown:

*   **Selenium**: deeper, cooler blacks.
*   **Sepia**: warm brown, from the highlights down.
*   **Gold**: blue-black. Over sepia, it gives orange-red highlights.
*   **Iron Blue**: blue shadows and navy blacks.
*   **Copper**: pink to brick red. Blacks get weaker.
*   **Vanadium**: green midtones and highlights. Blacks stay black.

On a lith print only Selenium and Gold work. On a cyanotype no toner works.

**Split Toning** (all prints): **Shadow Strength** and **Highlight Strength** add color to the shadows and the highlights. **Shadow Hue** and **Highlight Hue** pick the color, and show once the strength is above 0.

---

## 7. Finish tab

<!-- panel:leak -->
### 7.1 Light Leak

Reduces the glow a light leak adds to a negative or a slide, before any dust repair, only under the leaks you place. Burned-out film stays: heal or clone it. Not available on Positive frames and composites.

*   **Reduce Light Leak** turns the correction on. **Auto** (magic) places a shape on each leak the film supports: an edge where lit rebate carries on into the picture, a band along a straight leak. Glow that stops at the frame line is edge fog or edge print, and Auto leaves it.
*   **From Roll** (film icon): with other frames selected, carries this frame's leaks to each, and each frame reads the leak's strength and color itself. With none selected, it selects the frames Auto finds a leak on.
*   **Film Dmax** (Off, or 0.05 to 4.0 D): the film's maximum density, for a film whose dense end is saturated.
*   **Recover Layers**: rebuilds a color layer the leak drove to its densest from the layers that still hold the picture.
*   **Grain** and **Contrast** act only where leak came off: Grain calms stretched grain, Contrast restores flattened contrast.
*   **Band**: drag along the leak's strongest line, then drag each dashed edge to where it fades out.
*   **Edge**: drag along the side the leak comes in from; the line snaps to it. Where the crop takes in the lit gap between frames, the edge clears it along the whole side.
*   **Spot**: drag out an oval over a round leak.
*   **Paint**: brush over a leak with no line or center, wider than the leak.
*   **Selected leak**: **Strength** (0 to 400%) sets how much of the leak comes off; **Falloff** how a band, spot or painted region fades toward its edge; **Color** where its color comes from (**Auto**, **Back** for light through the base, **Front** for light on the emulsion side); **Color Trim** takes out more or less of its leading color. Color and Color Trim are hidden on B&W film.
*   **Show Burned** (flame) marks film the leak burned out in every layer.

<!-- panel:retouch -->
### 7.2 Retouch: dust, hairs, scratches

Removes dust, hairs and scratches. Each repair is rebuilt from the clean film around it.

*   **Overlay**: **Marked** shows what dust removal found, **IR** the scan's infrared channel.
*   **Optical Removal** (also on the Finish tab's bar): finds dust and hairs by itself. **Spot Threshold** and **Hair Threshold**: lower finds more but can remove real detail; 1 is off. **Size**: the largest dust to remove. To protect an area, right-drag over it or right-click a mark; the cursor button skips the right-click menu.
*   **IR Removal** (scans with an infrared channel, also on the Finish tab's bar): finds dust in infrared, where the picture does not show. **IR Threshold**: lower finds more. B&W and Kodachrome are skipped: their picture shows in infrared too. The channel comes from 4-channel TIFF and DNG scans or an `_IR.tif` file beside the scan.
*   **Heal**: click or paint over dust. Only marks that stand out are repaired, so paint generously.
*   **Scratch**: click points along a scratch or hair, then press Enter.
*   **Line**: click once on a long straight scratch, and NegPy finds and repairs the whole line. **Line Sensitivity**: lower finds fainter lines.
*   **Brush Size**: also `Alt`+scroll on the picture.
*   **Clone**: copies film from another area over a defect Heal cannot fix. The first click (or `Alt`-click) picks the source; then paint. **Set Source** picks a new one. **Match Tone** takes the brightness and color around the defect. **Strength** sets how much it covers, **Feather** how soft its edge is.
*   **Undo Last** and **Clear All** remove manual repairs; right-click one on the picture to delete it.

<!-- panel:finish -->
### 7.3 Finishing: vignette, carrier, border

How the print is presented. The other sliders of each effect show once its first slider is not 0.

*   **Vignette**: **Burn** darkens (positive) or lightens (negative) the edges. **Size** sets how far it reaches. **Roundness** goes from round (0) to the shape of the print (1).
*   **Filed Carrier**: **Width** draws a rough black frame around the picture, as a darkroom negative carrier that was filed wider does. **Roughness**, **Flare** (a gray glow outside the frame) and **Corners** shape it.
*   **Border**: **Width** puts a mat around the print. **Bottom Weight** makes the bottom wider, as on a framing mat. **Color**: **Paper White** matches the paper, **Custom** uses a color you pick.

---

## 8. Favorites tab

Your most-used sliders in one place. It is empty until you fill it: click **Edit Favorites**, turn sliders on at the left and drag them into order on the right, then click **Apply**. They are the same controls as in their own cards.

<!-- panel:presets -->
### Presets

Saved settings you can apply to other frames.

*   **Apply** (or double-click a preset): applies the selected preset to this frame.
*   **Save…**: choose which of the current settings to keep as a new preset. A preset can hold the Crop and Optics settings with the Flat Field profile, but never one frame's crop, rotation or flips.
*   **Pen** and **Trash**: edit or delete the selected preset.

---

## 9. History tab

<!-- panel:work_prints -->
### Work prints

A **work print** is a version of this frame that you keep under a name, like the test prints on the way to the final one.

*   **Save Work Print…** (`Ctrl+Shift+S`) keeps the current edit.
*   **Click** a work print to use it. `Ctrl+Z` goes back.
*   **⋮** or right-click: **Compare With This Version**, **Export This Version…**, **Rename…** or **Delete…**. **Compare With This Version** opens **Before / After** with the work print on the left.
*   **Load Metadata** (off at first): also restores the metadata saved with the work print.

<!-- panel:edit_history -->
### Edit history

Every change you made, newest on top (the last 100). **Now** marks the current state; after an undo, later steps show as **Undone** until your next edit. Click a step to go back to it. **⋮** or right-click → **Export This Version…** exports a past state. **Compare With This Step** opens **Before / After** with that step on the left, on your current crop. Press `\` to close it.

---

## 10. Roll tab

These cards hold what the frames of a roll share: the film, how it was scanned and the roll's measurements. Each card has the Frame / Roll pair ([§1](#frame-or-roll)). The line above the cards lists the cards this frame changes for itself.

<!-- panel:film -->
### 10.1 Setup

What the film is and how you scanned it. These choices decide which cards and controls show.

*   **Color / B&W / Slide**: color negative (C-41), a black-and-white print, or slide (E-6 and similar). B&W prints a B&W negative or a color negative in black and white. On a color negative, the Filtration card then gives a contrast Filter.
*   **Positive** (Slide only): for a file that is already a positive, such as a scanned print or another app's export. NegPy then skips inversion and metering. A Slide that is not a camera RAW, such as a TIFF, JPEG, JXL or a Pakon, Noritsu, Coolscan or Flextight scan, starts Positive. Turn it off to give a faded scan the slide tone curve. A JXL loads as linear.
*   **Encoding** (Positive TIFF): how the samples encode light. **Auto** reads the file's ICC profile, else takes linear above 8 bits and sRGB at 8 bits. **Linear** is a scanner's linear output, such as Nikon Scan with color management off. **2.2** is a 2.2 gamma in Adobe RGB primaries. Auto skips a scanner's own calibration profile and reads the file as untagged. Pick Linear or 2.2 to match the scanner software.
*   **Camera / Film Scanner**: what you scanned with. NegPy reads it from each file (a camera RAW is a camera) until you pick one. Your pick holds for every frame and sets the Scan tab's scanner.
*   **White Light / Narrowband Light** (not for a B&W negative): the light behind the negative. White light is a lightbox or LED panel. Narrowband is a light with pure red, green and blue, such as a Scanlight.

The scanning choices are one setting for the whole app. The list under them shows how a RAW file is decoded and which print stock is used.

A camera shows the Raw Decode and Optics cards. A camera with narrowband light also shows Trichrome and Calibration. A change updates every open frame (Ctrl+Z undoes it per frame), but a print stock you picked yourself stays. A camera Scan sets the decode and the print stock for each frame from its own light.

<!-- panel:demosaic -->
### 10.2 Raw Decode

How a camera RAW file is turned into pixels. Shown for a camera.

*   **Camera WB / Linear RAW**: Camera WB uses the camera's white balance. Linear RAW decodes neutral, for narrowband light. Setup sets it for you.
*   **Preview** / **Export**: the demosaic method, which fills in the colors each sensor pixel cannot see. **Auto** is fast on screen and uses **AHD** for export. **VNG** is smooth and **PPG** fast. **DCB** and **DHT** keep more fine detail but can turn grain into patterns. **AAHD** softens edges. Only the methods in your LibRaw build are listed.
*   **Recovery** (slides): rebuilds blown highlights in a camera RAW. **Blend** suits sun and sky. **Reconstruct** is stronger but can get the color wrong.

<!-- panel:optics -->
### 10.3 Optics

Corrections for the lens and the light, the same for every frame you scan with this setup. Shown for a camera.

#### Lens Correction

*   **Embedded**: uses the file's own lens correction (DNG, Sony ARW, Panasonic RW2). **Distortion** straightens curved lines and **CA** removes color fringes. Set it before you crop or retouch.
*   **Distortion Correction**: straightens lines that bow out (positive) or in (negative). Use the film edge as a straight guide.

#### Flat Field Correction: even out the light

Removes uneven light, such as darker corners, with a shot of the bare light source.

*   **Profile**: **+** saves a shot of the bare light as a named profile. The trash button deletes it, and every frame that uses it loses its correction.
*   **Apply Flat Field**: applies the selected profile to this roll.
*   **Check Flat Field**: shows how well the profile evens out its own shot. Even gray is good; a band or patch shows where it is off.

A frame keeps its own profile unless it follows its roll, so each light and film format can have one. A new frame, or a frame whose profile is missing, takes the last profile you chose.

<!-- panel:sensor -->
### 10.4 Calibration: colors from your scanning setup

Corrects color errors that come from your light, camera and film.

**Sensor Crosstalk Profile** (a Single Exposure under narrowband light, Linear RAW): removes the color that leaks between the sensor's channels. Run **Roll Analysis** again after a change.

*   **Profile**: the sensor crosstalk profile. Your own `.toml` profiles go in `<Documents>/NegPy/sensor/`.
*   **Method**: **Two-Scale** (default) removes the leak and keeps very strong colors from speckling. **Linear** is exact, but strong colors can speckle. **Density** has no speckle, but strong colors are a little duller.
*   **Calibrate**: makes a profile from three shots of the bare light (red, green and blue), from files or a tethered camera (see CAMERA_SCANNING.md).

**Crosstalk** (color film): film dyes, your light and your sensor mix the colors a little, and a matrix separates them again. Run **Roll Analysis** again after a change.

*   **Matrix**: the matrices for this film type. Those marked *(approx)* come from film data sheets and are a starting point. Your own `.toml` files go in `<Documents>/NegPy/crosstalk/` (see CROSSTALK.md). The slider button opens the editor; **+** makes a new matrix. No slide matrix comes with NegPy.
*   **Strength**: how much of the matrix to apply.
*   The swatch button opens **Measure from Chart** (color negatives): it measures a matrix from a ColorChecker 24 or SpyderCheckr 24 shot on the film. Drag the four dots onto the corner patches.

**Hue Trim** (camera with narrowband light): turns all colors around the color wheel, for a light that makes yellows orange or greens olive. Keep 0 for white light. It also works on slides, where the sensor crosstalk profile does not.

### 10.4.1 Fade Restoration: faded slides

Gives a faded slide its dye layers back. It is on the Calibration card, for slides (Transparency) only. It works together with Crosstalk, before Cast Removal. Set the fade first, then let Cast Removal ([§5.1](#51-filtration-color-balance)) remove the rest of the cast. Run **Roll Analysis** again after a change. The Survival sliders follow the roll: Roll on the Calibration card gives every frame this frame's values, and a frame you change keeps its own.

*   **Red Survival** (0.05 to 1.0, default 1.0): how much red dye is left. On a slide the cyan dye, read on red, fades fastest. At 1.0 a badly faded slide gets the right colors but looks washed out. Estimate with Midtone sets it; otherwise lower it by eye until the contrast comes back.
*   **Green Survival** / **Blue Survival** (0.2 to 5.0, default 1.0): how much green and blue dye is left, against red. At 1.0 that layer faded as much as red; below 1.0, more.
*   **Estimate**: sets the Survival sliders from the grays in this frame. **Midtone** reads the color of the middle grays and also sets Red Survival, so the best-kept layer is at 1.0. It suits a badly faded slide. **Gray Range** reads how far the grays reach into the shadows and leaves Red Survival as it is. It can suit a mild fade with deep shadows. A hint shows the result, or why there is none, and what the other method reads: a large gap means a cast that the fade does not explain. A value outside 0.2 to 5 stops at the limit, and the hint gives the value it read. Above 5, lower Red Survival by eye. A slide that is almost one color reads as faded: set the sliders by hand.

<!-- panel:process -->
### 10.5 Metering: negative → positive

To turn a negative into a positive, NegPy measures its darkest and brightest parts (the black and white points) and the color of the film. These controls set what it measures.

*   **Reanalyze Frame**: measures this frame, or each selected frame, again from its current crop. The frame then uses its own bounds, not the roll or scene baseline.
*   **Draw Region**: draw an area, and NegPy measures only that area. Double-click inside or press `Enter` to confirm, and `Esc` puts back the region you had before. **×** clears it.
*   **Film Base**: click the clear film between frames to set the film base for the Analysis readout. It is the roll's base, or this frame's own with no roll open. **×** clears it.
*   **Analysis Buffer**: ignores a border around the frame when measuring, so film edges and holders do not count.
*   **Clipping**: how much of the print is pure black or pure white.
*   **Lock Bounds**: keeps this frame's black and white points, even when the crop or Roll Analysis changes.
*   **Luma Range Clip** (fold): how tightly black and white are set. Positive gives more contrast, for dense or foggy negatives. Negative keeps more shadow and highlight detail.
*   **Color Clip** (fold): how tightly the color balance is set.
*   **White Point** / **Black Point** (fold): move the measured white and black points. A positive white point brightens the print; a positive black point lifts the blacks. **R / G / B** moves one color only.

A **slide** needs no inversion: the tone controls shape the picture directly. Auto Density and Auto Grade start off; turn them on for a faded slide. In Lightroom terms, Exposure is Print Density (lower is brighter), Contrast is ISO-R Grade, and Shadows and Highlights are Shadows and Highlights Density.

<!-- panel:baseline -->
### 10.6 Roll Analysis

Measures the whole roll once and shares the result, so the frames of one film print alike. NegPy measures the roll and each scene on its own, as frames arrive, but no frame uses a baseline until you turn on an average. With no roll open, the card is grayed out: open or save a roll first.

*   **Reanalyze**: measures every loaded frame (except scene and locked frames) saves the average as the roll's baseline and turns the averages on for those frames. Frames very different from the rest keep their own. It first crops the frames that wait for Auto Crop.
*   **Use This Frame**: makes this frame's measurement the roll's baseline.
*   **Use Average**: right-click any of the three to turn every average on or off. **Luma** uses the roll's black and white points, **Color** the roll's color balance, and **Cast** (color negatives) the roll's gray balance for Cast Removal. With Cast on, a frame shot under a different light keeps some of that light, as on a Frontier print. A magenta cast stays, a warm or green cast stays in part, and a blue cast goes. With Luma and Color on, the roll prints evenly; with both off, each frame is measured on its own.
*   **Baseline**: says where the baseline came from (roll, scene or frame). It warns when there is none, when it is out of date, or when this frame's color is far from the average.
*   **Rolls**: pick another roll's saved baseline. A ticked roll has one.
*   **Scenes**: groups of frames with their own baseline, such as one subject under one light ([Scenes](#scenes)). **Analyze** measures the scene, **Select** selects its frames and **Delete** removes the group.
*   **Detect Scenes**: finds the groups of frames under another light than the roll, in any frame order, and shows them on the Light Table. Each group waits under **Found**. To change a group, drag frames onto another group, or onto a frame with no scene to leave them out; right-click › **Scene** does the same. Give it a name, then **Accept** it as a scene with its own baseline, or **Discard** it.

<!-- panel:assembly -->
### 10.7 Frame Assembly

How files become frames. Each roll keeps its own settings here.

#### Trichrome

For a camera with narrowband light that takes three shots of each frame: red, green and blue.

*   **Trichrome Mode**: joins the three shots into one color frame. Shots are grouped by capture time (or file name), so take each frame's three shots one after another. Shots that do not group stay separate.
*   **Merge Exposures (Trichrome)** (group icon, or right-click): select a frame's three shots to join them. NegPy finds which shot is red, green and blue.
*   **Edit Trichrome…**: pick the three shots of the current frame by hand. **Align Channels** lines them up to remove color fringes. **Split Trichrome** (ungroup icon, or right-click) separates them again.
*   **Merge Frame to TIFF Negative…** (right-click): saves a triplet or a stitch as one TIFF beside its first source file, with its edit. An option moves the source files to the Trash.

#### Half Frame

**Half Frame Mode** splits each scan into two frames, for half-frame cameras. Each half has its own edit. Each roll keeps its own setting. With no roll open, it applies to frames in no roll until **Unload All**, and **Reopen your last session** keeps it. It is grayed while a frame from a roll is loaded, because that frame follows its own roll.

*   **Adjust…**: drag the white box to crop and the orange line to set the split. **Split direction** and **Cut Thickness** set the gap, **Auto-detect** finds them again, and **Apply** saves to this frame, the selection or all frames.
*   **Detect All**: finds the splits on every scan again.
*   **Unsplit** (on a split frame): makes it one frame again and deletes both halves' edits.

Right-click a half for **Adjust Split for This Frame…** and **Reset Split to Roll Default**. Turning Half Frame off keeps edited halves side by side as a **diptych**; turn it on again to edit them.

---

## 11. Metadata tab

Details about how you took and developed the picture: camera, lens, film and process. NegPy writes them into every export, so photo apps such as Lightroom show your film gear, not your scanner.

**Metadata Preview** is at the top. Under it, **Shot** holds Analog Gear, Capture and Exposure, and **Process** holds Process and Scanning. The two icon tabs are **Presets** and **My Gear**. These cards apply to the whole roll at first ([§1](#frame-or-roll)); the frame number is always per frame.

<!-- panel:metadata_gear -->
### Analog Gear

*   **Camera / Lens / Film stock**: pick from your own gear (the **My Gear** tab). **Other…** opens the full built-in catalog.
*   **Clear**: empties all three.
*   **Infer from Folder Name**: fills empty fields (camera, film, ISO, date) from the folder name.

<!-- panel:metadata_capture -->
### Capture

*   **Date**: when you took the picture, as `1998`, `1998-07`, `1998-07-14` or `1998-07-14 16:30`.
*   **Place**: the pin button opens a map. You can also paste coordinates or a map link. The map uses OpenStreetMap.

<!-- panel:metadata_process -->
### Process

How the film was developed.

*   **Saved process**: a recipe from My Gear that fills the fields below. Typing over a field unlinks it.
*   **Film format**: the film size, such as `35mm`, `120` or `4×5`.
*   **Developer** and **Dilution**: for example `D-76` and `1+1`.
*   **Push / Pull**: from `Push +3` to `Pull -3`.
*   **Time** and **Temperature (°C)**: time as `9:30` or in minutes.
*   **Clear**: empties the process fields.

<!-- panel:metadata_scanning -->
### Scanning

*   **Saved setup**: a scanning setup from My Gear.
*   **Scanning**: how you scanned, or notes.
*   **Roll / Frame**: the roll name and frame number. File names can use them (`{{ roll }}`, `{{ frame }}`).
*   **Sequence…**: numbers the selected frames, or the whole roll, in film order. Set the start, the step and the direction; **Skip Rejected Frames** leaves rejects out.
*   **Clear**: empties the setup and the note.

<!-- panel:metadata_exposure -->
### Exposure

The shutter speed, aperture, ISO and the camera's exposure bias (EV) of the original shot, if you know them. Click the lock to type free text such as `1/125s f/2.8 ISO 400 -1/3ev`. A bias is stops before `ev`, and sums like `1+2/3ev` add up.

<!-- panel:metadata_preview -->
### Metadata Preview

Shows what goes into the exported file. The shield button, **Protect Original Metadata**, exports the source file's own metadata unchanged instead. **Description…** picks the fields that make up the image description (camera, lens, film and ISO at first).

Camera, lens and film go into the standard EXIF fields, and your scanner into NegPy's own XMP fields. A star rating set in the camera carries over.

<!-- panel:gear_items -->
### My Gear

Your gear list, used by every gear picker. **Category** picks **Cameras**, **Lenses**, **Film Stocks**, **Processes** (development recipes) or **Scan Setups**. It is saved in `~/NegPy/gear/`.

*   **+**: copy an item from the built-in catalog, or **Add Custom** to type one in.
*   **Catalog**: also show the built-in models.
*   **Copy / Trash**: duplicate or delete the selected item. Built-in items cannot be deleted.

<!-- panel:gear_presets -->
### Presets

Saved sets of metadata, such as one camera, lens and film. They are saved in `~/NegPy/presets/metadata/`.

*   **Apply to This Frame**: writes the preset's fields onto this frame; other fields stay.
*   **+**: saves the current frame's metadata under a name.
*   **⋮** or right-click a preset: apply, duplicate, edit or delete it.
*   **Details**: edit the preset's fields. **Notes** is free text.

A film stock sets the format, so set a format such as `6×7` after the stock.

---

## 12. Export tab

The **Export** button is at the bottom of the tab. Above it, **File** holds the output settings and **Proof** the soft proof. The two icon tabs are **Presets** and **Alongside Exports**.

### Output intent

*   **Print** (default): the picture as you see it.
*   **Flat**: a flat, low-contrast master for editing in another app. It skips the print look and the effects, but keeps light leak reduction and a color Print stock's dye color, and writes a 16-bit file. **Preview Flat** shows it on the canvas. In a roll, run **Roll Analysis** first so the masters match. Without a roll, NegPy measures each frame on its own, so the masters can differ.
*   **Linear**: the raw scan data before NegPy's processing, as a 16-bit TIFF or JPEG XL, for a raw workflow in another app. It reads camera RAW, LinearRaw DNG, Coolscan NEF, Flextight, Noritsu and Pakon files and TIFF scans. Options for infrared, white balance, flat field, sensor and lens correction are off at first. File names get `_linear`, so a source is never overwritten.

### File settings

*   **Format**: JPEG, TIFF, PNG, JPEG XL or WebP, with quality options. **Bit depth**: 8 or 16 bits (TIFF, PNG, JPEG XL). TIFF and PNG compression is always lossless.
*   **Export profile**: the color space of the file, such as sRGB for the web or Adobe RGB, or a printer or paper profile. **Import ICC** adds a profile. **Input ICC** sets the color space of a scan that has none.
*   **Paper ratio**: the shape of the final print, or **Original**.
*   **Resolution**: **Original** (full size), **Print** (size in cm and DPI) or **Pixels** (the long edge in pixels). A Print or Pixels export applies the Lab Sharpening after the resize. **Preview Export Size** shows the frame at this size on the canvas, to judge sharpening and grain.
*   **Destination**: the **Filename Pattern** (see [TEMPLATING.md](TEMPLATING.md)), **Overwrite Existing Files**, and the folder: an `export` subfolder beside each source file (default), the source folder, or a path you pick.

### Export button

Its arrow picks what to export: this frame (`Ctrl+E`), the selected frames or all visible frames. NegPy asks first when a batch holds unedited frames.

<!-- panel:export_presets -->
### Presets

Saved export recipes: format, size, color, folder and file name. Turn on the ones you want, and **Export Presets** writes each frame once per preset. **Manage** edits them. Presets do not work with Linear output.

<!-- panel:export_alongside -->
### Alongside Exports

Extra files beside the print. Each has a switch on the left and a one-time action on the right.

*   **Printing Notes**: a marked-up copy of the frame with its dodge and burn and print settings, as a darkroom printer keeps. **Preview** shows it (`Shift+N`); **Export Notes** saves it.
*   **Auto Density / Grade**: **Bake on Export** fixes the automatic brightness and contrast of each exported frame, so they do not change later. **Bake to Roll** does it for every visible frame now.
*   **Edit Sidecars**: **Save on Export** writes a `.negpy` file with the edit beside each source. **Export Sidecars** writes them for all visible frames. Right-click a frame → **Load Edit from Sidecar…** reads one back.

<!-- panel:contact_sheet -->
### Contact Sheet

On the **Alongside Exports** tab. A contact sheet shows the whole roll as film strips on one sheet of paper, as a darkroom contact print does.

*   **Path**: the folder for the sheets. Empty uses the export folder.
*   **Contact Sheet…**: opens the sheet. Drag an edge of the paper to resize it.
*   **Film format**: 35mm, 35mm Half Frame or 120 (with **Frame size** for 120).
*   **Paper**, **Width**, **Height** and **DPI**: the sheet's size and resolution. The default paper holds a 36-exposure roll.
*   **Order**: by file name, by date, or by scene with a new strip or row for each scene.
*   **Print**: **As Edited**, or **Straight Proof**: the whole roll at one exposure and normal contrast, so you see which negatives are thin or dense.
*   **Pick Frames**: click a frame to leave it out or put it back. Rejects start out.
*   **Edge Print**: the film maker's markings along the edges, such as the film name, frame numbers and the colored lines a stock prints by its perforations.
*   **Roll Label**: the roll name, film, camera and date above the strips. Type your own text in **Label text** to print it instead.
*   **White Paper**: prints on white, which saves ink on a home printer.
*   **Film Edges**: off prints the frames upright, each at its own shape, in rows as on the Light Table. The sheet then has no perforations and no **Edge Print**.

Sheets are saved as `contact_sheet.jpg` (`contact_sheet_1of2.jpg` and on for a long roll).

<!-- panel:soft_proof -->
### Soft Proof

The **Proof** tab shows on screen how the picture will look when printed on a given printer and paper. It changes the preview only, never the export. Paper cannot show colors as bright as a screen, so a correct proof looks duller.

*   **Proof on Screen** (`Shift+P`): turns the proof on or off.
*   **Preset**: a saved printer and paper setup.
*   **Profile**: the printer or paper profile. It follows the export profile until you pick one.
*   **Intent**: **Relative Colorimetric** keeps printable colors exact and clips the rest. **Perceptual** compresses all colors, which suits very colorful pictures. **Saturation** is for charts and graphics.
*   **Black Point Compensation**: maps the darkest tone to the paper's black.
*   **Simulate Paper White** and **Simulate Ink Black**: show the paper's real white and black.
*   **Gamut Warning**: shows colors the printer cannot print as gray.
*   **Display**: your monitor's profile, found automatically.

---

## 13. Scan tab

Scan film straight into NegPy. The **Scanner** card picks the scanner; the tab then shows that scanner's cards, the shared **Output** card and the **Scan** button.

<!-- panel:scan_source -->
### Scanner

*   **Scanner**: **Film Scanner** for a dedicated film scanner (the Device, Film & Quality and Framing cards), or **Camera** for a camera on a copy stand (the Camera and Preset cards).

<!-- panel:scan_device -->
### Device

*   **Backend**: **SANE** (Linux and macOS), **Nikon Coolscan (nkscan)** or **pyOpticfilm (Plustek)**. The controls follow what the scanner reports.
*   **Device**: the scanner; the arrows refresh the list. The eject menu has **Eject Now** and, on a strip feeder, **Eject When Done**, which returns the strip after a batch.
*   **Debug log** (Coolscan): writes `nkscan.log` in the NegPy folder. For a bug report, set **Trace**, repeat the problem and attach the file.

NegPy keeps macOS and Windows awake while it scans. On Linux, turn off automatic suspend before a long roll.

**Plustek**: the OpticFilm 8200i SE, 8100 V2, 8300i SE and 7600i v1 work; for other models, try SANE. On Windows, set up the USB driver first ([PLUSTEK_WINDOWS.md](PLUSTEK_WINDOWS.md)). **Coolscan**: needs no SANE. On Linux, USB needs a udev rule for vendor `04b0`.

<!-- panel:scan_quality -->
### Film & Quality

*   **Film**: Color negative, B&W negative, Slide or Kodachrome. It sets frame detection, metering, and whether infrared is offered.
*   **Film format**: the frame length (135, 66, 645 and others). **Auto** works where the holder sets it.
*   **Resolution** and **Bit depth**: the scan's dpi and bits per channel.
*   **Scan mode** (Plustek): **Multi-Pass** averages several scans for less noise; **Adaptive Multi-Exposure** combines a short and a long exposure for more range. Plustek scanners do not offer them at 7200 dpi.
*   **Samples** (Coolscan): reads each line several times, for less noise in the shadows. Slower.
*   **Infrared**: **IR** scans an infrared channel, for dust removal on the Retouch card. **ICE** (Coolscan) removes the dust in the scanner instead. A scanner without ICE shows one **IR** toggle. Not for B&W or Kodachrome.
*   **Superfine** (Coolscan): scans one line per pass. Slower.
*   **Autofocus**, **Auto-exposure** and **Exposure**: focus and exposure on the scanner, before the scan.

<!-- panel:scan_framing -->
### Framing

*   **Frames**: `1-6`, `1,2,5`, or empty for all. The strip preview fills it in.
*   **Preview Strip…** (strip feeder): previews every frame, to place the frames and pick them. **Preview…** (manual holder): sets one crop window. **Clear** resets it. **Detect Frames** finds the frames again after the film moves.
*   **Meter Frame…** (Coolscan): measures the exposure on one frame and uses it for every later scan, so all frames of a roll match. **Unlock** ends it. Meter again for each roll.
*   **Prescan…** (Plustek): a quick full preview. Drag a crop, and the next scan reads only that area.

<!-- panel:scan_camera -->
### Camera

Scans with a camera on a copy stand, connected over USB in **PC Remote** mode (macOS and Linux; it needs `python-gphoto2`). With a NegPy **Scanlight**, NegPy also controls the light. CAMERA_SCANNING.md explains the setup.

*   **Camera** / **Light**: the connection status. The light turns amber when it runs warm. On Linux, the Scanlight needs your user in the `dialout` group.
*   **Live View**: a live picture from the camera. Click to place the focus magnifier, and focus until the **Focus meter** reads **at peak**. Set ISO, shutter and aperture here or on the camera. When it closes, the Scanlight goes back to how it was before.
*   **Histogram** and **Zebras**: the frame's histogram, and stripes on parts that are about to clip, in the clipping colors that Color vision sets.
*   **Meter**: sets the shutter so the brightest part that must not clip (a negative's clear film, a slide's highlights) lands just under clipping. Meter before the first scan. Without a calibrated preset, a scan uses the exposure set on the camera, and NegPy asks before it scans unmetered.
*   **Scan** shoots the next frame and imports it; **Retake** (the redo arrow) shoots the last frame again. When a scan clips, the status line tells you how much of the picture clipped. In the Live View window, `Space` or `S` scans and `R` retakes. With a Scanlight, **R**, **G**, **B** and **IR** pick which exposures Retake repeats.
*   **Capture Delay**: waits before the first exposure, so a shaky table can settle.
*   **Keep Light On** (the bulb): keeps the Scanlight lit after each scan, so you can frame the next shot.
*   **Film** (with a Lento Dock): what the dock does when you click Scan. **Next Frame** moves the film one frame after each scan. **Whole Strip** scans every frame, then ejects the strip; Stop ends it. Retake never moves the film. **Eject** moves the strip out. The dock's S button works as Scan.
*   **Shoot Flat Field**: after a whole strip ejects, shoots the empty dock and saves it as the roll's Flat Field profile.

<!-- panel:scan_light -->
### Preset

Shown while a Scanlight is connected. A preset holds the light levels and the camera exposure for one kind of film.

*   **Preset**: the light and exposure for each scan. **+** adds one in three ways:
    *   **Built-in Presets**: ready to use, with no calibration: **White Light**, **White Light + IR**, **Trichrome** and **Trichrome + IR**. They use the camera's exposure, which the meter sets. White Light needs a Scanlight with a white LED (v4, Big Scanlight).
    *   **Calibrate on the Film Base…**: place the rectangle on clear film and run it. NegPy finds the light levels and the shutter.
    *   **Set by Hand**: set the sliders yourself, then save them as a preset. With no camera connected, it saves no exposure, and a scan uses the camera's own.
*   **Red**, **Green**, **Blue**, **White**: the light levels (0 to 255). **Light On** turns the light on to frame the film, or off.
*   **Capture mode**: **Trichrome** takes one exposure per color and joins them; **Single Exposure** takes one exposure with all three colors on.
*   **Create Sensor Crosstalk Profile** (Single Exposure): also saves a sensor crosstalk profile for this preset (see Calibration, §10.4).
*   **ISO**, **Shutter**, **Aperture**: the preset's exposure. A change holds until you pick the preset again. The gauge button meters.
*   **IR Shutter** and **Infrared** (lights with infrared LEDs): adds an infrared exposure to each scan for dust removal, saved as an `_IR` file. It needs a camera that sees infrared (full-spectrum or converted) and manual focus. Color film only.
*   **Channel Delay**: a pause between the red, green and blue exposures, for cameras that need it.

<!-- panel:scan_output -->
### Output

Shared by both scanners.

*   **Folder**: where scans are saved.
*   **Scan as Roll**: the folder opens as a roll, and frames load as they are saved.
*   **Folder as Roll**: scans go into the folder itself. Off writes into a **Roll** subfolder, and a new Roll name starts a new roll. **+** beside Roll steps to the next name (`Roll001` to `Roll002`).
*   **Format** and **Filename** (film scanner): TIFF, or TIFF (mono) for B&W, and the file name pattern (`{{ date }}`, `{{ seq }}`).

<!-- panel:scan_strip -->
### Strip preview

Shows the whole strip, to place and pick the frames before the scan. **Apply Framing** keeps the framing and **Scan** scans. **Stop Preview** ends a preview early.

*   **Crop**: drag on a frame. **Clear Crops** removes all crops.
*   **Offset**: moves every frame along the film to line it up with the pictures. The white box shows the frame NegPy found.
*   **Drift**: an offset that grows from frame to frame.
*   The slider under each tile moves that frame alone.
*   **Size** sets the tile size. **All**, **None** and each frame's toggle pick frames. **Preview frame** scans one frame again.

---

## 14. Preferences

**⋯** menu → **Preferences…** (`Ctrl+,`). Changes apply at once; a row says when it needs a restart.

### Interface

*   **Language**: English, Simplified Chinese, Japanese, Korean or German, after a restart.
*   **UI scale**: 80% to 120%, after a restart.
*   **Canvas background**: from black to white. Mid gray is best for judging a print.
*   **Sticky Zoom**: keeps the zoom when you change frames. **Reverse Scroll Zoom**: scroll up zooms out.
*   **Shortcut Hints** (on at first): after you use a control with the mouse, a toast on the canvas shows its keyboard shortcut, once per session.
*   **Customize Shortcuts…**, **Edit Toolbar…** and **Reset Panel Layout**.

### Accessibility

*   **Color vision**: colors for marks that differ by color alone (dust overlay, masks, Zone Overlay, clipping). Pick the setting for protanopia or deuteranopia, tritanopia, or achromatopsia.

### Performance

*   **GPU Acceleration**: renders on the graphics card. Off uses the slower CPU, with the same result. If the GPU fails at launch, an amber line here says so.
*   **Search by Meaning**: find frames by description (§2).
*   **Update Thumbnails Automatically**: renders out-of-date thumbnails again in the background, when a roll opens and after a change to many frames. Off (the default), the frames get an amber dot until you use **Update Thumbnails**.
*   **Preview size**: the canvas resolution. Higher is sharper and uses more memory.
*   **Preview cache**, **HQ buffers**, **Rendered frames** and **GPU texture cap**: how much memory NegPy uses. Lower them on a computer with little memory, or if exports run out of GPU memory.
*   **Reduce Export Memory Use**: exports in smaller pieces, for a computer where exports run out of GPU memory.
*   **Show GPU Memory Warning**: tells you when an HQ preview is reduced to fit.

### Session & Storage

*   **Carry Settings Between Frames** and **Persistent Settings…**: what a new frame takes from the last one (§1).
*   **Manage Database…**: sizes, clearing saved edits, and library folders.

### Startup override (`override.toml`)

If NegPy crashes at launch or draws wrongly, edit `Documents/NegPy/override.toml` (made on first run) and restart. Its values win over Preferences.

| Setting | Values | Effect |
|---------|--------|--------|
| `rendering.backend` | `"auto"`, `"vulkan"`, `"dx12"`, `"metal"`, `"cpu"` | How images are processed. `"cpu"` turns the GPU off. |
| `display.qt_rhi_backend` | `"auto"`, `"vulkan"`, `"d3d12"`, `"metal"`, `"opengl"`, `"software"` | How the window is drawn. |
| `display.qt_platform` | `"auto"`, `"xcb"`, `"wayland"` | Window system (Linux only). |
| `performance.force_hq_preview` | `true` / `false` | Turns HQ Preview on or off. |
| `logging.level` | `"debug"`, `"info"`, `"warning"`, `"error"` | How much NegPy logs. Use `"debug"` for a bug report. |

The memory rows of Preferences can be set here too (`max_texture_size`, `preview_render_size`, `preview_cache_max_bytes`, `preview_cache_max_entries`, `preview_cache_max_full_res_entries`, `render_memo_max_entries`).

**Common fixes:**

*   **Crashes at once on Linux**: `backend = "cpu"` or `qt_rhi_backend = "opengl"`.
*   **Black or empty preview on Windows**: `backend = "dx12"` or `qt_rhi_backend = "software"`.
*   **Wayland drawing problems**: `qt_platform = "xcb"`.
*   **GPU out of memory during export**: `max_texture_size = 4096`.

---

## 15. Updating NegPy

NegPy checks for a new version at startup. A new release puts a green dot on **⋯** and adds **Update to vX.Y.Z…** to its **Help** menu. **Check for Updates…** checks by hand.

**Install Update** downloads the new version, closes NegPy, installs it and opens it again. On Windows, approve the administrator prompt *before* NegPy closes. If the update fails, your install stays as it was. Your edits, presets and settings are not touched.

When NegPy cannot update itself, for example when it runs from source, the button reads **Open Releases Page**.

---

## Additional Info

*   Nikon Z 8 and Z 9 NEFs in **HE** or **HE\*** compression cannot be opened. Shoot **Lossless Compressed**, or convert them with Adobe DNG Converter.
*   **Edits** are saved in a local database by file content, so files can move or be renamed. **Edit Sidecars** (Export tab) can also write them as `.negpy` files.
*   NegPy saves an edit when you change frames or export. Closing NegPy before that can lose the last changes.
*   [KEYBOARD.md](KEYBOARD.md) lists every shortcut, [TEMPLATING.md](TEMPLATING.md) explains file name patterns, and [PIPELINE.md](PIPELINE.md) explains the processing in depth.
