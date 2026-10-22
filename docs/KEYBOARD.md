# Keyboard Shortcuts

Every shortcut, including the slider steps, can be changed in the app. Press `?` for the shortcut overlay, then click **Customize**. Slider shortcuts are shown as merged rows (for example **Density ↑/↓**) with a **Step** column. The defaults are listed below.

A slider shortcut reports its new value in the canvas HUD, so you can keep a control on a hidden tab and still read what you set.

A control that the current mode or a lock has retired does not move by keyboard. Temperature on a B&W frame is an example: the shortcut changes nothing, and the HUD shows the control name instead of a value. A control on another tab, in a collapsed section or behind a closed panel still works.

`Ctrl` in this list is `⌘` on macOS. The app writes each key the way your platform writes it, so a menu, a tooltip and the overlay show `⇧⌘C` there and `Ctrl + Shift + C` elsewhere.

Numpad keys can be bound separately from the number row (for example `Num+9` and `9`). Num Lock must be on for numpad digits.

Actions with no default key are not listed; every one of them can still be bound in **Customize**.

<!-- shortcuts:start -->
## Navigation
| Key | Action |
|-----|--------|
| `Left` | Previous file |
| `Right` | Next file |
| `Home` | First file |
| `End` | Last file |
| `Ctrl + L` | Open the Library tab |
| `Ctrl + K` | Find a control, card or action |
| `Ctrl + F` | Focus the Film Strip search box |
| `Ctrl + Shift + F` | Search the whole library and load the matches |

## Triage
| Key | Action |
|-----|--------|
| `K` | Mark frame as keeper |
| `Shift + X` | Reject frame (skipped by batch export) |

## Tools
| Key | Action |
|-----|--------|
| `\` | Before/after split against the default conversion; hold to see the default alone |
| `Shift + W` | Toggle WB picker |
| `Shift + C` | Toggle manual crop |
| `Shift + D` | Toggle heal tool |
| `Shift + S` | Toggle scratch tool |
| `Shift + K` | Toggle transport-scratch line tool |
| `Shift + J` | Toggle clone tool |
| `Shift + B` | Toggle dodge & burn mask draw |
| `Ctrl + R` | Toggle analysis region draw |
| `M` | Peek flat scan (digital intermediate); hold to peek only while down |
| `N` | Peek negative (source as loaded); hold to peek only while down |
| `P` | Peek the camera's embedded preview; hold to peek only while down |
| `Shift + Z` | Adams zone overlay |
| `Shift + T` | Density × grade test strip |
| `Shift + F` | Color ring-around (M/Y filtration) |
| `Shift + L` | Grain focuser loupe |
| `Shift + N` | Printing notes (dodge/burn map + print recipe) |
| `Shift + I` | Probe at cursor (zone, print density and ΔD beside the pointer) |
| `Shift + P` | Soft proof the print on screen |
| `Esc` | Leave the current view (peek, split, strip, zone overlay) or cancel the active tool; Crop and Draw Region undo their changes |

## Geometry
| Key | Action |
|-----|--------|
| `]` | Rotate 90° CW |
| `[` | Rotate 90° CCW |
| `H` | Flip horizontal |
| `V` | Flip vertical |
| `X` / `Z` | Increase / decrease **Crop Offset** (default step 1 px) |
| `Alt + R` / `Alt + Shift + R` | Increase / decrease **Fine Rotation** (default step 0.01°) |
| `L` | Toggle the Level tool |
| `O` | Next crop guide overlay |
| `Shift + O` | Rotate crop guide orientation |
| `Shift + A` | Toggle Auto Crop |

## View
| Key | Action |
|-----|--------|
| `I` | Frame info on the canvas: off, basic or full |
| `Ctrl + [` | Collapse or expand the Film Strip |
| `Ctrl + ]` | Toggle controls panel (re-docks when floating) |
| `Ctrl + Shift + L` | Dock the controls panel and expand the Film Strip |
| `Shift + H` | Hide the controls panel and collapse the Film Strip, or bring both back |
| `Shift + G` | Light Table: the whole roll in rows, in place of the canvas |
| ` + ` | Light Table: larger frames, fewer to a row |
| `-` | Light Table: smaller frames, more to a row |
| `Shift + R` | Reference view: pin this frame beside the canvas to match others to it |
| `0` | Fit to window |
| `1` | Zoom 100% |
| `2` | Zoom 200% |

## Retouch
| Key | Action |
|-----|--------|
| `Alt + 9` / `Alt + Shift + 9` | Increase / decrease **Spot Threshold** (default step 0.01) |
| `Alt + 0` / `Alt + Shift + 0` | Increase / decrease **Size** (default step 1 px) |
| `Alt + M` / `Alt + Shift + M` | Increase / decrease **Brush Size** (default step 1 px) |

## Exposure
| Key | Action |
|-----|--------|
| `E` / `D` | Increase / decrease **Magenta** (default step 0.01) |
| `R` / `F` | Increase / decrease **Yellow** (default step 0.01) |
| `T` / `G` | Increase / decrease **Temperature** (default step 50 K) |
| `Q` / `A` | Increase / decrease **Print Density** (default step 0.01) |
| `W` / `S` | Increase / decrease **ISO-R Grade** (default step 10 ISO-R) |
| `Alt + T` / `Alt + Shift + T` | Increase / decrease **Toe** (default step 0.01) |
| `Alt + Y` / `Alt + Shift + Y` | Increase / decrease **Toe Width** (default step 0.01) |
| `Alt + U` / `Alt + Shift + U` | Increase / decrease **Shoulder** (default step 0.01) |
| `Alt + I` / `Alt + Shift + I` | Increase / decrease **Shoulder Width** (default step 0.01) |
| `Alt + P` / `Alt + Shift + P` | Increase / decrease **White Point** (default step 0.01) |
| `Alt + O` / `Alt + Shift + O` | Increase / decrease **Black Point** (default step 0.01) |

## Actions
| Key | Action |
|-----|--------|
| `Ctrl + E` | Export |
| `Ctrl + C` | Copy settings |
| `Ctrl + Shift + C` | Copy settings (with bounds) |
| `Ctrl + V` | Paste settings |
| `Ctrl + ,` | Open Preferences |
| `Ctrl + Shift + S` | Save the current edit as a named work print |
| `Ctrl + Z` | Undo |
| `Ctrl + Y` | Redo |

## Process
| Key | Action |
|-----|--------|
| `Alt + Q` | Toggle Lock Bounds |
| `Alt + B` / `Alt + Shift + B` | Increase / decrease **Analysis Buffer** (default step 0.01) |
| `Alt + N` / `Alt + Shift + N` | Increase / decrease **Luma Range Clip** (default step 1) |
| `Alt + E` / `Alt + Shift + E` | Increase / decrease **Color Clip** (default step 1) |
| `Alt + 1` / `Alt + Shift + 1` | Increase / decrease **Crosstalk** (default step 0.01) |

## Lab
| Key | Action |
|-----|--------|
| `Alt + 2` / `Alt + Shift + 2` | Increase / decrease **Chroma Denoise** (default step 0.01) |
| `Alt + 3` / `Alt + Shift + 3` | Increase / decrease **Chroma** (default step 0.01) |
| `Alt + 5` / `Alt + Shift + 5` | Increase / decrease **CLAHE** (default step 0.01) |
| `Alt + 6` / `Alt + Shift + 6` | Increase / decrease **Sharpening** (default step 0.01) |
| `Alt + 7` / `Alt + Shift + 7` | Increase / decrease **Glow** (default step 0.01) |
| `Alt + 8` / `Alt + Shift + 8` | Increase / decrease **Halation** (default step 0.01) |

## Toning
| Key | Action |
|-----|--------|
| `Alt + J` / `Alt + Shift + J` | Increase / decrease **Selenium** (default step 0.01) |
| `Alt + K` / `Alt + Shift + K` | Increase / decrease **Sepia** (default step 0.01) |
| `Alt + H` / `Alt + Shift + H` | Increase / decrease **Shadow hue** (default step 0.01) |
| `Alt + G` / `Alt + Shift + G` | Increase / decrease **Shadow strength** (default step 0.01) |
| `Alt + L` / `Alt + Shift + L` | Increase / decrease **Highlight hue** (default step 0.01) |
| `Alt + ;` / `Alt + Shift + ;` | Increase / decrease **Highlight strength** (default step 0.01) |

## Finishing
| Key | Action |
|-----|--------|
| `Alt + V` / `Alt + Shift + V` | Increase / decrease **Vignette burn** (default step 0.01) |
| `Alt + S` / `Alt + Shift + S` | Increase / decrease **Vignette size** (default step 0.01) |
| `Alt + D` / `Alt + Shift + D` | Increase / decrease **Border width** (default step 0.01) |

## Tabs
| Key | Action |
|-----|--------|
| `Ctrl + 1` | Roll tab |
| `Ctrl + 2` | Geometry tab |
| `Ctrl + 3` | Exposure tab |
| `Ctrl + 4` | Look tab |
| `Ctrl + 5` | Finish tab |
| `Ctrl + 6` | History tab |
| `Ctrl + 7` | Export tab |
| `Ctrl + 8` | Metadata tab |
| `Ctrl + 9` | Scan tab |
| `Ctrl + 0` | Favorites tab |

## Camera Live View
| Key | Action |
|-----|--------|
| `S` | Scan, or stop the capture |
| `Space` | Scan, or stop the capture (second key) |
| `R` | Retake the current frame |
| `M` | Meter: set the shutter from one RAW probe |
| `H` | Show the live histogram |
| `Z` | Show clipping zebras |

## Help
| Key | Action |
|-----|--------|
| `?` | Show shortcuts |
<!-- shortcuts:end -->

While a test strip or ring-around is up, `[` and `]` turn that proof's ladder instead of the image. While the paper test strip is up, `←` and `→` step to the neighbor paper instead of the neighbor file, `Enter` keeps the paper shown full frame, and `Esc` goes back to the strip. The first `Esc` clears in-progress points; the second puts the tool down.

## Mouse
| Input | Action |
|-----|--------|
| `Mouse Wheel` | Zoom in / out (up to 400%); **Reverse scroll zoom** in Preferences flips the direction |
| `Alt` + `Mouse Wheel` | Brush Size up / down, for the heal, scratch and exclusion brushes |
| `Pinch` | Zoom in / out; sizes the brush instead while the Heal or Scratch tool is live, or a right-click is set to exclude |
| `Middle Click` + `Drag` | Pan zoomed image |
| `Left Click` + `Drag` | Pan zoomed image (when no tool is active) |
| `Space` + `Left-Drag` | Pan zoomed image while a canvas tool is active |

## Menu bar (macOS only)

NegPy has a menu bar on macOS: **Preferences…** sits in the application menu, and there is a **Window** and a **Help** menu. These keys come from it. They are platform window commands, not NegPy actions, so they are fixed and do not appear in the shortcut editor.

| Key | Action |
|-----|--------|
| `⌘ + M` | Minimize the front window |
| `⌘ + W` | Close the front window (closing the main window quits NegPy) |

Every other menu item uses the binding listed above for its action, and shows it only when that binding uses `⌘`. A shortcut bound to a plain key still works from the keyboard; the menu just does not print it, because macOS gives a menu key priority over everything and a plain `?` would fire while you type into a text box.
