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
| `Ctrl + L` | Open the library folder |
| `Alt + Up` | Go up one library folder |
| `Ctrl + F` | Focus the film strip search box |
| `Ctrl + Shift + F` | Search every library folder and load the matches |

## Triage
| Key | Action |
|-----|--------|
| `K` | Mark frame as keeper |
| `Shift + X` | Reject frame (skipped by batch export) |

## Tools
| Key | Action |
|-----|--------|
| `\` | Before/after split (auto baseline) |
| `Shift + W` | Toggle WB picker |
| `Shift + C` | Toggle manual crop |
| `Shift + D` | Toggle heal tool |
| `Shift + S` | Toggle scratch tool |
| `Shift + K` | Toggle transport-scratch line tool |
| `Shift + B` | Toggle dodge & burn mask draw |
| `Shift + R` | Toggle analysis region draw |
| `M` | Peek flat scan (digital intermediate) |
| `N` | Peek negative (source as loaded) |
| `Shift + Z` | Adams zone overlay |
| `Shift + T` | Density × grade test strip |
| `Shift + F` | Color ring-around (M/Y filtration) |
| `Shift + L` | Grain focuser loupe |
| `Shift + N` | Printing notes (dodge/burn map + print recipe) |
| `Shift + P` | Soft proof the print on screen |
| `Esc` | Leave the current view (peek, split, strip) or cancel the active tool |

## Geometry
| Key | Action |
|-----|--------|
| `]` | Rotate 90° CW |
| `[` | Rotate 90° CCW |
| `H` | Flip horizontal |
| `V` | Flip vertical |
| `X` / `Z` | Increase / decrease **Crop offset** (default step 1 px) |
| `Alt + R` / `Alt + Shift + R` | Increase / decrease **Fine rotation** (default step 0.01°) |
| `L` | Toggle straighten line tool |
| `O` | Next crop guide overlay |
| `Shift + O` | Rotate crop guide orientation |
| `Shift + A` | Toggle autocrop |

## View
| Key | Action |
|-----|--------|
| `Ctrl + [` | Toggle session panel (re-docks when floating) |
| `Ctrl + ]` | Toggle controls panel (re-docks when floating) |
| `Ctrl + Shift + L` | Dock session and controls panels |
| `0` | Fit to window |
| `1` | Zoom 100% |
| `2` | Zoom 200% |

## Retouch
| Key | Action |
|-----|--------|
| `Alt + 9` / `Alt + Shift + 9` | Increase / decrease **Threshold** (default step 0.01) |
| `Alt + 0` / `Alt + Shift + 0` | Increase / decrease **Auto size** (default step 1 px) |
| `Alt + M` / `Alt + Shift + M` | Increase / decrease **Brush size** (default step 1 px) |

## Exposure
| Key | Action |
|-----|--------|
| `E` / `D` | Increase / decrease **Magenta** (default step 0.01) |
| `R` / `F` | Increase / decrease **Yellow** (default step 0.01) |
| `T` / `G` | Increase / decrease **Temperature** (default step 50 K) |
| `Q` / `A` | Increase / decrease **Density** (default step 0.01) |
| `W` / `S` | Increase / decrease **Grade** (default step 10 ISO-R) |
| `Alt + T` / `Alt + Shift + T` | Increase / decrease **Toe** (default step 0.01) |
| `Alt + Y` / `Alt + Shift + Y` | Increase / decrease **Toe width** (default step 0.01) |
| `Alt + U` / `Alt + Shift + U` | Increase / decrease **Shoulder** (default step 0.01) |
| `Alt + I` / `Alt + Shift + I` | Increase / decrease **Shoulder width** (default step 0.01) |

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
| `Alt + Q` | Toggle bounds lock |
| `Alt + B` / `Alt + Shift + B` | Increase / decrease **Analysis buffer** (default step 0.01) |
| `Alt + N` / `Alt + Shift + N` | Increase / decrease **Luma range clip** (default step 1) |
| `Alt + E` / `Alt + Shift + E` | Increase / decrease **Color range clip** (default step 1) |
| `Alt + P` / `Alt + Shift + P` | Increase / decrease **White point** (default step 0.01) |
| `Alt + O` / `Alt + Shift + O` | Increase / decrease **Black point** (default step 0.01) |
| `Alt + 1` / `Alt + Shift + 1` | Increase / decrease **Crosstalk** (default step 0.01) |

## Lab
| Key | Action |
|-----|--------|
| `Alt + 2` / `Alt + Shift + 2` | Increase / decrease **Denoise** (default step 0.01) |
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
| `Alt + Semicolon` / `Alt + Shift + Semicolon` | Increase / decrease **Highlight strength** (default step 0.01) |

## Finishing
| Key | Action |
|-----|--------|
| `Alt + V` / `Alt + Shift + V` | Increase / decrease **Vignette burn** (default step 0.01) |
| `Alt + S` / `Alt + Shift + S` | Increase / decrease **Vignette size** (default step 0.01) |
| `Alt + D` / `Alt + Shift + D` | Increase / decrease **Border width** (default step 0.01) |

## Tabs
| Key | Action |
|-----|--------|
| `Ctrl + 1` | Setup tab |
| `Ctrl + 2` | Geometry tab |
| `Ctrl + 3` | Tone tab |
| `Ctrl + 4` | Lab & Toning tab |
| `Ctrl + 5` | Finish tab |
| `Ctrl + 6` | History tab |
| `Ctrl + 7` | Export tab |
| `Ctrl + 8` | Metadata tab |
| `Ctrl + 9` | Scan tab |
| `Ctrl + 0` | Favorites tab |

## Help
| Key | Action |
|-----|--------|
| `?` | Show shortcuts |
<!-- shortcuts:end -->

While a test strip or ring-around is up, `[` and `]` turn that proof's ladder instead of the image. The first `Esc` clears in-progress points; the second puts the tool down.

## Mouse
| Input | Action |
|-----|--------|
| `Mouse Wheel` | Zoom in / out (up to 400%); **Reverse scroll zoom** in Preferences flips the direction |
| `Middle Click` + `Drag` | Pan zoomed image |
| `Left Click` + `Drag` | Pan zoomed image (when no tool is active) |

## Menu bar (macOS only)

NegPy has a menu bar on macOS: **Preferences…** sits in the application menu, and there is a **Window** and a **Help** menu. These keys come from it. They are platform window commands, not NegPy actions, so they are fixed and do not appear in the shortcut editor.

| Key | Action |
|-----|--------|
| `⌘ + M` | Minimize the front window |
| `⌘ + W` | Close the front window (closing the main window quits NegPy) |

Every other menu item uses the binding listed above for its action, and shows it only when that binding uses `⌘`. A shortcut bound to a plain key still works from the keyboard; the menu just does not print it, because macOS gives a menu key priority over everything and a plain `?` would fire while you type into a text box.
