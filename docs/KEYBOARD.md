# Keyboard Shortcuts

Every shortcut, including the slider steps, can be changed in the app. Press `?` for the shortcut overlay, then click **Customize**. Slider shortcuts are shown as merged rows (for example **Density ↑/↓**) with a **Step** column. The defaults are listed below.

A slider shortcut reports its new value in the canvas HUD, so you can keep a control on a hidden tab and still read what you set.

A control that the current mode or a lock has retired does not move by keyboard. Temperature on a B&W frame is an example: the shortcut changes nothing, and the HUD shows the control name instead of a value. A control on another tab, in a collapsed section or behind a closed panel still works.

Numpad keys can be bound separately from the number row (for example `Num+9` and `9`). Num Lock must be on for numpad digits.

## Navigation
| Key | Action |
|-----|--------|
| `Left Arrow` | Previous image |
| `Right Arrow` | Next image |

## Triage
| Key | Action |
|-----|--------|
| `K` | Mark frame as keeper |
| `Shift + X` | Reject frame (skipped by batch export and sidecar writes) |

## Image adjustments (high speed)
| Key | Action |
|-----|--------|
| `Q` / `A` | Increase / decrease **Density** (default step 0.01) |
| `W` / `S` | Increase / decrease **Grade** (default step 10 ISO-R) |
| `E` / `D` | Increase / decrease **Magenta** (default step 0.01) |
| `R` / `F` | Increase / decrease **Yellow** (default step 0.01) |
| `X` / `Z` | Increase / decrease **Crop Offset** (default step 1 px) |

## Tools
| Key | Action |
|-----|--------|
| `Shift + W` | Toggle White Balance picker |
| `Shift + C` | Toggle Manual Crop tool |
| `Shift + D` | Toggle Dust Spot picker |
| `Shift + S` | Toggle Scratch tool |
| `Shift + K` | Toggle Transport Line tool |
| `Shift + B` | Toggle Dodge & Burn mask draw |
| `Shift + R` | Toggle Analysis Region draw |
| `Shift + T` | Print the density × grade test strip |
| `Shift + F` | Print the color ring-around (M/Y filtration) |
| `\|` | Peek flat scan (digital intermediate preview) |
| `Esc` | First press clears in-progress points. Second press puts the tool down |

## Geometry and orientation
| Key | Action |
|-----|--------|
| `[` | Rotate 90° CCW |
| `]` | Rotate 90° CW |
| `H` | Flip horizontal |
| `V` | Flip vertical |

While a test strip or ring-around is up, `[` and `]` turn that proof's ladder instead of the image.

## System actions
| Key | Action |
|-----|--------|
| `Ctrl + E` | Export current image |
| `Ctrl + Z` | Undo last change |
| `Ctrl + Y` | Redo change |
| `Ctrl + C` | Copy settings from current image |
| `Ctrl + V` | Paste settings to current image |

## Viewport
| Key | Action |
|-----|--------|
| `Ctrl + [` | Toggle session panel (re-docks when floating) |
| `Ctrl + ]` | Toggle controls panel (re-docks when floating) |
| `Ctrl + Shift + L` | Dock session and controls panels |
| `Mouse Wheel` | Zoom in / out (up to 400%); **Reverse Scroll Zoom** in the toolbar **⋯** menu flips the direction |
| `Middle Click` + `Drag` | Pan zoomed image |
| `Left Click` + `Drag` | Pan zoomed image (when no tool is active) |

## Menu bar (macOS only)

NegPy has a menu bar on macOS, with a **Window** and a **Help** menu. These keys come from it. They are platform window commands, not NegPy actions, so they are fixed and do not appear in the shortcut editor.

| Key | Action |
|-----|--------|
| `⌘ + M` | Minimize the front window |
| `⌘ + W` | Close the front window (closing the main window quits NegPy) |

Every other menu item uses the binding listed above for its action, and shows it only when that binding uses `⌘`. A shortcut bound to a plain key still works from the keyboard; the menu just does not print it, because macOS gives a menu key priority over everything and a plain `?` would fire while you type into a text box.
