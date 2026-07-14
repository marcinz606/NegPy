# Keyboard Shortcuts

Summary of available keyboard shortcuts in NegPy.

All shortcuts, including slider adjustments, can be changed in-app from the `?` shortcut overlay via `Customize`. Slider shortcuts are shown as merged rows (e.g. **Density ↑/↓**) with a customizable **Step** column — defaults match the built-in keyboard increments below.

Numpad keys can be bound separately from the number row (e.g. `Num+9` vs `9`). Num Lock must be on for numpad digits to register.

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

## Image Adjustments (High Speed)
| Key | Action |
|-----|--------|
| `Q` / `A` | Increase / Decrease **Density** (default step 0.01) |
| `W` / `S` | Increase / Decrease **Grade** (default step 10 ISO-R) |
| `E` / `D` | Increase / Decrease **Magenta** (default step 0.01) |
| `R` / `F` | Increase / Decrease **Yellow** (default step 0.01) |
| `X` / `Z` | Increase / Decrease **Crop Offset** (default step 1 px) |

## Tools
| Key | Action |
|-----|--------|
| `Shift + W` | Toggle White Balance Picker |
| `Shift + C` | Toggle Manual Crop Tool |
| `Shift + D` | Toggle Dust Spot Picker |
| `Shift + S` | Toggle Scratch Tool |
| `Shift + B` | Toggle Dodge & Burn Mask Draw |
| `Shift + R` | Toggle Analysis Region Draw |
| `\|` | Peek flat scan (digital intermediate preview) |
| `Esc` | First press clears in-progress points, second puts the tool down |

## Geometry & Orientation
| Key | Action |
|-----|--------|
| `[` | Rotate 90° CCW |
| `]` | Rotate 90° CW |
| `H` | Flip Horizontal |
| `V` | Flip Vertical |

## System Actions
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
| `Mouse Wheel` | Zoom in / out (up to 400%) |
| `Middle Click` + `Drag` | Pan zoomed image |
| `Left Click` + `Drag` | Pan zoomed image (when no tool is active) |
