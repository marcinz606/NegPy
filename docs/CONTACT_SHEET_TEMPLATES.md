# Contact Sheet Templates

The **Contact Sheet** section in the Export sidebar loads layout presets from plain `.toml`
files. **Default** is your in-app baseline layout. It starts at the factory values
600 / 16 / 32 / 38.

---

## Folder

Place template files here:

```
~/NegPy/contact_sheets/
```

On Windows this is typically:

```
C:\Users\<you>\NegPy\contact_sheets\
```

NegPy creates the folder on startup. Click **Save as template** in the app to write a file
from the current layout settings.

---

## File format

Each template is a UTF-8 TOML file with a display name and a `[layout]` table:

```toml
name = "Tight 35mm"

[layout]
cell_px = 400
gap = 8
margin = 16
max_tiles = 48
```

| Key | Meaning | Allowed range |
|---|---|---|
| `cell_px` | Long edge of each tile cell (pixels) | 100–4000 |
| `gap` | Space between cells (pixels) | 0–200 |
| `margin` | Black border around the grid (pixels) | 0–500 |
| `max_tiles` | Frames per sheet before pagination | 1–200 |

An omitted key falls back to the built-in default (600 / 16 / 32 / 38).

The optional top-level `name` field appears in the **Template** dropdown. Without it,
NegPy shows the filename stem (without `.toml`).

---

## Examples

**NegPy factory default (reference only, not required as a file)**

```toml
name = "NegPy default"

[layout]
cell_px = 600
gap = 16
margin = 32
max_tiles = 38
```

**Large cells, fewer per page**

```toml
name = "Large cells"

[layout]
cell_px = 900
gap = 20
margin = 40
max_tiles = 12
```

---

## Behaviour in the app

- Select **Default** to load your saved Default layout. It starts at the factory 600 / 16 / 32 / 38.
- Select a **named template** to load that `.toml` file into the spinboxes.
- Change a spinbox while a template is selected and NegPy updates that template. Default goes
  into the app settings, a named template is rewritten as a `.toml` file. Changes debounce
  about 500 ms.
- **Save as template** creates a **new** named file from the current spinbox values.
- NegPy ignores invalid or unreadable files when it builds the list.
- If you delete a saved template file, the app falls back to **Default** on the next launch.

Templates control the grid layout only. The output folder, the tile rendering and the JPEG
naming do not change.
