# Filename Templating

NegPy uses **Jinja2** for dynamic file naming in both the **Export** and **Scan** sidebars.

---

## Export Sidebar

### Available Variables

| Variable | Description | Example Output |
| :--- | :--- | :--- |
| `{{ original_name }}` | Base filename of the source file (without extension). | `DSC0123` |
| `{{ colorspace }}` | Target export color space. | `sRGB`, `Adobe RGB` |
| `{{ format }}` | Export file format (JPEG, TIFF, …). | `JPEG`, `TIFF` |
| `{{ paper_ratio }}` | Selected aspect ratio. | `3:2`, `Original` |
| `{{ size }}` | Print size in cm (Empty if "Original Resolution" is used). | `30cm` |
| `{{ dpi }}` | Export DPI (Empty if "Original Resolution" is used). | `300dpi` |
| `{{ target_px }}` | Target long-edge size in pixels (Empty unless Pixels mode). | `2048px` |
| `{{ border }}` | Inserts "border" if width > 0, else empty. | `border` |
| `{{ date }}` | Current date in YYYYMMDD format. | `20260125` |
| `{{ roll }}` | Scanlight capture roll name (Metadata → Roll), or parsed from a `{roll}_Frame{NNN}` stem. This is not the Roll Analysis normalization name. | `Roll001` |
| `{{ frame }}` | Capture frame number (integer), or parsed from the stem. Unset (`none`) if unknown. Use `{{ frame\|pad(3) }}` or `{{ frame_padded }}` to zero-pad. `"%03d" % frame` works only when frame is set; if it is not, the whole pattern falls back to `original_name`. | `12` |
| `{{ frame_padded }}` | Zero-padded frame (`012`), or empty if unknown. Same as `{{ frame\|pad(3) }}`. | `012` |
| `{{ camera }}` | Camera make + model. | `Mamiya 7` |
| `{{ camera_make }}` / `{{ camera_model }}` | Camera make / model separately. | `Mamiya`, `7` |
| `{{ lens }}` | Lens model (or make if model is empty). | `80mm f/4` |
| `{{ lens_make }}` / `{{ lens_model }}` | Lens make / model separately. | |
| `{{ focal_length }}` | Lens focal length in mm. | `80` |
| `{{ film }}` | Film stock name. | `Portra 400` |
| `{{ film_iso }}` | Film ISO. | `400` |
| `{{ film_manufacturer }}` | Film manufacturer. | `Kodak` |
| `{{ film_color_type }}` | Film color type. | `Color negative` |
| `{{ film_format }}` | Film format (35mm, 120, …). Distinct from export `{{ format }}`. | `35mm` |
| `{{ developer }}` | Developer. | `D-76 1+1` |
| `{{ push_pull }}` | Push/pull as an integer (−3…+3, 0 = Normal). | `1` |
| `{{ scanning }}` | Scanning method note. | `DSLR copy-stand` |
| `{{ exposure }}` | Exposure override text from Metadata. | `1/125s f/2.8` |
| `{{ capture_date }}` | Original capture date in YYYYMMDD. A partial date pads to the first day. Empty if unset. | `19980714` |
| `{{ capture_year }}` | Original capture year. Empty if unset. | `1998` |

Gear and process values come from the **Metadata** panel, or from each file's saved metadata in a batch. An empty field renders as an empty string, so the separators around it collapse. NegPy strips path-unsafe characters from metadata values.

### Examples

| Pattern | Result |
| :--- | :--- |
| `{{ original_name }}` | `DSC0123.jpg` |
| `{{ date }}_{{ original_name }}_{{ colorspace }}` | `20260125_DSC0123_Adobe_RGB.jpg` |
| `{{ original_name }}_{{ size }}_{{ dpi }}_{{ border }}` | `DSC0123_30cm_300dpi_border.jpg` |
| `PRINT_{{ original_name }}_{{ paper_ratio }}` | `PRINT_DSC0123_3:2.jpg` |
| `{{ roll }}_Frame{{ frame\|pad(3) }}_{{ film }}_{{ film_iso }}` | `Roll001_Frame012_Portra_400_400.jpg` |
| `{{ film }}_{{ camera }}_{{ original_name }}` | `Portra_400_Mamiya_7_DSC0123.jpg` |

---

## Scan Sidebar

### Available Variables

| Variable | Description | Example Output |
| :--- | :--- | :--- |
| `{{ date }}` | Current date in YYYYMMDD format. | `20260125` |
| `{{ seq }}` | Sequence number (integer, auto-incremented to avoid overwriting). | `1`, `2`, … |

To zero-pad the sequence number use Python's `%` format operator: `{{ "%03d" % seq }}`.

### Examples

| Pattern | Result |
| :--- | :--- |
| `{{ date }}_{{ "%03d" % seq }}` | `20260125_001.tif` |
| `roll_{{ date }}_{{ seq }}` | `roll_20260125_1.tif` |
| `plustek_{{ date }}_{{ "%04d" % seq }}` | `plustek_20260125_0001.tif` |

### Auto-increment

The sequence starts at `1` for each scan session. It increments until the filename does not yet exist on disk. NegPy **never overwrites** an existing file.

---

## Filename cleanup

Both sidebars apply the same separator cleanup to the rendered template:

*   Spaces, dashes and underscores between variables collapse into a **single underscore** (`_`).
*   Leading and trailing separators are removed.
*   When a variable is empty (`{{ border }}` with no border set), the separators around it are cleaned up.
*   `{{ original_name }}` (export only) is inserted verbatim. Dashes, spaces and underscores in the original filename are kept exactly.

**Example:**
Pattern: `{{ original_name }} - {{ border }} - final`
*   With border: `DSC0123_border_final.jpg`
*   No border: `DSC0123_final.jpg`
