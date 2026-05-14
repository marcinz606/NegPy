# Filename Templating

NegPy uses **Jinja2** for dynamic file naming in both the **Export** and **Scan** sidebars.

---

## Export Sidebar

### Available Variables

| Variable | Description | Example Output |
| :--- | :--- | :--- |
| `{{ original_name }}` | Base filename of the source file (without extension). | `DSC0123` |
| `{{ colorspace }}` | Target export color space. | `sRGB`, `Adobe RGB` |
| `{{ format }}` | Export file format. | `JPEG`, `TIFF` |
| `{{ paper_ratio }}` | Selected aspect ratio. | `3:2`, `Original` |
| `{{ size }}` | Print size in cm (Empty if "Original Resolution" is used). | `30cm` |
| `{{ dpi }}` | Export DPI (Empty if "Original Resolution" is used). | `300dpi` |
| `{{ border }}` | Inserts "border" if width > 0, else empty. | `border` |
| `{{ date }}` | Current date in YYYYMMDD format. | `20260125` |

### Examples

| Pattern | Result |
| :--- | :--- |
| `{{ original_name }}` | `DSC0123.jpg` |
| `{{ date }}_{{ original_name }}_{{ colorspace }}` | `20260125_DSC0123_Adobe_RGB.jpg` |
| `{{ original_name }}_{{ size }}_{{ dpi }}_{{ border }}` | `DSC0123_30cm_300dpi_border.jpg` |
| `PRINT_{{ original_name }}_{{ paper_ratio }}` | `PRINT_DSC0123_3:2.jpg` |

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

The sequence starts at `1` for each scan session and increments automatically until a filename that does not yet exist on disk is found. Existing files are **never overwritten**.

---

## Filename Cleanup

Both sidebars apply the same separator cleanup to the rendered template:
*   Spaces, dashes, and underscores between variables are collapsed into a **single underscore** (`_`).
*   Leading or trailing separators are removed.
*   If a variable is empty (like `{{ border }}` when no border is set), surrounding separators are cleaned up automatically.
*   `{{ original_name }}` (export only) is always inserted verbatim — dashes, spaces, and underscores in the original filename are preserved exactly.

**Example:**
Pattern: `{{ original_name }} - {{ border }} - final`
*   With border: `DSC0123_border_final.jpg`
*   No border: `DSC0123_final.jpg`
