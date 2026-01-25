# Export Filename Templating

NegPy uses **Jinja2** to allow dynamic naming of exported files. You can configure this pattern in the **Export** sidebar.

## Available Variables

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

## Examples

| Pattern | Result |
| :--- | :--- |
| `positive_{{ original_name }}` | `positive_DSC0123.jpg` |
| `{{ date }}_{{ original_name }}_{{ colorspace }}` | `20260125_DSC0123_Adobe_RGB.jpg` |
| `{{ original_name }}_{{ size }}_{{ dpi }}_{{ border }}` | `DSC0123_30cm_300dpi_border.jpg` |
| `PRINT_{{ original_name }}_{{ paper_ratio }}` | `PRINT_DSC0123_3:2.jpg` |

## Filename Cleanup

NegPy automatically cleans up the resulting filename to ensure it is valid for your filesystem:
*   Any combination of spaces, dashes, or multiple underscores is collapsed into a **single underscore** (`_`).
*   Trailing or leading underscores are removed.
*   If a variable is empty (like `{{ border }}` when no border is set), it effectively disappears and the surrounding separators are cleaned up.

**Example:**
Pattern: `{{ original_name }} - {{ border }} - final`
*   With border: `DSC0123_border_final.jpg`
*   No border: `DSC0123_final.jpg`
