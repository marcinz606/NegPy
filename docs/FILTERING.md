# Filtering the Film Strip

The file panel (left sidebar) has a search box that filters the visible film strip by filename. Sorting, hot folder mode, and "Export All" all operate on the visible (filtered) set.

## Plain Mode (default)

Type any text — case-insensitive substring match against each filename.

| Input | Matches |
|-------|---------|
| `IMG` | `IMG_0001.cr2`, `img_test.NEF` |
| `.cr2` | every `.cr2` file |
| `_42` | files containing `_42` anywhere in the name |

Click the **×** in the box to clear the filter. The strip immediately restores all loaded files.

## Regex Mode

Click the **`.*`** toggle next to the search box to switch to regex mode. The pattern is compiled with `re.IGNORECASE` and matched via `re.search` (anchor with `^` / `$` for full-name match).

| Pattern | Matches |
|---------|---------|
| `^IMG_\d{4}` | `IMG_0001.cr2`, `IMG_0042.NEF` |
| `\.(cr2\|nef)$` | only `.cr2` or `.nef` files |
| `roll_\d+_scan` | files like `roll_3_scan.tif` |

Invalid regex (e.g. unclosed `[`) paints the input border red and leaves the previous filter in place — nothing disappears mid-typing.

## Behavior

- **Selection follows the filter.** Hidden files are dropped from the multi-select set; the active file moves to the first remaining visible selection (or clears if nothing matches). Sync Edits never touches an invisible file.
- **Export All exports only what's visible.** Filter to a subset, click Export All, only that subset writes out.
- **Hot Folder Mode** still ingests new files in the background. New files that don't match the active filter stay hidden until the filter is cleared or relaxed.
- **Sort is preserved.** Filter is applied after sort, so visible order matches your Name/Date + Asc/Desc choice.
- **Filter is session-only.** Closing and reopening NegPy starts with an empty filter.
