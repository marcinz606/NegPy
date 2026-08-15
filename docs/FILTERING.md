# Filtering the Film Strip

The search box in the left sidebar filters the film strip by filename. Sorting, hot folder mode and "Export All" all use the visible (filtered) set.

## Plain mode (default)

Type any text. NegPy matches it against each filename, and ignores case.

| Input | Matches |
|-------|---------|
| `IMG` | `IMG_0001.cr2`, `img_test.NEF` |
| `.cr2` | every `.cr2` file |
| `_42` | files that contain `_42` anywhere in the name |

Click the **×** in the box to clear the filter. All loaded files come back immediately.

## Regex mode

Click the **`.*`** toggle beside the search box. NegPy compiles the pattern with `re.IGNORECASE` and matches it with `re.search`. Use `^` and `$` to match the full name.

| Pattern | Matches |
|---------|---------|
| `^IMG_\d{4}` | `IMG_0001.cr2`, `IMG_0042.NEF` |
| `\.(cr2\|nef)$` | only `.cr2` or `.nef` files |
| `roll_\d+_scan` | files like `roll_3_scan.tif` |

An invalid regex (for example an unclosed `[`) turns the input border red and keeps the previous filter. Nothing disappears while you type.

## Behaviour

- **The selection follows the filter.** NegPy drops hidden files from the multi-select set. The active file moves to the first visible selection, or clears if nothing matches. Sync Edits never touches an invisible file.
- **Export All exports only what is visible.** Filter to a subset, then click Export All. Only that subset is written.
- **Hot Folder mode still ingests new files.** New files that do not match the active filter stay hidden until you clear or relax the filter.
- **The sort order is kept.** The filter is applied after the sort, so the visible order follows your Name/Date and Asc/Desc choice.
- **The filter is session-only.** NegPy starts with an empty filter.
