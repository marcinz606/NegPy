# Filtering the Film Strip

The search box in the left sidebar filters the film strip by filename. Sorting, hot folder mode and "Export All" use only the visible (filtered) set.

## Plain mode (default)

NegPy matches the text against each filename and ignores case.

| Input | Matches |
|-------|---------|
| `IMG` | `IMG_0001.cr2`, `img_test.NEF` |
| `.cr2` | every `.cr2` file |
| `_42` | files that contain `_42` anywhere in the name |

Click the **×** in the box to clear the filter.

## Regex mode

Click the **`.*`** toggle beside the search box. NegPy compiles the pattern with `re.IGNORECASE` and matches it with `re.search`. Use `^` and `$` to match the full name.

| Pattern | Matches |
|---------|---------|
| `^IMG_\d{4}` | `IMG_0001.cr2`, `IMG_0042.NEF` |
| `\.(cr2\|nef)$` | only `.cr2` or `.nef` files |
| `roll_\d+_scan` | files like `roll_3_scan.tif` |

An invalid regex (for example an unclosed `[`) turns the input border red and keeps the previous filter.

## Behavior

- NegPy drops hidden files from the multi-select set. The active file moves to the first visible selection, or clears if nothing matches. Sync Edits never changes a hidden file.
- Export All writes only the visible files.
- Hot Folder mode still ingests new files. A new file that does not match the filter stays hidden until you clear or relax the filter.
- The filter applies after the sort, so the visible order follows your Name/Date and Asc/Desc choice.
- The filter lasts for one session.
